import requests as req
import os
import time
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


HF_TOKEN = "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"
MODEL_NAME = "facebook/bart-large-cnn"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

N_FIRST_LINKS = 607
INPUT_CSV = "SB_publication_PMC.csv"
OUTPUT_CSV = "SB_publication_PMC_RESAULT.csv"


SLEEP_BETWEEN_CALLS = 2.0
REQUEST_TIMEDOUT = 120
MAX_RETRIES = 5

CHUNK_MAX_CHAR = 1800
CHUNK_SUM_MAXTOK = 180
CHUNK_SUM_MINTOK = 60
FINAL_SUM_MAXTOK = 300
FINAL_SUM_MINTOK = 120

GROUP_SIZE = 10   


def robust_print(*args , **kwargs ):
    print(f"[{datetime.utcnow().isoformat()}]" ,*args , **kwargs)

def read_csv_or_download(path_or_empty):
    if path_or_empty and Path(path_or_empty).exists():
        robust_print(f"Reading CSV from local file:{path_or_empty}")
        return pd.read_csv(path_or_empty)
    raw_url = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"
    robust_print("Downloading csv from github" , raw_url)
    r = req.get(raw_url , timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def fetch_pmc_soup(url):
    headers = {"User-Agent":"Mozilla/0.5 (compatible; batch-summarizer/1.0)"}
    r = req.get(url , headers=headers , timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text , "html.parser")


def extract_metadata_and_sections(soup):
    title_tag = soup.find(['h1','h2'] , attrs = {'class': re.compile(r'.*ArticleTitle.*' , re.I)})\
                or soup.find('h1') or soup.find('title')
    title = title_tag.get_text(" " , strip = True) if title_tag else ""

    abstract_text = ""
    abs_div = soup.find('div' , class_ = re.compile(r'abstract' , re.I))
    if abs_div:
        abstract_text = " ".join([p.get_text(" " , strip = True) for p in abs_div.find_all(['p' , 'div'])])\
                        or abs_div.get_text(" " , strip = True)
    def get_section_text_by_keywords(keywords):
        headers = soup.find_all(['h1' , 'h2' , 'h3' , 'h4'] , strip = True)
        matched = []
        for h in headers:
            txt = h.get_text(" " , strip = True). lower()
            if any(kw in txt for kw in keywords):
                matched.append(h)
        collected = []
        for h in matched :
            for sib in h.next_siblings:
                if getattr(sib, "name", None) and sib.name.startswith('h'):
                    break
                if getattr(sib , "get_text" , None):
                    s = sib.get_text(" ", strip = True)
                    if len(s) > 30 :
                        collected.append(s)
            return "\n".join(collected)
    results = get_section_text_by_keywords(['result' , 'results' , 'findings' ])
    discussion = get_section_text_by_keywords(['discussion', 'interpretation'])
    conclusion = get_section_text_by_keywords(['conclusion' , "conclusions" , 'summary'])
    introduction = get_section_text_by_keywords(['introduction' , 'introductions' , 'introduce'])

    if not(abstract_text or results or conclusion or discussion or introduction):
        body = soup.find ('div' , id = 'maincontent') or soup
        paras = [p.get_text(" " , strip = True) for p in body.find_all('p') if len(p.get_text(" " , strip = True)) > 40]
        return {"title": title, "selected_text": "\n".join(paras)}
    parts = []
    if abstract_text : parts.append("Abstract:" + abstract_text)
    if results : parts.append("results:" + results)
    if conclusion : parts.append("conclusion:" + conclusion)
    if introduction : parts.append(" introdution:" + introduction)
    return {"title " : title , "selected_text" : "\n\n".join(parts)}


def simple_text_tokenize(text):
    return re.split(r'(?<=[\.\!\?])\s+',text.strip())

def split_text_to_chunks(text , max_chars = CHUNK_MAX_CHAR):
    sents = simple_text_tokenize(text)
    chunks , cur = [] , " "
    for s in sents :
        if len(cur) + len(s) + 1 <= max_chars:
          cur +=(" " + s) if cur else s
        else :
            if cur : chunks.append(cur)
            cur = s
    if cur : chunks.append(cur)

    return chunks

def hf_summarize_one(text , max_length = CHUNK_SUM_MAXTOK , min_length = CHUNK_SUM_MINTOK):
    if not HF_TOKEN:
        raise RuntimeError("Token not found, go get it from the Hugging Face site")
    payload = {
        "inputs" : text ,
        "parameters" : {"max_length" : max_length , "min_length" : min_length , "do_sample" : False} 
    }
    attempt = 0
    while attempt < MAX_RETRIES:
        try :
            r = req.post(API_URL , headers=headers , json=payload , timeout=REQUEST_TIMEDOUT)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data , list) and "summary_text" in data[0]:
                    return data[0]["summary_text"]
                if isinstance(data , dict ) and data.get("error"):
                    raise RuntimeError("Token not found , NO summaries" + str(data.get("error")))
                
                
                return str(data)
            else : 
                r.raise_for_status()
        except Exception as e :
            attempt += 1
            wait = 2 ** attempt
            robust_print(f"HF call failed ({attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s ...", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Unfortunately, I tried my best, but it didn't work. Come back later and see what I can do for you.😢")

from sklearn.feature_extraction.text import TfidfVectorizer
def extractive_summary_tfidf(text , n_sentences = 8):
    sents = simple_text_tokenize(text)
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vect = TfidfVectorizer(stop_words='english')
    X = vect.fit_transform(sents)
    scores = X.sum(axis = 1).A1
    idx = np.argsort(-scores)[:n_sentences]
    idx_sorted = sorted(idx)
    return " ".join([sents[i] for i in idx_sorted])

def extract_keyword(text , top_n = 10):
    try : 
        v = TfidfVectorizer(stop_words='english' , ngram_range=(1,2) , max_features=500)
        X = v.fit_transform([text])
        arr = X.toarray()[0]
        if arr.sum() == 0:
            return []
        top_idx = arr.argsort()[-top_n:][::-1]
        feats = np.array(v.get_feature_names_out())
        return feats[top_idx].tolist()
    except Exception:
        return []
    
def extract_numberic_highlights(text , max_items = 10):
    # 1 / 5/ 67 / 234 / numbers / % / + / - / * ,....
    items = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:[A-Za-z]{1,6}|mmHg|mg|kg|g|cm|µm)?', text)
    seen = []
    for it in items:
        it = it.strip()
        if it not in seen :
            seen.append(it)
        if len(seen) >= max_items:
            break
    return seen


def summarize_long_article_pipline(url):
    start_time = datetime.utcnow()
    result = {
        "url": url,
        "title": "",
        "num_chunks": 0,
        "extracted_text_length": 0,
        "chunk_summaries": [],
        "group_summaries": [],
        "final_summary": "",
        "keywords": [],
        "numeric_highlights": [],
        "status": "pending",
        "error": "",
        "started_at": start_time.isoformat(),
        "finished_at": ""
    }

    try:
        soup = fetch_pmc_soup(url)
        meta = extract_metadata_and_sections(soup)
        text = meta.get("selected_text" , "")
        if not text or len (text.strip()) < 200 :
            body = soup.find('div' , id = 'maincontent') or soup
            paras = [p.get_text(" " , strip = True) for p in body.find_all('p') if len(p.get_text(" " , strip = True)) > 40]
            text = "\n".join(paras)

        result['title'] = meta.get("title" , "")
        result["extracted_text_length"] = len(text)

        chunks = split_text_to_chunks(text , max_chars=CHUNK_MAX_CHAR)
        result["num_chunk"] = len(chunks)
        robust_print(f"Title: {result['title'][:120]}... Len={len(text)} chars -> {len(chunks)} chunks")

       
        for i , ch in enumerate(chunks , start = 1):
            robust_print(f'summarizing chunks {i} /{len(chunks)}(chars = {len(ch)})')
            try :
                prefix = ("خلاصه کردن متن برای هر پاراگراف \n\n\n")
                chunk_sum = hf_summarize_one(prefix + ch , max_length=CHUNK_SUM_MAXTOK , min_length=CHUNK_SUM_MINTOK)
            except Exception as e : 
                robust_print(f"hf fail for chunk{i} : {e}. come back" , file = sys.stderr)
                chunk_sum = extractive_summary_tfidf(ch , n_sentences= 8)
            result["chunk_summaries"].append(chunk_sum)
            time.sleep(SLEEP_BETWEEN_CALLS)

        
        grouped = []
        if len(result["chunk_summaries"]) > GROUP_SIZE:
            for i in range(0 , len(result['chunk_summaries']) , GROUP_SIZE):
                group_text = "\n".join(result["chunk_summaries"][i:i+GROUP_SIZE])
                robust_print(f" Grouping chunk summaries {i+1}-{min(i+GROUP_SIZE, len(result['chunk_summaries']))} => summarizing group")
                try :
                    gsum = hf_summarize_one("summary of each chunks :\n\n\n" + group_text , max_length=180 , min_length=60)
                except Exception as e:
                    gsum = extractive_summary_tfidf(group_text,n_sentences=8)
                grouped.append(gsum)
                result["group_summaries"].append(gsum)
                time.sleep(SLEEP_BETWEEN_CALLS)
            combined_for_final = "\n".join(grouped)
        else:
            combined_for_final = "\n".join(result["chunk_summaries"])


       
        robust_print(" Doing final summarization of combined text ...")
        try:
            final_prefix = ("Produce a coherent multi-paragraph detailed summary from the input. "
                            "Include main findings, numeric outcomes, and implications for space biology.\n\n")
            final_summary = hf_summarize_one(final_prefix + combined_for_final,
                                            max_length=FINAL_SUM_MAXTOK, min_length=FINAL_SUM_MINTOK)
        except Exception as e:
            robust_print(f" HF failed for final summary: {e}. Using extractive fallback.", file=sys.stderr)
            final_summary = extractive_summary_tfidf(combined_for_final, n_sentences=12)

        result["final_summary"] = final_summary
        
        result["keywords"] = extract_keyword(text, top_n=12)
        result["numeric_highlights"] = extract_numberic_highlights(text, max_items=12)

        result["status"] = "done"
        result["finished_at"] = datetime.utcnow().isoformat()
    except Exception as ex:
        robust_print(f" ERROR processing {url}: {ex}", file=sys.stderr)
        result["status"] = "error"
        result["error"] = str(ex)
        result["finished_at"] = datetime.utcnow().isoformat()
    return result



def run_batch(first_n=N_FIRST_LINKS):
    df = read_csv_or_download(INPUT_CSV)

    possible_cols = [c for c in df.columns if 'pmc' in c.lower() or 'url' in c.lower() or 'link' in c.lower()]
    if not possible_cols:
        raise RuntimeError(f"Couldn't find URL-like column in CSV. Columns: {df.columns.tolist()}")
    url_col = possible_cols[0]
    urls = df[url_col].dropna().unique().tolist()
    robust_print(f"Found {len(urls)} unique URLs in column '{url_col}'. Will process first {first_n}.")

    out_path = Path(OUTPUT_CSV)
    if out_path.exists():
        existing = pd.read_csv(out_path)
    else:
        existing = pd.DataFrame()

    results = []
    for i, url in enumerate(urls[:first_n], start=1):
        robust_print(f"\n=== Processing {i}/{first_n}: {url} ===")
        if not existing.empty and url in existing['url'].astype(str).values:
            robust_print("  Skipping because already present in output CSV.")
            continue
        res = summarize_long_article_pipline(url)
        res_for_csv = res.copy()
        res_for_csv["chunk_summaries"] = json.dumps(res_for_csv["chunk_summaries"], ensure_ascii=False)
        res_for_csv["group_summaries"] = json.dumps(res_for_csv["group_summaries"], ensure_ascii=False)
        res_for_csv["keywords"] = json.dumps(res_for_csv["keywords"], ensure_ascii=False)
        res_for_csv["numeric_highlights"] = json.dumps(res_for_csv["numeric_highlights"], ensure_ascii=False)
    
        try:
            if existing.empty:
                pd.DataFrame([res_for_csv]).to_csv(out_path, index=False, encoding='utf-8')
                existing = pd.read_csv(out_path)
            else:
                pd.DataFrame([res_for_csv]).to_csv(out_path, index=False, header=False, mode='a', encoding='utf-8')
                existing = pd.read_csv(out_path)
            robust_print(f"  Saved result to {out_path} (status={res['status']})")
        except Exception as e:
            robust_print(f"  Failed to write CSV: {e}", file=sys.stderr)
        time.sleep(1.0)

    robust_print("Batch finished.")


if __name__ == "__main__":
    run_batch(N_FIRST_LINKS)
    
