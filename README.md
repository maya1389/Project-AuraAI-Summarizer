# Project-AuraAI-Summarizer
# 🚀 NASA Space Biology Summarization Challenge

This project was developed as part of a **NASA Space Biology Challenge**
and carried out by a dedicated **team of 6 researchers and engineers**.\
The goal of the project is to **automatically summarize scientific
publications** from the **NASA Space Biology dataset**, extracting key
findings, numeric results, and implications for space biology research.

------------------------------------------------------------------------

## 📂 Dataset

We used the
**[SB_publication_PMC.csv](https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv)**
dataset, which contains links to scientific publications in the **PubMed
Central (PMC)** repository related to space biology.

The script automatically downloads the dataset if it is not available
locally.

------------------------------------------------------------------------

## 🛠️ Project Workflow

1.  **Load Dataset**\
    Read publication links from the CSV file.

2.  **Fetch Articles**\
    Retrieve full-text articles from PubMed Central (PMC).

3.  **Extract Content**\
    Identify sections such as **Abstract, Introduction, Results,
    Discussion, Conclusion**.

4.  **Chunk Splitting**\
    Long texts are split into smaller manageable chunks.

5.  **Summarization**

    -   **Abstractive Summarization**: Uses two Hugging Face models:
        -   `facebook/bart-large-cnn` → **High accuracy model**,
            developed by **Facebook** (Meta).\
        -   `sshleifer/distilbart-cnn-12-6` → **Faster inference**, but
            with slightly lower accuracy.\
    -   **Fallback Extractive Summarization**: Uses **TF-IDF** if both
        API calls fail.

6.  **Keyword & Number Extraction**\
    Identify top keywords and numeric highlights (percentages,
    measurements, values).

7.  **Batch Processing**\
    Summarizes multiple publications and stores results in CSV format.

------------------------------------------------------------------------

## 📊 Output

Each processed publication generates:\
- **Title**\
- **Final Summary** (multi-paragraph)\
- **Chunk Summaries**\
- **Grouped Summaries**\
- **Keywords**\
- **Numeric Highlights**\
- **Status & Metadata**

Results are saved in:\
`SB_publication_PMC_RESAULT.csv`

------------------------------------------------------------------------

## 📦 Dependencies

The project requires the following Python libraries:

-   [requests](https://pypi.org/project/requests/)\
-   [beautifulsoup4](https://pypi.org/project/beautifulsoup4/)\
-   [numpy](https://pypi.org/project/numpy/)\
-   [pandas](https://pypi.org/project/pandas/)\
-   [scikit-learn](https://pypi.org/project/scikit-learn/)

Make sure to install them via:

``` bash
pip install requests beautifulsoup4 numpy pandas scikit-learn
```

------------------------------------------------------------------------

## ⚡ Usage

Run the summarization batch on the first *N* publications (default =
607):

``` bash
python main.py
```

------------------------------------------------------------------------

## 🌍 Notes

-   The project integrates with two **Hugging Face Inference API**
    models:
    -   `facebook/bart-large-cnn` (accurate, developed by
        Facebook/Meta)\
    -   `sshleifer/distilbart-cnn-12-6` (faster but slightly less
        accurate)\
-   If the Hugging Face API is unavailable, the fallback **TF-IDF
    extractive method** is used.\
-   Results include both **high-level insights** and **fine-grained
    numeric outcomes** relevant to **NASA Space Biology** research.

------------------------------------------------------------------------

## 👨‍🚀 Team

This project was completed by a **team of 6 members** as part of NASA's
open science and innovation challenges.
