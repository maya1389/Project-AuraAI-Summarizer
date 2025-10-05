from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
from main import run_batch, OUTPUT_CSV  # از main.py می‌گیریم

app = Flask(__name__)

@app.route('/')
def index():
    # نمایش فایل HTML رابط کاربری
    return render_template('Article Summarizer3.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file or not file.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a valid CSV file.'}), 400

    # ذخیره ورودی
    input_path = "uploaded.csv"
    file.save(input_path)

    # اجرای پردازش با main.py
    run_batch(first_n=3)  # مثلاً 3 تا لینک اول رو خلاصه کن

    # خواندن خروجی نهایی
    df = pd.read_csv(OUTPUT_CSV)
    results = df[['url', 'title', 'final_summary']].to_dict(orient='records')

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
