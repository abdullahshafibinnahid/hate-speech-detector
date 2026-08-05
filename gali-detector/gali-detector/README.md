# Multilingual Gali Detector (Bangla + English)

Hate speech detection for Bangla, English, and Banglish (mixed) comments.

## Dataset
- `full_mixed_dataset.csv` - 65,007 rows
  - Merged from `train.csv` (BDSHS - 40,224 Bangla) + `labeled_data.csv` (24,783 English)
  - Columns: `text` (comment), `label` (0=NOT_HATE, 1=HATE)

## Project Structure
```
.
├── full_mixed_dataset.csv      # Full merged dataset
├── logistic_model.py           # Baseline: TF-IDF + Logistic Regression (~80% acc)
├── bert_model.py               # Best: Bangla BERT (sagorsarker/bangla-bert-base) (~90%+ acc)
├── app.py                      # FastAPI inference API
├── requirements.txt
├── README.md
└── .gitignore
```

## 1. Baseline Model (CPU)
```bash
pip install -r requirements.txt
python logistic_model.py
```
This will generate `gali_detector_model.pkl` and `tfidf_vectorizer.pkl`

## 2. Bangla BERT Model (Kaggle - GPU Recommended)
1. Create new Kaggle Notebook with GPU T4 x2
2. Upload `full_mixed_dataset.csv` and `bert_model.py`
3. Run:
```bash
pip install transformers datasets -q
python bert_model.py
```
4. Output folder `bangla-bert-gali-final` contains the final model

## 3. API
```bash
uvicorn app:app --reload
```
Open: http://localhost:8000/docs

Example:
```json
POST /predict
{
  "text": "tui ekta faltu"
}
Response:
{
  "prediction": "HATE",
  "confidence": 0.92
}
```

## Labels
- 0 = NOT_HATE / VALO
- 1 = HATE / GALI

## GitHub Push
```bash
git init
git add .
git commit -m "Add multilingual hate speech detector"
git branch -M main
git remote add origin https://github.com/USERNAME/gali-detector.git
git push -u origin main
```
