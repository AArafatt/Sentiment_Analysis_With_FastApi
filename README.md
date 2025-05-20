# Bangla Social Media Comment Classification

This project aims to classify Bangla social media comments into one of five categories using both a traditional machine learning approach (TF-IDF + Logistic Regression) and a deep learning approach (Bangla-BERT).

## Dataset

* **Source**: Custom dataset of Bangla social media comments
* **Columns**: `comment`, `Category`, `Gender`, `comment react number`, `label`
* **Purpose**: Detect toxic and harmful content in Bangla to enable better moderation

### Classes:

* `sexual`
* `not bully`
* `troll`
* `religious`
* `threat`

## Preprocessing

* Dropped missing and duplicate texts
* Applied Unicode normalization (using BUET NLP `normalizer`)
* Removed URLs, mentions, hashtags, digits, non-Bangla characters
* Applied label encoding and random oversampling for class balance

## Models Implemented

### 1. TF-IDF + Logistic Regression (Baseline)

* Feature extraction: TF-IDF with bigrams
* Classifier: Logistic Regression with balanced class weights

### 2. BanglaBERT (Transformer-based)

* Model: `csebuetnlp/banglabert` from HuggingFace
* Trained with early stopping
* Saved in HuggingFace format and ONNX format

Model Saved Link: https://drive.google.com/drive/folders/1Y4-6ZoRgB_VX0l2rH10obl3VmbZouIvN?usp=sharing

## Evaluation Metrics

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix

## Project Structure

```
project-folder/
│
├── data/                  # Preprocessed train/test CSVs
│   ├── train.csv
│   └── test.csv
│
├── models/                # Trained model + tokenizer + ONNX
│   ├── bangla_comment_classifier/
│   └── model.onnx
│
├── api/                   # FastAPI app
│   └── main.py
│
├── model_training.ipynb  # Jupyter Notebook for training and evaluation
│
├── streamlit_app.py       # Streamlit UI app
│
├── requirements.txt      # Python package requirements
│
├── .gitignore            # Git ignore file
└── README.md             # Project overview and documentation
```

## How to Run the API (FastAPI)

```bash
cd api
uvicorn main:app --reload
```

### Endpoint

**POST** `/predict`

**Request:**

```json
{
  "text": "এই মেয়েটা কি বলছে এসব বাজে কথা?"
}
```

**Response:**

```json
{
  "label": "troll",
  "label_id": 2
}
```

## How to Run Streamlit

```bash
streamlit run streamlit_app.py
```

## Limitations

* Requires large-scale diverse Bangla data for further generalization
* Spelling variations in informal Bangla may still impact results
* Resource-intensive for low-end systems when using BERT

## Future Improvements

* Integrate spell correction and dialect normalization
* Support multi-label and multilanguage inputs
* Optimize ONNX for real-time inference
* Explore adversarial examples and robustness

## License

This project is open for academic/research purposes only.

