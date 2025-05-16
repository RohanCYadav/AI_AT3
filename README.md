# AI_AT3
# 📰 AI_AT3 — Fake News Detection with BERT

This project fine-tunes a BERT-based model to detect whether a news article is **REAL** or **FAKE**, and presents an interactive chatbot UI for prediction and explanation.

Prototype Built for AI Assessment Task 3 using:
- Hugging Face Transformers (`bert-base-uncased`)
- PyTorch
- Scikit-learn for evaluation metrics
- Streamlit for real-time interface
- LIME for local explanation of predictions

---

## 📂 Project Structure

```
AI_AT3/
├── app.py                         # Streamlit chatbot UI
├── main.py                        # BERT training script
├── combined_fake_real_news.csv    # Merged and labeled dataset
├── requirements.txt               # Dependencies for the app
├── training_log.csv               # Epoch-wise loss log
├── loss_curve.png                 # Training/validation loss plot
├── fake_news_bert_model/         # Saved model + tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
└── README.md
```

---

## ⚙️ How to Run the Project

### 1. Download the Prototype zip and unzip into your system

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
### 4. **Download the model** from Google Drive and into the project folder:
   >  [Download]([https://drive.google.com/file/d/1GLVQlTcB396-B2oJawBjntGYiYJ5QZqE/view?usp=sharing])


### 4. Run the Streamlit Chatbot

```bash
streamlit run app.py
```

---

## How It Works

- **Training:** `main.py` trains a `BertForSequenceClassification` model on a binary fake/real news dataset using a PyTorch loop with early stopping and logging.
- **Prediction:** `app.py` loads the model and lets users input news snippets to classify via BERT.
- **Explanation:** LIME highlights the most influential words contributing to the model’s prediction.

---

## Sample Output

**Classification Report (on validation set):**

```
precision    recall  f1-score   support
    FAKE       0.99      0.98      0.99       1000
    REAL       0.98      0.99      0.98       1000
```

## Acknowledgments

- Dataset originally from Kaggle’s Fake and True news datasets
- Pretrained model: [bert-base-uncased](https://huggingface.co/bert-base-uncased)

---

## Authors
