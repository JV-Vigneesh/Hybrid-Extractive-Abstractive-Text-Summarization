# 🧠 Hybrid Extractive-Abstractive Text Summarization

A web-based NLP application that generates high-quality summaries by combining **extractive (TF-IDF)** and **abstractive (Transformer/BART)** techniques. The system produces concise, coherent, and context-aware summaries from raw text or uploaded documents.

---

## 🚀 Features

* 🔹 Hybrid Summarization (TF-IDF + BART)
* 🔹 Extractive Summary (TF-IDF based)
* 🔹 Abstractive Summary (Transformer-based)
* 🔹 Redundancy Removal using Semantic Similarity
* 🔹 Key Points Extraction
* 🔹 Multi-format Input Support:

  * TXT
  * PDF
  * DOCX
* 🔹 Evaluation Metrics:

  * ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
  * BERTScore (semantic similarity)
* 🔹 Interactive Web Interface (Flask + Bootstrap)
* 🔹 Summary Download Option

---

## 🧠 System Architecture

```
Input (Text / File)
        ↓
Text Extraction
        ↓
Preprocessing (Noise Removal, Normalization)
        ↓
Sentence Tokenization
        ↓
TF-IDF Sentence Ranking
        ↓
Top Sentence Selection
        ↓
BART Abstractive Summarization
        ↓
Redundancy Removal (Sentence Embeddings)
        ↓
Final Summary + Evaluation
```

---

## ⚙️ Technologies Used

* Python
* Flask (Web Framework)
* NLTK
* Scikit-learn (TF-IDF)
* Hugging Face Transformers (BART)
* Sentence Transformers (MiniLM)
* ROUGE Score
* BERTScore
* HTML, CSS, Bootstrap
* Chart.js

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hybrid-text-summarization.git
cd hybrid-text-summarization
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000/
```

---

## 📊 Sample Output

* Original Length: ~1000 words
* Summary Length: ~100 words
* Reduction: ~85–90%
* BERTScore: ~0.88–0.90

---

## 📁 Project Structure

```
Hybrid-Text-Summarization/
│
├── app.py
├── utils.py
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── uploads/
```

---

## 📌 Key Modules

* `tfidf_summary()` → Extractive summarization
* `bart_summary()` → Abstractive summarization
* `hybrid_summary()` → Combined approach
* `remove_redundancy()` → Semantic filtering
* `compute_rouge()` → ROUGE evaluation
* `compute_bertscore()` → Semantic evaluation

---

## ⚠️ Notes

* Internet is required only for the first run (model download).
* After that, the system works offline.
* GPU support is optional but improves performance.

---

## 🔮 Future Enhancements

* Multilingual summarization
* Fine-tuned transformer models
* Real-time API integration
* Improved evaluation metrics
* Deployment on cloud platforms
