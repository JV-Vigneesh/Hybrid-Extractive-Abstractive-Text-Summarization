# ==============================
# Hybrid Text Summarization Project
# ==============================

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from rouge_score import rouge_scorer

# -----------------------------
# Download tokenizer (only first time)
# -----------------------------
nltk.download('punkt')

# -----------------------------
# LOAD MODEL ONCE (IMPORTANT)
# -----------------------------
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",  # lightweight (good for 16GB laptop)
    device=-1
)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s) > 20]

# -----------------------------
# TF-IDF Ranking
# -----------------------------
def rank_sentences_tfidf(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    scores = tfidf_matrix.sum(axis=1)
    scores = np.array(scores).flatten()

    ranked = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    return ranked

# -----------------------------
# Extractive Summary
# -----------------------------
def tfidf_only_summary(text, k=3):
    sentences = preprocess_text(text)
    ranked = rank_sentences_tfidf(sentences)
    return " ".join([s for _, s in ranked[:k]])

# -----------------------------
# Abstractive Summary (BART)
# -----------------------------
def abstractive_summary(text):
    input_len = len(text.split())

    max_len = int(input_len * 0.6)
    min_len = int(input_len * 0.3)

    max_len = max(30, min(max_len, 120))
    min_len = max(10, min(min_len, 60))

    result = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return result[0]['summary_text']

# -----------------------------
# Hybrid Summary
# -----------------------------
def hybrid_summarize(text, k=3):
    sentences = preprocess_text(text)

    if len(sentences) == 0:
        return "Text too short."

    ranked = rank_sentences_tfidf(sentences)
    extracted = " ".join([s for _, s in ranked[:k]])

    return abstractive_summary(extracted)

# -----------------------------
# ROUGE Evaluation
# -----------------------------
def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, generated)

    return {
        "ROUGE-1": scores['rouge1'].fmeasure,
        "ROUGE-2": scores['rouge2'].fmeasure,
        "ROUGE-L": scores['rougeL'].fmeasure
    }

# -----------------------------
# MAIN TEST
# -----------------------------
if __name__ == "__main__":

    text = """
    Artificial Intelligence is transforming industries across the globe.
    It enables machines to learn from data and perform tasks that traditionally required human intelligence.
    Machine learning and deep learning are subsets of AI that have gained significant attention.
    Applications include natural language processing, computer vision, and robotics.
    However, challenges such as data privacy, ethical concerns, and model interpretability remain critical.
    Researchers are actively working to address these issues while improving model performance.
    """

    reference_summary = """
    Artificial Intelligence enables machines to perform intelligent tasks.
    It is widely used in NLP, computer vision, and robotics.
    Challenges like ethics and privacy remain important.
    """

    print("\n==============================")
    print("TF-IDF SUMMARY")
    print("==============================\n")
    tfidf_sum = tfidf_only_summary(text)
    print(tfidf_sum)

    print("\n==============================")
    print("BART SUMMARY")
    print("==============================\n")
    bart_sum = abstractive_summary(text)
    print(bart_sum)

    print("\n==============================")
    print("HYBRID SUMMARY")
    print("==============================\n")
    hybrid_sum = hybrid_summarize(text)
    print(hybrid_sum)

    print("\n==============================")
    print("ROUGE SCORES")
    print("==============================")

    models = {
        "TF-IDF": tfidf_sum,
        "BART": bart_sum,
        "HYBRID": hybrid_sum
    }

    for name, summary in models.items():
        scores = compute_rouge(reference_summary, summary)

        print(f"\n{name}:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")