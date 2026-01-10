import streamlit as st
import torch
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import gc
from transformers import XLMRobertaForSequenceClassification


torch.set_num_threads(1)

# ===============================
# PAGE
# ===============================
st.set_page_config(page_title="Deteksi Depresi", layout="centered")
st.title("🔍 Deteksi Tingkat Depresi")

# ===============================
# STYLE (CSS)
# ===============================
st.markdown(
    """
<style>
body, .stApp {
    background-color: #0f1115 !important;
    color: #E6E6E6 !important;
    font-family: Inter, sans-serif !important;
}

h1, h2, h3 {
    font-weight: 700 !important;
    color: #E6E6E6 !important;
}

.stRadio > div[role="radiogroup"] {
    display: flex !important;
    gap: 10px !important;
}

.stRadio > div[role="radiogroup"] > label {
    padding: 10px 22px !important;
    border-radius: 40px !important;
    background: #1c1f24 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #dfe7fd !important;
    font-weight: 600 !important;
}

.stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background:#4c4fed !important;
    color:white !important;
}

textarea {
    background:#1c1f24 !important;
    color:#E6E6E6 !important;
    border-radius:10px !important;
}

.stButton > button {
    background:#1c1f24 !important;
    color:#E6E6E6 !important;
    border-radius:10px !important;
    font-weight:600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# TITLE
# ===============================
st.title("🔍 Deteksi Tingkat Depresi dari Teks")
st.write(
    "Perbandingan **IndoBERT** dan **XLM-RoBERTa** dengan preprocessing **Normal** dan **Light**."
)

# ===============================
# SLANG
# ===============================
BASE_DIR = Path(__file__).parent
SLANG_PATH = BASE_DIR / "tweet" / "000_colloquial-indonesian-lexicon.csv"


@st.cache_data
def load_slang():
    df = pd.read_csv(SLANG_PATH, usecols=["slang", "formal"])
    return dict(zip(df.slang, df.formal))


slang = load_slang()


def normalize_slang(t):
    return " ".join(slang.get(w, w) for w in t.split())


def reduce_repeat(t):
    return re.sub(r"(.)\1{2,}", r"\1", t)


# ===============================
# NLP SETUP
# ===============================
nltk.download("stopwords", quiet=True)
stop_id = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()


def preprocess_normal(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|\d+", "", text)
    text = normalize_slang(text)
    text = reduce_repeat(text)
    text = " ".join(w for w in text.split() if w not in stop_id)
    text = " ".join(stemmer.stem(w) for w in text.split())
    return text


def preprocess_light(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    text = normalize_slang(text)
    text = reduce_repeat(text)
    return text


# ===============================
# MODEL CONFIG
# ===============================
MODEL_PATHS = {
    "IndoBERT": {
        "Normal": ("winsonn13/indobert-normal", preprocess_normal),
        "Light": ("winsonn13/indobert-light", preprocess_light),
    },
    "XLM-RoBERTa": {
        "Normal": ("winsonn13/xlmroberta-normal", preprocess_normal),
        "Light": ("winsonn13/xlmroberta-light", preprocess_light),
    },
}

LABELS = ["Tidak Depresi", "Depresi Ringan", "Depresi Sedang", "Depresi Berat"]
COLORS = ["#1b5e20", "#f9a825", "#ef6c00", "#c62828"]

# ===============================
# UI INPUT
# ===============================
arch = st.radio("Model", ["IndoBERT", "XLM-RoBERTa"], horizontal=True)
prep = st.radio("Preprocessing", ["Normal", "Light"], horizontal=True)
user_text = st.text_area("Masukkan teks", height=160)


# ===============================
# MODEL
# ===============================
@st.cache_resource
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


# ===============================
# RUN
# ===============================
if st.button("Analisis"):
    if not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
        st.stop()

    model_path, preprocess_fn = MODEL_PATHS[arch][prep]

    tok, model = load_model(model_path)

    processed = preprocess_fn(user_text)

    with torch.no_grad():
        enc = tok(processed, return_tensors="pt", truncation=True, max_length=256)
        probs = torch.softmax(model(**enc).logits, dim=1)[0].cpu().numpy()
        pred = int(np.argmax(probs))

    st.markdown(
        f"<div style='background:{COLORS[pred]};"
        f"padding:12px;border-radius:12px;"
        f"color:white;font-weight:700'>"
        f"{LABELS[pred]}</div>",
        unsafe_allow_html=True,
    )

    st.write("Confidence:", float(probs[pred]))
