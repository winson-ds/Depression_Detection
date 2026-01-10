# ===============================
# app.py — STREAMLIT CLOUD STABLE
# ===============================

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

torch.set_num_threads(1)

# ===============================
# PAGE CONFIG (HARUS PALING ATAS)
# ===============================
st.set_page_config(page_title="Deteksi Tingkat Depresi", layout="centered")

# ===============================
# STYLE
# ===============================
st.markdown(
    """
<style>
body, .stApp {
    background-color: #0f1115;
    color: #E6E6E6;
    font-family: Inter, sans-serif;
}
textarea {
    background:#1c1f24;
    color:#E6E6E6;
}
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# TITLE
# ===============================
st.title("🔍 Deteksi Tingkat Depresi dari Teks")
st.write("Perbandingan IndoBERT dan XLM-RoBERTa dengan preprocessing Normal & Light.")

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
# NLP
# ===============================
nltk.download("stopwords", quiet=True)
stop_id = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()


def preprocess_normal(text):
    steps = {"Original": text}
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|\d+", "", text)
    text = normalize_slang(text)
    text = reduce_repeat(text)
    text = " ".join(w for w in text.split() if w not in stop_id)
    text = " ".join(stemmer.stem(w) for w in text.split())
    steps["Final"] = text
    return text, steps


def preprocess_light(text):
    steps = {"Original": text}
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    text = normalize_slang(text)
    text = reduce_repeat(text)
    steps["Final"] = text
    return text, steps


# ===============================
# MODEL CONFIG
# ===============================
MODEL_CONFIG = {
    "IndoBERT Normal": {
        "path": "winsonn13/indobert-normal",
        "preprocess": preprocess_normal,
        "arch": "indobert",
    },
    "IndoBERT Light": {
        "path": "winsonn13/indobert-light",
        "preprocess": preprocess_light,
        "arch": "indobert",
    },
    "XLM-R Normal": {
        "path": "winsonn13/xlmroberta-normal",
        "preprocess": preprocess_normal,
        "arch": "xlmr",
    },
    "XLM-R Light": {
        "path": "winsonn13/xlmroberta-light",
        "preprocess": preprocess_light,
        "arch": "xlmr",
    },
}

LABELS = ["Tidak Depresi", "Depresi Ringan", "Depresi Sedang", "Depresi Berat"]
COLORS = ["#1b5e20", "#f9a825", "#ef6c00", "#c62828"]

# ===============================
# UI
# ===============================
model_choice = st.radio("Model", list(MODEL_CONFIG.keys()))
user_text = st.text_area("Masukkan teks", height=160)

# ===============================
# MODEL STORE (AMAN)
# ===============================
if "models" not in st.session_state:
    st.session_state.models = {}


def load_model(model_key):
    if model_key not in st.session_state.models:
        cfg = MODEL_CONFIG[model_key]
        with st.spinner("Memuat model..."):
            if cfg["arch"] == "xlmr":
                tokenizer = AutoTokenizer.from_pretrained(
                    "xlm-roberta-base", use_fast=False
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(cfg["path"], use_fast=False)

            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["path"],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.eval()

            st.session_state.models[model_key] = (tokenizer, model)

    return st.session_state.models[model_key]


# ===============================
# RUN
# ===============================
if st.button("Analisis"):
    if not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
        st.stop()

    cfg = MODEL_CONFIG[model_choice]
    tokenizer, model = load_model(model_choice)

    processed, steps = cfg["preprocess"](user_text)

    with torch.no_grad():
        enc = tokenizer(processed, return_tensors="pt", truncation=True, max_length=256)
        probs = torch.softmax(model(**enc).logits, dim=1)[0].numpy()
        pred = int(np.argmax(probs))

    st.markdown(
        f"<div style='background:{COLORS[pred]};padding:12px;border-radius:12px;"
        f"color:white;font-weight:700'>{LABELS[pred]}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Confidence")
    st.write(float(probs[pred]))

    st.subheader("Preprocessing")
    for k, v in steps.items():
        st.code(f"{k}: {v}")

    st.caption("⚠️ Alat skrining, bukan diagnosis klinis.")
