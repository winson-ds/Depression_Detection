import streamlit as st
import torch
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import warnings
import gc
import os

# =========================
# BASIC SAFETY
# =========================
torch.set_num_threads(1)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Cache dir (penting supaya tidak download ulang)
BASE_DIR = Path(__file__).parent
HF_CACHE = BASE_DIR / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)

# =========================
# SESSION STATE INIT
# =========================
if "current_model_path" not in st.session_state:
    st.session_state.current_model_path = None
    st.session_state.tokenizer = None
    st.session_state.model = None

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Deteksi Depresi", layout="centered")

st.markdown(
    """
<style>

body, .stApp {
    background-color: #0f1115 !important;
    color: #E6E6E6 !important;
    font-family: Inter, sans-serif !important;
}

/* Titles */
h1, h2, h3, h4 {
    font-weight: 700 !important;
    color: #E6E6E6 !important;
}


/* Horizontal pill layout */
.stRadio > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 10px !important;
}

/* pill base */
.stRadio > div[role="radiogroup"] > label {
    padding: 10px 22px !important;
    border-radius: 40px !important;
    background: #1c1f24 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #dfe7fd !important;
    cursor: pointer !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-align:center !important;
    display:flex !important;
    align-items:center !important;
    justify-content:center !important;
    min-width:100px !important;
    transition:0.15s ease-in-out;
}

/* HOVER */
.stRadio > div[role="radiogroup"] > label:hover {
    background:#2a3038 !important;
}

/* ACTIVE (FIXED & CLEAN) */
.stRadio > div[role="radiogroup"] > label:has(input:checked) {
    background:#4c4fed !important;       /* ungu / biru sesuai tema Anda */
    border-color:#4c4fed !important;
    color:white !important;
}

/* Spacing between label text and radio */
.stMarkdown h3 {
    margin-bottom: 8px !important;
}

/* Textarea and input */
textarea, input {
    background:#1c1f24 !important;
    border:1px solid rgba(255,255,255,0.10) !important;
    color:#E6E6E6 !important;
    border-radius:10px !important;
}

/* Buttons */
.stButton > button {
    background:#1c1f24 !important;
    border:1px solid rgba(255,255,255,0.12) !important;
    color:#E6E6E6 !important;
    border-radius:10px !important;
    padding:8px 20px !important;
    font-weight:600 !important;
}
.stButton > button:hover {
    background:#2a3038 !important;
}

/* Preprocessing box */
.pre-box {
    background:#1c1f24 !important;
    border:1px solid rgba(255,255,255,0.12);
    padding:12px;
    border-radius:12px;
    white-space:pre-wrap;
    font-family:monospace;
    margin-bottom:10px;
}

/* result pill */
.result-pill {
    padding:12px 18px;
    border-radius:12px;
    font-weight:700;
    color:white;
    display:inline-block;
    margin-top:12px;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================
# SLANG
# =========================
SLANG_PATH = BASE_DIR / "tweet" / "000_colloquial-indonesian-lexicon.csv"


@st.cache_resource
def load_slang():
    df = pd.read_csv(SLANG_PATH, usecols=["slang", "formal"])
    return dict(zip(df.slang, df.formal))


slang = load_slang()


def normalize_slang(t):
    return " ".join(slang.get(w, w) for w in t.split())


def reduce_repeat(t):
    return re.sub(r"(.)\1{2,}", r"\1", t)


# =========================
# PREPROCESS
# =========================
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stop_id = set(stopwords.words("indonesian"))
stemmer = StemmerFactory().create_stemmer()


def preprocess_normal(text):
    steps = {}
    steps["Original"] = text
    text = text.lower()
    steps["Lowercase"] = text
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+|\d+", "", text)
    steps["Cleaning"] = text
    text = normalize_slang(text)
    steps["Slang"] = text
    text = reduce_repeat(text)
    steps["Repeat"] = text
    text = " ".join(w for w in text.split() if w not in stop_id)
    steps["Stopwords"] = text
    text = " ".join(stemmer.stem(w) for w in text.split())
    steps["Stemming"] = text
    return text, steps


def preprocess_light(text):
    steps = {}
    steps["Original"] = text
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    text = normalize_slang(text)
    text = reduce_repeat(text)
    steps["Processed"] = text
    return text, steps


# =========================
# MODEL CONFIG
# =========================
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


# =========================
# LOAD MODEL (AMAN)
# =========================
@st.cache_resource(show_spinner="Memuat model…")
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False, cache_dir=HF_CACHE)
    model = AutoModelForSequenceClassification.from_pretrained(
        path, low_cpu_mem_usage=True, cache_dir=HF_CACHE
    )
    model.eval()
    return tok, model


def predict(text, tok, model):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0].numpy()
    return int(probs.argmax()), probs


# =========================
# UI
# =========================
st.title("🔍 Deteksi Tingkat Depresi")

arch = st.radio("Model", ["IndoBERT", "XLM-RoBERTa"], horizontal=True)
prep = st.radio("Preprocessing", ["Normal", "Light"], horizontal=True)

user_text = st.text_area("Masukkan teks:", height=160)

selected_model_path, preprocess_fn = MODEL_PATHS[arch][prep]

# 🔑 LOAD MODEL HANYA JIKA BERUBAH
if st.session_state.current_model_path != selected_model_path:
    if st.session_state.model is not None:
        del st.session_state.model
        del st.session_state.tokenizer
        gc.collect()

    tok, model = load_model(selected_model_path)
    st.session_state.tokenizer = tok
    st.session_state.model = model
    st.session_state.current_model_path = selected_model_path

# =========================
# RUN
# =========================
if st.button("Analisis"):
    if not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        try:
            processed, steps = preprocess_fn(user_text)
            pred, probs = predict(
                processed,
                st.session_state.tokenizer,
                st.session_state.model,
            )

            st.markdown(
                f"<div class='result-pill' style='background:{COLORS[pred]}'>{LABELS[pred]}</div>",
                unsafe_allow_html=True,
            )

            st.subheader("Confidence")
            st.write(float(probs[pred]))

            st.subheader("Preprocessing")
            for k, v in steps.items():
                st.markdown(
                    f"<div class='pre-box'><b>{k}</b><br>{v}</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.exception(e)
