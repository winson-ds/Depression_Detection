# app.py
import streamlit as st
import torch
import re
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# =====================================================================
# SLANG LOAD
# =====================================================================
SLANG_PATH = "tweet/000_colloquial-indonesian-lexicon.csv"


@st.cache_resource
def load_slang():
    df = pd.read_csv(SLANG_PATH, usecols=["slang", "formal"])
    return dict(zip(df["slang"], df["formal"]))


slang = load_slang()


def normalize_slang(t):
    return " ".join([slang.get(w, w) for w in t.split()])


def reduce_repeat(t):
    return re.sub(r"(.)\\1{2,}", r"\\1", t)


# =====================================================================
# PREPROCESS
# =====================================================================
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

stop_id = set(stopwords.words("indonesian"))

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()


def preprocess_normal(text):
    s = {}
    s["Original"] = text
    text = text.lower()
    s["Lowercase"] = text
    text = re.sub(r"http\\S+|www\\S+|@\\w+|#\\w+|\\d+", "", text)
    s["Cleaning"] = text
    text = re.sub(r"\\s+", " ", text).strip()
    s["Normalize"] = text
    text = normalize_slang(text)
    s["Slang"] = text
    text = reduce_repeat(text)
    s["Reduce Repeated Char"] = text
    text = " ".join([w for w in text.split() if w not in stop_id])
    s["Stopwords"] = text
    text = " ".join([stemmer.stem(w) for w in text.split()])
    s["Stemming"] = text
    return text, s


def preprocess_light(text):
    s = {}
    s["Original"] = text
    text = text.lower()
    s["Lowercase"] = text
    text = re.sub(r"http\\S+|www\\S+|@\\w+", "", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    s["Clean"] = text
    text = reduce_repeat(text)
    s["Reduce Repeated Char"] = text
    text = normalize_slang(text)
    s["Slang"] = text
    text = re.sub(r"\\s+", " ", text).strip()
    s["Normalize"] = text
    return text, s


# =====================================================================
# MODEL CONFIG
# =====================================================================
MODEL_PATHS = {
    "IndoBERT": {
        "Normal": ("models/indobert_normal", preprocess_normal),
        "Light": ("models/indobert_light", preprocess_light),
    },
    "XLM-RoBERTa": {
        "Normal": ("models/xlmroberta_normal", preprocess_normal),
        "Light": ("models/xlmroberta_light", preprocess_light),
    },
}

label_map = ["Tidak Depresi", "Depresi Ringan", "Depresi Sedang", "Depresi Berat"]
label_colors = ["#1b5e20", "#f9a825", "#ef6c00", "#c62828"]


# =====================================================================
# LOAD MODEL
# =====================================================================
@st.cache_resource
def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tok, model


def predict(txt, tok, model):
    enc = tok(txt, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**enc)
        p = torch.softmax(out.logits, dim=1).numpy()[0]
    return int(np.argmax(p)), p


# =====================================================================
# UI LAYOUT — STACKED (VERTICAL) LIKE NAVY VERSION
# =====================================================================

st.title("🔍 Deteksi Tingkat Depresi dari Teks")
st.write(
    "Aplikasi ini membandingkan arsitektur **IndoBERT** dan **XLM-RoBERTa** dengan dua jenis preprocessing."
)

# Radio (vertical stacked, NOT side-by-side)
st.markdown("### Arsitektur Model")
arch = st.radio(
    "", ["IndoBERT", "XLM-RoBERTa"], horizontal=True, label_visibility="collapsed"
)

st.markdown("### Tipe Preprocessing")
prep = st.radio("", ["Normal", "Light"], horizontal=True, label_visibility="collapsed")

# Input
user_text = st.text_area("Masukkan teks:", height=180)

# Run button
if st.button("Analisis"):
    if not user_text.strip():
        st.warning("Masukkan teks terlebih dahulu.")
    else:
        path, fn = MODEL_PATHS[arch][prep]
        tok, model = load_model(path)
        processed, steps = fn(user_text)
        pred, probs = predict(processed, tok, model)

        # Output badge
        st.markdown(
            f"<div class='result-pill' style='background:{label_colors[pred]}'>{label_map[pred]}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Confidence Score")
        st.write(f"{probs[pred]:.4f}")

        st.subheader("Tahapan Preprocessing")
        for k, v in steps.items():
            st.markdown(
                f"<div class='pre-box'><strong>{k}</strong><br>{v}</div>",
                unsafe_allow_html=True,
            )

        st.caption("Catatan: Ini adalah alat skrining, bukan alat diagnosis klinis.")
