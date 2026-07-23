# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np
from collections import Counter

st.set_page_config(page_title="বাংলা কমেন্ট ক্লাসিফায়ার", page_icon="🇧🇩")
st.title("🇧🇩 বাংলা কমেন্ট ক্লাসিফায়ার")
st.caption("১৬,০০০+ আসল ফেসবুক কমেন্ট দিয়ে ট্রেইন করা মেশিন লার্নিং মডেল। "
           "কমেন্ট লিখুন — AI বলবে সেটা পজিটিভ, নেগেটিভ, টক্সিক নাকি নিউট্রাল। "
           "বাংলা ও বাংলিশ দুটোই বোঝে!")

@st.cache_resource
def load_model():
    with open("baseline_model.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["vectorizer"], obj["model"]

vectorizer, model = load_model()

LABEL_BN = {"positive": "✅ পজিটিভ", "negative": "⚠️ নেগেটিভ",
            "toxic": "🚫 টক্সিক/গালি", "neutral": "💬 নিউট্রাল"}

text = st.text_area("কমেন্ট লিখুন (একাধিক হলে প্রতি লাইনে একটা):",
                    height=150, placeholder="যেমন: ভাই ভিডিওটা সেরা হইছে")

if st.button("বিশ্লেষণ করুন", type="primary") and text.strip():
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    counts = Counter()
    for line in lines:
        probs = model.predict_proba(vectorizer.transform([line]))[0]
        i = int(np.argmax(probs))
        label = model.classes_[i]
        conf = round(float(probs[i]) * 100, 1)
        counts[LABEL_BN[label]] += 1
        st.write(f"**{LABEL_BN[label]}** ({conf}%) — {line}")
    if len(lines) > 1:
        st.divider()
        st.subheader("📊 সারসংক্ষেপ")
        cols = st.columns(len(counts))
        for col, (lbl, n) in zip(cols, counts.items()):
            col.metric(lbl, f"{n}টা")

st.divider()
st.caption("⚙️ TF-IDF (character n-gram) + Logistic Regression | Test Accuracy: 88.3%")
