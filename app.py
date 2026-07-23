# -*- coding: utf-8 -*-
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

MODEL_ID = "gulamsakaria/bangla-comment-classifier1"

st.set_page_config(page_title="বাংলা কমেন্ট ক্লাসিফায়ার", page_icon="🇧🇩")
st.title("🇧🇩 বাংলা কমেন্ট ক্লাসিফায়ার")
st.caption("১৬,০০০+ আসল ফেসবুক কমেন্ট দিয়ে BanglaBERT ফাইন-টিউন করে বানানো। "
           "কমেন্ট লিখুন — AI বলবে সেটা পজিটিভ, নেগেটিভ, টক্সিক নাকি নিউট্রাল। "
           "বাংলা ও বাংলিশ দুটোই বোঝে!")

@st.cache_resource(show_spinner="মডেল লোড হচ্ছে (প্রথমবার ২-৩ মিনিট লাগতে পারে)...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()
id2label = model.config.id2label

LABEL_BN = {"positive": "✅ পজিটিভ", "negative": "⚠️ নেগেটিভ",
            "toxic": "🚫 টক্সিক/গালি", "neutral": "💬 নিউট্রাল"}

text = st.text_area("কমেন্ট লিখুন (একাধিক হলে প্রতি লাইনে একটা):",
                    height=150, placeholder="যেমন: ভাই ভিডিওটা সেরা হইছে")

if st.button("বিশ্লেষণ করুন", type="primary") and text.strip():
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    counts = Counter()
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        label = id2label[pred_id]
        conf = round(float(probs[pred_id]) * 100, 1)
        counts[LABEL_BN.get(label, label)] += 1
        st.write(f"**{LABEL_BN.get(label, label)}** ({conf}%) — {line}")
    if len(lines) > 1:
        st.divider()
        st.subheader("📊 সারসংক্ষেপ")
        cols = st.columns(len(counts))
        for col, (lbl, n) in zip(cols, counts.items()):
            col.metric(lbl, f"{n}টা")

st.divider()
st.caption("⚙️ Fine-tuned BanglaBERT (csebuetnlp/banglabert) | নিজস্ব সংগৃহীত ডেটাসেটে ট্রেইন করা")
