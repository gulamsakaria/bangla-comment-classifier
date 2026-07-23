# -*- coding: utf-8 -*-
# ============================================================
#  SANS1 — বাংলা কমেন্ট ক্লাসিফায়ার
#  Developed by Gulam Sakaria
# ============================================================
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

MODEL_ID = "gulamsakaria/bangla-comment-classifier1"
FB_LINK = "https://www.facebook.com/gulamsakaria2017"
GH_LINK = "https://github.com/gulamsakaria/bangla-comment-classifier"

st.set_page_config(page_title="SANS1 — বাংলা কমেন্ট ক্লাসিফায়ার",
                   page_icon="🧠", layout="centered")

# ---------------- ডিজাইন (CSS) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+Da+2:wght@500;700;800&family=Hind+Siliguri:wght@400;500;600&display=swap');

:root{
  --bg:#0B241E;          /* গাঢ় বটল-গ্রিন */
  --surface:#11312A;     /* কার্ড */
  --surface2:#173B32;
  --ink:#F2EFE4;         /* উষ্ণ অফ-হোয়াইট */
  --muted:#9DB8AE;
  --flag-red:#E8443A;    /* পতাকার লাল */
  --pos:#3FB97C; --neg:#E0A93E; --tox:#E8443A; --neu:#5FA8D3;
}

html, body, [class*="css"], .stApp { font-family:'Hind Siliguri',sans-serif; }
.stApp { background: radial-gradient(1100px 500px at 50% -10%, #14403455 0%, transparent 60%), var(--bg); }

#MainMenu, footer, header {visibility:hidden;}
.block-container{ padding-top:2.2rem; max-width:760px; }

/* ---------- হিরো / লোগো ---------- */
.sans-hero{ text-align:center; animation:fadeUp .8s ease both; }
.sans-logo{
  font-family:'Baloo Da 2',cursive; font-weight:800; font-size:4.2rem; line-height:1;
  letter-spacing:.02em; margin:0;
  background:linear-gradient(90deg,#F2EFE4 0%, #7FD8AC 35%, var(--flag-red) 70%, #F2EFE4 100%);
  background-size:250% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:shine 6s linear infinite;
}
.sans-dot{ display:inline-block; width:.55em; height:.55em; border-radius:50%;
  background:var(--flag-red); margin-left:.12em;
  box-shadow:0 0 18px #E8443A88; animation:pulse 2.4s ease-in-out infinite; }
.sans-sub{ color:var(--ink); font-size:1.25rem; font-weight:600; margin:.4rem 0 .2rem; }
.sans-tag{ color:var(--muted); font-size:.95rem; max-width:560px; margin:0 auto 1.4rem; }
.sans-rule{ width:120px; height:3px; margin:0 auto 1.6rem; border-radius:2px;
  background:linear-gradient(90deg, transparent, var(--flag-red), transparent); }

/* ---------- ইনপুট ---------- */
.stTextArea textarea{
  background:var(--surface) !important; color:var(--ink) !important;
  border:1px solid #24544877 !important; border-radius:14px !important;
  font-family:'Hind Siliguri',sans-serif !important; font-size:1.02rem !important;
  transition:border .25s, box-shadow .25s;
}
.stTextArea textarea:focus{ border-color:var(--flag-red) !important;
  box-shadow:0 0 0 3px #E8443A22 !important; }
.stTextArea label{ color:var(--muted) !important; font-weight:600; }

.stButton>button{
  width:100%; background:linear-gradient(135deg,#E8443A,#B92F27);
  color:#fff; font-family:'Baloo Da 2',cursive; font-weight:700; font-size:1.15rem;
  border:none; border-radius:14px; padding:.7rem 0;
  box-shadow:0 6px 22px #E8443A44; transition:transform .18s, box-shadow .18s;
}
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 10px 28px #E8443A66; color:#fff; }

/* ---------- রেজাল্ট কার্ড ---------- */
.res-card{
  display:flex; align-items:center; gap:.9rem;
  background:var(--surface); border:1px solid #24544855;
  border-left:5px solid var(--neu);
  border-radius:14px; padding:.85rem 1.1rem; margin:.5rem 0;
  animation:slideIn .5s ease both; transition:transform .18s;
}
.res-card:hover{ transform:translateX(4px); }
.res-card.pos{ border-left-color:var(--pos); } .res-card.neg{ border-left-color:var(--neg); }
.res-card.tox{ border-left-color:var(--tox); } .res-card.neu{ border-left-color:var(--neu); }
.res-badge{ font-weight:700; font-size:.95rem; white-space:nowrap; padding:.25rem .7rem;
  border-radius:999px; color:#08201A; }
.res-card.pos .res-badge{ background:var(--pos);} .res-card.neg .res-badge{ background:var(--neg);}
.res-card.tox .res-badge{ background:var(--tox); color:#fff;} .res-card.neu .res-badge{ background:var(--neu);}
.res-text{ color:var(--ink); font-size:1rem; }
.res-conf{ color:var(--muted); font-size:.85rem; margin-left:auto; white-space:nowrap; }

/* ---------- সারসংক্ষেপ ---------- */
.sum-wrap{ display:flex; gap:.6rem; flex-wrap:wrap; justify-content:center; margin-top:1rem;
  animation:fadeUp .6s ease both; }
.sum-chip{ background:var(--surface2); border:1px solid #24544866; border-radius:12px;
  padding:.55rem 1rem; text-align:center; }
.sum-chip .n{ font-family:'Baloo Da 2',cursive; font-weight:800; font-size:1.5rem; color:var(--ink); }
.sum-chip .l{ color:var(--muted); font-size:.82rem; }

/* ---------- ফুটার ---------- */
.sans-footer{ margin-top:3rem; padding-top:1.2rem; border-top:1px solid #24544855;
  text-align:center; color:var(--muted); font-size:.9rem; animation:fadeUp 1s ease both; }
.sans-footer b{ color:var(--ink); }
.sans-footer a{ color:#7FD8AC; text-decoration:none; font-weight:600; }
.sans-footer a:hover{ color:var(--flag-red); }
.dev-line{ font-family:'Baloo Da 2',cursive; font-size:1.05rem; margin-bottom:.3rem; }

@keyframes shine{ to{ background-position:250% center; } }
@keyframes pulse{ 0%,100%{ transform:scale(1); opacity:1;} 50%{ transform:scale(.75); opacity:.6;} }
@keyframes fadeUp{ from{ opacity:0; transform:translateY(14px);} to{ opacity:1; transform:none;} }
@keyframes slideIn{ from{ opacity:0; transform:translateX(-16px);} to{ opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

# ---------------- হিরো ----------------
st.markdown("""
<div class="sans-hero">
  <h1 class="sans-logo">SANS1<span class="sans-dot"></span></h1>
  <div class="sans-sub">বাংলা কমেন্ট ক্লাসিফায়ার</div>
  <p class="sans-tag">১৬,০০০+ আসল ফেসবুক কমেন্ট দিয়ে ফাইন-টিউন করা BanglaBERT মডেল —
  যেকোনো বাংলা বা বাংলিশ কমেন্ট দিন, SANS1 বলে দেবে সেটা পজিটিভ, নেগেটিভ, টক্সিক নাকি নিউট্রাল।</p>
  <div class="sans-rule"></div>
</div>
""", unsafe_allow_html=True)

# ---------------- মডেল ----------------
@st.cache_resource(show_spinner="🧠 SANS1 জাগছে... (প্রথমবার ২-৩ মিনিট লাগতে পারে)")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()
id2label = model.config.id2label

LABELS = {
    "positive": ("✅ পজিটিভ", "pos"),
    "negative": ("⚠️ নেগেটিভ", "neg"),
    "toxic":    ("🚫 টক্সিক",  "tox"),
    "neutral":  ("💬 নিউট্রাল","neu"),
}

# ---------------- ইনপুট ----------------
text = st.text_area("কমেন্ট লিখুন — একাধিক হলে প্রতি লাইনে একটা:",
                    height=150, placeholder="যেমন: ভাই ভিডিওটা সেরা হইছে")

if st.button("🔍 বিশ্লেষণ করুন") and text.strip():
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    counts = Counter()
    for i, line in enumerate(lines):
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        raw = id2label[pred_id]
        conf = round(float(probs[pred_id]) * 100, 1)
        bn, cls = LABELS.get(raw, (raw, "neu"))
        counts[bn] += 1
        safe = line.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="res-card {cls}" style="animation-delay:{i*0.08}s">'
            f'<span class="res-badge">{bn}</span>'
            f'<span class="res-text">{safe}</span>'
            f'<span class="res-conf">{conf}%</span></div>',
            unsafe_allow_html=True)

    if len(lines) > 1:
        chips = "".join(
            f'<div class="sum-chip"><div class="n">{n}</div><div class="l">{lbl}</div></div>'
            for lbl, n in counts.items())
        st.markdown(f'<div class="sum-wrap">{chips}</div>', unsafe_allow_html=True)

# ---------------- ফুটার ----------------
st.markdown(f"""
<div class="sans-footer">
  <div class="dev-line">Developed by <b>Gulam Sakaria</b></div>
  <div>🔗 <a href="{FB_LINK}" target="_blank">Facebook</a> &nbsp;•&nbsp;
  <a href="{GH_LINK}" target="_blank">GitHub</a></div>
  <div style="margin-top:.5rem;">⚙️ SANS1 — Fine-tuned BanglaBERT | নিজস্ব সংগৃহীত ডেটাসেটে ট্রেইন করা</div>
</div>
""", unsafe_allow_html=True)
