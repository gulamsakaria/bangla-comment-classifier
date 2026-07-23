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

st.set_page_config(page_title="SANS1 — বাংলা কমেন্ট AI",
                   page_icon="🧠", layout="centered")

# ================= ডিজাইন =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+Da+2:wght@500;700;800&family=Hind+Siliguri:wght@400;500;600;700&display=swap');

:root{
  --ink:#EDF2FA; --muted:#8FA3BF;
  --glass:rgba(255,255,255,.055); --stroke:rgba(255,255,255,.12);
  --cyan:#39D0FF; --violet:#8B7CFF; --rose:#FF5E8A;
  --pos:#2FD98B; --neg:#FFB454; --tox:#FF4D5E; --neu:#39D0FF;
}

html, body, [class*="css"], .stApp{ font-family:'Hind Siliguri',sans-serif; }
#MainMenu, footer, header{visibility:hidden;}
.block-container{ padding-top:2rem; max-width:780px; }

/* ---------- অ্যানিমেটেড অরোরা ব্যাকগ্রাউন্ড ---------- */
.stApp{ background:#070B14; overflow-x:hidden; }
.stApp::before, .stApp::after{
  content:""; position:fixed; z-index:0; border-radius:50%;
  filter:blur(110px); opacity:.5; pointer-events:none;
}
.stApp::before{
  width:560px; height:560px; top:-180px; left:-140px;
  background:radial-gradient(circle, #1B6BFF66, transparent 65%);
  animation:drift1 16s ease-in-out infinite alternate;
}
.stApp::after{
  width:520px; height:520px; bottom:-200px; right:-140px;
  background:radial-gradient(circle, #8B2FBF55, transparent 65%);
  animation:drift2 20s ease-in-out infinite alternate;
}
@keyframes drift1{ to{ transform:translate(70px,50px) scale(1.12);} }
@keyframes drift2{ to{ transform:translate(-60px,-60px) scale(1.08);} }

.block-container > div{ position:relative; z-index:1; }

/* ---------- হিরো ---------- */
.hero{
  text-align:center; padding:2.4rem 1.4rem 2rem; margin-bottom:1.4rem;
  background:var(--glass); border:1px solid var(--stroke); border-radius:24px;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 22px 60px rgba(0,0,0,.45);
  animation:fadeUp .8s ease both;
}
.brand{
  font-family:'Baloo Da 2',cursive; font-weight:800;
  font-size:4.6rem; line-height:1; margin:0; letter-spacing:.01em;
  background:linear-gradient(100deg, var(--cyan) 10%, var(--violet) 50%, var(--rose) 90%);
  background-size:220% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:sheen 5s linear infinite;
  text-shadow:0 0 60px rgba(90,120,255,.15);
}
.brand-chip{
  display:inline-block; margin-top:.7rem; padding:.3rem 1rem;
  font-size:.85rem; font-weight:700; letter-spacing:.14em; color:var(--cyan);
  border:1px solid rgba(57,208,255,.4); border-radius:999px;
  background:rgba(57,208,255,.08);
}
.hero h2{ color:var(--ink); font-weight:700; font-size:1.4rem; margin:1rem 0 .4rem; }
.hero p{ color:var(--muted); font-size:.98rem; max-width:540px; margin:0 auto; }
.scanline{
  width:180px; height:2px; margin:1.3rem auto 0; border-radius:2px; position:relative;
  background:rgba(255,255,255,.08); overflow:hidden;
}
.scanline::after{
  content:""; position:absolute; inset:0; width:45%;
  background:linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation:scan 2.6s ease-in-out infinite;
}
@keyframes scan{ 0%{left:-45%;} 100%{left:100%;} }
@keyframes sheen{ to{ background-position:220% center; } }

/* ---------- ইনপুট ---------- */
.stTextArea textarea{
  background:rgba(255,255,255,.05) !important; color:var(--ink) !important;
  border:1px solid var(--stroke) !important; border-radius:16px !important;
  font-family:'Hind Siliguri',sans-serif !important; font-size:1.03rem !important;
  backdrop-filter:blur(10px); transition:border .25s, box-shadow .25s;
}
.stTextArea textarea:focus{
  border-color:var(--cyan) !important;
  box-shadow:0 0 0 3px rgba(57,208,255,.18), 0 0 30px rgba(57,208,255,.12) !important;
}
.stTextArea label{ color:var(--muted) !important; font-weight:600; }

.stButton>button{
  width:100%; border:none; border-radius:16px; padding:.75rem 0;
  font-family:'Baloo Da 2',cursive; font-weight:700; font-size:1.18rem; color:#fff;
  background:linear-gradient(120deg, #1E90FF, var(--violet), var(--rose));
  background-size:200% auto;
  box-shadow:0 10px 34px rgba(90,110,255,.35);
  transition:background-position .4s, transform .18s, box-shadow .18s;
}
.stButton>button:hover{
  background-position:100% center; transform:translateY(-2px);
  box-shadow:0 14px 44px rgba(120,90,255,.5); color:#fff;
}

/* ---------- রেজাল্ট কার্ড ---------- */
.res{
  display:flex; align-items:center; gap:.9rem;
  background:var(--glass); border:1px solid var(--stroke);
  border-radius:16px; padding:.9rem 1.1rem; margin:.55rem 0;
  backdrop-filter:blur(12px);
  animation:slideIn .5s ease both; transition:transform .2s, border-color .2s;
}
.res:hover{ transform:translateX(5px); }
.res .badge{
  font-weight:700; font-size:.92rem; white-space:nowrap;
  padding:.3rem .8rem; border-radius:999px; color:#06101E;
}
.res.pos .badge{ background:var(--pos); box-shadow:0 0 16px rgba(47,217,139,.4);}
.res.neg .badge{ background:var(--neg); box-shadow:0 0 16px rgba(255,180,84,.4);}
.res.tox .badge{ background:var(--tox); color:#fff; box-shadow:0 0 16px rgba(255,77,94,.5);}
.res.neu .badge{ background:var(--neu); box-shadow:0 0 16px rgba(57,208,255,.4);}
.res .txt{ color:var(--ink); font-size:1rem; }
.res .conf{ margin-left:auto; color:var(--muted); font-size:.85rem; white-space:nowrap;
  font-variant-numeric:tabular-nums; }

/* ---------- সারসংক্ষেপ ---------- */
.sum{ display:flex; gap:.6rem; flex-wrap:wrap; justify-content:center; margin-top:1.1rem;
  animation:fadeUp .6s ease both; }
.chip{ min-width:96px; text-align:center; padding:.6rem 1rem;
  background:var(--glass); border:1px solid var(--stroke); border-radius:14px;
  backdrop-filter:blur(10px); }
.chip .n{ font-family:'Baloo Da 2',cursive; font-weight:800; font-size:1.6rem; color:var(--ink);}
.chip .l{ color:var(--muted); font-size:.82rem; }

/* ---------- ফুটার ---------- */
.foot{
  margin-top:3rem; padding:1.4rem 0 .6rem; text-align:center;
  border-top:1px solid var(--stroke); animation:fadeUp 1s ease both;
}
.foot .dev{ font-family:'Baloo Da 2',cursive; font-size:1.1rem; color:var(--ink); }
.foot .dev b{
  background:linear-gradient(90deg, var(--cyan), var(--violet));
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.foot a{ color:var(--cyan); text-decoration:none; font-weight:600; }
.foot a:hover{ color:var(--rose); }
.foot .fine{ margin-top:.6rem; color:#5A6B85; font-size:.78rem; }

@keyframes fadeUp{ from{opacity:0; transform:translateY(16px);} to{opacity:1; transform:none;} }
@keyframes slideIn{ from{opacity:0; transform:translateX(-18px);} to{opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important;} }
@media (max-width:520px){ .brand{ font-size:3.4rem; } }
</style>
""", unsafe_allow_html=True)

# ================= হিরো =================
st.markdown("""
<div class="hero">
  <h1 class="brand">SANS1</h1>
  <div class="brand-chip">BANGLA COMMENT AI</div>
  <h2>বাংলা কমেন্ট ক্লাসিফায়ার</h2>
  <p>১৬,০০০+ আসল ফেসবুক কমেন্ট দিয়ে ট্রেইন করা AI —
  যেকোনো বাংলা বা বাংলিশ কমেন্ট দিন, SANS1 মুহূর্তেই বলে দেবে
  সেটা পজিটিভ, নেগেটিভ, টক্সিক নাকি নিউট্রাল।</p>
  <div class="scanline"></div>
</div>
""", unsafe_allow_html=True)

# ================= মডেল =================
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

# ================= ইনপুট =================
text = st.text_area("কমেন্ট লিখুন — একাধিক হলে প্রতি লাইনে একটা:",
                    height=150, placeholder="যেমন: ভাই ভিডিওটা সেরা হইছে")

if st.button("⚡ SANS1 দিয়ে বিশ্লেষণ করুন") and text.strip():
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
            f'<div class="res {cls}" style="animation-delay:{i*0.07}s">'
            f'<span class="badge">{bn}</span>'
            f'<span class="txt">{safe}</span>'
            f'<span class="conf">{conf}%</span></div>',
            unsafe_allow_html=True)

    if len(lines) > 1:
        chips = "".join(
            f'<div class="chip"><div class="n">{n}</div><div class="l">{lbl}</div></div>'
            for lbl, n in counts.items())
        st.markdown(f'<div class="sum">{chips}</div>', unsafe_allow_html=True)

# ================= ফুটার =================
st.markdown(f"""
<div class="foot">
  <div class="dev">Developed by <b>Gulam Sakaria</b></div>
  <div style="margin-top:.4rem;">
    <a href="{FB_LINK}" target="_blank">Facebook</a> &nbsp;•&nbsp;
    <a href="{GH_LINK}" target="_blank">GitHub</a>
  </div>
  <div class="fine">SANS1 v1.0 — নিজস্ব সংগৃহীত ১৬,০০০+ কমেন্টের ডেটাসেটে ট্রেইন করা · built on BanglaBERT</div>
</div>
""", unsafe_allow_html=True)
