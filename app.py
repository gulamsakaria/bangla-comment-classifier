# -*- coding: utf-8 -*-
# ============================================================
#  SANS1 — বাংলা কমেন্ট ক্লাসিফায়ার  (Final Edition)
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;800;900&family=Baloo+Da+2:wght@600;700;800&family=Hind+Siliguri:wght@400;500;600;700&display=swap');

:root{
  --ink:#EDF2FA; --muted:#8899BB; --faint:#556688;
  --glass:rgba(255,255,255,.055); --stroke:rgba(255,255,255,.10);
  --cyan:#00D4FF; --violet:#7C5CFC; --rose:#FF6B9D;
  --pos:#00D4AA; --neg:#FF9A56; --tox:#FF5E6C; --neu:#4A9EFF;
  --grad:linear-gradient(120deg,var(--cyan) 0%,var(--violet) 55%,var(--rose) 100%);
}

html, body, [class*="css"], .stApp{ font-family:'Hind Siliguri','Inter',sans-serif; }
#MainMenu, footer, header{ display:none; }
.block-container{ padding-top:1.8rem; max-width:800px; }

::-webkit-scrollbar{ width:6px; } ::-webkit-scrollbar-track{ background:#0A0E17; }
::-webkit-scrollbar-thumb{ background:var(--cyan); border-radius:10px; }

/* ---------- অরোরা ব্যাকগ্রাউন্ড ---------- */
.stApp{ background:#070B14; overflow-x:hidden; }
.stApp::before, .stApp::after{
  content:""; position:fixed; z-index:0; border-radius:50%;
  filter:blur(110px); opacity:.5; pointer-events:none;
}
.stApp::before{ width:560px; height:560px; top:-180px; left:-140px;
  background:radial-gradient(circle,#1B6BFF55,transparent 65%);
  animation:drift1 16s ease-in-out infinite alternate; }
.stApp::after{ width:520px; height:520px; bottom:-200px; right:-140px;
  background:radial-gradient(circle,#8B2FBF4d,transparent 65%);
  animation:drift2 20s ease-in-out infinite alternate; }
@keyframes drift1{ to{ transform:translate(70px,50px) scale(1.12);} }
@keyframes drift2{ to{ transform:translate(-60px,-60px) scale(1.08);} }
.block-container > div{ position:relative; z-index:1; }

/* ---------- হিরো ---------- */
.hero{
  text-align:center; padding:2.4rem 1.4rem 1.8rem; margin-bottom:1rem;
  background:var(--glass); border:1px solid var(--stroke); border-radius:24px;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 22px 60px rgba(0,0,0,.45); animation:fadeUp .8s ease both;
}
.hero-badge{
  display:inline-block; padding:.28rem 1.1rem; margin-bottom:.9rem;
  font-family:'Inter',sans-serif; font-size:.68rem; font-weight:800;
  letter-spacing:.16em; text-transform:uppercase; color:var(--cyan);
  border:1px solid rgba(0,212,255,.35); border-radius:999px;
  background:rgba(0,212,255,.07);
}
.brand{
  font-family:'Inter',sans-serif; font-weight:900; font-size:4.6rem;
  line-height:1; margin:0; letter-spacing:-.02em;
  background:var(--grad); background-size:220% auto;
  -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:sheen 5s linear infinite; display:inline-block; position:relative;
}
.brand::after{
  content:""; position:absolute; top:-6px; right:-18px; width:14px; height:14px;
  border-radius:50%; background:var(--rose);
  box-shadow:0 0 34px rgba(255,107,157,.55); animation:pulse 2.2s ease-in-out infinite;
}
.hero h2{ color:var(--ink); font-weight:700; font-size:1.35rem; margin:.9rem 0 .3rem;
  font-family:'Baloo Da 2',cursive; }
.hero p{ color:var(--muted); font-size:.95rem; max-width:560px; margin:0 auto; line-height:1.7; }
.scanline{ width:170px; height:2px; margin:1.2rem auto 0; border-radius:2px;
  background:rgba(255,255,255,.08); overflow:hidden; position:relative; }
.scanline::after{ content:""; position:absolute; inset:0; width:45%;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  animation:scan 2.6s ease-in-out infinite; }

/* ---------- স্ট্যাটস ---------- */
.stats{ display:flex; justify-content:center; gap:1.6rem; flex-wrap:wrap;
  margin:.4rem 0 1.6rem; animation:fadeUp 1s ease both; }
.stat{ display:flex; align-items:center; gap:.45rem;
  color:var(--muted); font-size:.85rem; font-weight:500; }
.stat b{ color:var(--ink); font-weight:800; font-family:'Inter',sans-serif; }

/* ---------- ইনপুট ---------- */
.stTextArea textarea{
  background:rgba(255,255,255,.05) !important; color:var(--ink) !important;
  border:1px solid var(--stroke) !important; border-radius:16px !important;
  font-family:'Hind Siliguri',sans-serif !important; font-size:1.02rem !important;
  line-height:1.7 !important; backdrop-filter:blur(10px);
  transition:border .25s, box-shadow .25s;
}
.stTextArea textarea:focus{ border-color:var(--cyan) !important;
  box-shadow:0 0 0 3px rgba(0,212,255,.15), 0 0 28px rgba(0,212,255,.10) !important; }
.stTextArea label{ color:var(--muted) !important; font-weight:600; }

.stButton>button{
  width:100%; border:none; border-radius:16px; padding:.75rem 0;
  font-family:'Baloo Da 2',cursive; font-weight:700; font-size:1.12rem; color:#fff;
  background:var(--grad); background-size:200% auto;
  box-shadow:0 10px 34px rgba(60,120,255,.32);
  transition:background-position .4s, transform .18s, box-shadow .18s;
  position:relative; overflow:hidden;
}
.stButton>button::before{
  content:""; position:absolute; top:0; left:-100%; width:100%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent);
  transition:left .6s ease;
}
.stButton>button:hover{ background-position:100% center; transform:translateY(-2px);
  box-shadow:0 14px 44px rgba(110,90,255,.45); color:#fff; }
.stButton>button:hover::before{ left:100%; }

/* ---------- রেজাল্ট কার্ড ---------- */
.res{
  display:flex; align-items:center; gap:.85rem;
  background:var(--glass); border:1px solid var(--stroke);
  border-radius:16px; padding:.85rem 1.1rem; margin:.5rem 0;
  backdrop-filter:blur(12px); position:relative; overflow:hidden;
  animation:slideIn .45s ease both; transition:transform .2s, background .2s;
}
.res::before{ content:""; position:absolute; top:0; left:0; width:4px; height:100%; }
.res:hover{ transform:translateX(5px); background:rgba(255,255,255,.08); }
.res.pos::before{ background:var(--pos);} .res.neg::before{ background:var(--neg);}
.res.tox::before{ background:var(--tox);} .res.neu::before{ background:var(--neu);}
.badge{ font-family:'Inter',sans-serif; font-weight:800; font-size:.76rem;
  letter-spacing:.03em; white-space:nowrap; padding:.28rem .8rem;
  border-radius:999px; color:#06101E; flex-shrink:0; }
.res.pos .badge{ background:var(--pos); box-shadow:0 0 15px rgba(0,212,170,.35);}
.res.neg .badge{ background:var(--neg); box-shadow:0 0 15px rgba(255,154,86,.35);}
.res.tox .badge{ background:var(--tox); color:#fff; box-shadow:0 0 15px rgba(255,94,108,.45);}
.res.neu .badge{ background:var(--neu); box-shadow:0 0 15px rgba(74,158,255,.35);}
.txt{ color:var(--ink); font-size:.98rem; flex:1; word-break:break-word; }
.conf{ margin-left:auto; color:var(--faint); font-size:.82rem; white-space:nowrap;
  font-variant-numeric:tabular-nums; flex-shrink:0; }

/* ---------- সারসংক্ষেপ ---------- */
.sum{ display:flex; gap:.7rem; flex-wrap:wrap; justify-content:center;
  margin-top:1.2rem; padding:1.1rem;
  background:var(--glass); border:1px solid var(--stroke); border-radius:16px;
  backdrop-filter:blur(10px); animation:fadeUp .6s ease both; }
.chip{ display:flex; align-items:center; gap:.5rem; padding:.42rem 1rem;
  border-radius:999px; background:rgba(255,255,255,.05);
  border:1px solid var(--stroke); }
.dot{ width:9px; height:9px; border-radius:50%; }
.dot.pos{ background:var(--pos);} .dot.neg{ background:var(--neg);}
.dot.tox{ background:var(--tox);} .dot.neu{ background:var(--neu);}
.chip .n{ font-family:'Inter',sans-serif; font-weight:800; font-size:1.15rem;
  color:var(--ink); font-variant-numeric:tabular-nums; }
.chip .l{ font-size:.8rem; color:var(--muted); font-weight:600; }

/* ---------- ফুটার ---------- */
.foot{ margin-top:3rem; padding:1.4rem 0 .5rem; text-align:center;
  border-top:1px solid var(--stroke); animation:fadeUp 1s ease both; }
.foot .dev{ font-family:'Baloo Da 2',cursive; font-size:1.05rem; color:var(--muted); }
.foot .dev b{ background:var(--grad); -webkit-background-clip:text;
  background-clip:text; color:transparent; }
.foot .links{ margin-top:.4rem; }
.foot a{ color:var(--cyan); text-decoration:none; font-weight:600; font-size:.9rem; }
.foot a:hover{ color:var(--rose); }
.foot .fine{ margin-top:.6rem; color:#44526B; font-size:.74rem; letter-spacing:.02em; }

@keyframes sheen{ to{ background-position:220% center; } }
@keyframes scan{ 0%{left:-45%;} 100%{left:100%;} }
@keyframes pulse{ 0%,100%{ transform:scale(1); opacity:1;} 50%{ transform:scale(.7); opacity:.55;} }
@keyframes fadeUp{ from{opacity:0; transform:translateY(16px);} to{opacity:1; transform:none;} }
@keyframes slideIn{ from{opacity:0; transform:translateX(-18px);} to{opacity:1; transform:none;} }
@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important;} }
@media (max-width:560px){ .brand{ font-size:3.3rem; } .stats{ gap:.9rem; } .stat{ font-size:.76rem; } }
</style>
""", unsafe_allow_html=True)

# ================= হিরো =================
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ Bangla Comment AI</div><br>
  <h1 class="brand">SANS1</h1>
  <h2>বাংলা কমেন্ট ক্লাসিফায়ার</h2>
  <p>১৬,০০০+ আসল ফেসবুক কমেন্ট দিয়ে ট্রেইন করা AI — যেকোনো বাংলা বা বাংলিশ
  কমেন্ট দিন, SANS1 মুহূর্তেই বলে দেবে সেটা Positive, Negative, Toxic নাকি Neutral।</p>
  <div class="scanline"></div>
</div>

<div class="stats">
  <div class="stat">📊 <b>16K+</b> ডেটাসেট</div>
  <div class="stat">🎯 <b>4</b> ক্লাস</div>
  <div class="stat">⚡ <b>Real-time</b> ফলাফল</div>
  <div class="stat">🧠 <b>SANS1</b> v1.0</div>
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
    "positive": ("✅ Positive", "pos"),
    "negative": ("⚠️ Negative", "neg"),
    "toxic":    ("🚫 Toxic",    "tox"),
    "neutral":  ("💬 Neutral",  "neu"),
}

# ================= ইনপুট =================
def clear_text():
    st.session_state.comment_input = ""

text = st.text_area("কমেন্ট লিখুন — একাধিক হলে প্রতি লাইনে একটা:",
                    height=150, key="comment_input",
                    placeholder="যেমন: ভাই ভিডিওটা সেরা হইছে")

col_btn, col_clear = st.columns([5, 1])
with col_btn:
    analyze = st.button("⚡ বিশ্লেষণ করুন", use_container_width=True)
with col_clear:
    st.button("✕ মুছুন", use_container_width=True, on_click=clear_text)

# ================= বিশ্লেষণ =================
if analyze and text.strip():
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    counts = Counter()
    results = []

    progress = st.progress(0, text="SANS1 বিশ্লেষণ করছে...")
    for idx, line in enumerate(lines):
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
        pred_id = int(probs.argmax())
        raw = id2label[pred_id]
        conf = float(probs[pred_id]) * 100
        bn, cls = LABELS.get(raw, (raw, "neu"))
        counts[bn] += 1
        results.append((line, bn, cls, conf))
        progress.progress((idx + 1) / len(lines),
                          text=f"SANS1 বিশ্লেষণ করছে... {idx+1}/{len(lines)}")
    progress.empty()

    for i, (line, bn, cls, conf) in enumerate(results):
        safe = line.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="res {cls}" style="animation-delay:{i*0.06}s">'
            f'<span class="badge">{bn}</span>'
            f'<span class="txt">{safe}</span>'
            f'<span class="conf">{conf:.1f}%</span></div>',
            unsafe_allow_html=True)

    if len(lines) > 1:
        chips = "".join(
            f'<div class="chip"><span class="dot {LABELS[k][1] if k in LABELS else "neu"}"></span>'
            f'<span class="n">{n}</span><span class="l">{lbl.split(" ",1)[1]}</span></div>'
            for lbl, n in counts.most_common()
            for k in [next((kk for kk, vv in LABELS.items() if vv[0] == lbl), "neutral")])
        st.markdown(f'<div class="sum">{chips}</div>', unsafe_allow_html=True)

elif analyze:
    st.info("বিশ্লেষণের জন্য আগে একটা কমেন্ট লিখুন।")

# ================= ফুটার =================
st.markdown(f"""
<div class="foot">
  <div class="dev">Developed by <b>Gulam Sakaria</b></div>
  <div class="links">
    <a href="{FB_LINK}" target="_blank">📘 Facebook</a> &nbsp;•&nbsp;
    <a href="{GH_LINK}" target="_blank">🐙 GitHub</a>
  </div>
  <div class="fine">SANS1 v1.0 · নিজস্ব সংগৃহীত ১৬,০০০+ কমেন্টের ডেটাসেট · built on BanglaBERT</div>
</div>
""", unsafe_allow_html=True)
