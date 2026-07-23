# -*- coding: utf-8 -*-
# ============================================================
#  SANS1 — বাংলা কমেন্ট ক্লাসিফায়ার  (Blue Edition)
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
                   page_icon="🔷", layout="centered")

# ================= ডিজাইন =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@700;800;900&family=Hind+Siliguri:wght@400;500;600;700&display=swap');

:root{
  --bg:#F5F8FC;          /* হালকা নীলচে সাদা */
  --card:#FFFFFF;
  --soft:#EAF1F9;
  --ink:#0F2440;         /* গাঢ় নেভি */
  --muted:#4A6285;
  --faint:#8DA2BE;
  --blue:#1D6FE0;        /* প্রধান নীল */
  --blue-dark:#1558B8;
  --blue-soft:#E3EEFC;
  --line:#DCE6F2;
  --pos:#1E7E54; --pos-bg:#E1F3EA;
  --neg:#A9691B; --neg-bg:#FBF0DC;
  --tox:#C03546; --tox-bg:#FBE5E8;
  --neu:#1D6FE0; --neu-bg:#E3EEFC;
}

html, body, [class*="css"], .stApp{
  font-family:'Hind Siliguri',sans-serif; color:var(--ink);
}
.stApp{
  background:
    radial-gradient(900px 420px at 50% -8%, rgba(29,111,224,.07), transparent 62%),
    var(--bg);
}
#MainMenu, footer, header{ display:none; }
.block-container{ padding-top:2.6rem; max-width:720px; }

/* ---------- হিরো ---------- */
.hero{ text-align:center; margin-bottom:2rem; animation:rise .7s ease both; }
.mark{
  display:inline-flex; align-items:center; gap:.55rem;
  font-family:'Inter',sans-serif; font-weight:900;
  font-size:3.2rem; line-height:1; letter-spacing:-.02em; color:var(--ink);
}
.mark .spark{
  color:var(--blue); font-size:2.1rem; display:inline-block;
  animation:breathe 3.2s ease-in-out infinite;
}
.mark .one{ color:var(--blue); }
.hero .sub{ margin-top:.55rem; font-size:1.12rem; font-weight:600; color:var(--ink); }
.hero .desc{
  margin:.55rem auto 0; max-width:530px;
  color:var(--muted); font-size:.95rem; line-height:1.75;
}
.hero .meta{
  display:flex; justify-content:center; gap:.5rem; flex-wrap:wrap; margin-top:1.1rem;
}
.pill{
  font-size:.78rem; font-weight:600; color:var(--muted);
  background:var(--card); border:1px solid var(--line);
  border-radius:999px; padding:.3rem .85rem;
  box-shadow:0 1px 2px rgba(15,36,64,.04);
}
.pill b{ color:var(--blue); }

/* ---------- ইনপুট ---------- */
.stTextArea textarea{
  background:var(--card) !important; color:var(--ink) !important;
  caret-color:var(--blue) !important;
  border:1.5px solid var(--line) !important; border-radius:16px !important;
  font-family:'Hind Siliguri',sans-serif !important;
  font-size:1.03rem !important; line-height:1.75 !important;
  padding:1rem !important;
  box-shadow:0 1px 3px rgba(15,36,64,.06);
  transition:border .2s, box-shadow .2s;
}
.stTextArea textarea::placeholder{ color:var(--faint) !important; }
.stTextArea textarea:focus{
  border-color:var(--blue) !important;
  box-shadow:0 0 0 3px rgba(29,111,224,.15) !important;
}
.stTextArea label{
  color:var(--muted) !important; font-weight:600 !important; font-size:.9rem !important;
}

/* ---------- বাটন ---------- */
.stButton>button{
  width:100%; border-radius:12px; padding:.68rem 0;
  font-family:'Hind Siliguri',sans-serif; font-weight:700; font-size:1.02rem;
  transition:transform .15s, box-shadow .15s, background .15s;
}
div[data-testid="column"]:first-child .stButton>button{
  background:linear-gradient(135deg, var(--blue), var(--blue-dark));
  color:#fff; border:none;
  box-shadow:0 3px 12px rgba(29,111,224,.32);
}
div[data-testid="column"]:first-child .stButton>button:hover{
  transform:translateY(-1px);
  box-shadow:0 6px 18px rgba(29,111,224,.42); color:#fff;
}
div[data-testid="column"]:last-child .stButton>button{
  background:var(--card); color:var(--muted); border:1.5px solid var(--line);
}
div[data-testid="column"]:last-child .stButton>button:hover{
  border-color:var(--blue); color:var(--blue);
}

/* ---------- রেজাল্ট কার্ড ---------- */
.res{
  display:flex; align-items:center; gap:.85rem;
  background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:.85rem 1.05rem; margin:.5rem 0;
  box-shadow:0 1px 3px rgba(15,36,64,.05);
  animation:rise .45s ease both;
  transition:box-shadow .2s, border-color .2s, transform .2s;
}
.res:hover{
  box-shadow:0 5px 16px rgba(15,36,64,.10);
  border-color:#C6D6EA; transform:translateY(-1px);
}
.badge{
  font-weight:700; font-size:.8rem; white-space:nowrap;
  padding:.28rem .8rem; border-radius:999px; flex-shrink:0;
}
.res.pos .badge{ background:var(--pos-bg); color:var(--pos); }
.res.neg .badge{ background:var(--neg-bg); color:var(--neg); }
.res.tox .badge{ background:var(--tox-bg); color:var(--tox); }
.res.neu .badge{ background:var(--neu-bg); color:var(--neu); }
.txt{ color:var(--ink); font-size:.98rem; flex:1; word-break:break-word; line-height:1.55; }
.conf{
  margin-left:auto; flex-shrink:0; white-space:nowrap;
  color:var(--faint); font-size:.82rem; font-variant-numeric:tabular-nums;
}

/* ---------- সারসংক্ষেপ ---------- */
.sum{
  display:flex; gap:.6rem; flex-wrap:wrap; justify-content:center;
  margin-top:1.1rem; padding:1rem;
  background:var(--soft); border:1px solid var(--line); border-radius:14px;
  animation:rise .5s ease both;
}
.chip{
  display:flex; align-items:center; gap:.5rem;
  background:var(--card); border:1px solid var(--line);
  border-radius:999px; padding:.4rem 1rem;
}
.dot{ width:9px; height:9px; border-radius:50%; }
.dot.pos{ background:var(--pos);} .dot.neg{ background:var(--neg);}
.dot.tox{ background:var(--tox);} .dot.neu{ background:var(--neu);}
.chip .n{ font-weight:800; font-size:1.05rem; color:var(--ink);
  font-family:'Inter',sans-serif; font-variant-numeric:tabular-nums; }
.chip .l{ font-size:.8rem; color:var(--muted); font-weight:600; }

/* ---------- প্রগ্রেস ---------- */
.stProgress > div > div > div{ background:var(--blue) !important; }

/* ---------- ফুটার ---------- */
.foot{
  margin-top:3.2rem; padding-top:1.4rem; text-align:center;
  border-top:1px solid var(--line);
}
.foot .dev{ font-size:.95rem; color:var(--muted); }
.foot .dev b{ color:var(--blue); }
.foot .links{ margin-top:.35rem; }
.foot a{ color:var(--blue); text-decoration:none; font-weight:600; font-size:.88rem; }
.foot a:hover{ text-decoration:underline; }
.foot .fine{ margin-top:.55rem; color:var(--faint); font-size:.74rem; }

@keyframes rise{ from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;} }
@keyframes breathe{ 0%,100%{ transform:scale(1) rotate(0deg); opacity:1;}
  50%{ transform:scale(1.12) rotate(10deg); opacity:.8;} }
@media (prefers-reduced-motion: reduce){ *{animation:none !important; transition:none !important;} }
@media (max-width:560px){ .mark{ font-size:2.5rem; } .mark .spark{ font-size:1.7rem; } }
</style>
""", unsafe_allow_html=True)

# ================= হিরো =================
st.markdown("""
<div class="hero">
  <div class="mark"><span class="spark">🔷</span>SANS<span class="one">1</span></div>
  <div class="sub">বাংলা কমেন্ট ক্লাসিফায়ার</div>
  <p class="desc">১৬,০০০+ আসল ফেসবুক কমেন্ট দিয়ে ট্রেইন করা AI।
  যেকোনো বাংলা বা বাংলিশ কমেন্ট লিখুন — SANS1 বলে দেবে সেটা
  Positive, Negative, Toxic নাকি Neutral।</p>
  <div class="meta">
    <span class="pill">📊 <b>16K+</b> ডেটাসেট</span>
    <span class="pill">🎯 <b>৪টি</b> ক্লাস</span>
    <span class="pill">⚡ <b>Real-time</b></span>
    <span class="pill">🔷 SANS1 <b>v1.0</b></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ================= মডেল =================
@st.cache_resource(show_spinner="🔷 SANS1 প্রস্তুত হচ্ছে... (প্রথমবার ২-৩ মিনিট লাগতে পারে)")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()
id2label = model.config.id2label

LABELS = {
    "positive": ("Positive", "pos"),
    "negative": ("Negative", "neg"),
    "toxic":    ("Toxic",    "tox"),
    "neutral":  ("Neutral",  "neu"),
}

# ================= ইনপুট =================
def clear_text():
    st.session_state.comment_input = ""

text = st.text_area("কমেন্ট লিখুন — একাধিক হলে প্রতি লাইনে একটা:",
                    height=150, key="comment_input",
                    placeholder="যেমন: ভাই ভিডিওটা সেরা হইছে")

col_btn, col_clear = st.columns([5, 1])
with col_btn:
    analyze = st.button("বিশ্লেষণ করুন", use_container_width=True)
with col_clear:
    st.button("মুছুন", use_container_width=True, on_click=clear_text)

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
        label_en, cls = LABELS.get(raw, (raw, "neu"))
        counts[(label_en, cls)] += 1
        results.append((line, label_en, cls, conf))
        progress.progress((idx + 1) / len(lines),
                          text=f"SANS1 বিশ্লেষণ করছে... {idx+1}/{len(lines)}")
    progress.empty()

    for i, (line, label_en, cls, conf) in enumerate(results):
        safe = line.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="res {cls}" style="animation-delay:{i*0.05}s">'
            f'<span class="badge">{label_en}</span>'
            f'<span class="txt">{safe}</span>'
            f'<span class="conf">{conf:.1f}%</span></div>',
            unsafe_allow_html=True)

    if len(lines) > 1:
        chips = "".join(
            f'<div class="chip"><span class="dot {cls}"></span>'
            f'<span class="n">{n}</span><span class="l">{lbl}</span></div>'
            for (lbl, cls), n in counts.most_common())
        st.markdown(f'<div class="sum">{chips}</div>', unsafe_allow_html=True)

elif analyze:
    st.info("বিশ্লেষণের জন্য আগে একটা কমেন্ট লিখুন।")

# ================= ফুটার =================
st.markdown(f"""
<div class="foot">
  <div class="dev">Developed by <b>Gulam Sakaria</b></div>
  <div class="links">
    <a href="{FB_LINK}" target="_blank">Facebook</a> &nbsp;·&nbsp;
    <a href="{GH_LINK}" target="_blank">GitHub</a>
  </div>
  <div class="fine">SANS1 v1.0 · নিজস্ব সংগৃহীত ১৬,০০০+ কমেন্টের ডেটাসেট · built on BanglaBERT</div>
</div>
""", unsafe_allow_html=True)
