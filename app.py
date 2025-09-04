# app.py — جدول الأهداف الأسبوعية فقط (كامل)
# =========================================================
# TriplePower — جدول الأهداف (الأسبوعي فقط — سطر واحد لكل رمز)
# يعتمد اختيار "الشمعة البيعية المعتبرة" على كسر الشمعة الشرائية
# (بنفسها أو لاحقًا) وفق شرط 55% — أسبوعي.
# فلتر الاختراق الثلاثي اختياري (يومي مؤكَّد + أسبوعي إيجابي + أول شهري)،
# لكن الجدول النهائي يعرض الأعمدة الأسبوعية فقط.
# الأهداف تُحسب وفق مدرسة القوة الثلاثية من قمة/قاع الشمعة البيعية المعتبرة:
#   T1 = H + 1*R,  T2 = H + 2*R,  T3 = H + 3*R  حيث R = H - L
# مع استخدام إغلاق الأسبوع الأخير المغلق كسعرٍ افتراضي،
# ويمكن عرض آخر سعر يومي متاح (اختياري) للعرض فقط.
# =========================================================

import os, re, hashlib, secrets, base64
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# =============================
# تحميل متغيرات البيئة
# =============================
load_dotenv()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL")
if not SHEET_CSV_URL:
    st.error("⚠️ لم يتم ضبط SHEET_CSV_URL في متغيرات البيئة. أضفه ثم أعد التشغيل.")
    st.stop()

# =============================
# تهيئة الصفحة + RTL
# =============================
st.set_page_config(page_title="🎯 جدول الأهداف (اليومي + الأسبوعي) | TriplePower", layout="wide")
st.markdown("""
<style>
  :root, html, body, .stApp { direction: rtl; }
  .stApp { text-align: right; }
  input, textarea, select { direction: rtl; text-align: right; }
  .stTextInput input, .stTextArea textarea, .stSelectbox div[role="combobox"],
  .stNumberInput input, .stDateInput input, .stMultiSelect [data-baseweb],
  label, .stButton button { text-align: right; }
  table { direction: rtl; }
  .stAlert { direction: rtl; }
  /* تلوين الرأس */
  table thead th { background:#0B7; color:#fff; padding:8px; }
  table, th, td { border:1px solid #ddd; border-collapse:collapse; }
  td { padding:8px; text-align:center; }
  tr:nth-child(even){ background:#f9f9f9; }
  tr:hover { background:#f1f1f1; }
  .positive { background:#d4edda; color:#155724; font-weight:bold; }
  .negative { background:#f8d7da; color:#721c24; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# =============================
# دوال مساعدة عامة
# =============================

def linkify(text: str) -> str:
    if not text: return ""
    return re.sub(r"(https?://[^\s]+)", r"[\1](\1)", text)


def load_important_links() -> str:
    try:
        with open("روابط مهمة.txt","r",encoding="utf-8") as f: return f.read()
    except FileNotFoundError:
        return "⚠️ ملف 'روابط مهمة.txt' غير موجود."


def load_symbols_names(file_path: str, market_type: str) -> dict:
    mapping = {}
    try:
        with open(file_path,"r",encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts=line.split('\t',1)
                if len(parts)==2:
                    symbol,name=parts
                    if market_type=="سعودي": mapping[symbol.strip()]=name.strip()
                    else: mapping[symbol.strip().upper()]=name.strip()
        return mapping
    except Exception as e:
        st.warning(f"⚠️ خطأ في تحميل ملف {file_path}: {e}")
        return {}

# ===== مصادقة (PBKDF2) =====
PBKDF_ITER = 100_000

def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, algoname, iters, b64salt, b64hash = stored.split("$",4)
        if algo!="pbkdf2" or algoname!="sha256": return False
        iters=int(iters); salt=base64.b64decode(b64salt); expected=base64.b64decode(b64hash)
        test=hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
        return secrets.compare_digest(test, expected)
    except Exception: return False

@st.cache_data(ttl=600)
def load_users():
    df=pd.read_csv(SHEET_CSV_URL, dtype=str)
    return df.to_dict("records")

def check_login(username, password, users):
    username=(username or "").strip(); password=(password or "")
    for u in users:
        if u.get("username")==username:
            h=u.get("password_hash")
            if h: return u if _pbkdf2_verify(password,h) else None
            if u.get("password")==password: return u
    return None

def is_expired(expiry_date: str)->bool:
    try: return datetime.strptime(expiry_date.strip(),"%Y-%m-%d").date()<date.today()
    except Exception: return True

# =============================
# تحميل البيانات من ياهو
# =============================
@st.cache_data(ttl=300)
def fetch_data(symbols, sd, ed, iv):
    if not symbols or not str(symbols).strip(): return None
    try:
        return yf.download(
            tickers=symbols, start=sd, end=ed+timedelta(days=1),
            interval=iv, group_by="ticker", auto_adjust=False,
            progress=False, threads=True
        )
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}"); return None


def extract_symbol_df(batch_df: pd.DataFrame, code: str)->pd.DataFrame|None:
    if batch_df is None or batch_df.empty: return None
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            if code in set(batch_df.columns.get_level_values(0)):
                return batch_df[code].reset_index()
            return None
        else:
            cols=set(map(str.lower,batch_df.columns.astype(str)))
            if {"open","high","low","close"}.issubset(cols):
                return batch_df.reset_index()
    except Exception: return None
    return None


def drop_last_if_incomplete(df: pd.DataFrame, tf: str, suffix: str, allow_intraday_daily: bool=False)->pd.DataFrame:
    if df is None or df.empty: return df
    dfx=df.copy()
    if dfx.iloc[-1][["Open","High","Low","Close"]].isna().any():
        return dfx.iloc[:-1] if len(dfx)>1 else dfx.iloc[0:0]
    last_dt=pd.to_datetime(dfx["Date"].iloc[-1]).date()
    if tf=="1d":
        if allow_intraday_daily: return dfx
        if suffix==".SR":
            now=datetime.now(ZoneInfo("Asia/Riyadh"))
            after_close=(now.hour>15) or (now.hour==15 and now.minute>=10)
            if last_dt==now.date() and not after_close:
                return dfx.iloc[:-1] if len(dfx)>1 else dfx.iloc[0:0]
        else:
            now=datetime.now(ZoneInfo("America/New_York"))
            after_close=(now.hour>16) or (now.hour==16 and now.minute>=5)
            if last_dt==now.date() and not after_close:
                return dfx.iloc[:-1] if len(dfx)>1 else dfx.iloc[0:0]
        return dfx
    if tf=="1wk": return dfx
    if tf=="1mo":
        now=datetime.now(ZoneInfo("Asia/Riyadh" if suffix==".SR" else "America/New_York"))
        today=now.date()
        if last_dt.year==today.year and last_dt.month==today.month:
            return dfx.iloc[:-1] if len(dfx)>1 else dfx.iloc[0:0]
        return dfx
    return dfx

# =============================
# منطق 55% (بيعية/شرائية) مع "الكسر الآن أو لاحقًا"
# =============================

def _body_ratio(c,o,h,l):
    rng=(h-l)
    return np.where(rng!=0, np.abs(c-o)/rng, 0.0), rng


def last_sell_anchor_info(_df: pd.DataFrame, pct: float = 0.55):
    """
    تُرجع dict تحتوي idx/H/L/R لآخر شمعة بيعية 55% كسرت قاع شمعة شرائية 55%
    (بنفسها أو لاحقًا) — وفق منطق المدرسة الأصلي.
    لا يتم تقريب H/L هنا للحفاظ على الدقة؛ التقريب يكون عند الإخراج فقط.
    """
    if _df is None or _df.empty:
        return None
    df = _df[["Open","High","Low","Close"]].dropna().copy()
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy();  c = df["Close"].to_numpy()

    # تعريف شموع 55%
    rng = (h - l)
    br  = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)  # بيعية معتبرة
    win55  = (c > o) & (br >= pct) & (rng != 0)  # شرائية معتبرة

    # آخر قاع شرائي 55% قبل كل نقطة
    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur

    # أصغر قاع مستقبلي (لتحقيق شرط "الكسر لاحقًا")
    future_min = np.minimum.accumulate(l[::-1])[::-1]

    # الشمعة البيعية المعتبرة: كسرت القاع الشرائي الآن أو لاحقًا
    considered_sell = (
        lose55 &
        ~np.isnan(last_win_low) &
        ((l <= last_win_low) | (future_min <= last_win_low))
    )

    idx = np.where(considered_sell)[0]
    if len(idx) == 0:
        return None

    j = int(idx[-1])
    H = float(h[j]); L = float(l[j]); R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return {"idx": j, "H": H, "L": L, "R": R}


def last_sell_anchor_targets(_df: pd.DataFrame, pct: float = 0.55):
    """
    تُرجع (H, T1, T2, T3) بحسب آخر شمعة بيعية 55% المعتبرة.
    الأهداف وفق مدرسة القوة الثلاثية (مقياس +100/+200/+300 من H):
      T1 = H + 1*R,  T2 = H + 2*R,  T3 = H + 3*R  حيث R = H - L.
    الحساب يتم بقيم H/L الدقيقة دون تقريب مسبق، ثم يُقرّب الناتج للعرض.
    """
    info = last_sell_anchor_info(_df, pct=pct)
    if info is None:
        return None
    H = float(info["H"]); L = float(info["L"])
    R = H - L
    return (
        round(H, 2),
        round(H + 1.0*R, 2),
        round(H + 2.0*R, 2),
        round(H + 3.0*R, 2)
    )

# جديد: اختيار المرساة الأسبوعية بناءً على "آخر اختراق أسبوعي"
# المرساة المطلوبة هي قمة الشمعة البيعية الأسبوعية التي كان اختراقُ قمتها
# هو "آخر" إشارة اختراق (أحدث إغلاق أعلى من قمتها) عبر السلسلة الأسبوعية.

def weekly_latest_breakout_anchor_targets(_df: pd.DataFrame, pct: float = 0.55):
    if _df is None or _df.empty:
        return None
    df = _df[["Open","High","Low","Close"]].dropna().copy()
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy();  c = df["Close"].to_numpy()

    # تعريف شموع 55%
    rng = (h - l)
    br  = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)   # بيعية 55%
    win55  = (c > o) & (br >= pct) & (rng != 0)   # شرائية 55%

    # آخر قاع شرائي 55% قبل كل نقطة
    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur

    # أصغر قاع مستقبلي (تحقيق شرط كسر القاع لاحقًا)
    future_min = np.minimum.accumulate(l[::-1])[::-1]

    # جميع الشموع البيعية المعتبرة (التي كسرت قاع شرائية الآن أو لاحقًا)
    considered_sell = (
        lose55 &
        ~np.isnan(last_win_low) &
        ((l <= last_win_low) | (future_min <= last_win_low))
    )
    anchors = np.where(considered_sell)[0]
    if len(anchors) == 0:
        return None

    # لكل مرساة، ابحث أول إغلاق أسبوعي لاحق أعلى من قمتها (لحظة الاختراق)
    breakout_events = []  # عناصرها (t_break, j_anchor)
    for j in anchors:
        # أول t > j بحيث Close[t] > High[j]
        later = np.where(c[j+1:] > h[j])[0]
        if len(later) == 0:
            continue
        t_break = int(j + 1 + later[0])
        breakout_events.append((t_break, j))

    if len(breakout_events) == 0:
        return None

    # اختر "آخر" اختراق أسبوعي (أحدث t_break)
    t_last, j_last = max(breakout_events, key=lambda x: x[0])

    H = float(h[j_last]); L = float(l[j_last]); R = H - L
    if not np.isfinite(R) or R <= 0:
        return None

    return (
        round(H, 2),
        round(H + 1.0*R, 2),
        round(H + 2.0*R, 2),
        round(H + 3.0*R, 2)
    )

