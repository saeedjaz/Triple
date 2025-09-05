# app.py — جدول الأهداف (اليومي + الأسبوعي)
# =========================================================
# TriplePower — جدول الأهداف (سطر واحد لكل رمز)
# ✅ كسر الشرائية بالإغلاق (Close)
# ✅ سلبية شهرية صريحة عند كسر قاع الشرائية الشهرية 55%
# ✅ المرساة الحالية = أحدث "مقاومة غير مخترقة"
# 🆕 عمود أخير: "الدعم الاسبوعي" = قاع آخر شمعة شرائية 55% أسبوعية مغلقة
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
# منطق 55% (بيعية/شرائية) + مراسي الاختراق (Close)
# =============================

def _body_ratio(c,o,h,l):
    rng=(h-l)
    return np.where(rng!=0, np.abs(c-o)/rng, 0.0), rng


def last_sell_anchor_info(_df: pd.DataFrame, pct: float = 0.55):
    if _df is None or _df.empty:
        return None
    df = _df[["Open","High","Low","Close"]].dropna().copy()
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy();  c = df["Close"].to_numpy()

    rng = (h - l)
    br  = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55  = (c > o) & (br >= pct) & (rng != 0)

    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur

    future_min_close = np.minimum.accumulate(c[::-1])[::-1]

    considered_sell = (
        lose55 &
        ~np.isnan(last_win_low) &
        ((c <= last_win_low) | (future_min_close <= last_win_low))
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

# -------- مطابقة TradingView: المرساة الحالية --------

def _enumerate_sell_anchors_with_break(df: pd.DataFrame, pct: float=0.55):
    if df is None or df.empty:
        return []
    arr = []
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy();  c = df["Close"].to_numpy()

    rng = (h - l)
    br  = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55  = (c > o) & (br >= pct) & (rng != 0)

    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur

    future_min_close = np.minimum.accumulate(c[::-1])[::-1]

    considered_sell = (
        lose55 & ~np.isnan(last_win_low) &
        ((c <= last_win_low) | (future_min_close <= last_win_low))
    )

    idx = np.where(considered_sell)[0]
    for j in idx:
        later = np.where(c[j+1:] > h[j])[0]
        t_break = int(j + 1 + later[0]) if len(later) else None
        arr.append({
            "j": int(j),
            "H": float(h[j]),
            "L": float(l[j]),
            "R": float(h[j]-l[j]),
            "t_break": t_break,
        })
    return arr


def _select_current_anchor(anchors, mode: str):
    """اختيار المرساة بحسب السياسة:
    - "unbroken": أحدث مرساة غير مخترقة؛ وإن لم توجد فـ "أول اختراق".
    - "first_break": أقدم مرساة تم اختراقها (أول اختراق في الموجة)؛ وإن لم توجد فـ أحدث غير مخترقة.
    - "last_break": أحدث مرساة تم اختراقها تاريخيًا؛ وإن لم توجد فـ أحدث غير مخترقة.
    """
    if not anchors:
        return None
    if mode == "unbroken":
        unbroken = [a for a in anchors if a["t_break"] is None]
        if unbroken:
            return max(unbroken, key=lambda a: a["j"])  # الأحدث زمنيًا من غير المخترقات
        broken = [a for a in anchors if a["t_break"] is not None]
        return min(broken, key=lambda a: a["t_break"]) if broken else None
    elif mode == "first_break":
        broken = [a for a in anchors if a["t_break"] is not None]
        if broken:
            return min(broken, key=lambda a: a["t_break"])  # أول اختراق في الموجة
        return max(anchors, key=lambda a: a["j"])  # fallback: أحدث غير مخترقة
    elif mode == "last_break":
        broken = [a for a in anchors if a["t_break"] is not None]
        if broken:
            return max(broken, key=lambda a: a["t_break"])  # آخر اختراق تاريخي
        return max(anchors, key=lambda a: a["j"])  # fallback
    # افتراضيًا: سلوك المقاومة الحالية
    unbroken = [a for a in anchors if a["t_break"] is None]
    if unbroken:
        return max(unbroken, key=lambda a: a["j"]) 
    broken = [a for a in anchors if a["t_break"] is not None]
    return min(broken, key=lambda a: a["t_break"]) if broken else None


def weekly_latest_breakout_anchor_targets(_df: pd.DataFrame, pct: float = 0.55, mode: str = "first_break"):
    """ترجيع أهداف الأسبوعي وفق سياسة اختيار المرساة.
    - في وضع "first_break" نقيّد الاختيار بكون الاختراق حدث *بعد آخر إشارة شراء أسبوعية* (بداية الموجة)،
      لتطابق TradingView.
    يعيد: ((H, T1, T2, T3), info) أو (None, None)
    """
    if _df is None or _df.empty:
        return None, None
    # إطار عمل داخلي نظيف
    df = _df[["Open","High","Low","Close"]].dropna().copy()
    # حساب جميع المراسي
    anchors = _enumerate_sell_anchors_with_break(df, pct=pct)

    # وظيفة داخلية: إيجاد مؤشر آخر إشارة شراء أسبوعية (بداية موجة)
    def _last_weekly_first_buy_index(_df_ohlc: pd.DataFrame) -> int:
        tmp = detect_breakout_with_state(_df_ohlc.copy(), pct=pct)
        if tmp is None or tmp.empty or "FirstBuySig" not in tmp.columns:
            return 0
        idx = np.where(tmp["FirstBuySig"].to_numpy())[0]
        return int(idx[-1]) if len(idx) else 0

    pick = None
    if mode == "first_break":
        start_i = _last_weekly_first_buy_index(df)
        broken_after = [a for a in anchors if a["t_break"] is not None and a["t_break"] >= start_i]
        if broken_after:
            pick = min(broken_after, key=lambda a: a["t_break"])  # أول اختراق بعد بداية الموجة
        else:
            broken = [a for a in anchors if a["t_break"] is not None]
            if broken:
                pick = min(broken, key=lambda a: a["t_break"])  # أقدم اختراق إجمالاً
            else:
                unbroken = [a for a in anchors if a["t_break"] is None]
                if unbroken:
                    pick = max(unbroken, key=lambda a: a["j"])       # أحدث غير مخترقة
    else:
        # بقية الأوضاع تستعمل المحدِّد العام
        pick = _select_current_anchor(anchors, mode)

    if (not pick) or (not np.isfinite(pick["R"])) or (pick["R"] <= 0):
        return None, None

    H = float(pick["H"]) ; L = float(pick["L"]) ; R = float(pick["R"]) ; j = int(pick["j"])  
    # تاريخ الشمعة الأسبوعية من _df الأصلي
    try:
        if "Date" in _df.columns:
            date_val = pd.to_datetime(_df["Date"].iloc[j]).date()
        else:
            date_val = pd.to_datetime(_df.index[j]).date()
    except Exception:
        date_val = None

    info = {"date": str(date_val) if date_val else None,
            "H": round(H,2), "L": round(L,2), "R": round(R,2)}

    t1 = round(H + 1.0 * R, 2)
    t2 = round(H + 2.0 * R, 2)
    t3 = round(H + 3.0 * R, 2)
    return (round(H, 2), t1, t2, t3), info



def daily_latest_breakout_anchor_targets(_df: pd.DataFrame, pct: float = 0.55, mode: str = "first_break"):
    """ترجيع أهداف اليومي وفق سياسة اختيار المرساة.
    - في وضع "first_break" نقيّد الاختيار بكون الاختراق حدث *بعد آخر إشارة شراء يومية* لبداية الموجة.
    يعيد: (H, T1, T2, T3) أو None
    """
    if _df is None or _df.empty:
        return None
    df = _df[["Open","High","Low","Close"]].dropna().copy()
    anchors = _enumerate_sell_anchors_with_break(df, pct=pct)

    def _last_daily_first_buy_index(_df_ohlc: pd.DataFrame) -> int:
        tmp = detect_breakout_with_state(_df_ohlc.copy(), pct=pct)
        if tmp is None or tmp.empty or "FirstBuySig" not in tmp.columns:
            return 0
        idx = np.where(tmp["FirstBuySig"].to_numpy())[0]
        return int(idx[-1]) if len(idx) else 0

    pick = None
    if mode == "first_break":
        start_i = _last_daily_first_buy_index(df)
        broken_after = [a for a in anchors if a["t_break"] is not None and a["t_break"] >= start_i]
        if broken_after:
            pick = min(broken_after, key=lambda a: a["t_break"])  # أول اختراق بعد بداية الموجة
        else:
            broken = [a for a in anchors if a["t_break"] is not None]
            if broken:
                pick = min(broken, key=lambda a: a["t_break"])  # أقدم اختراق إجمالاً
            else:
                unbroken = [a for a in anchors if a["t_break"] is None]
                if unbroken:
                    pick = max(unbroken, key=lambda a: a["j"])       # أحدث غير مخترقة
    else:
        pick = _select_current_anchor(anchors, mode)

    if (not pick) or (not np.isfinite(pick["R"])) or (pick["R"] <= 0):
        return None

    H = float(pick["H"])  
    R = float(pick["R"])  
    t1 = round(H + 1.0 * R, 2)
    t2 = round(H + 2.0 * R, 2)
    t3 = round(H + 3.0 * R, 2)
    return (round(H, 2), t1, t2, t3)



def daily_latest_breakout_anchor_targets(_df: pd.DataFrame, pct: float = 0.55, mode: str = "first_break"):
    """ترجيع أهداف اليومي وفق سياسة اختيار المرساة.
    يعيد: (H, T1, T2, T3) أو None
    """
    if _df is None or _df.empty:
        return None
    df = _df[["Open", "High", "Low", "Close"]].dropna().copy()
    anchors = _enumerate_sell_anchors_with_break(df, pct=pct)
    pick = _select_current_anchor(anchors, mode)
    if (not pick) or (not np.isfinite(pick["R"])) or (pick["R"] <= 0):
        return None
    H = float(pick["H"])  
    R = float(pick["R"])  
    t1 = round(H + 1.0 * R, 2)
    t2 = round(H + 2.0 * R, 2)
    t3 = round(H + 3.0 * R, 2)
    return (round(H, 2), t1, t2, t3)



def _is_current_week_closed(suffix: str) -> tuple[bool, date]:
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    end_weekday = 3 if suffix == ".SR" else 4   # Thu=3, Fri=4
    days_to_end = (end_weekday - now.weekday()) % 7
    week_end_date = now.date() + timedelta(days=days_to_end)
    close_h, close_m = (15, 10) if suffix == ".SR" else (16, 5)
    closed = (
        now.date() > week_end_date or
        (now.date() == week_end_date and (now.hour > close_h or (now.hour == close_h and now.minute >= close_m)))
    )
    return closed, week_end_date


def resample_weekly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return df_daily.iloc[0:0]
    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty:
        return df_daily.iloc[0:0]

    dfw = df_daily[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    dfw.set_index("Date", inplace=True)
    rule = "W-THU" if suffix == ".SR" else "W-FRI"
    dfw = dfw.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna().reset_index()

    is_closed, current_week_end = _is_current_week_closed(suffix)
    if (not is_closed) and (not dfw.empty):
        last_week_label = pd.to_datetime(dfw["Date"].iat[-1]).date()
        if last_week_label == current_week_end:
            dfw = dfw.iloc[:-1]
    return dfw


def resample_monthly_from_daily(df_daily: pd.DataFrame, suffix: str)->pd.DataFrame:
    if df_daily is None or df_daily.empty: return df_daily.iloc[0:0]
    df_daily=drop_last_if_incomplete(df_daily,"1d",suffix,False)
    if df_daily.empty: return df_daily.iloc[0:0]
    dfm=df_daily[["Date","Open","High","Low","Close"]].dropna().copy()
    dfm.set_index("Date",inplace=True)
    dfm=dfm.resample("M").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna().reset_index()
    tz=ZoneInfo("Asia/Riyadh" if suffix==".SR" else "America/New_York")
    now=datetime.now(tz)
    if not dfm.empty and (dfm["Date"].iat[-1].year==now.year and dfm["Date"].iat[-1].month==now.month):
        dfm=dfm.iloc[:-1]
    return dfm

# =============================
# فلتر اختياري (اختراقات) + دعم أسبوعي 55%
# =============================

def detect_breakout_with_state(df: pd.DataFrame, pct: float=0.55)->pd.DataFrame:
    if df is None or df.empty: return df
    o=df["Open"].values; h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    rng=(h-l); br=np.where(rng!=0, np.abs(c-o)/rng, 0.0)
    lose55=(c<o) & (br>=pct) & (rng!=0)
    win55 =(c>o) & (br>=pct) & (rng!=0)

    last_win_low=np.full(c.shape, np.nan); cur=np.nan
    for i in range(len(c)):
        if win55[i]: cur=l[i]
        last_win_low[i]=cur
    valid_sell_now = lose55 & ~np.isnan(last_win_low) & (c <= last_win_low)

    state=0; states=[]; first_buy=[]; lose_high=np.nan; win_low=np.nan
    for i in range(len(df)):
        buy  = (state==0) and (not np.isnan(lose_high)) and (c[i]>lose_high)
        stop = (state==1) and (not np.isnan(win_low))   and (c[i]<win_low)
        if buy: state=1; first_buy.append(True)
        elif stop: state=0; first_buy.append(False); lose_high=np.nan
        else: first_buy.append(False)
        if valid_sell_now[i]: lose_high=h[i]
        if win55[i]: win_low=l[i]
        states.append(state)

    df=df.copy()
    df["State"]=states; df["FirstBuySig"]=first_buy
    df["LoseCndl55"]=valid_sell_now; df["WinCndl55"]=win55
    return df


def weekly_state_from_daily(df_daily: pd.DataFrame, suffix: str)->bool:
    dfw=resample_weekly_from_daily(df_daily,suffix)
    if dfw.empty: return False
    dfw=detect_breakout_with_state(dfw)
    return bool(dfw["State"].iat[-1]==1)


def monthly_first_breakout_from_daily(df_daily: pd.DataFrame, suffix: str)->bool:
    dfm=resample_monthly_from_daily(df_daily,suffix)
    if dfm is None or dfm.empty: return False
    dfm=detect_breakout_with_state(dfm)
    return bool(dfm["FirstBuySig"].iat[-1])

# ——— الدعم الأسبوعي: قاع آخر شمعة شرائية 55% أسبوعية مغلقة ———

def weekly_last_bullish55_low_value(df_w: pd.DataFrame, pct: float=0.55):
    if df_w is None or df_w.empty:
        return None
    df = df_w[["Open","High","Low","Close"]].dropna().copy()
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy();  c = df["Close"].to_numpy()
    rng = (h - l)
    br  = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    win55 = (c > o) & (br >= pct) & (rng != 0)
    idx = np.where(win55)[0]
    if len(idx)==0:
        return None
    return float(l[int(idx[-1])])

# =============================
# تنسيق العرض + تقريب التيك
# =============================

def _fmt_num(x):
    try: return f"{float(x):.2f}"
    except Exception: return "—"

# تقريب إلى أقرب تيك (0.01/0.05/0.1 ...)
def round_to_tick(x, tick=0.01):
    try:
        fx = float(x)
        return round(round(fx / tick) * tick, 2)
    except Exception:
        return x


def render_table(df: pd.DataFrame)->str:
    from html import escape as esc
    html=["<table><thead><tr>"]
    for col in df.columns: html.append(f"<th>{esc(str(col))}</th>")
    html.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        # سعر الإغلاق المعتمد للمقارنة
        try:
            close_val=float(str(r["سعر الإغلاق"]).replace(",",""))
        except Exception:
            close_val=None
        # قيمة الدعم الأسبوعي (قد تكون "—") لاستخدامها في التلوين الشرطي
        support_val=None
        try:
            sv=str(r.get("الدعم الاسبوعي","—")).strip()
            if sv not in ("—","",None):
                support_val=float(sv.replace(",",""))
        except Exception:
            support_val=None

        html.append("<tr>")
        for col in df.columns:
            val=r[col]; cls=""
            # تلوين قمة الشمعة البيعية (يومي/أسبوعي) وفق المقارنة مع الإغلاق
            if close_val is not None and col in {"قمة الشمعة البيعية الاسبوعية","قمة الشمعة البيعية اليومية"}:
                try:
                    top=float(str(val).replace(",",""))
                    cls="positive" if close_val>=top else "negative"
                except Exception:
                    cls=""
            # 🆕 تلوين الدعم الأسبوعي بالأحمر عند الكسر (إغلاق أدنى من الدعم)
            if col == "الدعم الاسبوعي" and close_val is not None and support_val is not None:
                if close_val < support_val:
                    cls = "negative"
            html.append(f'<td class="{cls}">{esc(str(val))}</td>')
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

# =============================
# جلسة العمل (تسجيل الدخول)
# =============================

st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("login_error", None)
st.session_state.setdefault("login_attempts", 0)

def do_login():
    if st.session_state.login_attempts>=5:
        st.session_state.login_error="too_many"; return
    users=load_users()
    me=check_login(st.session_state.login_username, st.session_state.login_password, users)
    if me is None:
        st.session_state.login_attempts+=1; st.session_state.login_error="bad"
    elif is_expired(me.get("expiry","")):
        st.session_state.login_error="expired"
    else:
        st.session_state.authenticated=True; st.session_state.user=me; st.session_state.login_error=None

if not st.session_state.authenticated:
    c1,c2=st.columns([2,1])
    with c2:
        st.markdown('<h3 style="font-size:20px;">🔒 تسجيل دخول المشتركين</h3>', unsafe_allow_html=True)
        st.text_input("اسم المستخدم", key="login_username")
        st.text_input("كلمة المرور", type="password", key="login_password")
        st.button("دخول", key="login_btn", on_click=do_login)
        if st.session_state.login_error=="bad": st.error("⚠️ اسم المستخدم أو كلمة المرور غير صحيحة.")
        elif st.session_state.login_error=="expired": st.error("⚠️ انتهى اشتراكك. يرجى التجديد.")
        elif st.session_state.login_error=="too_many": st.error("⛔ محاولات كثيرة. حاول لاحقًا.")
    with c1:
        st.markdown(
            "<div style='background:#f0f2f6;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgb(0 0 0 / 0.1);line-height:1.6;'>"
            "<h3 style='font-size:20px;'>منصة القوة الثلاثية TriplePower</h3>"
            + linkify(load_important_links()) + "</div>",
            unsafe_allow_html=True,
        )
    st.stop()

# تحقق الاشتراك
if is_expired(st.session_state.user["expiry"]):
    st.warning("⚠️ انتهى اشتراكك. تم تسجيل خروجك تلقائيًا.")
    st.session_state.authenticated=False; st.session_state.user=None; st.rerun()

# =============================
# بعد الدخول
# =============================
me=st.session_state.user
st.markdown("---")
with st.sidebar:
    st.markdown(f"""<div style=\"background:#28a745;padding:10px;border-radius:5px;color:#fff;
                     font-weight:bold;text-align:center;margin-bottom:10px;\">
                     ✅ اشتراكك سارٍ حتى: {me['expiry']}</div>""", unsafe_allow_html=True)

    try:
        expiry_dt=datetime.strptime(me["expiry"].strip(),"%Y-%m-%d").date()
        today_riyadh=datetime.now(ZoneInfo("Asia/Riyadh")).date()
        days_left=(expiry_dt-today_riyadh).days
        if 0<=days_left<=3: st.warning(f"⚠️ تبقّى {days_left} يومًا على انتهاء الاشتراك.")
    except Exception: pass

    market=st.selectbox("اختر السوق", ["السوق السعودي","السوق الأمريكي"], key="market_select") 
    suffix=".SR" if market=="السوق السعودي" else ""
    apply_triple_filter=st.checkbox(
        "اشتراط الاختراق الثلاثي (اختياري)", value=False, key="triple_filter",
        help="لن يُعرض الرمز إلا إذا تحقق (اختراق يومي مؤكد + أسبوعي إيجابي + أول اختراق شهري)."
    )
    start_date=st.date_input("من", date(2020,1,1), key="start_date_input")
    end_date  =st.date_input("إلى", date.today(), key="end_date_input")
    allow_intraday_daily=st.checkbox("👁️ عرض السعر اليومي الحالي بدل إغلاق الأسبوع (اختياري)", value=False, key="intraday_price",
                                     help="إذا فُعل، يعرض آخر سعر يومي متاح؛ وإلا يعرض إغلاق آخر أسبوع مغلق.")
    batch_size=st.slider("حجم الدُفعة عند الجلب", 20, 120, 60, 10, key="batch_size_slider")

    enable_tick_round = st.checkbox("تقريب الأهداف حسب تيك السعر", value=False, key="tick_round_enable")
    tick_value = st.selectbox("قيمة التيك", [0.01, 0.05, 0.1], index=0, key="tick_value") if enable_tick_round else None

    # 🆕 سياسة اختيار المرساة المستخدمة لحساب الأهداف
    anchor_policy = st.selectbox(
        "سياسة اختيار المرساة",
        [
            "المقاومة الحالية (غير مخترقة)",
            "أول اختراق في الموجة (نمط TradingView)",
            "آخر اختراق تاريخي",
        ],
        index=1,  # ⚠️ الافتراضي الآن = أول اختراق في الموجة لمطابقة الشارت
        help="اختَر القاعدة التي تُحدِّد منها الشمعة البيعية المعتبرة لحساب الأهداف الأسبوعية/اليومية."
    )
    _MODE_MAP = {
        "المقاومة الحالية (غير مخترقة)": "unbroken",
        "أول اختراق في الموجة (نمط TradingView)": "first_break",
        "آخر اختراق تاريخي": "last_break",
    }
    anchor_mode = _MODE_MAP.get(anchor_policy, "first_break")

    # 🧪 خيار تشخيصي لإظهار تفاصيل المرساة المختارة
    show_anchor_debug = st.checkbox("إظهار معلومات المرساة الأسبوعية في الجدول (تشخيص)", value=False)

    symbol_name_dict = load_symbols_names("saudiSY.txt","سعودي") if suffix==".SR" else load_symbols_names("usaSY.txt","امريكي")

    if st.button("🎯 رموز تجريبية", key="demo_symbols_btn"):
        st.session_state.symbols = "3080 4003 4013" if suffix==".SR" else "AAPL MSFT GOOGL"

    try:
        with open("رموز الاسواق العالمية.xlsx","rb") as f:
            st.download_button("📥 تحميل ملف رموز الأسواق", f, "رموز الاسواق العالمية.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except FileNotFoundError:
        st.warning("⚠️ ملف الرموز غير موجود بجانب app.py")

    if st.button("تسجيل الخروج", key="logout_btn"):
        st.session_state.authenticated=False; st.session_state.user=None; st.rerun()

# إدخال الرموز
symbols_input=st.text_area("أدخل الرموز (مفصولة بمسافة أو سطر)", st.session_state.get("symbols",""), key="symbols_textarea")
symbols=[s.strip()+suffix for s in symbols_input.replace("\n"," ").split() if s.strip()]

# =============================
# تنفيذ التحليل — بناء الجدول
# =============================
if st.button("🔎 إنشاء جدول الأهداف (اليومي + الأسبوعي)", key="run_weekly_btn"):
    if not symbols:
        st.warning("⚠️ الرجاء إدخال رموز أولًا."); st.stop()

    with st.spinner("⏳ نجلب البيانات ونحسب الأهداف..."):
        rows=[]
        total=len(symbols); prog=st.progress(0, text=f"بدء التحليل... (0/{total})"); processed=0

        for i in range(0,total,batch_size):
            chunk_syms=symbols[i:i+batch_size]
            ddata_chunk=fetch_data(" ".join(chunk_syms), start_date, end_date, "1d")
            if ddata_chunk is None or (isinstance(ddata_chunk,pd.DataFrame) and ddata_chunk.empty):
                processed+=len(chunk_syms); prog.progress(min(processed/total,1.0), text=f"تمت معالجة {processed}/{total}"); continue

            for code in chunk_syms:
                try:
                    df_d_raw=extract_symbol_df(ddata_chunk, code)
                    if df_d_raw is None or df_d_raw.empty: continue

                    df_d_conf = drop_last_if_incomplete(df_d_raw, "1d", suffix, allow_intraday_daily=False)
                    if df_d_conf is None or df_d_conf.empty: continue

                    df_w = resample_weekly_from_daily(df_d_conf, suffix)
                    if df_w is None or df_w.empty: continue

                    weekly_close = float(df_w["Close"].iat[-1])
                    last_close = float(df_d_raw["Close"].iat[-1]) if allow_intraday_daily else weekly_close

                    df_d = detect_breakout_with_state(df_d_conf)
                    daily_state_pos = bool((not df_d.empty) and (df_d["State"].iat[-1] == 1))
                    weekly_pos = weekly_state_from_daily(df_d_conf, suffix)
                    monthly_first = monthly_first_breakout_from_daily(df_d_conf, suffix)
                    if apply_triple_filter and not (daily_state_pos and weekly_pos and monthly_first):
                        continue

                    sym=code.replace(suffix,"").upper()
                    company=(symbol_name_dict.get(sym,"غير معروف") or "غير معروف")[:20]

                    weekly_H, weekly_t1, weekly_t2, weekly_t3 = ("—","—","—","—")
                    t_w, info_w = weekly_latest_breakout_anchor_targets(df_w, pct=0.55, mode=anchor_mode)
                    if t_w is not None:
                        weekly_H, weekly_t1, weekly_t2, weekly_t3 = t_w

                    daily_H, daily_t1, daily_t2, daily_t3 = ("—","—","—","—")
                    t_d = daily_latest_breakout_anchor_targets(df_d_conf, pct=0.55, mode=anchor_mode)
                    if t_d is not None: daily_H, daily_t1, daily_t2, daily_t3 = t_d

                    # 🆕 الدعم الأسبوعي
                    weekly_support = weekly_last_bullish55_low_value(df_w, pct=0.55)
                    if tick_value and isinstance(weekly_support,(int,float)):
                        weekly_support = round_to_tick(weekly_support, tick_value)

                    # تقريب حسب التيك للأهداف (اختياري)
                    if tick_value:
                        if isinstance(weekly_t1, (int, float)): weekly_t1 = round_to_tick(weekly_t1, tick_value)
                        if isinstance(weekly_t2, (int, float)): weekly_t2 = round_to_tick(weekly_t2, tick_value)
                        if isinstance(weekly_t3, (int, float)): weekly_t3 = round_to_tick(weekly_t3, tick_value)
                        if isinstance(daily_t1,  (int, float)): daily_t1  = round_to_tick(daily_t1,  tick_value)
                        if isinstance(daily_t2,  (int, float)): daily_t2  = round_to_tick(daily_t2,  tick_value)
                        if isinstance(daily_t3,  (int, float)): daily_t3  = round_to_tick(daily_t3,  tick_value)

                    # شهري: القوة والتسارع الشهري
                    df_m = resample_monthly_from_daily(df_d_conf, suffix)
                    monthly_text = "لا توجد شمعة بيعية شهرية معتبرة"
                    info_m = last_sell_anchor_info(df_m, pct=0.55) if (df_m is not None and not df_m.empty) else None
                    if info_m is not None:
                        Hm = float(info_m["H"]); Lm = float(info_m["L"]) 
                        month_close = float(df_m["Close"].iat[-1])
                        dfm_calc = df_m[["Open","High","Low","Close"]].dropna().copy()
                        oM = dfm_calc["Open"].to_numpy(); hM = dfm_calc["High"].to_numpy()
                        lM = dfm_calc["Low"].to_numpy();  cM = dfm_calc["Close"].to_numpy()
                        rngM = (hM - lM)
                        brM  = np.where(rngM != 0, np.abs(cM - oM) / rngM, 0.0)
                        win55M = (cM > oM) & (brM >= 0.55) & (rngM != 0)
                        idx_win = np.where(win55M)[0]
                        last_win_low_val = float(lM[idx_win[-1]]) if len(idx_win)>0 else Lm

                        if month_close < last_win_low_val:
                            monthly_text = f"سلبية شهرية قوية — الإغلاق الشهري دون {last_win_low_val:.2f}"
                        elif month_close < Hm:
                            monthly_text = f"غير متواجدة ويجب الإغلاق الشهري فوق {Hm:.2f}"
                        else:
                            monthly_text = f"متواجدة بشرط الحفاظ على {last_win_low_val:.2f}"

                    row = {
                        "اسم الشركة": company,
                        "الرمز": sym,
                        "سعر الإغلاق": round(last_close,2),
                        "قمة الشمعة البيعية اليومية": daily_H,
                        "الهدف الأول (يومي)": daily_t1,
                        "الهدف الثاني (يومي)": daily_t2,
                        "الهدف الثالث (يومي)": daily_t3,
                        "قمة الشمعة البيعية الاسبوعية": weekly_H,
                        "الهدف الأول (اسبوعي)": weekly_t1,
                        "الهدف الثاني (اسبوعي)": weekly_t2,
                        "الهدف الثالث (اسبوعي)": weekly_t3,
                        "القوة والتسارع الشهري": monthly_text,
                        "الدعم الاسبوعي": weekly_support if weekly_support is not None else "—",
                    }
                    if info_w is not None and show_anchor_debug:
                        row["تاريخ المرساة الأسبوعية"] = str(info_w["date"]) 
                        row["H/L/R (أسبوعي)"] = f"H={info_w['H']} L={info_w['L']} R={info_w['R']}"
                    rows.append(row)

                except Exception:
                    continue

            processed+=len(chunk_syms)
            prog.progress(min(processed/total,1.0), text=f"تمت معالجة {processed}/{total}")

        if rows:
            columns_order = [
                "اسم الشركة",
                "الرمز",
                "سعر الإغلاق",
                "قمة الشمعة البيعية اليومية",
                "الهدف الأول (يومي)",
                "الهدف الثاني (يومي)",
                "الهدف الثالث (يومي)",
                "قمة الشمعة البيعية الاسبوعية",
                "الهدف الأول (اسبوعي)",
                "الهدف الثاني (اسبوعي)",
                "الهدف الثالث (اسبوعي)",
                "القوة والتسارع الشهري",
                "الدعم الاسبوعي",
            ]
            # أضف أعمدة التشخيص إن فُعّلت
            if show_anchor_debug:
                columns_order += ["تاريخ المرساة الأسبوعية", "H/L/R (أسبوعي)"]
            df_final = pd.DataFrame(rows)[columns_order]

            non_numeric_cols = {"اسم الشركة", "الرمز", "القوة والتسارع الشهري"}
            for col in df_final.columns:
                if col in non_numeric_cols: continue
                df_final[col] = df_final[col].apply(lambda x: _fmt_num(x))

            market_name="السوق السعودي" if suffix==".SR" else "السوق الأمريكي"
            day_str=f"{end_date.day}-{end_date.month}-{end_date.year}"
            filt_note="— فلترة بالاختراق مفعّلة" if apply_triple_filter else "— بدون اشتراط الاختراق"
            if allow_intraday_daily: filt_note += " — عرض السعر اليومي الحالي"
            st.subheader(f"🎯 جدول الأهداف (اليومي + الأسبوعي) — {market_name} — {day_str} — عدد الرموز: {len(df_final)} {filt_note}")

            st.markdown(render_table(df_final), unsafe_allow_html=True)
            st.download_button(
                "📥 تنزيل جدول الأهداف (اليومي + الأسبوعي) CSV",
                df_final.to_csv(index=False).encode("utf-8-sig"),
                file_name="TriplePower_Targets_DailyWeekly.csv",
                mime="text/csv"
            )
        else:
            st.info("لا توجد بيانات كافية لحساب الأهداف على الفواصل المحددة.")
