# app.py
# =========================================================
# منصة TriplePower - جدول الأهداف فقط (Wide: يومي + أسبوعي)
# - "بداية الحركة بالإغلاق أعلى (أسبوعي)" مطابق تمامًا لليومي
# - عمودا "القوة والتسارع الشهري" و "F:M"
# - إسقاط الأسبوع الجاري غير المغلق (KSA: الخميس) والشهر الجاري غير المغلق
# - إصلاح الدمج بأنواع الأعمدة
# - تصحيح اختيار "الشمعة البيعية المعتبرة" (بنفسها أو ما بعدها) وفق نموذج TriplePower
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
# إعداد عام
# =============================
load_dotenv()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL")
if not SHEET_CSV_URL:
    st.error("⚠️ لم يتم ضبط SHEET_CSV_URL في متغيرات البيئة. أضفه ثم أعد التشغيل.")
    st.stop()

st.set_page_config(page_title="🎯 جدول الأهداف | TriplePower", layout="wide")
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
        with open("روابط مهمة.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "⚠️ ملف 'روابط مهمة.txt' غير موجود."

def load_symbols_names(file_path: str, market_type: str) -> dict:
    mapping = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    symbol, name = parts
                    if market_type == "سعودي":
                        mapping[symbol.strip()] = name.strip()
                    else:
                        mapping[symbol.strip().upper()] = name.strip()
    except Exception as e:
        st.warning(f"⚠️ خطأ في تحميل ملف {file_path}: {e}")
    return mapping

# ===== مصادقة (PBKDF2) =====
PBKDF_ITER = 100_000
def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, algoname, iters, b64salt, b64hash = stored.split("$", 4)
        if algo != "pbkdf2" or algoname != "sha256": return False
        iters = int(iters)
        salt = base64.b64decode(b64salt)
        expected = base64.b64decode(b64hash)
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return secrets.compare_digest(test, expected)
    except Exception:
        return False

@st.cache_data(ttl=600)
def load_users():
    df = pd.read_csv(SHEET_CSV_URL, dtype=str)
    return df.to_dict("records")

def check_login(username, password, users):
    username = (username or "").strip()
    password = (password or "")
    for u in users:
        if u.get("username") == username:
            pwd_hash = u.get("password_hash")
            if pwd_hash:
                return u if _pbkdf2_verify(password, pwd_hash) else None
            if u.get("password") == password:
                return u
    return None

def is_expired(expiry_date: str) -> bool:
    try:
        exp = datetime.strptime(expiry_date.strip(), "%Y-%m-%d").date()
        return exp < date.today()
    except Exception:
        return True

# =============================
# جلب البيانات
# =============================
@st.cache_data(ttl=300)
def fetch_data(symbols, sd, ed, iv):
    if not symbols or not str(symbols).strip(): return None
    try:
        return yf.download(
            tickers=symbols,
            start=sd, end=ed + timedelta(days=1),
            interval=iv, group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None

def extract_symbol_df(batch_df: pd.DataFrame, code: str) -> pd.DataFrame | None:
    if batch_df is None or batch_df.empty: return None
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            if code in set(batch_df.columns.get_level_values(0)):
                return batch_df[code].reset_index()
            return None
        else:
            cols = set(map(str.lower, batch_df.columns.astype(str)))
            if {"open","high","low","close"}.issubset(cols):
                return batch_df.reset_index()
    except Exception:
        return None
    return None

def drop_last_if_incomplete(df: pd.DataFrame, tf: str, suffix: str, allow_intraday_daily: bool = False) -> pd.DataFrame:
    if df is None or df.empty: return df
    dfx = df.copy()
    if dfx.iloc[-1][["Open","High","Low","Close"]].isna().any():
        return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
    last_dt = pd.to_datetime(dfx["Date"].iloc[-1]).date()
    if tf == "1d":
        if allow_intraday_daily: return dfx
        if suffix == ".SR":
            now = datetime.now(ZoneInfo("Asia/Riyadh"))
            after_close = (now.hour > 15) or (now.hour == 15 and now.minute >= 10)
            if last_dt == now.date() and not after_close:
                return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        else:
            now = datetime.now(ZoneInfo("America/New_York"))
            after_close = (now.hour > 16) or (now.hour == 16 and now.minute >= 5)
            if last_dt == now.date() and not after_close:
                return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        return dfx
    if tf == "1wk": return dfx
    if tf == "1mo":
        now = datetime.now(ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York"))
        today = now.date()
        if last_dt.year == today.year and last_dt.month == today.month:
            return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        return dfx
    return dfx

# =============================
# منطق الشموع (بيعية معتبرة 55%) + الحالات
# =============================
def _qualify_sell55(c, o, h, l, pct=0.55):
    rng = (h - l)
    br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55  = (c > o) & (br >= pct) & (rng != 0)

    last_win_low = np.full(c.shape, np.nan, dtype=float)
    cur_low = np.nan
    for i in range(len(c)):
        if win55[i]: cur_low = l[i]
        last_win_low[i] = cur_low

    # بيعية معتبرة: كسرت قاع آخر شمعة شرائية 55% (هنا التحقق "في نفس الشمعة")
    valid_sell_now = lose55 & ~np.isnan(last_win_low) & (l <= last_win_low)
    return valid_sell_now, win55

def detect_breakout_with_state(df: pd.DataFrame, pct: float = 0.55) -> pd.DataFrame:
    if df is None or df.empty: return df
    o = df["Open"].values; h = df["High"].values; l = df["Low"].values; c = df["Close"].values
    valid_sell55, win55 = _qualify_sell55(c, o, h, l, pct)

    state = 0
    states, first_buy = [], []
    lose_high_55_const = np.nan  # قمة آخر شمعة بيعية معتبرة
    win_low_55_const   = np.nan  # قاع آخر شمعة رابحة 55%

    for i in range(len(df)):
        buy_sig  = (state == 0) and (not np.isnan(lose_high_55_const)) and (c[i] > lose_high_55_const)
        stop_sig = (state == 1) and (not np.isnan(win_low_55_const))   and (c[i] < win_low_55_const)
        if buy_sig:
            state = 1; first_buy.append(True)
        elif stop_sig:
            state = 0; first_buy.append(False); lose_high_55_const = np.nan
        else:
            first_buy.append(False)

        if valid_sell55[i]: lose_high_55_const = h[i]
        if win55[i]:        win_low_55_const   = l[i]
        states.append(state)

    df["State"] = states
    df["FirstBuySig"] = first_buy
    df["LoseCndl55"] = valid_sell55
    df["WinCndl55"]  = win55
    return df

# === اختيار "آخر شمعة بيعية 55%" التي كسرت (بنفسها أو ما بعدها) قاع آخر شمعة شرائية 55% ===
def last_considered_sell_index(df: pd.DataFrame, pct: float = 0.55):
    """
    تعيد فهرس 'آخر شمعة بيعية 55%' التي كسرت (بنفسها أو ما بعدها) قاع آخر شمعة شرائية 55%.
    """
    if df is None or df.empty:
        return None

    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)

    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)

    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55  = (c > o) & (br >= pct) & (rng != 0)

    # prev_win_low[i]: قاع آخر شمعة شرائية 55% قبل i
    prev_win_low = np.full_like(l, np.nan, dtype=float)
    last_low = np.nan
    for i in range(len(l)):
        prev_win_low[i] = last_low
        if win55[i]:
            last_low = l[i]

    # أدنى Low من i وحتى النهاية (لكشف "كسرت ما بعدها")
    fwd_min_low = np.empty_like(l)
    cur_min = np.inf
    for i in range(len(l) - 1, -1, -1):
        cur_min = min(cur_min, l[i])
        fwd_min_low[i] = cur_min

    considered = lose55 & ~np.isnan(prev_win_low) & (fwd_min_low <= prev_win_low)
    idx = np.where(considered)[0]
    if len(idx) == 0:
        return None
    return int(idx[-1])

# =============================
# تجميع أسبوعي/شهري (مع إسقاط غير المغلق)
# =============================
def _is_current_week_closed(suffix: str):
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    end_weekday = 3 if suffix == ".SR" else 4  # Thu / Fri
    days_ahead = (end_weekday - now.weekday()) % 7
    week_end_date = (now + timedelta(days=days_ahead)).date()
    close_h, close_m = (15, 10) if suffix == ".SR" else (16, 5)
    closed = (now.date() > week_end_date) or (
        now.date() == week_end_date and (now.hour > close_h or (now.hour == close_h and now.minute >= close_m))
    )
    return closed, week_end_date

def resample_weekly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """نفس منطق اليومي لكن على بيانات أسبوعية مؤكدة فقط."""
    if df_daily is None or df_daily.empty: return df_daily.iloc[0:0]
    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty: return df_daily.iloc[0:0]

    dfw = df_daily[["Date","Open","High","Low","Close"]].dropna().copy()
    dfw.set_index("Date", inplace=True)
    rule = "W-THU" if suffix == ".SR" else "W-FRI"
    dfw = dfw.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna().reset_index()

    is_closed, week_end = _is_current_week_closed(suffix)
    if not is_closed and not dfw.empty:
        if pd.to_datetime(dfw["Date"].iat[-1]).date() == week_end:
            dfw = dfw.iloc[:-1]
    return dfw

def resample_monthly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df_daily is None or df_daily.empty: return df_daily.iloc[0:0]
    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty: return df_daily.iloc[0:0]
    dfm = df_daily[["Date","Open","High","Low","Close"]].dropna().copy()
    dfm.set_index("Date", inplace=True)
    dfm = dfm.resample("M").agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna().reset_index()
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    if not dfm.empty and (dfm["Date"].iat[-1].year == now.year and dfm["Date"].iat[-1].month == now.month):
        dfm = dfm.iloc[:-1]
    return dfm

# =============================
# حساب البداية والأهداف (موحّد للفواصل: يومي/أسبوعي)
# =============================
def compute_start_and_targets_any_tf(df_tf: pd.DataFrame):
    """
    اختيار 'آخر شمعة بيعية 55%' التي كسرت (بنفسها أو ما بعدها) قاع آخر شمعة شرائية 55%.
    البداية = H
    الأهداف = H + n*(H-L), n = 1..3
    """
    if df_tf is None or df_tf.empty:
        return None
    j = last_considered_sell_index(df_tf, pct=0.55)
    if j is None:
        return None
    H = float(df_tf["High"].iat[j])
    L = float(df_tf["Low"].iat[j])
    R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return round(H, 2), round(H + R, 2), round(H + 2*R, 2), round(H + 3*R, 2)

# =============================
# HTML للجدول العريض
# =============================
def _fmt_num(x):
    try: return f"{float(x):.2f}"
    except Exception: return "—"

def generate_targets_html_table_wide(df: pd.DataFrame) -> str:
    html = """
    <style>
      table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
      th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
      th {background-color: #04AA6D; color: white;}
      tr:nth-child(even){background-color: #f9f9f9;}
      tr:hover {background-color: #f1f1f1;}
      .positive {background-color: #d4edda; color: #155724; font-weight: bold;}
      .negative {background-color: #f8d7da; color: #721c24; font-weight: bold;}
    </style>
    <table><thead><tr>"""
    from html import escape as _esc
    for col in df.columns: html += f"<th>{_esc(str(col))}</th>"
    html += "</tr></thead><tbody>"

    color_cols = [c for c in df.columns if c.startswith("بداية الحركة بالإغلاق أعلى")]
    for _, r in df.iterrows():
        try: close_val = float(str(r["سعر الإغلاق"]).replace(",", ""))
        except Exception: close_val = None
        html += "<tr>"
        for col in df.columns:
            val = r[col]; cell_cls = ""
            if close_val is not None and col in color_cols:
                try:
                    start_val = float(str(val).replace(",", ""))
                    cell_cls = "positive" if close_val >= start_val else "negative"
                except Exception:
                    cell_cls = ""
            html += f'<td class="{cell_cls}">{_esc(str(val))}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# حالة الجلسة + تسجيل الدخول
# =============================
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("login_error", None)
st.session_state.setdefault("login_attempts", 0)

def do_login():
    if st.session_state.login_attempts >= 5:
        st.session_state.login_error = "too_many"; return
    users = load_users()
    me = check_login(st.session_state.login_username, st.session_state.login_password, users)
    if me is None:
        st.session_state.login_attempts += 1; st.session_state.login_error = "bad"
    elif is_expired(me.get("expiry","")):
        st.session_state.login_error = "expired"
    else:
        st.session_state.authenticated = True; st.session_state.user = me; st.session_state.login_error = None

if not st.session_state.authenticated:
    col_left, col_right = st.columns([2, 1])
    with col_right:
        st.markdown('<h3 style="font-size:20px;">🔒 تسجيل دخول المشتركين</h3>', unsafe_allow_html=True)
        st.text_input("اسم المستخدم", key="login_username", placeholder="أدخل اسم المستخدم")
        st.text_input("كلمة المرور", type="password", key="login_password", placeholder="أدخل كلمة المرور")
        st.button("دخول", on_click=do_login)
        if st.session_state.login_error == "bad":
            st.error("⚠️ اسم المستخدم أو كلمة المرور غير صحيحة.")
        elif st.session_state.login_error == "expired":
            st.error("⚠️ انتهى اشتراكك. يرجى التجديد.")
        elif st.session_state.login_error == "too_many":
            st.error("⛔ تم تجاوز محاولات الدخول مؤقتًا.")
    with col_left:
        st.markdown(
            "<div style='background-color:#f0f2f6;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgb(0 0 0 / 0.1);line-height:1.6;'>"
            "<h3 style='font-size:20px;'>منصة القوة الثلاثية TriplePower</h3>"
            + linkify(load_important_links()) + "</div>",
            unsafe_allow_html=True,
        )
    st.stop()

if is_expired(st.session_state.user["expiry"]):
    st.warning("⚠️ انتهى اشتراكك. تم تسجيل خروجك تلقائيًا.")
    st.session_state.authenticated = False; st.session_state.user = None; st.rerun()

# =============================
# واجهة التحكم
# =============================
me = st.session_state.user
st.markdown("---")
with st.sidebar:
    st.markdown(f"""<div style="background-color:#28a745;padding:10px;border-radius:5px;color:white;
                    font-weight:bold;text-align:center;margin-bottom:10px;">
                    ✅ اشتراكك سارٍ حتى: {me['expiry']}</div>""", unsafe_allow_html=True)

    try:
        expiry_dt = datetime.strptime(me["expiry"].strip(), "%Y-%m-%d").date()
        today_riyadh = datetime.now(ZoneInfo("Asia/Riyadh")).date()
        days_left = (expiry_dt - today_riyadh).days
        if 0 <= days_left <= 3:
            st.warning(f"⚠️ تبقّى {days_left} يومًا على انتهاء الاشتراك.")
    except Exception:
        pass

    market = st.selectbox("اختر السوق", ["السوق السعودي", "السوق الأمريكي"])
    suffix = ".SR" if market == "السوق السعودي" else ""
    apply_triple_filter = st.checkbox(
        "اشتراط الاختراق الثلاثي (اختياري)", value=False,
        help="عند التفعيل: لن يُعرض الرمز إلا إذا تحقق (اختراق يومي مؤكد + أسبوعي إيجابي + أول اختراق شهري)."
    )
    start_date = st.date_input("من", date(2020, 1, 1))
    end_date   = st.date_input("إلى", date.today())
    allow_intraday_daily = st.checkbox("👁️ عرض اختراقات اليوم قبل الإغلاق (يومي) — للعرض فقط", value=False)
    batch_size = st.slider("حجم الدُفعة عند الجلب", 20, 120, 60, 10)

    symbol_name_dict = (
        load_symbols_names("saudiSY.txt", "سعودي") if suffix == ".SR" else load_symbols_names("usaSY.txt", "امريكي")
    )

    if st.button("🎯 رموز تجريبية"):
        st.session_state.symbols = "1010 1020 1030" if suffix == ".SR" else "AAPL MSFT GOOGL"

    try:
        with open("رموز الاسواق العالمية.xlsx", "rb") as file:
            st.download_button("📥 تحميل ملف رموز الأسواق", file, "رموز الاسواق العالمية.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except FileNotFoundError:
        st.warning("⚠️ ملف الرموز غير موجود بجانب app.py")

    if st.button("تسجيل الخروج"):
        st.session_state.authenticated = False; st.session_state.user = None; st.rerun()

symbols_input = st.text_area("أدخل الرموز (مفصولة بمسافة أو سطر)", st.session_state.get("symbols", ""))
symbols = [s.strip() + suffix for s in symbols_input.replace("\n", " ").split() if s.strip()]

# =============================
# تنفيذ التحليل — جدول الأهداف فقط
# =============================
if st.button("🔎 إنشاء جدول الأهداف"):
    if not symbols:
        st.warning("⚠️ الرجاء إدخال رموز أولًا."); st.stop()

    with st.spinner("⏳ نجلب البيانات ونحسب الأهداف..."):
        targets_rows = []       # صفوف طولي (يومي/أسبوعي) لكل رمز -> Pivot إلى Wide
        monthly_power_rows = [] # لكل رمز: القوة الشهرية + F:M

        total = len(symbols)
        prog = st.progress(0, text=f"بدء التحليل... (0/{total})")
        processed = 0

        for i in range(0, total, batch_size):
            chunk_syms = symbols[i:i + batch_size]
            ddata_chunk = fetch_data(" ".join(chunk_syms), start_date, end_date, "1d")
            if ddata_chunk is None or (isinstance(ddata_chunk, pd.DataFrame) and ddata_chunk.empty):
                processed += len(chunk_syms)
                prog.progress(min(processed / total, 1.0), text=f"تمت معالجة {processed}/{total}")
                continue

            for code in chunk_syms:
                try:
                    df_d_raw = extract_symbol_df(ddata_chunk, code)
                    if df_d_raw is None or df_d_raw.empty: continue

                    # يومي مؤكد
                    df_d_conf = drop_last_if_incomplete(df_d_raw, "1d", suffix, allow_intraday_daily=False)
                    if df_d_conf is None or df_d_conf.empty: continue

                    # تشغيل منطق الشموع لاحتياجات الفلاتر
                    df_d = detect_breakout_with_state(df_d_conf)
                    if df_d is None or df_d.empty: continue

                    # فلتر اختياري: يومي + أسبوعي إيجابي + أول شهري
                    daily_first_break = bool(df_d["FirstBuySig"].iat[-1])

                    df_w = resample_weekly_from_daily(df_d_conf, suffix)
                    weekly_positive = False
                    if not df_w.empty:
                        df_w_state = detect_breakout_with_state(df_w.copy())
                        weekly_positive = bool(df_w_state["State"].iat[-1] == 1)

                    df_m = resample_monthly_from_daily(df_d_conf, suffix)
                    monthly_first = False
                    if not df_m.empty:
                        df_m_state = detect_breakout_with_state(df_m.copy())
                        monthly_first = bool(df_m_state["FirstBuySig"].iat[-1])

                    if apply_triple_filter and not (daily_first_break and weekly_positive and monthly_first):
                        continue

                    # بيانات عامة
                    last_close = float(df_d["Close"].iat[-1])
                    sym = code.replace(suffix, '').upper()
                    company_name = (symbol_name_dict.get(sym, "غير معروف") or "غير معروف")[:20]

                    # ---- (1) أهداف اليومي - المنطق الموحد ----
                    tp_d = compute_start_and_targets_any_tf(df_d_conf)
                    if tp_d is not None:
                        d_start, d_t1, d_t2, d_t3 = tp_d
                    else:
                        d_start = d_t1 = d_t2 = d_t3 = "—"

                    targets_rows.append({
                        "اسم الشركة": company_name,
                        "الرمز": sym,
                        "سعر الإغلاق": round(last_close, 2),
                        "الفاصل": "يومي",
                        "بداية الحركة بالإغلاق أعلى": d_start,
                        "الهدف الأول": d_t1,
                        "الهدف الثاني": d_t2,
                        "الهدف الثالث": d_t3,
                    })

                    # ---- (2) أهداف الأسبوعي - نفس المنطق تمامًا ----
                    tp_w = compute_start_and_targets_any_tf(df_w)
                    if tp_w is not None:
                        w_start, w_t1, w_t2, w_t3 = tp_w
                    else:
                        w_start = w_t1 = w_t2 = w_t3 = "—"

                    targets_rows.append({
                        "اسم الشركة": company_name,
                        "الرمز": sym,
                        "سعر الإغلاق": round(last_close, 2),
                        "الفاصل": "أسبوعي",
                        "بداية الحركة بالإغلاق أعلى": w_start,
                        "الهدف الأول": w_t1,
                        "الهدف الثاني": w_t2,
                        "الهدف الثالث": w_t3,
                    })

                    # ---- (3) القوة والتسارع الشهري + F:M ----
                    monthly_text = "لا توجد شمعة بيعية شهرية معتبرة"
                    fm_value = np.nan
                    if df_m is not None and not df_m.empty:
                        df_m2 = detect_breakout_with_state(df_m.copy())
                        if "LoseCndl55" in df_m2.columns and df_m2["LoseCndl55"].any():
                            idx_m = np.where(df_m2["LoseCndl55"].values)[0]; j = int(idx_m[-1])
                            Hm = float(df_m2["High"].iat[j]); Lm = float(df_m2["Low"].iat[j])
                            if last_close < Hm:
                                monthly_text = f"غير متواجدة ويجب الإغلاق فوق {Hm:.2f}"; fm_value = Hm
                            else:
                                monthly_text = f"متواجدة بشرط الحفاظ على {Lm:.2f}"; fm_value = Lm

                    monthly_power_rows.append({
                        "اسم الشركة": company_name,
                        "الرمز": sym,
                        "سعر الإغلاق": round(last_close, 2),
                        "القوة والتسارع الشهري": monthly_text,
                        "F:M": fm_value,
                    })

                except Exception:
                    continue

            processed += len(chunk_syms)
            prog.progress(min(processed / total, 1.0), text=f"تمت معالجة {processed}/{total}")

        # ===== Pivot إلى Wide + دمج القوة الشهرية =====
        if targets_rows:
            df_targets_long = pd.DataFrame(targets_rows)[
                ["اسم الشركة","الرمز","سعر الإغلاق","الفاصل","بداية الحركة بالإغلاق أعلى","الهدف الأول","الهدف الثاني","الهدف الثالث"]
            ]
            df_wide = df_targets_long.pivot_table(
                index=["اسم الشركة","الرمز","سعر الإغلاق"],
                columns="الفاصل",
                values=["بداية الحركة بالإغلاق أعلى","الهدف الأول","الهدف الثاني","الهدف الثالث"],
                aggfunc="first"
            )
            df_wide.columns = [f"{metric} ({tf})" for metric, tf in df_wide.columns.to_flat_index()]
            df_wide = df_wide.reset_index()

            df_monthly_cols = pd.DataFrame(monthly_power_rows)[
                ["اسم الشركة","الرمز","سعر الإغلاق","القوة والتسارع الشهري","F:M"]
            ].drop_duplicates(subset=["اسم الشركة","الرمز","سعر الإغلاق"], keep="last")

            # توحيد أنواع المفاتيح قبل الدمج
            for col in ["اسم الشركة","الرمز"]:
                df_wide[col] = df_wide[col].astype(str)
                df_monthly_cols[col] = df_monthly_cols[col].astype(str)
            df_wide["سعر الإغلاق"] = pd.to_numeric(df_wide["سعر الإغلاق"], errors="coerce")
            df_monthly_cols["سعر الإغلاق"] = pd.to_numeric(df_monthly_cols["سعر الإغلاق"], errors="coerce")

            df_final = pd.merge(df_wide, df_monthly_cols, on=["اسم الشركة","الرمز","سعر الإغلاق"], how="left")

            # تنسيق الأرقام للعرض فقط
            for col in df_final.columns:
                if col == "سعر الإغلاق" or col.startswith("بداية الحركة") or col.startswith("الهدف") or col == "F:M":
                    df_final[col] = df_final[col].map(_fmt_num)

            ordered = [
                "اسم الشركة","الرمز","سعر الإغلاق",
                "بداية الحركة بالإغلاق أعلى (يومي)","الهدف الأول (يومي)","الهدف الثاني (يومي)","الهدف الثالث (يومي)",
                "بداية الحركة بالإغلاق أعلى (أسبوعي)","الهدف الأول (أسبوعي)","الهدف الثاني (أسبوعي)","الهدف الثالث (أسبوعي)",
                "القوة والتسارع الشهري","F:M"
            ]
            existing = [c for c in ordered if c in df_final.columns]
            existing += [c for c in df_final.columns if c not in existing]
            df_final = df_final[existing]

            market_name = "السوق السعودي" if suffix == ".SR" else "السوق الأمريكي"
            day_str = f"{end_date.day}-{end_date.month}-{end_date.year}"
            filt_note = "— فلترة بالاختراق مفعّلة" if apply_triple_filter else "— بدون اشتراط الاختراق"
            st.subheader(f"🎯 جدول الأهداف ({market_name}) — {day_str} — عدد الرموز: {len(df_final)} {filt_note}")

            html_targets = generate_targets_html_table_wide(df_final)
            st.markdown(html_targets, unsafe_allow_html=True)

            st.download_button(
                "📥 تنزيل جدول الأهداف CSV",
                df_final.to_csv(index=False).encode("utf-8-sig"),
                file_name="TriplePower_Targets_Wide_WithMonthlyPower.csv",
                mime="text/csv"
            )
        else:
            st.info("لا توجد بيانات كافية لحساب الأهداف على الفواصل المحددة.")
