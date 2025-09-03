# app.py
# =========================================================
# منصة TriplePower - جدول الأهداف بنمط الصورة (صفّان: يومي + أسبوعي)
# الافتراضي: لا يُشترط الاختراق؛ يمكن تفعيل فلتر الاختراق الثلاثي اختياريًا
# =========================================================

import os
import re
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from html import escape
from zoneinfo import ZoneInfo  # لضبط التوقيت المحلي
import hashlib, secrets, base64  # تشفير كلمات المرور

# =============================
# تحميل متغيرات البيئة
# =============================
load_dotenv()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL")

# إيقاف آمن إذا لم يتم ضبط متغير البيئة
if not SHEET_CSV_URL:
    st.error("⚠️ لم يتم ضبط SHEET_CSV_URL في متغيرات البيئة. أضفه ثم أعد التشغيل.")
    st.stop()

# =============================
# تهيئة الصفحة العامة + دعم RTL
# =============================
st.set_page_config(page_title="🔒🔍 فلتر الاشتراكات واختراق الشموع | TriplePower", layout="wide")

# حقن CSS عالمي لجعل الاتجاه RTL في كامل التطبيق
RTL_CSS = """
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
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# =============================
# دوال مساعدة
# =============================

def linkify(text: str) -> str:
    """تحويل أي رابط نصي إلى رابط Markdown قابل للنقر."""
    if not text:
        return ""
    pattern = r"(https?://[^\s]+)"
    return re.sub(pattern, r"[\1](\1)", text)

def load_important_links() -> str:
    """تحميل محتوى ملف الروابط المهمة (إن وُجد)."""
    try:
        with open("روابط مهمة.txt", "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "⚠️ ملف 'روابط مهمة.txt' غير موجود."

def load_symbols_names(file_path: str, market_type: str) -> dict:
    """تحميل قاموس (الرمز → الاسم). يدعم السعودية/أمريكا."""
    mapping = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    symbol, name = parts
                    if market_type == "سعودي":
                        mapping[symbol.strip()] = name.strip()
                    else:
                        mapping[symbol.strip().upper()] = name.strip()
        return mapping
    except Exception as e:
        st.warning(f"⚠️ خطأ في تحميل ملف {file_path}: {e}")
        return {}

# ===== تشفير كلمات المرور (PBKDF2) =====
PBKDF_ITER = 100_000

def _pbkdf2_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF_ITER)
    return f"pbkdf2$sha256${PBKDF_ITER}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"

def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, algoname, iters, b64salt, b64hash = stored.split("$", 4)
        if algo != "pbkdf2" or algoname != "sha256":
            return False
        iters = int(iters)
        salt = base64.b64decode(b64salt)
        expected = base64.b64decode(b64hash)
        test = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return secrets.compare_digest(test, expected)
    except Exception:
        return False

# ===== كاش لتحميل بيانات المستخدمين =====
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
            if pwd_hash:  # المسار الآمن
                return u if _pbkdf2_verify(password, pwd_hash) else None
            # توافق خلفي مع العمود القديم
            if u.get("password") == password:
                return u
    return None

def is_expired(expiry_date: str) -> bool:
    try:
        exp = datetime.strptime(expiry_date.strip(), "%Y-%m-%d").date()
        return exp < date.today()
    except Exception:
        return True

@st.cache_data(ttl=300)
def fetch_data(symbols, sd, ed, iv):
    """تنزيل بيانات من yfinance لدفعة واحدة."""
    if not symbols or not str(symbols).strip():
        return None
    try:
        return yf.download(
            tickers=symbols,
            start=sd,
            end=ed + timedelta(days=1),
            interval=iv,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None

def extract_symbol_df(batch_df: pd.DataFrame, code: str) -> pd.DataFrame | None:
    """
    استخراج DataFrame لرمز محدد من نتيجة yfinance سواءً كانت MultiIndex (عدة رموز)
    أو DataFrame أعمدة مسطّحة (رمز واحد).
    """
    if batch_df is None or batch_df.empty:
        return None
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            lvl0 = batch_df.columns.get_level_values(0)
            if code in set(lvl0):
                return batch_df[code].reset_index()
            else:
                return None
        else:
            cols = set(map(str.lower, batch_df.columns.astype(str)))
            if {"open","high","low","close"}.issubset(cols):
                return batch_df.reset_index()
    except Exception:
        return None
    return None

def drop_last_if_incomplete(df: pd.DataFrame, tf: str, suffix: str, allow_intraday_daily: bool = False) -> pd.DataFrame:
    """إسقاط الشمعة غير المكتملة (مع خيار السماح باليومي الحالي)."""
    if df is None or df.empty:
        return df
    dfx = df.copy()

    # لو كان آخر صف ناقص قيماً (OHLC) نحذفه
    if dfx.iloc[-1][["Open","High","Low","Close"]].isna().any():
        return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]

    last_dt = pd.to_datetime(dfx["Date"].iloc[-1]).date()

    if tf == "1d":
        if allow_intraday_daily:
            return dfx
        if suffix == ".SR":
            now = datetime.now(ZoneInfo("Asia/Riyadh"))
            after_close = (now.hour > 15) or (now.hour == 15 and now.minute >= 10)  # تداول
            if last_dt == now.date() and not after_close:
                return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        else:
            now = datetime.now(ZoneInfo("America/New_York"))
            after_close = (now.hour > 16) or (now.hour == 16 and now.minute >= 5)  # السوق الأمريكي
            if last_dt == now.date() and not after_close:
                return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        return dfx

    if tf == "1wk":
        return dfx  # الأسبوعي يُفحص في التجميع من اليومي

    if tf == "1mo":
        now = datetime.now(ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York"))
        today = now.date()
        if last_dt.year == today.year and last_dt.month == today.month:
            return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        return dfx

    return dfx

# =============================
# منطق الإشارة (مع اشتراط "بيعية معتبرة")
# =============================

def _qualify_sell55(c, o, h, l, pct=0.55):
    """
    نعتبر الشمعة البيعية 55% "معتبرة" إذا كسرت قاع آخر شمعة شرائية 55% (الآن).
    يمكن توسيع المنطق لاحقًا ليشمل الكسر بما بعدها.
    """
    rng = (h - l)
    br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55  = (c > o) & (br >= pct) & (rng != 0)

    # نتتبّع قاع آخر شمعة شرائية 55%
    last_win_low = np.full(c.shape, np.nan, dtype=float)
    cur_low = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur_low = l[i]
        last_win_low[i] = cur_low

    valid_sell_now = lose55 & ~np.isnan(last_win_low) & (l <= last_win_low)
    return valid_sell_now, win55

def detect_breakout_with_state(df: pd.DataFrame, pct: float = 0.55) -> pd.DataFrame:
    """
    - شراء: إغلاق > قمة آخر شمعة بيعية "معتبرة" 55%.
    - خروج: إغلاق < قاع آخر شمعة رابحة 55%.
    - بعد الخروج: نصفر مرجع البيع لإجبار ظهور شمعة بيعية معتبرة جديدة قبل أي دخول لاحق.
    """
    if df is None or df.empty:
        return df

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values

    valid_sell55, win55 = _qualify_sell55(c, o, h, l, pct)

    state = 0
    states, first_buy_signals = [], []
    lose_high_55_const = np.nan   # قمة آخر شمعة بيعية معتبرة
    win_low_55_const   = np.nan   # قاع آخر شمعة رابحة 55%

    for i in range(len(df)):
        buy_sig  = (state == 0) and (not np.isnan(lose_high_55_const)) and (c[i] > lose_high_55_const)
        stop_sig = (state == 1) and (not np.isnan(win_low_55_const))   and (c[i] < win_low_55_const)

        if buy_sig:
            state = 1
            first_buy_signals.append(True)
        elif stop_sig:
            state = 0
            first_buy_signals.append(False)
            lose_high_55_const = np.nan  # لا نسمح بإعادة استخدام قمة قديمة بعد الخروج
        else:
            first_buy_signals.append(False)

        if valid_sell55[i]:
            lose_high_55_const = h[i]
        if win55[i]:
            win_low_55_const = l[i]

        states.append(state)

    df["State"] = states
    df["FirstBuySig"] = first_buy_signals
    df["LoseCndl55"] = valid_sell55
    df["WinCndl55"]  = win55
    return df

# =============================
# إعادة التجميع الأسبوعي/الشهري من اليومي المؤكَّد
# =============================

def _week_is_closed_by_data(df_daily: pd.DataFrame, suffix: str) -> bool:
    """تحقق عملي لإغلاق الأسبوع من توفّر آخر شمعة يومية مؤكدة قبل/بعد الإغلاق."""
    df = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df is None or df.empty:
        return False
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    last_dt = pd.to_datetime(df["Date"].iat[-1])
    if last_dt.date() < now.date():
        return True
    close_h, close_m = (15,10) if suffix==".SR" else (16,5)
    return (last_dt.date() == now.date()) and (now.hour > close_h or (now.hour == close_h and now.minute >= close_m))

def resample_weekly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """أسبوعي من اليومي المؤكَّد + استبعاد الأسبوع الجاري إذا لم يُغلق."""
    if df_daily is None or df_daily.empty:
        return df_daily.iloc[0:0]

    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty:
        return df_daily.iloc[0:0]

    dfw = df_daily[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    dfw.set_index("Date", inplace=True)
    rule = "W-THU" if suffix == ".SR" else "W-FRI"
    dfw = dfw.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna().reset_index()

    # حذف الأسبوع الجاري إن لم يُغلق حسب البيانات
    if not _week_is_closed_by_data(df_daily, suffix) and not dfw.empty:
        dfw = dfw.iloc[:-1]
    return dfw

def resample_monthly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """شهري من اليومي المؤكَّد + استبعاد الشهر الجاري إذا لم يُغلق."""
    if df_daily is None or df_daily.empty:
        return df_daily.iloc[0:0]

    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty:
        return df_daily.iloc[0:0]

    dfm = df_daily[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    dfm.set_index("Date", inplace=True)
    dfm = dfm.resample("M").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna().reset_index()

    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    if not dfm.empty and (dfm["Date"].iat[-1].year == now.year and dfm["Date"].iat[-1].month == now.month):
        dfm = dfm.iloc[:-1]
    return dfm

def weekly_state_from_daily(df_daily: pd.DataFrame, suffix: str) -> bool:
    df_w = resample_weekly_from_daily(df_daily, suffix)
    if df_w.empty:
        return False
    df_w = detect_breakout_with_state(df_w)
    return bool(df_w["State"].iat[-1] == 1)

def monthly_state_from_daily(df_daily: pd.DataFrame, suffix: str) -> bool:
    df_m = resample_monthly_from_daily(df_daily, suffix)
    if df_m.empty:
        return False
    df_m = detect_breakout_with_state(df_m)
    return bool(df_m["State"].iat[-1] == 1)

def monthly_first_breakout_from_daily(df_daily: pd.DataFrame, suffix: str) -> bool:
    """True إذا كان آخر شمعه شهرية (المؤكدة) سجّلت أول اختراق (FirstBuySig) حسب منطق 55%."""
    df_m = resample_monthly_from_daily(df_daily, suffix)
    if df_m is None or df_m.empty:
        return False
    df_m = detect_breakout_with_state(df_m)
    return bool(df_m["FirstBuySig"].iat[-1])

def generate_html_table(df: pd.DataFrame) -> str:
    html = """
    <style>
    table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
    th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    th {background-color: #04AA6D; color: white;}
    tr:nth-child(even){background-color: #f2f2f2;}
    tr:hover {background-color: #ddd;}
    a {color: #1a73e8; text-decoration: none;}
    a:hover {text-decoration: underline;}
    .positive {background-color: #d4edda; color: #155724; font-weight: bold;}
    .negative {background-color: #f8d7da; color: #721c24; font-weight: bold;}
    </style>
    <table>
    <thead><tr>"""
    for col in df.columns:
        html += f"<th>{escape(col)}</th>"
    html += "</tr></thead><tbody>"
    status_cols = ["يومي", "أسبوعي", "شهري"]
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            cell_class = ""
            if col in status_cols:
                if str(val).strip() == "إيجابي":
                    cell_class = "positive"
                elif str(val).strip() == "سلبي":
                    cell_class = "negative"
            if col == "رابط TradingView":
                safe_url = escape(val)
                html += f'<td><a href="{safe_url}" target="_blank" rel="noopener">{safe_url}</a></td>'
            else:
                html += f'<td class="{cell_class}">{escape(str(val))}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# جدول الأهداف (نمط الصورة)
# =============================

TF_LABELS = {"1d": "يومي", "1wk": "أسبوعي", "1mo": "شهري"}

def _last_valid_sell55_idx(df: pd.DataFrame) -> int | None:
    """آخر شمعة بيعية معتبرة 55% (مؤشر الصف)."""
    if df is None or df.empty or "LoseCndl55" not in df.columns:
        return None
    idx = np.where(df["LoseCndl55"].values)[0]
    return int(idx[-1]) if len(idx) else None

def compute_tp_targets_from_last_sell(df_tf: pd.DataFrame) -> tuple[float, float, float, float] | None:
    """
    يحسب: (start_above, t1, t2, t3) على فاصل محدد.
    start_above = قمة الشمعة البيعية المعتبرة.
    tN = start_above + N * (مدى الشمعة).
    """
    if df_tf is None or df_tf.empty:
        return None
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df_tf.columns:
            return None

    df_tf = detect_breakout_with_state(df_tf)  # يضيف LoseCndl55
    i = _last_valid_sell55_idx(df_tf)
    if i is None:
        return None

    H = float(df_tf["High"].iat[i])
    L = float(df_tf["Low"].iat[i])
    R = H - L
    if not np.isfinite(R) or R <= 0:
        return None

    start_above = round(H, 2)
    t1 = round(H + 1 * R, 2)
    t2 = round(H + 2 * R, 2)
    t3 = round(H + 3 * R, 2)
    return start_above, t1, t2, t3

def _fmt_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"

def generate_targets_html_table(df: pd.DataFrame) -> str:
    """جدول HTML مُلوَّن كما في الصورة، ويتحمّل نقص الأرقام."""
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
    <table><thead><tr>
    """
    from html import escape as _esc
    for col in df.columns:
        html += f"<th>{_esc(str(col))}</th>"
    html += "</tr></thead><tbody>"

    for _, r in df.iterrows():
        # تلوين خانة "بداية الحركة" فقط إذا كانت رقمية
        try:
            start_val = float(r["بداية الحركة بالإغلاق أعلى"])
            cur_close = float(r["سعر الإغلاق"])
            row_cls = "positive" if cur_close >= start_val else "negative"
        except Exception:
            row_cls = ""

        html += "<tr>"
        for col in df.columns:
            val = r[col]
            cell_cls = row_cls if col == "بداية الحركة بالإغلاق أعلى" else ""
            html += f'<td class="{cell_cls}">{_esc(str(val))}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# جلسة العمل (حالة المستخدم)
# =============================
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("login_error", None)
st.session_state.setdefault("login_attempts", 0)

def do_login():
    if st.session_state.login_attempts >= 5:
        st.session_state.login_error = "too_many"
        return
    users = load_users()
    me = check_login(st.session_state.login_username, st.session_state.login_password, users)
    if me is None:
        st.session_state.login_attempts += 1
        st.session_state.login_error = "bad"
    elif is_expired(me.get("expiry","")):
        st.session_state.login_error = "expired"
    else:
        st.session_state.authenticated = True
        st.session_state.user = me
        st.session_state.login_error = None

# =============================
# شاشة تسجيل الدخول
# =============================
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
            st.error("⛔ تم تجاوز محاولات الدخول المسموح بها مؤقتًا. حاول لاحقًا.")
    with col_left:
        important_links = load_important_links()
        st.markdown(
            "<div style='background-color:#f0f2f6;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgb(0 0 0 / 0.1);line-height:1.6;'>"
            "<h3 style='font-size:20px;'>فلتر منصة القوة الثلاثية للتداول في الأسواق المالية TriplePower</h3>"
            + linkify(important_links) + "</div>",
            unsafe_allow_html=True,
        )
    st.stop()

# =============================
# تحقق دوري من الاشتراك
# =============================
if is_expired(st.session_state.user["expiry"]):
    st.warning("⚠️ انتهى اشتراكك. تم تسجيل خروجك تلقائيًا.")
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

# =============================
# بعد تسجيل الدخول
# =============================
me = st.session_state.user
st.markdown("---")
with st.sidebar:
    # بطاقة صلاحية الاشتراك
    st.markdown(
        f"""<div style="
            background-color:#28a745;padding:10px;border-radius:5px;color:white;
            font-weight:bold;text-align:center;margin-bottom:10px;">
            ✅ اشتراكك سارٍ حتى: {me['expiry']}
            </div>""",
        unsafe_allow_html=True,
    )

    # 🔔 تنبيه انتهاء الاشتراك خلال 3 أيام أو أقل (بحسب توقيت الرياض)
    try:
        expiry_dt = datetime.strptime(me["expiry"].strip(), "%Y-%m-%d").date()
        today_riyadh = datetime.now(ZoneInfo("Asia/Riyadh")).date()
        days_left = (expiry_dt - today_riyadh).days
        if 0 <= days_left <= 3:
            st.warning(f"⚠️ تنبيه: تبقّى {days_left} يومًا على انتهاء الاشتراك. يُرجى التجديد لتجنّب انقطاع الخدمة.")
    except Exception:
        pass

    # اختراقات الساعة — اختيارية لتقليل الضغط
    st.markdown("### ⚡ أبرز اختراقات الساعة في السوق الأمريكي")
    show_intraday = st.checkbox("عرض اختراقات الساعة (تجريبي)", value=False, help="قد يبطئ التحميل.")
    intraday_syms = """AAPL MSFT NVDA AMD TSLA META GOOGL AMZN NFLX AVGO QCOM TXN LRCX INTC MU ADI ORLY COST PEP PYPL QQQ""".split()

    @st.cache_data(ttl=300)
    def get_intraday_breakouts(symbols):
        data = fetch_data(" ".join(symbols), date.today()-timedelta(days=5), date.today(), "60m")
        out = []
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            return out
        for s in symbols:
            try:
                df = extract_symbol_df(data, s)
                if df is None or df.empty:
                    continue
                df = detect_breakout_with_state(df)
                if not df.empty and bool(df["FirstBuySig"].iat[-1]):
                    out.append(s)
            except Exception:
                continue
        return out

    if show_intraday:
        breakout_list = get_intraday_breakouts(intraday_syms)
        st.sidebar.markdown(
            ", ".join([f"[{s}](https://www.tradingview.com/symbols/{s}/)" for s in breakout_list]) if breakout_list else "لا توجد اختراقات ساعة حالياً."
        )
    else:
        st.sidebar.caption("فعّل الخيار أعلاه لعرضها.")

    st.markdown("### ⚙️ إعدادات التحليل")
    market = st.sidebar.selectbox("اختر السوق", ["السوق السعودي", "السوق الأمريكي"])
    suffix = ".SR" if market == "السوق السعودي" else ""
    # خيار اختياري لاشتراط الاختراق الثلاثي
    apply_triple_filter = st.sidebar.checkbox(
        "اشتراط الاختراق الثلاثي (اختياري)",
        value=False,
        help="عند التفعيل: تُعرض فقط الرموز التي تحقق (اختراق يومي مؤكد + أسبوعي إيجابي + أول اختراق شهري). عند التعطيل: تُعرض كل الرموز."
    )

    start_date = st.sidebar.date_input("من", date(2020, 1, 1))
    end_date = st.sidebar.date_input("إلى", date.today())

    allow_intraday_daily = st.sidebar.checkbox(
        "👁️ عرض اختراقات اليوم قبل الإغلاق (يومي) — للعرض فقط",
        value=False,
        help="الفلتر الأساسي يشترط إغلاق يومي مؤكد. هذا الخيار لا يؤثر على الفلترة، فقط على أي عرض اختياري.",
    )

    # حجم الدُفعة عند الجلب (لجميع الرموز)
    batch_size = st.sidebar.slider("حجم الدُفعة عند الجلب", min_value=20, max_value=120, value=60, step=10,
                                   help="تكبيرها يسرّع الجلب ولكن قد يستهلك ذاكرة أكبر.")

    # تحميل قاموس الأسماء
    symbol_name_dict = (
        load_symbols_names("saudiSY.txt", "سعودي") if suffix == ".SR" else load_symbols_names("usaSY.txt", "امريكي")
    )

    if st.sidebar.button("🎯 رموز تجريبية"):
        st.session_state.symbols = "1120 2380 1050" if suffix == ".SR" else "AAPL MSFT GOOGL"
    try:
        with open("رموز الاسواق العالمية.xlsx", "rb") as file:
            st.sidebar.download_button(
                "📥 تحميل ملف رموز الأسواق",
                file,
                "رموز الاسواق العالمية.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except FileNotFoundError:
        st.sidebar.warning("⚠️ الملف غير موجود. الرجاء رفعه بجانب app.py")
    if st.sidebar.button("تسجيل الخروج"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

# =============================
# إدخال الرموز
# =============================
symbols_input = st.text_area("أدخل الرموز (مفصولة بمسافة أو سطر)", st.session_state.get("symbols", ""))
symbols = [s.strip() + suffix for s in symbols_input.replace("\n", " ").split() if s.strip()]

# =============================
# تنفيذ التحليل
# =============================
if st.button("🔎 تنفيذ التحليل"):
    if not symbols:
        st.warning("⚠️ الرجاء إدخال رموز أولًا.")
        st.stop()

    with st.spinner("⏳ نجلب البيانات ونحسب الشروط والأهداف..."):
        results = []
        targets_rows = []

        total = len(symbols)
        prog = st.progress(0, text=f"بدء التحليل... (0/{total})")
        processed = 0

        # نجلب ونعالج على دفعات لتقليل استهلاك الذاكرة
        for i in range(0, total, batch_size):
            chunk_syms = symbols[i:i + batch_size]
            ddata_chunk = fetch_data(" ".join(chunk_syms), start_date, end_date, "1d")
            if ddata_chunk is None or (isinstance(ddata_chunk, pd.DataFrame) and ddata_chunk.empty):
                processed += len(chunk_syms)
                prog.progress(min(processed / total, 1.0), text=f"تمت معالجة {processed}/{total}")
                continue

            for code in chunk_syms:
                try:
                    # استخراج اليومي للرمز من الدفعة الحالية
                    df_d_raw = extract_symbol_df(ddata_chunk, code)
                    if df_d_raw is None or df_d_raw.empty:
                        continue

                    # يومي مؤكد فقط (لا نسمح بمعاينة مبكرة هنا لأنه شرط أساسي)
                    df_d_conf = drop_last_if_incomplete(
                        df_d_raw,
                        "1d",
                        suffix,
                        allow_intraday_daily=False,
                    )
                    if df_d_conf is None or df_d_conf.empty:
                        continue

                    # منطق 55% على اليومي المؤكد
                    df_d = detect_breakout_with_state(df_d_conf)
                    if df_d is None or df_d.empty:
                        continue

                    # حالات الفواصل (لا نفلتر عليها إلا إذا طُلب)
                    daily_positive    = bool(df_d["State"].iat[-1] == 1)
                    daily_first_break = bool(df_d["FirstBuySig"].iat[-1])
                    weekly_positive   = weekly_state_from_daily(df_d_conf, suffix)
                    monthly_first     = monthly_first_breakout_from_daily(df_d_conf, suffix)

                    # تطبيق الفلتر الاختياري (إن تم تفعيله)
                    if apply_triple_filter:
                        if not (daily_first_break and weekly_positive and monthly_first):
                            continue  # تجاهل هذا الرمز عند الفلترة الصارمة

                    # بيانات العرض
                    last_close = float(df_d["Close"].iat[-1])
                    sym = code.replace(suffix, '').upper()
                    company_name = (symbol_name_dict.get(sym, "غير معروف") or "غير معروف")[:20]
                    tv = f"TADAWUL-{sym}" if suffix == ".SR" else sym
                    url = f"https://www.tradingview.com/symbols/{tv}/"

                    results.append(
                        {
                            "م": 0,
                            "الرمز": sym,
                            "اسم الشركة": company_name,
                            "سعر الإغلاق": round(last_close, 2),
                            "يومي": "إيجابي" if daily_positive else "سلبي",
                            "أسبوعي": "إيجابي" if weekly_positive else "سلبي",
                            "شهري": "اختراق أول مرة" if monthly_first else "—",
                            "رابط TradingView": url,
                        }
                    )

                    # ===== جدول الأهداف: صفّان (يومي + أسبوعي) =====
                    intervals_for_targets = ["1d", "1wk"]  # أضِف "1mo" لو أردت صفًا شهريًا أيضًا
                    for tf in intervals_for_targets:
                        if tf == "1d":
                            df_tf = df_d_conf.copy()
                        elif tf == "1wk":
                            df_tf = resample_weekly_from_daily(df_d_conf, suffix)
                        else:
                            df_tf = resample_monthly_from_daily(df_d_conf, suffix)

                        tp = compute_tp_targets_from_last_sell(df_tf)
                        if tp is not None:
                            start_above, t1, t2, t3 = tp
                        else:
                            start_above = t1 = t2 = t3 = "—"  # لا توجد شمعة بيعية معتبرة على هذا الفاصل

                        targets_rows.append({
                            "اسم الشركة": company_name,
                            "الرمز": sym,
                            "سعر الإغلاق": round(last_close, 2),  # نعرض الإغلاق اليومي كما في المثال
                            "الفاصل": {"1d":"يومي","1wk":"أسبوعي","1mo":"شهري"}.get(tf, tf),
                            "بداية الحركة بالإغلاق أعلى": start_above,
                            "الهدف الأول": t1,
                            "الهدف الثاني": t2,
                            "الهدف الثالث": t3,
                        })

                except Exception:
                    continue

            processed += len(chunk_syms)
            prog.progress(min(processed / total, 1.0), text=f"تمت معالجة {processed}/{total}")

        # ===== جدول الرموز =====
        if results:
            df_results = pd.DataFrame(results)[
                ["م", "الرمز", "اسم الشركة", "سعر الإغلاق", "يومي", "أسبوعي", "شهري", "رابط TradingView"]
            ]
            # فرز وترقيم
            df_results = df_results.sort_values(by="الرمز").reset_index(drop=True)
            df_results["م"] = range(1, len(df_results) + 1)
            # تنسيق السعر
            df_results["سعر الإغلاق"] = df_results["سعر الإغلاق"].map(lambda x: f"{x:,.2f}")

            # ===== العنوان الديناميكي =====
            market_name = "السوق السعودي" if suffix == ".SR" else "السوق الأمريكي"
            day_str = f"{end_date.day}-{end_date.month}-{end_date.year}"
            filt_note = "— فلترة بالاختراق مفعّلة" if apply_triple_filter else "— بدون اشتراط الاختراق"

            with st.container():
                st.subheader(f"نتائج ({market_name}) — {day_str} — العدد: {len(df_results)} {filt_note}")
                html_out = generate_html_table(df_results)
                st.markdown(html_out, unsafe_allow_html=True)

                # أزرار تنزيل
                csv_bytes = df_results.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 تنزيل النتائج CSV",
                    csv_bytes,
                    file_name=f"TriplePower_{('KSA' if suffix=='.SR' else 'USA')}_{day_str}.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "📥 تنزيل النتائج HTML",
                    html_out.encode("utf-8"),
                    file_name=f"TriplePower_{('KSA' if suffix=='.SR' else 'USA')}_{day_str}.html",
                    mime="text/html"
                )
        else:
            st.info("لا توجد رموز ضمن قائمتك لعرضها (تحقق من الإدخال أو من توفر البيانات).")

        # ===== جدول الأهداف بنمط الصورة =====
        if targets_rows:
            df_targets = pd.DataFrame(targets_rows)[
                ["اسم الشركة","الرمز","سعر الإغلاق","الفاصل","بداية الحركة بالإغلاق أعلى","الهدف الأول","الهدف الثاني","الهدف الثالث"]
            ]
            # ترتيب: يومي ثم أسبوعي لكل رمز
            order_map = {"يومي": 0, "أسبوعي": 1, "شهري": 2}
            df_targets["_ord"] = df_targets["الفاصل"].map(order_map).fillna(9)
            df_targets = df_targets.sort_values(["الرمز", "_ord"]).drop(columns="_ord").reset_index(drop=True)

            # تنسيق الأرقام مع تحمّل الفراغ
            for col in ["سعر الإغلاق","بداية الحركة بالإغلاق أعلى","الهدف الأول","الهدف الثاني","الهدف الثالث"]:
                df_targets[col] = df_targets[col].map(_fmt_num)

            st.markdown("### 🎯 جدول الأهداف (TriplePower) — يومي + أسبوعي")
            html_targets = generate_targets_html_table(df_targets)
            st.markdown(html_targets, unsafe_allow_html=True)

            # تنزيل
            st.download_button(
                "📥 تنزيل جدول الأهداف CSV",
                df_targets.to_csv(index=False).encode("utf-8-sig"),
                file_name="TriplePower_Targets.csv",
                mime="text/csv"
            )
        else:
            st.info("لا توجد بيانات كافية لحساب الأهداف على الفواصل المحددة.")
