
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

# =============================
# تحميل متغيرات البيئة
# =============================
load_dotenv()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL")

# =============================
# تهيئة الصفحة العامة + دعم RTL
# =============================
st.set_page_config(page_title="🔒🔍 فلتر منصة القوة الثلاثية للتداول في الأسواق المالية  | TriplePowerFilter", layout="wide")

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
  .positive {background-color: #d4edda; color: #155724; font-weight: bold;}
  .negative {background-color: #f8d7da; color: #721c24; font-weight: bold;}
  .halal  { background-color:#d1fae5; color:#065f46; font-weight:bold; }
  .haram  { background-color:#fee2e2; color:#991b1b; font-weight:bold; }
  .review { background-color:#fff7ed; color:#92400e; font-weight:bold; }
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
    return re.sub(pattern, r"[\\1](\\1)", text)

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
                parts = line.split('\\t', 1)
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

# ===== كاش لتحميل بيانات المستخدمين =====
@st.cache_data(ttl=3600)
def load_users():
    df = pd.read_csv(SHEET_CSV_URL, dtype=str)
    return df.to_dict("records")

def check_login(username, password, users):
    return next((u for u in users if u.get("username") == username and u.get("password") == password), None)

def is_expired(expiry_date: str) -> bool:
    try:
        exp = datetime.strptime(expiry_date.strip(), "%Y-%m-%d").date()
        return exp < date.today()
    except Exception:
        return True

@st.cache_data(ttl=300)
def fetch_data(symbols, sd, ed, iv):
    """تنزيل بيانات من yfinance. حارس للمدخلات لتجنّب مكالمات فارغة."""
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

def drop_last_if_incomplete(df: pd.DataFrame, tf: str, suffix: str, allow_intraday_daily: bool = False) -> pd.DataFrame:
    """إسقاط الشمعة غير المكتملة (مع خيار السماح باليومي الحالي)."""
    if df is None or df.empty:
        return df
    dfx = df.copy()
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
        # لن نستخدم 1mo من مزوّد البيانات بعد الآن، لكن نترك الحارس إن استُخدم.
        now = datetime.now(ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York"))
        today = now.date()
        if last_dt.year == today.year and last_dt.month == today.month:
            return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        return dfx

    return dfx

# =============================
# منطق الإشارة (مبسّط مع تصفير مرجع البيع بعد الخروج)
# =============================
def detect_breakout_with_state(df: pd.DataFrame, pct: float = 0.55) -> pd.DataFrame:
    """
    - شراء: إغلاق > قمة آخر شمعة خاسرة 55%.
    - خروج: إغلاق < قاع آخر شمعة رابحة 55%.
    - بعد الخروج: نصفر مرجع البيع لإجبار ظهور شمعة خاسرة 55% جديدة قبل أي دخول لاحق.
    """
    if df is None or df.empty:
        return df

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    rng = (h - l)

    # جعل النسبة مستقرة عدديًا
    br = np.where(rng != 0, np.round(np.abs(c - o) / rng, 6), 0.0)
    lose_cndl_55 = (c < o) & (br >= pct + 1e-9) & (rng != 0)
    win_cndl_55  = (c > o) & (br >= pct + 1e-9) & (rng != 0)

    state = 0
    states, first_buy_signals = [], []
    lose_high_55_const = np.nan   # قمة آخر شمعة خاسرة 55%
    win_low_55_const   = np.nan   # قاع آخر شمعة رابحة 55%

    for i in range(len(df)):
        prev_lose_high = lose_high_55_const
        prev_win_low   = win_low_55_const

        buy_sig  = (state == 0) and (not np.isnan(prev_lose_high)) and (c[i] > prev_lose_high)
        stop_sig = (state == 1) and (not np.isnan(prev_win_low))   and (c[i] < prev_win_low)

        if buy_sig:
            state = 1
            first_buy_signals.append(True)
        elif stop_sig:
            state = 0
            first_buy_signals.append(False)
            # لا نسمح بإعادة استخدام قمة قديمة بعد الخروج
            lose_high_55_const = np.nan
        else:
            first_buy_signals.append(False)

        # تحديث المراجع بعد القرار
        if lose_cndl_55[i]:
            lose_high_55_const = h[i]
        if win_cndl_55[i]:
            win_low_55_const = l[i]

        states.append(state)

    df["State"] = states
    df["FirstBuySig"] = first_buy_signals
    df["LoseCndl55"] = lose_cndl_55
    df["WinCndl55"]  = win_cndl_55
    return df

# =============================
# إعادة التجميع الأسبوعي/الشهري من اليومي المؤكَّد
# =============================

def _is_current_week_closed(suffix: str) -> tuple[bool, date]:
    """يرجع (هل أُغلق أسبوع التداول الحالي؟, تاريخ نهاية هذا الأسبوع)."""
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    # Monday=0 .. Sunday=6 -> Thu=3 (السعودي), Fri=4 (الأمريكي)
    end_weekday = 3 if suffix == ".SR" else 4
    days_ahead = (end_weekday - now.weekday()) % 7
    week_end_date = now.date() + timedelta(days=days_ahead)
    close_h, close_m = (15, 10) if suffix == ".SR" else (16, 5)
    closed = (now.date() > week_end_date) or (
        now.date() == week_end_date and (now.hour > close_h or (now.hour == close_h and now.minute >= close_m))
    )
    return closed, week_end_date

def resample_weekly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """أسبوعي من اليومي المؤكَّد + استبعاد الأسبوع الجاري إذا لم يُغلق."""
    if df_daily is None or df_daily.empty:
        return df_daily.iloc[0:0]

    # فلترة اليومي من أي شمعة غير مكتملة أولاً (لا يسمح باليومي الجاري)
    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty:
        return df_daily.iloc[0:0]

    dfw = df_daily[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    dfw.set_index("Date", inplace=True)

    rule = "W-THU" if suffix == ".SR" else "W-FRI"
    dfw = dfw.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna().reset_index()

    # حذف الأسبوع الجاري إن لم يُغلق بعد
    is_closed, current_week_end = _is_current_week_closed(suffix)
    if not is_closed and not dfw.empty and pd.to_datetime(dfw["Date"].iat[-1]).date() == current_week_end:
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
    # نهاية شهر Gregorian
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
    .halal  { background-color:#d1fae5; color:#065f46; font-weight:bold; }
    .haram  { background-color:#fee2e2; color:#991b1b; font-weight:bold; }
    .review { background-color:#fff7ed; color:#92400e; font-weight:bold; }
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
            if col == "الحكم الشرعي":
                v = str(val).strip()
                cell_class = "review"
                if v == "مباح": cell_class = "halal"
                elif v == "غير مباح": cell_class = "haram"
            if col == "رابط TradingView":
                safe_url = escape(val)
                html += f'<td><a href="{safe_url}" target="_blank" rel="noopener">{safe_url}</a></td>'
            else:
                html += f'<td class="{cell_class}">{escape(str(val))}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# 🕌 فلتر شرعي (قرار 485) لرموز ناسداك
# =============================

BANNED_PATTERNS = [
    r"bank", r"insurance", r"reinsurance", r"mortgage", r"capital markets",
    r"consumer finance", r"casino", r"gaming", r"gambling", r"tobacco",
    r"winery|distiller|alcohol", r"adult"
]

def _is_banned_sector(sector: str, industry: str) -> bool:
    text = f"{sector or ''} {industry or ''}".lower()
    return any(re.search(p, text) for p in BANNED_PATTERNS)

def _safe_first(x):
    import pandas as _pd, numpy as _np
    if x is None: return _np.nan
    try:
        if hasattr(x, "dropna"):
            s = x.dropna()
            if len(s) > 0: return s.iloc[0]
    except Exception:
        pass
    try:
        if _pd.isna(x): return _np.nan
        return x
    except Exception:
        return _np.nan

def _try_rows(df, names):
    if df is None or getattr(df, "empty", True): return None
    idx = [str(i).lower() for i in df.index]
    for name in names:
        if name.lower() in idx:
            return df.loc[df.index[idx.index(name.lower())]]
    return None

@st.cache_data(ttl=86400)
def shariah_screen_nasdaq(symbol: str) -> dict:
    """
    يُرجع: verdict (مباح/غير مباح/يحتاج مراجعة)، نسب debt_ratio/haram_ratio، sector/industry، reasons[]
    """
    t = yf.Ticker(symbol)

    # معلومات عامة
    try:
        info = t.get_info() if hasattr(t, "get_info") else t.info
    except Exception:
        info = getattr(t, "info", {}) or {}
    sector = info.get("sector")
    industry = info.get("industry")

    # القوائم المالية (سنوي ثم ربعي احتياطًا)
    def _pick_first_available(getter_names):
        for g in getter_names:
            try:
                df = getattr(t, g)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        return None

    bs = _pick_first_available(["balance_sheet", "quarterly_balance_sheet"])
    is_ = _pick_first_available(["income_stmt", "quarterly_income_stmt"])

    # القيمة السوقية والدفترية
    market_cap = None
    try:
        market_cap = getattr(t, "fast_info", {}).get("market_cap", None)
    except Exception:
        pass
    if not market_cap:
        market_cap = info.get("marketCap")

    equity = _try_rows(bs, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"])
    book_value = _safe_first(equity)

    # دين = طويل + قصير (تقريبي)
    long_debt = _safe_first(_try_rows(bs, ["Long Term Debt", "Long-Term Debt", "Long Term Debt And Capital Lease Obligation"]))
    short_debt = _safe_first(_try_rows(bs, [
        "Short Long Term Debt", "Short/Current Long Term Debt", "Current Portion Of Long Term Debt", "Short Term Debt"
    ]))
    total_debt = 0.0
    for v in [long_debt, short_debt]:
        if pd.notna(v):
            total_debt += float(v)

    # إجمالي الإيراد
    total_rev = _try_rows(is_, ["Total Revenue", "TotalRevenue"])
    total_rev_val = float(_safe_first(total_rev)) if total_rev is not None and pd.notna(_safe_first(total_rev)) else np.nan

    # تقدير الإيراد المحرم (دخل فوائد/استثمار/غير تشغيلي موجَب فقط)
    interest_income   = _safe_first(_try_rows(is_, ["Interest Income", "Net Interest Income", "Interest And Similar Income"]))
    investment_income = _safe_first(_try_rows(is_, ["Investment Income, Net", "Net Investment Income"]))
    other_nonop       = _safe_first(_try_rows(is_, ["Other Non Operating Income", "Total Other Income/Expense Net", "Other Income (Expense)"]))
    haram_income_est = float(sum([
        v for v in [interest_income, investment_income, other_nonop] if (pd.notna(v) and v > 0)
    ])) if any(pd.notna(v) for v in [interest_income, investment_income, other_nonop]) else np.nan

    # مانع القطاع
    sector_ok = not _is_banned_sector(sector, industry)

    # مقام نسبة الدين = الأكبر بين (القيمة السوقية، الدفترية) إن وُجدا
    denom = None
    if pd.notna(market_cap) and market_cap and market_cap > 0:
        denom = market_cap
    if pd.notna(book_value) and book_value and book_value > 0:
        denom = max(denom or 0, float(book_value))
    debt_ratio = (total_debt / denom) if (denom and denom > 0 and total_debt is not None) else np.nan

    # نسبة الإيراد المحرّم
    haram_ratio = (haram_income_est / total_rev_val) if (pd.notna(haram_income_est) and pd.notna(total_rev_val) and total_rev_val > 0) else np.nan

    needs_review, reasons = False, []

    if not sector_ok:
        reasons.append("النشاط/الصناعة ضمن المحرّمات الواضحة.")
    if pd.isna(debt_ratio):
        needs_review = True
        reasons.append("تعذّر حساب نسبة الدَّين (بيانات ناقصة).")
    elif debt_ratio > 0.30:
        reasons.append(f"نسبة الدَّين {debt_ratio:.2%} تتجاوز 30%.")

    if pd.isna(haram_ratio):
        needs_review = True
        reasons.append("تعذّر تقدير نسبة الإيراد المحرّم بدقة.")
    elif haram_ratio > 0.05:
        reasons.append(f"نسبة الإيراد المحرّم {haram_ratio:.2%} تتجاوز 5%.")

    if sector_ok and (pd.notna(debt_ratio) and debt_ratio <= 0.30) and (pd.notna(haram_ratio) and haram_ratio <= 0.05):
        verdict = "مباح"
    elif needs_review and not any("تتجاوز" in r for r in reasons):
        verdict = "يحتاج مراجعة"
    else:
        verdict = "غير مباح"

    return {
        "verdict": verdict,
        "debt_ratio": None if pd.isna(debt_ratio) else float(debt_ratio),
        "haram_ratio": None if pd.isna(haram_ratio) else float(haram_ratio),
        "sector": sector, "industry": industry,
        "reasons": reasons
    }

# =============================
# جلسة العمل (حالة المستخدم)
# =============================
st.session_state.setdefault("authenticated", False)
st.session_state.setdefault("user", None)
st.session_state.setdefault("login_error", None)

def do_login():
    users = load_users()
    me = check_login(st.session_state.login_username, st.session_state.login_password, users)
    if me is None:
        st.session_state.login_error = "bad"
    elif is_expired(me["expiry"]):
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
    with col_left:
        important_links = load_important_links()
        st.markdown(
            "<div style='background-color:#f0f2f6;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgb(0 0 0 / 0.1);line-height:1.6;'>"
            "<h3 style='font-size:20px;'>فلتر منصة القوة الثلاثية للتداول في الأسواق المالية TriplePowerFilter</h3>"
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

    st.markdown("### ⚡ أبرز اختراقات الساعة في السوق الأمريكي")
    sample_symbols = """AAL AAPL ADBE ADI ADP ADSK AEP AKAM ALGN AMAT AMD AMGN AMZN ANSS APA AVGO AYRO BIDU BIIB BKNG BKR BMRN BNTX BYND CAR CDNS CDW CHKP CHRW CHTR CINF CLOV CMCSA CME COO COST CPB CPRT CSCO CSX CTAS CTSH DLTR DPZ DXCM EA EBAY ENPH EQIX ETSY EVRG EXC EXPE FANG FAST FFIV FITB FOX FOXA FSLR FTNT GEN GILD GOOG GOOGL GT HAS HBAN HOLX HON HSIC HST IDXX ILMN INCY INTC INTU IPGP ISRG JBHT JD JKHY KDP KHC KLAC LBTYA LBTYK LILA LILAK LIN LKQ LNT LRCX LULU MAR MAT MCHP MDLZ META MKTX MNST MSFT MU NAVI NDAQ NFLX NTAP NTES NTRS NVAX NVDA NWL NWS NWSA NXPI ODFL ORLY PARA PAYX PCAR PDCO PEP PFG POOL PYPL QCOM QQQ QRVO QVCGA REG REGN ROKU ROP ROST SBAC SBUX SIRI SNPS STX SWKS TCOM TER TMUS TRIP TRMB TROW TSCO TSLA TTD TTWO TXN UAL ULTA URBN VOD VRSK VRSN VRTX VTRS WBA WBD WDC WTW WYNN XEL XRAY XRX ZBRA ZION ZM""".split()
    intraday_data = fetch_data(" ".join(sample_symbols), date.today() - timedelta(days=5), date.today(), "60m")
    breakout_list = []
    if intraday_data is not None:
        for sym in sample_symbols:
            try:
                df_sym = intraday_data[sym].reset_index()
                df_sym = detect_breakout_with_state(df_sym)
                if not df_sym.empty and df_sym["FirstBuySig"].iat[-1]:
                    breakout_list.append(sym)
            except Exception:
                continue
    st.sidebar.markdown(
        ", ".join([f"[{s}](https://www.tradingview.com/symbols/{s}/)" for s in breakout_list]) if breakout_list else "لا توجد اختراقات ساعة حالياً."
    )

    st.markdown("### ⚙️ إعدادات التحليل")
    market = st.sidebar.selectbox("اختر السوق", ["السوق السعودي", "السوق الأمريكي"])
    suffix = ".SR" if market == "السوق السعودي" else ""
    interval = st.sidebar.selectbox("الفاصل الزمني", ["1d", "1wk", "1mo"])
    start_date = st.sidebar.date_input("من", date(2020, 1, 1))
    end_date = st.sidebar.date_input("إلى", date.today())

    # زر/خيار: عرض اختراقات اليوم دون انتظار إغلاق الشمعة
    allow_intraday_daily = st.sidebar.checkbox(
        "👁️ عرض اختراقات اليوم قبل الإغلاق (يومي)",
        value=False,
        help="اليومي فقط. الأسبوعي/الشهري مؤكدان دائمًا ويعاد تجميعهما من اليومي.",
    )

    # الفلتر الشرعي (ناسداك)
    st.markdown("### 🕌 الفلتر الشرعي (NASDAQ)")
    enable_shariah = st.sidebar.checkbox(
        "تفعيل الفلتر الشرعي (ناسداك فقط)",
        value=False,
        help="يطبّق قرار (485) وتحديثه على رموز السوق الأمريكي. سيضيف أعمدة الحكم الشرعي للجدول."
    )
    show_shariah_details = st.sidebar.checkbox(
        "عرض تفاصيل النِّسَب الشرعية داخل الجدول",
        value=True
    )

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
symbols = [s.strip() + suffix for s in symbols_input.replace("\\n", " ").split() if s.strip()]

# =============================
# تنفيذ التحليل
# =============================
if st.button("🔎 تنفيذ التحليل"):
    if not symbols:
        st.warning("⚠️ الرجاء إدخال رموز أولًا.")
        st.stop()

    # مصدر واحد: اليومي
    ddata = fetch_data(" ".join(symbols), start_date, end_date, "1d")
    if ddata is None:
        st.error("⚠️ لم تتم تحميل بيانات السوق؛ يرجى المحاولة لاحقًا.")
        st.stop()
    if isinstance(ddata, pd.DataFrame) and ddata.empty:
        st.info("ℹ️ البيانات فارغة للفترة/الأسواق المحددة.")
        st.stop()

    results = []
    for code in symbols:
        try:
            # --- 1) تجهيز داتا الفاصل المختار من اليومي المؤكَّد ---
            df_d_raw = ddata[code].reset_index()             # يومي خام
            if interval == "1wk":
                df_sel = resample_weekly_from_daily(df_d_raw, suffix)
            elif interval == "1mo":
                df_sel = resample_monthly_from_daily(df_d_raw, suffix)
            else:
                df_sel = drop_last_if_incomplete(
                    df_d_raw,
                    "1d",
                    suffix,
                    allow_intraday_daily=allow_intraday_daily,   # فقط لليومي
                )

            df_sel = detect_breakout_with_state(df_sel)
            if df_sel.empty or not df_sel["FirstBuySig"].iat[-1]:
                continue

            # --- 2) بناء الحالة لكل فاصل (من نفس المصدر اليومي المؤكَّد) ---
            # يومي
            df_d = drop_last_if_incomplete(
                df_d_raw,
                "1d",
                suffix,
                allow_intraday_daily=allow_intraday_daily,
            )
            df_d = detect_breakout_with_state(df_d)
            if df_d.empty:
                continue

            # أسبوعي/شهري من اليومي المؤكَّد فقط
            weekly_positive  = weekly_state_from_daily(df_d_raw, suffix)
            monthly_positive = monthly_state_from_daily(df_d_raw, suffix)

            daily_positive = df_d["State"].iat[-1] == 1
            last_close = float(df_d["Close"].iat[-1])

            sym = code.replace(suffix, '').upper()
            company_name = (symbol_name_dict.get(sym, "غير معروف") or "غير معروف")[:15]
            tv = f"TADAWUL-{sym}" if suffix == ".SR" else sym.upper()
            url = f"https://www.tradingview.com/symbols/{tv}/"

            results.append(
                {
                    "م": 0,  # سنحدّثه لاحقًا
                    "الرمز": sym,
                    "اسم الشركة": company_name,
                    "سعر الإغلاق": round(last_close, 2),
                    "يومي": "إيجابي" if daily_positive else "سلبي",
                    "أسبوعي": "إيجابي" if weekly_positive else "سلبي",
                    "شهري": "إيجابي" if monthly_positive else "سلبي",
                    "رابط TradingView": url,
                }
            )
        except Exception:
            continue

    if results:
        df_results = pd.DataFrame(results)[
            ["م", "الرمز", "اسم الشركة", "سعر الإغلاق", "يومي", "أسبوعي", "شهري", "رابط TradingView"]
        ]
        # ترقيم تسلسلي
        df_results["م"] = range(1, len(df_results) + 1)

        # ===== دمج الفلتر الشرعي (ناسداك) في نفس الجدول =====
        if enable_shariah:
            if suffix != "":
                st.info("🕌 الفلتر الشرعي مفعّل، لكنه متاح حاليًا للسوق الأمريكي (ناسداك) فقط.")
            else:
                st.write("🕌 جاري احتساب الحكم الشرعي… (يُستفاد من الكاش لتسريع التكرار)")
                sh_cols_verdict, sh_cols_ratios, sh_cols_notes = [], [], []
                progress = st.progress(0)
                total = len(df_results)
                for i, sym in enumerate(df_results["الرمز"].tolist(), start=1):
                    try:
                        sh = shariah_screen_nasdaq(sym)
                        sh_cols_verdict.append(sh["verdict"])
                        dr = "غير متاح" if sh["debt_ratio"] is None else f"{sh['debt_ratio']*100:.2f}%"
                        hr = "غير متاح" if sh["haram_ratio"] is None else f"{sh['haram_ratio']*100:.2f}%"
                        if show_shariah_details:
                            sh_cols_ratios.append(f"دين: {dr} | محرم: {hr}")
                        else:
                            sh_cols_ratios.append("")
                        notes = "؛ ".join(sh["reasons"]) if sh.get("reasons") else ""
                        sh_cols_notes.append(notes)
                    except Exception:
                        sh_cols_verdict.append("يحتاج مراجعة")
                        sh_cols_ratios.append("دين: غير متاح | محرم: غير متاح")
                        sh_cols_notes.append("تعذّر التحليل.")
                    progress.progress(i/total)
                progress.empty()

                insert_at = 4  # بعد سعر الإغلاق
                df_results.insert(insert_at, "الحكم الشرعي", sh_cols_verdict)
                if show_shariah_details:
                    df_results.insert(insert_at + 1, "نِسَب شرعية", sh_cols_ratios)
                # ملاحظات اختيارية
                # df_results.insert(insert_at + (2 if show_shariah_details else 1), "ملاحظات شرعية", sh_cols_notes)

        # ===== العنوان الديناميكي أعلى الجدول =====
        market_name = "السوق السعودي" if suffix == ".SR" else "السوق الأمريكي"
        day_str = f"{end_date.day}-{end_date.month}-{end_date.year}"
        suffix_note = " (حتى الآن)" if (interval == "1d" and allow_intraday_daily) else ""
        tf_label_map = {"1d": "اليومي (D)", "1wk": "الأسبوعي (W)", "1mo": "الشهري (M)"}
        tf_label = tf_label_map.get(interval, str(interval))

        with st.container():
            st.subheader(f"أبرز اختراقات ({market_name}) - فاصل {tf_label} ليوم {day_str}{suffix_note}")
            st.markdown(generate_html_table(df_results), unsafe_allow_html=True)
    else:
        st.info("🔎 لا توجد اختراقات جديدة.")
