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

    # مصدر واحد: اليومي
    ddata = fetch_data(" ".join(symbols), start_date, end_date, "1d")
    if ddata is None:
        st.error("⚠️ لم تتم تحميل بيانات السوق؛ يرجى المحاولة لاحقًا.")
        st.stop()
    if isinstance(ddata, pd.DataFrame) and ddata.empty:
        st.info("ℹ️ البيانات فارغة للفترة/الأسواق المحددة.")
        st.stop()

    # لو احتاج المستخدم عرض اليومي نفسه، نحمّل أيضًا data_sel لليومي (اختياري)
    data_sel = ddata if interval == "1d" else None

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
