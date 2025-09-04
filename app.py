# app.py — جدول الأهداف (اليومي + الأسبوعي) — مُحدّث لمطابقة منطق مؤشر Pine v6
# =========================================================
# TriplePower — جدول الأهداف (سطر واحد لكل رمز)
# — توافق تام مع منطق الشمعة 55% في مؤشر Pine:
#   • LoseCndl55/WinCndl55 على كل فاصل مع اعتماد الشمعة «المؤكَّدة» فقط (بيانات الإغلاق).
#   • حالة State: تتحول إلى 1 عندما يغلق السعر فوق قمة آخر شمعة بيعية 55%، وتعود 0 عند إغلاق تحت قاع آخر شمعة شرائية 55%.
#   • FirstBuySig: أول انتقال من 0 → 1.
#   • «فرصة» = (شهري إيجابي) و (أسبوعي إيجابي) و (أول اختراق يومي) و (الإغلاق اليومي > قمة آخر شمعة بيعية يومية 55%).
# — أهداف المرساة: R = H - L ⇒ T1 = H + R ، T2 = H + 2R ، T3 = H + 3R.
# — إعادة التجميع أسبوعي/شهري تُراعي المنطقة الزمنية للبورصة وحدود الأسبوع/الشهر المقفلة.
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
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# =============================
# دوال مساعدة عامة
# =============================

def linkify(text: str) -> str:
    if not text:
        return ""
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
                if not line:
                    continue
                parts = line.split("\t", 1)
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

# ===== مصادقة (PBKDF2) =====
PBKDF_ITER = 100_000


def _pbkdf2_verify(password: str, stored: str) -> bool:
    try:
        algo, algoname, iters, b64salt, b64hash = stored.split("$", 4)
        if algo != "pbkdf2" or algoname != "sha256":
            return False
        iters = int(iters)
        salt = base64.b64decode(b64salt)
        expected = base64.b64decode(b64hash)
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
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
            h = u.get("password_hash")
            if h:
                return u if _pbkdf2_verify(password, h) else None
            if u.get("password") == password:
                return u
    return None


def is_expired(expiry_date: str) -> bool:
    try:
        return datetime.strptime(expiry_date.strip(), "%Y-%m-%d").date() < date.today()
    except Exception:
        return True

# =============================
# تحميل البيانات من ياهو
# =============================

@st.cache_data(ttl=300)
def fetch_data(symbols, sd, ed, iv):
    if not symbols or not str(symbols).strip():
        return None
    try:
        return yf.download(
            tickers=symbols,
            start=sd,
            end=ed + timedelta(days=1),
            interval=iv,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {e}")
        return None


def extract_symbol_df(batch_df: pd.DataFrame, code: str) -> pd.DataFrame | None:
    if batch_df is None or batch_df.empty:
        return None
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            if code in set(batch_df.columns.get_level_values(0)):
                return batch_df[code].reset_index()
            return None
        else:
            cols = set(map(str.lower, batch_df.columns.astype(str)))
            if {"open", "high", "low", "close"}.issubset(cols):
                return batch_df.reset_index()
    except Exception:
        return None
    return None


def drop_last_if_incomplete(
    df: pd.DataFrame, tf: str, suffix: str, allow_intraday_daily: bool = False
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    dfx = df.copy()
    if dfx.iloc[-1][["Open", "High", "Low", "Close"]].isna().any():
        return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
    last_dt = pd.to_datetime(dfx["Date"].iloc[-1]).date()
    if tf == "1d":
        if allow_intraday_daily:
            return dfx
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
    if tf == "1wk":
        return dfx
    if tf == "1mo":
        now = datetime.now(ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York"))
        today = now.date()
        if last_dt.year == today.year and last_dt.month == today.month:
            return dfx.iloc[:-1] if len(dfx) > 1 else dfx.iloc[0:0]
        return dfx
    return dfx

# =============================
# منطق 55% (بيعية/شرائية) + مراسي «آخر اختراق»
# =============================

def _body_ratio(c, o, h, l):
    rng = h - l
    return np.where(rng != 0, np.abs(c - o) / rng, 0.0), rng


# ——— مرساة «شمعة بيعية 55% كسرت شرائية 55% (بنفسها أو لاحقًا)» — للاستخدام في قياس R فقط ———
# (مطابق لفكرة Dynamic في Pine عند ظهور LoseCndl55 مع كسر لاحق لقاع الشرائية)

def last_sell_anchor_info(_df: pd.DataFrame, pct: float = 0.55):
    if _df is None or _df.empty:
        return None
    df = _df[["Open", "High", "Low", "Close"]].dropna().copy()
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    rng = h - l
    br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55 = (c > o) & (br >= pct) & (rng != 0)

    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur

    future_min = np.minimum.accumulate(l[::-1])[::-1]
    considered_sell = lose55 & ~np.isnan(last_win_low) & (
        (l <= last_win_low) | (future_min <= last_win_low)
    )

    idx = np.where(considered_sell)[0]
    if len(idx) == 0:
        return None

    j = int(idx[-1])
    H = float(h[j])
    L = float(l[j])
    R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return {"idx": j, "H": H, "L": L, "R": R}


def last_sell_anchor_targets(_df: pd.DataFrame, pct: float = 0.55):
    info = last_sell_anchor_info(_df, pct=pct)
    if info is None:
        return None
    H = float(info["H"])
    L = float(info["L"])
    R = H - L
    return (round(H, 2), round(H + R, 2), round(H + 2 * R, 2), round(H + 3 * R, 2))


# ——— «آخر اختراق» للفواصل (اختيار المرساة التي كان اختراق قمتها أحدث إغلاق فوقها) ———

def weekly_latest_breakout_anchor_targets(
    _df: pd.DataFrame, pct: float = 0.55, tick: float | None = None
):
    if _df is None or _df.empty:
        return None
    df = _df[["Open", "High", "Low", "Close"]].dropna().copy()
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    rng = h - l
    br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55 = (c > o) & (br >= pct) & (rng != 0)

    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur
    future_min = np.minimum.accumulate(l[::-1])[::-1]

    anchors = np.where(
        lose55 & ~np.isnan(last_win_low) & ((l <= last_win_low) | (future_min <= last_win_low))
    )[0]
    if len(anchors) == 0:
        return None

    events = []  # (t_break, j_anchor)
    for j in anchors:
        if tick:
            H = float(h[j])
            level = round_to_tick(H, tick)
            closes = np.array([round_to_tick(x, tick) for x in c[j + 1 :]])
            later = np.where(closes >= level)[0]
        else:
            later = np.where(c[j + 1 :] >= h[j])[0]
        if len(later) == 0:
            continue
        t_break = int(j + 1 + later[0])
        events.append((t_break, j))
    if len(events) == 0:
        return None

    _, j_last = max(events, key=lambda x: x[0])
    H = float(h[j_last])
    L = float(l[j_last])
    R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return (round(H, 2), round(H + R, 2), round(H + 2 * R, 2), round(H + 3 * R, 2))


def daily_latest_breakout_anchor_targets(
    _df: pd.DataFrame, pct: float = 0.55, tick: float | None = None
):
    if _df is None or _df.empty:
        return None
    df = _df[["Open", "High", "Low", "Close"]].dropna().copy()
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    rng = h - l
    br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55 = (c > o) & (br >= pct) & (rng != 0)

    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur
    future_min = np.minimum.accumulate(l[::-1])[::-1]

    anchors = np.where(
        lose55 & ~np.isnan(last_win_low) & ((l <= last_win_low) | (future_min <= last_win_low))
    )[0]
    if len(anchors) == 0:
        return None

    events = []  # (t_break, j_anchor)
    for j in anchors:
        if tick:
            level = round_to_tick(h[j], tick)
            closes = np.array([round_to_tick(x, tick) for x in c[j + 1 :]])
            later = np.where(closes >= level)[0]
        else:
            later = np.where(c[j + 1 :] >= h[j])[0]
        if len(later) == 0:
            continue
        t_break = int(j + 1 + later[0])
        events.append((t_break, j))
    if len(events) == 0:
        return None

    _, j_last = max(events, key=lambda x: x[0])
    H = float(h[j_last])
    L = float(l[j_last])
    R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return (round(H, 2), round(H + R, 2), round(H + 2 * R, 2), round(H + 3 * R, 2))

# =============================
# التجميع الأسبوعي/الشهري من اليومي المؤكد + توحيد المنطقة الزمنية
# =============================

def ensure_local_tz(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """توحيد المنطقة الزمنية قبل إعادة التجميع لتتوافق حدود الأسبوع/الشهر مع المنصة."""
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    dfx = df.copy()
    dfx["Date"] = pd.to_datetime(dfx["Date"])
    # تحويل/توطين حسب الحاجة
    if getattr(dfx["Date"].dt, "tz", None) is None:
        dfx["Date"] = dfx["Date"].dt.tz_localize(tz)
    else:
        dfx["Date"] = dfx["Date"].dt.tz_convert(tz)
    return dfx


def _is_current_week_closed(suffix: str) -> tuple[bool, date]:
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    end_weekday = 3 if suffix == ".SR" else 4  # Thu=3, Fri=4
    days_to_end = (end_weekday - now.weekday()) % 7
    week_end_date = now.date() + timedelta(days=days_to_end)
    close_h, close_m = (15, 10) if suffix == ".SR" else (16, 5)
    closed = (
        (now.date() > week_end_date)
        or (
            now.date() == week_end_date
            and (now.hour > close_h or (now.hour == close_h and now.minute >= close_m))
        )
    )
    return closed, week_end_date


def resample_weekly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return df_daily.iloc[0:0]
    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, allow_intraday_daily=False)
    if df_daily.empty:
        return df_daily.iloc[0:0]
    dfw = df_daily[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    dfw = ensure_local_tz(dfw, suffix)
    dfw.set_index("Date", inplace=True)
    rule = "W-THU" if suffix == ".SR" else "W-FRI"
    dfw = (
        dfw.resample(rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
        .reset_index()
    )
    is_closed, current_week_end = _is_current_week_closed(suffix)
    if (not is_closed) and (not dfw.empty):
        last_week_label = pd.to_datetime(dfw["Date"].iat[-1]).date()
        if last_week_label == current_week_end:
            dfw = dfw.iloc[:-1]
    return dfw


def resample_monthly_from_daily(df_daily: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if df_daily is None or df_daily.empty:
        return df_daily.iloc[0:0]
    df_daily = drop_last_if_incomplete(df_daily, "1d", suffix, False)
    if df_daily.empty:
        return df_daily.iloc[0:0]
    dfm = df_daily[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    dfm = ensure_local_tz(dfm, suffix)
    dfm.set_index("Date", inplace=True)
    dfm = (
        dfm.resample("M")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
        .reset_index()
    )
    tz = ZoneInfo("Asia/Riyadh" if suffix == ".SR" else "America/New_York")
    now = datetime.now(tz)
    if not dfm.empty and (
        dfm["Date"].iat[-1].year == now.year and dfm["Date"].iat[-1].month == now.month
    ):
        dfm = dfm.iloc[:-1]
    return dfm

# =============================
# كشف الحالة والاختراق — مطابق لمنطق Pine (بدون شرط «اعتبارية» للبيعية)
# =============================

def detect_breakout_with_state(df: pd.DataFrame, pct: float = 0.55) -> pd.DataFrame:
    """
    • LoseCndl55: (C<O) & body_ratio>=pct & range!=0 — تُلوّن باللون الأحمر في Pine.
    • WinCndl55 : (C>O) & body_ratio>=pct & range!=0 — تُلوّن باللون الأخضر في Pine.
    • lose_high_const: آخر قمة لشمعة LoseCndl55 (ثابتة بالإغلاق).
    • win_low_const : آخر قاع لشمعة WinCndl55  (ثابتة بالإغلاق).
    • State: 0→1 عندما Close>lose_high_prev  ، والعودة 1→0 عندما Close<win_low_prev.
    • FirstBuySig: (State==1 & State(prev)==0).
    
    ملاحظة مهمة: لا نقيّد تحديث lose_high_const بشرط كسر قاع شرائية — تمامًا كما في Pine.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    o = df["Open"].to_numpy()
    h = df["High"].to_numpy()
    l = df["Low"].to_numpy()
    c = df["Close"].to_numpy()

    rng = (h - l)
    br = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55 = (c > o) & (br >= pct) & (rng != 0)

    state = 0
    states = []
    first_buy = []
    lose_high_const = []
    win_low_const = []

    lose_high = np.nan
    win_low = np.nan

    for i in range(len(df)):
        prev_lose_high = lose_high
        prev_win_low = win_low

        # إشارات الشراء/الوقف حسب قيم «البار السابق» كما في Pine
        buy = (state == 0) and (not np.isnan(prev_lose_high)) and (c[i] > prev_lose_high)
        stop = (state == 1) and (not np.isnan(prev_win_low)) and (c[i] < prev_win_low)

        if buy:
            state = 1
            first_buy.append(True)
        elif stop:
            state = 0
            first_buy.append(False)
        else:
            first_buy.append(False)

        # تحديث الثوابت بعد اتخاذ القرار (مطابقة «confirmed bar»)
        if lose55[i]:
            lose_high = h[i]
        if win55[i]:
            win_low = l[i]

        states.append(state)
        lose_high_const.append(lose_high)
        win_low_const.append(win_low)

    df["State"] = states
    df["FirstBuySig"] = first_buy
    df["LoseCndl55"] = lose55
    df["WinCndl55"] = win55
    df["LoseHigh55Const"] = lose_high_const
    df["WinLow55Const"] = win_low_const
    return df


def weekly_state_from_daily(df_daily: pd.DataFrame, suffix: str) -> bool:
    dfw = resample_weekly_from_daily(df_daily, suffix)
    if dfw.empty:
        return False
    dfw = detect_breakout_with_state(dfw)
    return bool(dfw["State"].iat[-1] == 1)


def monthly_state_from_daily(df_daily: pd.DataFrame, suffix: str) -> bool:
    dfm = resample_monthly_from_daily(df_daily, suffix)
    if dfm.empty:
        return False
    dfm = detect_breakout_with_state(dfm)
    return bool(dfm["State"].iat[-1] == 1)


def monthly_first_breakout_from_daily(df_daily: pd.DataFrame, suffix: str) -> bool:
    dfm = resample_monthly_from_daily(df_daily, suffix)
    if dfm is None or dfm.empty:
        return False
    dfm = detect_breakout_with_state(dfm)
    return bool(dfm["FirstBuySig"].iat[-1])

# =============================
# تنسيق العرض + تقريب التيك
# =============================

def _fmt_num(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "—"


def round_to_tick(x, tick=0.01):
    try:
        fx = float(x)
        return round(round(fx / tick) * tick, 2)
    except Exception:
        return x


def render_table(df: pd.DataFrame) -> str:
    from html import escape as esc

    html = ["<table><thead><tr>"]
    for col in df.columns:
        html.append(f"<th>{esc(str(col))}</th>")
    html.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        try:
            close_val = float(str(r["سعر الإغلاق"]).replace(",", ""))
        except Exception:
            close_val = None
        html.append("<tr>")
        for col in df.columns:
            val = r[col]
            cls = ""
            if close_val is not None and col in {
                "قمة الشمعة البيعية الاسبوعية",
                "قمة الشمعة البيعية اليومية",
            }:
                try:
                    top = float(str(val).replace(",", ""))
                    cls = "positive" if close_val >= top else "negative"
                except Exception:
                    cls = ""
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
    if st.session_state.login_attempts >= 5:
        st.session_state.login_error = "too_many"
        return
    users = load_users()
    me = check_login(
        st.session_state.login_username, st.session_state.login_password, users
    )
    if me is None:
        st.session_state.login_attempts += 1
        st.session_state.login_error = "bad"
    elif is_expired(me.get("expiry", "")):
        st.session_state.login_error = "expired"
    else:
        st.session_state.authenticated = True
        st.session_state.user = me
        st.session_state.login_error = None


if not st.session_state.authenticated:
    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown(
            '<h3 style="font-size:20px;">🔒 تسجيل دخول المشتركين</h3>',
            unsafe_allow_html=True,
        )
        st.text_input("اسم المستخدم", key="login_username")
        st.text_input("كلمة المرور", type="password", key="login_password")
        st.button("دخول", key="login_btn", on_click=do_login)
        if st.session_state.login_error == "bad":
            st.error("⚠️ اسم المستخدم أو كلمة المرور غير صحيحة.")
        elif st.session_state.login_error == "expired":
            st.error("⚠️ انتهى اشتراكك. يرجى التجديد.")
        elif st.session_state.login_error == "too_many":
            st.error("⛔ محاولات كثيرة. حاول لاحقًا.")
    with c1:
        st.markdown(
            "<div style='background:#f0f2f6;padding:20px;border-radius:8px;box-shadow:0 2px 5px rgb(0 0 0 / 0.1);line-height:1.6;'>"
            "<h3 style='font-size:20px;'>منصة القوة الثلاثية TriplePower</h3>"
            + linkify(load_important_links())
            + "</div>",
            unsafe_allow_html=True,
        )
    st.stop()

# تحقق الاشتراك
if is_expired(st.session_state.user["expiry"]):
    st.warning("⚠️ انتهى اشتراكك. تم تسجيل خروجك تلقائيًا.")
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

# =============================
# بعد الدخول
# =============================
me = st.session_state.user
st.markdown("---")
with st.sidebar:
    st.markdown(
        f"""<div style="background:#28a745;padding:10px;border-radius:5px;color:#fff;
                     font-weight:bold;text-align:center;margin-bottom:10px;">
                     ✅ اشتراكك سارٍ حتى: {me['expiry']}</div>""",
        unsafe_allow_html=True,
    )

    # تنبيه قرب الانتهاء
    try:
        expiry_dt = datetime.strptime(me["expiry"].strip(), "%Y-%m-%d").date()
        today_riyadh = datetime.now(ZoneInfo("Asia/Riyadh")).date()
        days_left = (expiry_dt - today_riyadh).days
        if 0 <= days_left <= 3:
            st.warning(f"⚠️ تبقّى {days_left} يومًا على انتهاء الاشتراك.")
    except Exception:
        pass

    market = st.selectbox("اختر السوق", ["السوق السعودي", "السوق الأمريكي"], key="market_select")
    suffix = ".SR" if market == "السوق السعودي" else ""

    apply_triple_filter = st.checkbox(
        "اشتراط \"فرصة\" (شهري+أسبوعي إيجابي + أول اختراق يومي)",
        value=False,
        key="triple_filter",
        help=(
            "لن يُعرض الرمز إلا إذا تحقق: شهري إيجابي، وأسبوعي إيجابي، وFirstBuySig يومي، "
            "مع إغلاق يومي فوق قمة آخر شمعة بيعية يومية 55%."
        ),
    )

    start_date = st.date_input("من", date(2020, 1, 1), key="start_date_input")
    end_date = st.date_input("إلى", date.today(), key="end_date_input")
    allow_intraday_daily = st.checkbox(
        "👁️ عرض السعر اليومي الحالي بدل إغلاق الأسبوع (اختياري)",
        value=False,
        key="intraday_price",
        help="إذا فُعل، يعرض آخر سعر يومي متاح؛ وإلا يعرض إغلاق آخر أسبوع مغلق.",
    )
    batch_size = st.slider("حجم الدُفعة عند الجلب", 20, 120, 60, 10, key="batch_size_slider")

    # تقريب الأهداف حسب التيك (اختياري)
    enable_tick_round = st.checkbox("تقريب الأهداف حسب تيك السعر", value=False, key="tick_round_enable")
    tick_value = (
        st.selectbox("قيمة التيك", [0.01, 0.05, 0.1], index=0, key="tick_value")
        if enable_tick_round
        else None
    )

    symbol_name_dict = (
        load_symbols_names("saudiSY.txt", "سعودي") if suffix == ".SR" else load_symbols_names("usaSY.txt", "امريكي")
    )

    if st.button("🎯 رموز تجريبية", key="demo_symbols_btn"):
        st.session_state.symbols = (
            "3080 4003 4013" if suffix == ".SR" else "AAPL MSFT GOOGL"
        )

    try:
        with open("رموز الاسواق العالمية.xlsx", "rb") as f:
            st.download_button(
                "📥 تحميل ملف رموز الأسواق",
                f,
                "رموز الاسواق العالمية.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except FileNotFoundError:
        st.warning("⚠️ ملف الرموز غير موجود بجانب app.py")

    if st.button("تسجيل الخروج", key="logout_btn"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

# إدخال الرموز
symbols_input = st.text_area(
    "أدخل الرموز (مفصولة بمسافة أو سطر)", st.session_state.get("symbols", ""), key="symbols_textarea"
)
symbols = [s.strip() + suffix for s in symbols_input.replace("\n", " ").split() if s.strip()]

# =============================
# تنفيذ التحليل — بناء الجدول
# =============================
if st.button("🔎 إنشاء جدول الأهداف (اليومي + الأسبوعي)", key="run_weekly_btn"):
    if not symbols:
        st.warning("⚠️ الرجاء إدخال رموز أولًا.")
        st.stop()

    with st.spinner("⏳ نجلب البيانات ونحسب الأهداف..."):
        rows = []
        total = len(symbols)
        prog = st.progress(0, text=f"بدء التحليل... (0/{total})")
        processed = 0

        for i in range(0, total, batch_size):
            chunk_syms = symbols[i : i + batch_size]
            ddata_chunk = fetch_data(" ".join(chunk_syms), start_date, end_date, "1d")
            if ddata_chunk is None or (
                isinstance(ddata_chunk, pd.DataFrame) and ddata_chunk.empty
            ):
                processed += len(chunk_syms)
                prog.progress(min(processed / total, 1.0), text=f"تمت معالجة {processed}/{total}")
                continue

            for code in chunk_syms:
                try:
                    df_d_raw = extract_symbol_df(ddata_chunk, code)
                    if df_d_raw is None or df_d_raw.empty:
                        continue

                    # يومي «مؤكَّد» لاشتقاق أسبوعي/شهري مغلقين
                    df_d_conf = drop_last_if_incomplete(
                        df_d_raw, "1d", suffix, allow_intraday_daily=False
                    )
                    if df_d_conf is None or df_d_conf.empty:
                        continue

                    # تجميع أسبوعي صحيح (الأسبوع غير المغلق يُستبعد)
                    df_w = resample_weekly_from_daily(df_d_conf, suffix)
                    if df_w is None or df_w.empty:
                        continue

                    # السعر المعروض
                    weekly_close = float(df_w["Close"].iat[-1])
                    last_close = (
                        float(df_d_raw["Close"].iat[-1]) if allow_intraday_daily else weekly_close
                    )

                    # كشف الحالة اليومية/الأسبوعية/الشهرية بمنطق Pine
                    df_d_state = detect_breakout_with_state(df_d_conf)
                    daily_state_pos = bool((not df_d_state.empty) and (df_d_state["State"].iat[-1] == 1))
                    weekly_pos = weekly_state_from_daily(df_d_conf, suffix)
                    monthly_pos = monthly_state_from_daily(df_d_conf, suffix)

                    # عناصر «فرصة» كما في Pine
                    daily_first = bool(df_d_state["FirstBuySig"].iat[-1])
                    daily_sell_high = df_d_state["LoseHigh55Const"].iat[-1]
                    d_close_conf = float(df_d_conf["Close"].iat[-1])
                    has_daily_sell_high = np.isfinite(daily_sell_high)
                    is_opportunity = (
                        monthly_pos and weekly_pos and daily_first and has_daily_sell_high and (d_close_conf > float(daily_sell_high))
                    )

                    # فلتر اختياري: «فرصة»
                    if apply_triple_filter and not is_opportunity:
                        continue

                    # أسماء
                    sym = code.replace(suffix, "").upper()
                    company = (symbol_name_dict.get(sym, "غير معروف") or "غير معروف")[:20]

                    # أسبوعي: المرساة = آخر اختراق أسبوعي
                    weekly_H, weekly_t1, weekly_t2, weekly_t3 = ("—", "—", "—", "—")
                    t_w = weekly_latest_breakout_anchor_targets(
                        df_w, pct=0.55, tick=(tick_value if tick_value else None)
                    )
                    if t_w is not None:
                        weekly_H, weekly_t1, weekly_t2, weekly_t3 = t_w

                    # يومي (مدرسة القوة): المرساة = آخر شمعة بيعية 55% كسرت قاع شمعة شرائية 55% (بنفسها أو لاحقًا)
                    daily_H, daily_t1, daily_t2, daily_t3 = ("—", "—", "—", "—")
                    t_d = last_sell_anchor_targets(
                        df_d_conf, pct=0.55
                    )
                    if t_d is not None:
                        daily_H, daily_t1, daily_t2, daily_t3 = t_d

                    # تقريب حسب التيك (اختياري) — الأهداف فقط
                    if tick_value:
                        if isinstance(weekly_t1, (int, float)):
                            weekly_t1 = round_to_tick(weekly_t1, tick_value)
                        if isinstance(weekly_t2, (int, float)):
                            weekly_t2 = round_to_tick(weekly_t2, tick_value)
                        if isinstance(weekly_t3, (int, float)):
                            weekly_t3 = round_to_tick(weekly_t3, tick_value)
                        if isinstance(daily_t1, (int, float)):
                            daily_t1 = round_to_tick(daily_t1, tick_value)
                        if isinstance(daily_t2, (int, float)):
                            daily_t2 = round_to_tick(daily_t2, tick_value)
                        if isinstance(daily_t3, (int, float)):
                            daily_t3 = round_to_tick(daily_t3, tick_value)

                    # شهري: نص الوقف/التسارع الشهري — وفق مدرسة TriplePower
                    df_m = resample_monthly_from_daily(df_d_conf, suffix)
                    monthly_text = "لا توجد شمعة بيعية شهرية معتبرة"
                    info_m = (
                        last_sell_anchor_info(df_m, pct=0.55)
                        if (df_m is not None and not df_m.empty)
                        else None
                    )
                    if info_m is not None:
                        Hm = float(info_m["H"])
                        Lm = float(info_m["L"])  # احتياطي عند عدم وجود شرائية 55%
                        if d_close_conf < Hm:
                            monthly_text = f"غير متواجدة ويجب الإغلاق فوق {Hm:.2f}"
                        else:
                            dfm_calc = df_m[["Open", "High", "Low", "Close"]].dropna().copy()
                            oM = dfm_calc["Open"].to_numpy()
                            hM = dfm_calc["High"].to_numpy()
                            lM = dfm_calc["Low"].to_numpy()
                            cM = dfm_calc["Close"].to_numpy()
                            rngM = (hM - lM)
                            brM = np.where(rngM != 0, np.abs(cM - oM) / rngM, 0.0)
                            win55M = (cM > oM) & (brM >= 0.55) & (rngM != 0)
                            idx_win = np.where(win55M)[0]
                            last_win_low_val = float(lM[idx_win[-1]]) if len(idx_win) > 0 else Lm
                            monthly_text = f"متواجدة بشرط الحفاظ على {last_win_low_val:.2f}"

                    rows.append(
                        {
                            "اسم الشركة": company,
                            "الرمز": sym,
                            "سعر الإغلاق": round(last_close, 2),
                            "قمة الشمعة البيعية اليومية": daily_H,
                            "الهدف الأول (يومي)": daily_t1,
                            "الهدف الثاني (يومي)": daily_t2,
                            "الهدف الثالث (يومي)": daily_t3,
                            "قمة الشمعة البيعية الاسبوعية": weekly_H,
                            "الهدف الأول (اسبوعي)": weekly_t1,
                            "الهدف الثاني (اسبوعي)": weekly_t2,
                            "الهدف الثالث (اسبوعي)": weekly_t3,
                            "القوة والتسارع الشهري": monthly_text,
                        }
                    )

                except Exception:
                    continue

            processed += len(chunk_syms)
            prog.progress(
                min(processed / total, 1.0), text=f"تمت معالجة {processed}/{total}"
            )

        # ===== إخراج الجدول =====
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
            ]
            df_final = pd.DataFrame(rows)[columns_order]

            # تنسيق أرقام للعرض — استثناء العمود النصّي الشهري
            non_numeric_cols = {"اسم الشركة", "الرمز", "القوة والتسارع الشهري"}
            for col in df_final.columns:
                if col in non_numeric_cols:
                    continue
                df_final[col] = df_final[col].apply(lambda x: _fmt_num(x))

            market_name = "السوق السعودي" if suffix == ".SR" else "السوق الأمريكي"
            day_str = f"{end_date.day}-{end_date.month}-{end_date.year}"
            filt_note = (
                "— فلترة \"فرصة\" مفعّلة"
                if apply_triple_filter
                else "— بدون اشتراط الفرصة"
            )
            if allow_intraday_daily:
                filt_note += " — عرض السعر اليومي الحالي"
            st.subheader(
                f"🎯 جدول الأهداف (اليومي + الأسبوعي) — {market_name} — {day_str} — عدد الرموز: {len(df_final)} {filt_note}"
            )

            st.markdown(render_table(df_final), unsafe_allow_html=True)
            st.download_button(
                "📥 تنزيل جدول الأهداف (اليومي + الأسبوعي) CSV",
                df_final.to_csv(index=False).encode("utf-8-sig"),
                file_name="TriplePower_Targets_DailyWeekly.csv",
                mime="text/csv",
            )
        else:
            st.info("لا توجد بيانات كافية لحساب الأهداف على الفواصل المحددة.")
