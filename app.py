
import re
import time
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from html import escape

# =============================
# تهيئة الصفحة + RTL
# =============================
st.set_page_config(page_title="🕌 Shariah NASDAQ Filter | قرار (485)", layout="wide")
RTL_CSS = """
<style>
  :root, html, body, .stApp { direction: rtl; }
  .stApp { text-align: right; }
  input, textarea, select { direction: rtl; text-align: right; }
  .stTextInput input, .stTextArea textarea, .stSelectbox div[role="combobox"],
  .stNumberInput input, .stMultiSelect [data-baseweb], label, .stButton button { text-align: right; }
  table { direction: rtl; }
  .halal  { background-color:#d1fae5; color:#065f46; font-weight:bold; }
  .haram  { background-color:#fee2e2; color:#991b1b; font-weight:bold; }
  .review { background-color:#fff7ed; color:#92400e; font-weight:bold; }
  .btn-link { display:inline-block; padding:6px 10px; border-radius:8px; background:#1a73e8; color:#fff !important; font-weight:600; text-decoration:none; }
  .badges { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 16px; }
  .badge { padding:6px 12px; border-radius:999px; font-weight:700; }
  .badge.halal { background:#d1fae5; color:#065f46; }
  .badge.haram { background:#fee2e2; color:#991b1b; }
  .badge.review { background:#fff7ed; color:#92400e; }
</style>
"""
st.markdown(RTL_CSS, unsafe_allow_html=True)

# =============================
# جدول HTML
# =============================
def generate_html_table(df: pd.DataFrame) -> str:
    html = """
    <style>
    table {border-collapse: collapse; width: 100%; direction: rtl; font-family: Arial, sans-serif;}
    th, td {border: 1px solid #ddd; padding: 8px; text-align: center;}
    th {background-color: #04AA6D; color: white;}
    tr:nth-child(even){background-color: #f2f2f2;}
    tr:hover {background-color: #ddd;}
    a {text-decoration: none;}
    </style>
    <table>
    <thead><tr>"""
    for col in df.columns:
        html += f"<th>{escape(col)}</th>"
    html += "</tr></thead><tbody>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            val = row[col]
            cell_class = ""
            if col == "الحكم الشرعي":
                v = str(val).strip()
                cell_class = "review"
                if v == "مباح": cell_class = "halal"
                elif v == "غير مباح": cell_class = "haram"
            if col == "رابط TradingView":
                url = escape(val)
                html += f'<td><a dir="ltr" class="btn-link" href="{url}" target="_blank" rel="noopener">TradingView</a></td>'
            elif col == "نِسَب شرعية" and isinstance(val, str):
                parts = []
                for seg in val.split("|"):
                    seg = seg.strip()
                    if ":" in seg:
                        k, v2 = seg.split(":", 1)
                        parts.append(f'{escape(k.strip())}: <span dir="ltr">{escape(v2.strip())}</span>')
                    else:
                        parts.append(f'<span dir="ltr">{escape(seg)}</span>')
                html += f'<td class="{cell_class}">' + " | ".join(parts) + "</td>"
            else:
                html += f'<td class="{cell_class}">{escape(str(val))}</td>'
        html += "</tr>"
    html += "</tbody></table>"
    return html

# =============================
# 🕌 خوارزمية القرار (485) + تحديث 2004
# =============================
BANNED_PATTERNS = [
    r"bank", r"insurance", r"reinsurance", r"mortgage", r"capital markets",
    r"consumer finance", r"casino", r"gaming", r"gambling", r"tobacco",
    r"winery|distiller|alcohol", r"adult"
]

def _sector_haram(sector: str, industry: str) -> bool:
    s = f"{sector or ''} {industry or ''}".lower()
    import re as _re
    return any(_re.search(p, s) for p in BANNED_PATTERNS)

def _try_rows(df, variants):
    if df is None or getattr(df, "empty", True): return None
    idx_map = {str(i).strip().lower(): i for i in df.index}
    for v in variants:
        key = v.strip().lower()
        if key in idx_map:
            return df.loc[idx_map[key]]
    for v in variants:
        vv = v.strip().lower()
        for k, orig in idx_map.items():
            if vv in k:
                return df.loc[orig]
    return None

def _first_scalar(x):
    import numpy as _np, pandas as _pd
    if x is None: return _np.nan
    try:
        s = x.dropna()
        if len(s) > 0: return float(s.iloc[0])
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return _np.nan

@st.cache_data(ttl=86400)
def shariah_screen_nasdaq(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    # معلومات عامة
    try:
        info = t.get_info() if hasattr(t, "get_info") else t.info
    except Exception:
        info = getattr(t, "info", {}) or {}
    name = info.get("shortName") or info.get("longName") or "غير معروف"
    sector = info.get("sector")
    industry = info.get("industry")

    # القوائم
    def pick(getters):
        for g in getters:
            try:
                df = getattr(t, g)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        return None
    bs = pick(["balance_sheet", "quarterly_balance_sheet"])
    is_ = pick(["income_stmt", "quarterly_income_stmt"])

    # السوقية
    market_cap = None
    try:
        market_cap = getattr(t, "fast_info", {}).get("market_cap", None)
    except Exception:
        pass
    if not market_cap:
        market_cap = info.get("marketCap")

    # الدفترية
    equity_row = _try_rows(bs, [
        "Total Stockholder Equity", "Total Equity Gross Minority Interest",
        "Stockholders Equity", "Total Shareholder Equity", "Total Stockholders' Equity"
    ])
    book_value = _first_scalar(equity_row)

    # الدَّين
    total_debt_row = _try_rows(bs, [
        "Total Debt", "Total Debt Net", "Total Borrowings", "Total Interest-Bearing Debt"
    ])
    total_debt = _first_scalar(total_debt_row)
    if np.isnan(total_debt):
        long_debt = _first_scalar(_try_rows(bs, [
            "Long Term Debt", "Long-Term Debt", "Long Term Debt And Capital Lease Obligation"
        ]))
        short_debt = _first_scalar(_try_rows(bs, [
            "Short Long Term Debt", "Short/Current Long Term Debt",
            "Current Portion Of Long Term Debt", "Short Term Debt", "Short Term Borrowings"
        ]))
        if pd.notna(long_debt) or pd.notna(short_debt):
            total_debt = (0 if pd.isna(long_debt) else long_debt) + (0 if pd.isna(short_debt) else short_debt)

    # الإيرادات
    total_rev_val = _first_scalar(_try_rows(is_, [
        "Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue", "Sales/Revenue"
    ]))

    # المحرَّم (موجب فقط)
    interest_income   = _first_scalar(_try_rows(is_, [
        "Interest Income", "Net Interest Income", "Interest And Similar Income",
        "Non-Operating Interest Income", "Net Non Operating Interest Income Expense"
    ]))
    investment_income = _first_scalar(_try_rows(is_, [
        "Investment Income, Net", "Net Investment Income", "Income From Investment", "Investment Income"
    ]))
    other_nonop       = _first_scalar(_try_rows(is_, [
        "Other Non Operating Income", "Total Other Income/Expense Net", "Other Income (Expense)",
        "Other income (expense)", "Other Income"
    ]))
    haram_parts = [v for v in [interest_income, investment_income, other_nonop] if pd.notna(v) and v > 0]
    haram_income_est = np.nan if not haram_parts else float(sum(haram_parts))

    # الموانع
    sector_ok = not _sector_haram(sector, industry)

    # النِّسَب
    denom = None
    if pd.notna(market_cap) and market_cap and market_cap > 0:
        denom = float(market_cap)
    if pd.notna(book_value) and book_value and book_value > 0:
        denom = max(denom or 0, float(book_value))
    debt_ratio  = (float(total_debt) / denom) if (denom and pd.notna(total_debt)) else np.nan
    haram_ratio = (float(haram_income_est) / float(total_rev_val)) if (pd.notna(haram_income_est) and pd.notna(total_rev_val) and total_rev_val > 0) else np.nan

    reasons, needs_review = [], False
    if not sector_ok:
        reasons.append("النشاط/الصناعة ضمن المحرّمات.")
    if pd.isna(debt_ratio):
        needs_review = True; reasons.append("تعذّر حساب نسبة الدَّين (بيانات ناقصة).")
    elif debt_ratio > 0.30:
        reasons.append(f"نسبة الدَّين {debt_ratio:.2%} تتجاوز 30%.")
    if pd.isna(haram_ratio):
        needs_review = True; reasons.append("تعذّر تقدير نسبة الإيراد المحرّم بدقة.")
    elif haram_ratio > 0.05:
        reasons.append(f"نسبة الإيراد المحرّم {haram_ratio:.2%} تتجاوز 5%.")

    if sector_ok and (pd.notna(debt_ratio) and debt_ratio <= 0.30) and (pd.notna(haram_ratio) and haram_ratio <= 0.05):
        verdict = "مباح"
    elif needs_review and not any("تتجاوز" in r for r in reasons):
        verdict = "يحتاج مراجعة"
    else:
        verdict = "غير مباح"

    return {
        "name": name, "sector": sector, "industry": industry,
        "verdict": verdict,
        "debt_ratio": None if pd.isna(debt_ratio) else float(debt_ratio),
        "haram_ratio": None if pd.isna(haram_ratio) else float(haram_ratio),
        "reasons": reasons
    }

def _fmt_ratio(x):
    return "غير متاح" if x is None or pd.isna(x) else f"{x*100:.2f}%"

# =============================
# واجهة المستخدم
# =============================
st.title("🕌 الفلتر الشرعي لأسهم ناسداك (قرار 485) — مستقل")
st.caption("حساب تقريبي يعتمد بيانات Yahoo Finance. ليس فتوى.")

with st.form("shariah_form"):
    symbols_text = st.text_area("أدخل رموز ناسداك (مفصولة بمسافة أو سطر):", "AAPL MSFT GOOGL NVDA")
    throttle = st.slider("إبطاء الطلبات (ثوانٍ بين كل رمز):", 0.0, 2.0, 0.6, 0.1)
    show_reasons = st.checkbox("عرض الملاحظات/الأسباب", True)
    force_refresh = st.checkbox("تحديث البيانات (تجاهل الكاش)", False)
    submitted = st.form_submit_button("تشغيل التحليل الشرعي")

if submitted:
    syms = [s.strip().upper() for s in symbols_text.replace("\n", " ").split() if s.strip()]
    if not syms:
        st.warning("⚠️ الرجاء إدخال رموز.")
    else:
        if force_refresh:
            try:
                shariah_screen_nasdaq.clear()
            except Exception:
                try:
                    st.cache_data.clear()
                except Exception:
                    pass

        rows, errors = [], 0
        progress = st.progress(0)
        total = len(syms)

        for i, sym in enumerate(syms, start=1):
            try:
                if throttle and throttle > 0:
                    time.sleep(throttle)
                sh = shariah_screen_nasdaq(sym)
                dr = _fmt_ratio(sh.get("debt_ratio"))
                hr = _fmt_ratio(sh.get("haram_ratio"))
                rows.append({
                    "م": i,
                    "الرمز": sym,
                    "اسم الشركة": sh.get("name", "غير معروف"),
                    "القطاع": sh.get("sector", ""),
                    "الصناعة": sh.get("industry", ""),
                    "الحكم الشرعي": sh.get("verdict", "يحتاج مراجعة"),
                    "نِسَب شرعية": f"دين: {dr} | محرم: {hr}",
                    "ملاحظات": "؛ ".join(sh.get("reasons", [])) if show_reasons else "",
                    "رابط TradingView": f"https://www.tradingview.com/symbols/{sym}/",
                })
            except Exception as e:
                errors += 1
                rows.append({
                    "م": i, "الرمز": sym, "اسم الشركة": "غير معروف",
                    "القطاع": "", "الصناعة": "",
                    "الحكم الشرعي": "يneeds مراجعة",
                    "نِسَب شرعية": "دين: غير متاح | محرم: غير متاح",
                    "ملاحظات": f"تعذّر التحليل: {type(e).__name__}" if show_reasons else "",
                    "رابط TradingView": f"https://www.tradingview.com/symbols/{sym}/",
                })
            progress.progress(i/total)

        progress.empty()

        df = pd.DataFrame(rows)[
            ["م","الرمز","اسم الشركة","القطاع","الصناعة","الحكم الشرعي","نِسَب شرعية","ملاحظات","رابط TradingView"]
        ]

        counts = df["الحكم الشرعي"].value_counts().to_dict()
        st.markdown(f"""
<div class="badges">
  <span class="badge halal">مباح: {counts.get("مباح",0)}</span>
  <span class="badge haram">غير مباح: {counts.get("غير مباح",0)}</span>
  <span class="badge review">يحتاج مراجعة: {counts.get("يحتاج مراجعة",0)}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown(generate_html_table(df), unsafe_allow_html=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 تنزيل النتائج CSV", data=csv, file_name="Shariah_NASDAQ_results.csv", mime="text/csv")

        if errors:
            st.info(f"ℹ️ تمت معالجة {total} رمزًا مع {errors} نتيجة تحتاج مراجعة بسبب تعذّر جلب بيانات كاملة.")
