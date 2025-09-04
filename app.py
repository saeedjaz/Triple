# =============================================
# استبدل الدوال أدناه بنظيراتها في app.py
# خطوة 1 — ضبط معيار "كسر الشمعة الشرائية" ليكون بالإغلاق لا بالـ Low
# =============================================
import numpy as np
import pandas as pd

# ---------------------------------------------
# 1) اختيار آخر شمعة بيعية معتبرة 55%:
#    تعديل معيار الكسر من (l <= last_win_low) إلى (c <= last_win_low)
#    وكذلك عند فحص الكسر اللاحق نعتمد future_min_close بدل future_min للـ Lows
# ---------------------------------------------

def last_sell_anchor_info(_df: pd.DataFrame, pct: float = 0.55):
    """
    تُرجع dict تحتوي idx/H/L/R لآخر شمعة بيعية 55% كسرت قاع شمعة شرائية 55%
    (بنفسها بالإغلاق أو لاحقًا بالإغلاق) — وفق مدرسة TriplePower.
    لا يتم تقريب H/L هنا للحفاظ على الدقة؛ التقريب يكون عند الإخراج فقط.
    """
    if _df is None or _df.empty:
        return None
    df = _df[["Open","High","Low","Close"]].dropna().copy()
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy();  c = df["Close"].to_numpy()

    rng = (h - l)
    br  = np.where(rng != 0, np.abs(c - o) / rng, 0.0)
    lose55 = (c < o) & (br >= pct) & (rng != 0)
    win55  = (c > o) & (br >= pct) & (rng != 0)

    # آخر قاع شرائي 55% قبل كل نقطة
    last_win_low = np.full(c.shape, np.nan)
    cur = np.nan
    for i in range(len(c)):
        if win55[i]:
            cur = l[i]
        last_win_low[i] = cur

    # أصغر إغلاق مستقبلي (لتحقيق "الكسر لاحقًا" بالإغلاق)
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
    """
    تُرجع (H, T1, T2, T3) بحسب آخر شمعة بيعية 55% المعتبرة بالإغلاق.
    الأهداف: T1=H+R، T2=H+2R، T3=H+3R حيث R=H-L.
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

# ---------------------------------------------
# 2) آخر اختراق أسبوعي/يومي — أيضًا نعتمد تحقق الكسر السابق بالإغلاق
# ---------------------------------------------

def weekly_latest_breakout_anchor_targets(_df: pd.DataFrame, pct: float = 0.55):
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
        if win55[i]: cur = l[i]
        last_win_low[i] = cur

    future_min_close = np.minimum.accumulate(c[::-1])[::-1]

    anchors = np.where(
        lose55 & ~np.isnan(last_win_low) &
        ((c <= last_win_low) | (future_min_close <= last_win_low))
    )[0]
    if len(anchors) == 0:
        return None

    # نحتفظ بالمرساة التي كان اختراق قمتها بالإغلاق هو الأحدث
    events = []  # (t_break, j_anchor)
    for j in anchors:
        later = np.where(c[j+1:] > h[j])[0]
        if len(later) == 0: continue
        t_break = int(j + 1 + later[0])
        events.append((t_break, j))
    if len(events) == 0:
        return None

    _, j_last = max(events, key=lambda x: x[0])
    H = float(h[j_last]); L = float(l[j_last]); R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return (
        round(H, 2),
        round(H + 1.0*R, 2),
        round(H + 2.0*R, 2),
        round(H + 3.0*R, 2)
    )


def daily_latest_breakout_anchor_targets(_df: pd.DataFrame, pct: float = 0.55):
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
        if win55[i]: cur = l[i]
        last_win_low[i] = cur

    future_min_close = np.minimum.accumulate(c[::-1])[::-1]

    anchors = np.where(
        lose55 & ~np.isnan(last_win_low) &
        ((c <= last_win_low) | (future_min_close <= last_win_low))
    )[0]
    if len(anchors) == 0:
        return None

    events = []  # (t_break, j_anchor)
    for j in anchors:
        later = np.where(c[j+1:] > h[j])[0]
        if len(later) == 0: continue
        t_break = int(j + 1 + later[0])
        events.append((t_break, j))
    if len(events) == 0:
        return None

    _, j_last = max(events, key=lambda x: x[0])
    H = float(h[j_last]); L = float(l[j_last]); R = H - L
    if not np.isfinite(R) or R <= 0:
        return None
    return (
        round(H, 2),
        round(H + 1.0*R, 2),
        round(H + 2.0*R, 2),
        round(H + 3.0*R, 2)
    )

# ---------------------------------------------
# 3) منطق الحالة/الإشارة — ضبط اعتبار "البيعية المعتبرة الآن": بالإغلاق
# ---------------------------------------------

def detect_breakout_with_state(df: pd.DataFrame, pct: float=0.55) -> pd.DataFrame:
    if df is None or df.empty: return df
    o=df["Open"].values; h=df["High"].values; l=df["Low"].values; c=df["Close"].values
    rng=(h-l); br=np.where(rng!=0, np.abs(c-o)/rng, 0.0)
    lose55=(c<o) & (br>=pct) & (rng!=0)
    win55 =(c>o) & (br>=pct) & (rng!=0)

    last_win_low=np.full(c.shape, np.nan); cur=np.nan
    for i in range(len(c)):
        if win55[i]: cur=l[i]
        last_win_low[i]=cur

    # معيار اعتبار البيعية الآن: إغلاق تحت قاع الشرائية 55%
    valid_sell_now = lose55 & ~np.isnan(last_win_low) & (c <= last_win_low)

    state=0; states=[]; first_buy=[]; lose_high=np.nan; win_low=np.nan
    for i in range(len(df)):
        buy  = (state==0) and (not np.isnan(lose_high)) and (c[i]>lose_high)
        stop = (state==1) and (not np.isnan(win_low))   and (c[i]<win_low)
        if buy:
            state=1; first_buy.append(True)
        elif stop:
            state=0; first_buy.append(False); lose_high=np.nan
        else:
            first_buy.append(False)
        if valid_sell_now[i]: lose_high=h[i]
        if win55[i]: win_low=l[i]
        states.append(state)

    df = df.copy()
    df["State"]=states; df["FirstBuySig"]=first_buy
    df["LoseCndl55"]=valid_sell_now; df["WinCndl55"]=win55
    return df
