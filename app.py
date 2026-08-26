import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v10.4", layout="wide")
st.title("🧪 단기스윙 v10 · E1+X3 엣지 검증")
st.caption("v10.3 상위 진입특성 기반 · 단독 필터/2개 조합/OOS/연도별 강건성 검증")

TARGETS = [3.0, 5.0, 7.0, 10.0]

@st.cache_data(ttl=86400, show_spinner=False)
def stock_listing():
    kospi = fdr.StockListing("KOSPI")
    kosdaq = fdr.StockListing("KOSDAQ")

    def normalize(df, market):
        out = df.copy()
        if "Code" not in out.columns and "Symbol" in out.columns:
            out["Code"] = out["Symbol"]
        if "Name" not in out.columns:
            out["Name"] = out["Code"]
        out["Market"] = market
        return out[["Code", "Name", "Market"]]

    return pd.concat(
        [normalize(kospi, "KOSPI"), normalize(kosdaq, "KOSDAQ")],
        ignore_index=True
    )

@st.cache_data(ttl=86400, show_spinner=False)
def load_data(code, start, end):
    return fdr.DataReader("NAVER:" + str(code).zfill(6), start, end)

def add_indicators(d):
    x = d.copy()
    x["거래대금"] = x["Close"] * x["Volume"]
    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA60"] = x["Close"].rolling(60).mean()
    x["AVG_VOL20"] = x["Volume"].shift(1).rolling(20).mean()
    x["등락률"] = x["Close"].pct_change() * 100
    x["MA20기울기5일(%)"] = (x["MA20"] / x["MA20"].shift(5) - 1) * 100
    x["전고점20"] = x["High"].shift(1).rolling(20).max()
    x["전고점60"] = x["High"].shift(1).rolling(60).max()
    return x

def base_ok(d, b, p):
    x = d.iloc[b]
    return (
        pd.notna(x["AVG_VOL20"])
        and x["등락률"] >= p["base_rise"]
        and x["거래대금"] >= p["value_eok"] * 1e8
        and x["Volume"] >= x["AVG_VOL20"] * p["volume_mult"]
        and x["Close"] > x["MA5"]
        and x["MA5"] > d["MA5"].iloc[b - 1]
        and x["MA20"] > d["MA20"].iloc[b - 1]
    )

def find_setups(d, code, name, market, p, cut, end):
    """공통 setup 생성.
    기준봉 -> 2~5거래일 눌림 -> 다음날 눌림고가 돌파.
    돌파 거래량/눌림 거래량 필터를 여기서 적용한다.
    """
    d = add_indicators(d)
    rows = []

    for b in range(21, len(d)):
        if d.index[b] < cut or d.index[b] > end:
            continue
        if not base_ok(d, b, p):
            continue

        x = d.iloc[b]

        for k in range(2, 6):
            pull_i = b + k
            breakout_i = pull_i + 1
            if breakout_i >= len(d):
                break

            pull = d.iloc[pull_i]
            br = d.iloc[breakout_i]

            pull_ok = (
                pull["Volume"] <= x["Volume"] * p["pullback_ratio"]
                and pull["Close"] > x["Low"]
            )
            if not pull_ok:
                continue

            breakout_level = float(pull["High"])
            if float(br["High"]) < breakout_level:
                continue

            breakout_vol_ratio = float(br["Volume"]) / max(float(pull["Volume"]), 1.0)
            if breakout_vol_ratio < p["breakout_vol_cut"]:
                continue

            rows.append({
                "시장": market,
                "종목명": name,
                "코드": str(code).zfill(6),
                "기준봉일": d.index[b].date(),
                "기준봉상승률(%)": round(float(x["등락률"]), 2),
                "거래대금(억)": round(float(x["거래대금"]) / 1e8, 0),
                "기준봉거래량배수": round(float(x["Volume"] / x["AVG_VOL20"]), 2),

                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(breakout_level),
                "눌림저가": round(float(pull["Low"])),
                "눌림종가": round(float(pull["Close"])),
                "눌림깊이(%)": round((float(pull["Close"]) / float(x["Close"]) - 1) * 100, 2),

                "돌파일": d.index[breakout_i].date(),
                "돌파시가": round(float(br["Open"])),
                "돌파고가": round(float(br["High"])),
                "돌파저가": round(float(br["Low"])),
                "돌파종가": round(float(br["Close"])),
                "돌파종가수익률(%)": round((float(br["Close"]) / breakout_level - 1) * 100, 2),
                "돌파거래량vs눌림(배)": round(breakout_vol_ratio, 2),

                "_b": b,
                "_pull_i": pull_i,
                "_breakout_i": breakout_i,
            })
            break

    return rows, d

def make_entry(d, setup, entry_mode, p):
    pull_i = int(setup["_pull_i"])
    breakout_i = int(setup["_breakout_i"])
    level = float(setup["눌림고가"])

    # E1: 현재 B 방식. 돌파 당일 눌림고가 진입.
    if entry_mode == "E1 돌파당일":
        return {
            "진입일": d.index[breakout_i].date(),
            "진입가": level,
            "_entry_i": breakout_i,
            "_eval_i": breakout_i,
            "진입설명": "눌림고가 돌파 당일 진입",
        }

    # E2: 돌파 종가 확인 -> 다음 거래일 시가 진입.
    if entry_mode == "E2 종가확인":
        if float(d.iloc[breakout_i]["Close"]) < level:
            return None
        entry_i = breakout_i + 1
        if entry_i >= len(d):
            return None
        return {
            "진입일": d.index[entry_i].date(),
            "진입가": float(d.iloc[entry_i]["Open"]),
            "_entry_i": entry_i,
            "_eval_i": entry_i,
            "진입설명": "돌파 종가 확인 후 다음날 시가",
        }

    # E3: 돌파 후 1~3거래일 안에 돌파선 재확인(retest) 후 다음날 시가.
    # retest: 저가가 돌파선 +2% 안쪽까지 내려오고 종가는 돌파선 위 유지.
    if entry_mode == "E3 재눌림":
        start = breakout_i + 1
        end_i = min(len(d) - 2, breakout_i + int(p["retest_days"]))

        for i in range(start, end_i + 1):
            r = d.iloc[i]
            touched = float(r["Low"]) <= level * (1 + p["retest_touch_pct"] / 100.0)
            held = float(r["Close"]) >= level
            if touched and held:
                entry_i = i + 1
                return {
                    "진입일": d.index[entry_i].date(),
                    "진입가": float(d.iloc[entry_i]["Open"]),
                    "_entry_i": entry_i,
                    "_eval_i": entry_i,
                    "진입설명": "돌파 후 재눌림 지지 확인 다음날 시가",
                    "재눌림일": d.index[i].date(),
                }
        return None

    return None

def initial_stop(entry_price, setup, exit_mode, p):
    if exit_mode in ["X1 고정-3%", "X3 +3%후트레일"]:
        return entry_price * (1 - 0.03)

    # X2 구조손절: 눌림 저점 아래. 단 리스크가 너무 크면 거래 제외.
    structural = float(setup["눌림저가"])
    risk_pct = (structural / entry_price - 1) * 100
    if risk_pct < -p["max_structural_risk"]:
        return None
    return structural

def evaluate_trade(d, setup, entry, exit_mode, p):
    entry_i = int(entry["_entry_i"])
    entry_price = float(entry["진입가"])
    stop_price = initial_stop(entry_price, setup, exit_mode, p)

    if stop_price is None or stop_price >= entry_price:
        return None

    hit = {t: False for t in TARGETS}
    mfe, mae = 0.0, 0.0

    last_i = min(len(d) - 1, entry_i + p["holding_days"] - 1)
    exit_i = None
    exit_price = None
    reason = None

    # X3 state:
    activated = False
    highest_before_today = entry_price
    active_stop = stop_price

    for i in range(entry_i, last_i + 1):
        r = d.iloc[i]
        high = float(r["High"])
        low = float(r["Low"])

        # X3은 오늘 시작 시점에 '전일까지 알고 있던 최고가'만 사용해 스톱 계산.
        if exit_mode == "X3 +3%후트레일" and activated:
            trail_from_prior = highest_before_today * (1 - p["trail_pct"] / 100.0)
            active_stop = max(entry_price, trail_from_prior)
        else:
            active_stop = stop_price

        high_ret = (high / entry_price - 1) * 100
        low_ret = (low / entry_price - 1) * 100

        if low <= active_stop:
            mae = min(mae, (active_stop / entry_price - 1) * 100)
            # 같은 날 상승과 손절 순서는 알 수 없으므로 손절 우선.
            if high < entry_price * (1 + p.get("activation_pct", 3.0) / 100.0):
                mfe = max(mfe, max(0.0, high_ret))
            exit_i = i
            exit_price = active_stop
            if exit_mode == "X3 +3%후트레일" and activated and active_stop >= entry_price:
                reason = "본전/트레일"
            else:
                reason = "손절"
            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)

        for t in TARGETS:
            if high >= entry_price * (1 + t / 100.0):
                hit[t] = True

        # +3% 활성화는 그 거래일 종료 후부터 적용.
        if exit_mode == "X3 +3%후트레일" and high >= entry_price * (1 + p.get("activation_pct", 3.0) / 100.0):
            activated = True

        highest_before_today = max(highest_before_today, high)

    if exit_i is None:
        exit_i = last_i
        exit_price = float(d.iloc[last_i]["Close"])
        reason = "기간종료"

    ret = (exit_price / entry_price - 1) * 100

    return {
        "초기손절가": round(stop_price),
        "청산일": d.index[exit_i].date(),
        "청산가": round(exit_price),
        "청산사유": reason,
        "손절": reason == "손절",
        "최종수익률(%)": round(ret, 2),
        "MFE(%)": round(mfe, 2),
        "MAE(%)": round(mae, 2),
        "+3%": hit[3.0],
        "+5%": hit[5.0],
        "+7%": hit[7.0],
        "+10%": hit[10.0],
    }

def dedup_setups(df):
    if df.empty:
        return df
    return (
        df.sort_values(["코드", "돌파일", "기준봉일"])
        .drop_duplicates(subset=["코드", "돌파일"], keep="last")
        .reset_index(drop=True)
    )

def summarize(df):
    if df.empty:
        return {
            "신호":0, "승률(%)":0.0, "평균수익률(%)":0.0, "손절률(%)":0.0,
            "+3%(%)":0.0, "+5%(%)":0.0, "+7%(%)":0.0, "+10%(%)":0.0,
            "평균MFE(%)":0.0, "평균MAE(%)":0.0
        }

    return {
        "신호": len(df),
        "승률(%)": round((df["최종수익률(%)"] > 0).mean() * 100, 1),
        "평균수익률(%)": round(df["최종수익률(%)"].mean(), 2),
        "손절률(%)": round(df["손절"].mean() * 100, 1),
        "+3%(%)": round(df["+3%"].mean() * 100, 1),
        "+5%(%)": round(df["+5%"].mean() * 100, 1),
        "+7%(%)": round(df["+7%"].mean() * 100, 1),
        "+10%(%)": round(df["+10%"].mean() * 100, 1),
        "평균MFE(%)": round(df["MFE(%)"].mean(), 2),
        "평균MAE(%)": round(df["MAE(%)"].mean(), 2),
    }

def apply_costs(df, buy_slip_pct, sell_slip_pct, fees_tax_pct):
    """실전비용 단순 보수모델.
    순수익률 = 총수익률 - 매수 슬리피지 - 매도 슬리피지 - 왕복 수수료/세금.
    개별 증권사/시장별 실제 비용과는 다를 수 있으므로 사용자가 값을 조정한다.
    """
    q = df.copy()
    total_cost = float(buy_slip_pct) + float(sell_slip_pct) + float(fees_tax_pct)
    q["총비용(%)"] = round(total_cost, 4)
    q["순수익률(%)"] = q["최종수익률(%)"] - total_cost
    q["순이익"] = q["순수익률(%)"] > 0
    return q

def performance_stats(df, ret_col="순수익률(%)"):
    if df.empty:
        return {
            "신호":0, "승률(%)":0.0, "평균수익률(%)":0.0, "중앙수익률(%)":0.0,
            "MDD(%)":0.0, "최대연속손실":0, "ProfitFactor":None,
            "누적복리수익률(%)":0.0
        }

    q = df.sort_values(["진입일", "코드"]).copy()
    r = q[ret_col].astype(float) / 100.0

    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    mdd = float(dd.min()) if len(dd) else 0.0

    streak = 0
    max_streak = 0
    for x in q[ret_col].astype(float):
        if x <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    gains = q.loc[q[ret_col] > 0, ret_col].sum()
    losses = q.loc[q[ret_col] <= 0, ret_col].sum()
    pf = float(gains / abs(losses)) if losses < 0 else None

    return {
        "신호": len(q),
        "승률(%)": round((q[ret_col] > 0).mean() * 100, 1),
        "평균수익률(%)": round(q[ret_col].mean(), 2),
        "중앙수익률(%)": round(q[ret_col].median(), 2),
        "MDD(%)": round(mdd, 2),
        "최대연속손실": int(max_streak),
        "ProfitFactor": None if pf is None else round(pf, 2),
        "누적복리수익률(%)": round((equity.iloc[-1] - 1.0) * 100.0, 2),
    }

def chronological_oos(df, train_ratio=0.60):
    """시간순 holdout. 파라미터를 재최적화하지 않고 앞 구간/뒤 구간을 분리해 동일 전략의 지속성을 본다."""
    q = df.sort_values(["진입일", "코드"]).reset_index(drop=True).copy()
    if len(q) < 10:
        return pd.DataFrame(), pd.DataFrame(), None

    split_i = max(1, min(len(q)-1, int(len(q) * train_ratio)))
    train = q.iloc[:split_i].copy()
    test = q.iloc[split_i:].copy()
    split_date = test.iloc[0]["진입일"]
    return train, test, split_date

def yearly_walkforward(df, ret_col="순수익률(%)"):
    q = df.copy()
    q["연도"] = pd.to_datetime(q["진입일"]).dt.year
    rows = []
    years = sorted(q["연도"].dropna().unique())

    for y in years:
        test = q[q["연도"] == y].copy()
        prior = q[q["연도"] < y].copy()

        test_stats = performance_stats(test, ret_col)
        prior_avg = prior[ret_col].mean() if not prior.empty else float("nan")

        rows.append({
            "OOS연도": int(y),
            "이전연도신호": len(prior),
            "이전연도평균(%)": None if pd.isna(prior_avg) else round(prior_avg, 2),
            **test_stats
        })
    return pd.DataFrame(rows)

def equity_curve_table(df, ret_col="순수익률(%)"):
    q = df.sort_values(["진입일", "코드"]).reset_index(drop=True).copy()
    if q.empty:
        return q
    q["거래순번"] = range(1, len(q)+1)
    q["자산배수"] = (1.0 + q[ret_col].astype(float)/100.0).cumprod()
    q["누적수익률(%)"] = (q["자산배수"] - 1.0) * 100.0
    q["고점자산배수"] = q["자산배수"].cummax()
    q["Drawdown(%)"] = (q["자산배수"] / q["고점자산배수"] - 1.0) * 100.0
    return q

def entry_features(d, setup, entry):
    """진입 직전/당일에 알 수 있었던 정보만 추출한다."""
    entry_date = pd.Timestamp(entry["진입일"])
    matches = d.index[d.index.normalize() == entry_date.normalize()]
    if not len(matches):
        return {}
    entry_i = d.index.get_loc(matches[0])

    br_date = pd.Timestamp(setup["돌파일"])
    br_matches = d.index[d.index.normalize() == br_date.normalize()]
    if not len(br_matches):
        return {}
    breakout_i = d.index.get_loc(br_matches[0])

    e = d.iloc[entry_i]
    br = d.iloc[breakout_i]

    def safe_pct(a, b):
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return float("nan")
        return (float(a) / float(b) - 1) * 100.0

    br_high = float(br["High"])
    br_low = float(br["Low"])
    br_range = max(br_high - br_low, 1.0)

    return {
        "진입거래대금(억)": round(float(e["거래대금"]) / 1e8, 2) if pd.notna(e["거래대금"]) else None,
        "진입거래량배수20": round(float(e["Volume"] / e["AVG_VOL20"]), 2) if pd.notna(e["AVG_VOL20"]) and e["AVG_VOL20"] != 0 else None,
        "진입5일선이격(%)": round(safe_pct(e["Close"], e["MA5"]), 2),
        "진입20일선이격(%)": round(safe_pct(e["Close"], e["MA20"]), 2),
        "진입60일선이격(%)": round(safe_pct(e["Close"], e["MA60"]), 2),
        "20일선5일기울기(%)": round(float(e["MA20기울기5일(%)"]), 2) if pd.notna(e["MA20기울기5일(%)"]) else None,
        "전고점20이격(%)": round(safe_pct(e["Close"], e["전고점20"]), 2),
        "전고점60이격(%)": round(safe_pct(e["Close"], e["전고점60"]), 2),
        "돌파봉등락률(%)": round(float(br["등락률"]), 2) if pd.notna(br["등락률"]) else None,
        "돌파봉종가위치(%)": round((float(br["Close"]) - br_low) / br_range * 100.0, 2),
        "돌파봉거래대금(억)": round(float(br["거래대금"]) / 1e8, 2) if pd.notna(br["거래대금"]) else None,
        "돌파봉거래량배수20": round(float(br["Volume"] / br["AVG_VOL20"]), 2) if pd.notna(br["AVG_VOL20"]) and br["AVG_VOL20"] != 0 else None,
        "돌파거래량vs눌림(배)": setup.get("돌파거래량vs눌림(배)"),
        "돌파종가수익률(%)": setup.get("돌파종가수익률(%)"),
        "기준봉상승률(%)": setup.get("기준봉상승률(%)"),
        "기준봉거래량배수": setup.get("기준봉거래량배수"),
        "기준봉거래대금(억)": setup.get("거래대금(억)"),
        "눌림깊이(%)": setup.get("눌림깊이(%)"),
    }

def classify_trade(ret):
    if ret >= 10:
        return "대박(+10%↑)"
    if ret > 0:
        return "수익(0~10%)"
    return "손절/손실"

def standardized_group_compare(df, features):
    rows = []
    groups = ["대박(+10%↑)", "수익(0~10%)", "손절/손실"]
    for f in features:
        vals = {}
        for g in groups:
            q = df.loc[df["결과군"] == g, f].dropna()
            vals[g] = {"mean": q.mean() if len(q) else float("nan"), "median": q.median() if len(q) else float("nan"), "n": len(q)}
        all_std = df[f].std()
        jm = vals["대박(+10%↑)"]["mean"]
        lm = vals["손절/손실"]["mean"]
        effect = ((jm-lm)/all_std if pd.notna(jm) and pd.notna(lm) and pd.notna(all_std) and all_std != 0 else float("nan"))
        rows.append({
            "변수": f,
            "대박평균": None if pd.isna(vals["대박(+10%↑)"]["mean"]) else round(vals["대박(+10%↑)"]["mean"],2),
            "대박중앙": None if pd.isna(vals["대박(+10%↑)"]["median"]) else round(vals["대박(+10%↑)"]["median"],2),
            "수익평균": None if pd.isna(vals["수익(0~10%)"]["mean"]) else round(vals["수익(0~10%)"]["mean"],2),
            "수익중앙": None if pd.isna(vals["수익(0~10%)"]["median"]) else round(vals["수익(0~10%)"]["median"],2),
            "손실평균": None if pd.isna(vals["손절/손실"]["mean"]) else round(vals["손절/손실"]["mean"],2),
            "손실중앙": None if pd.isna(vals["손절/손실"]["median"]) else round(vals["손절/손실"]["median"],2),
            "대박-손실 표준화차이": None if pd.isna(effect) else round(effect,2),
            "대박N": vals["대박(+10%↑)"]["n"], "수익N": vals["수익(0~10%)"]["n"], "손실N": vals["손절/손실"]["n"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_abs"] = out["대박-손실 표준화차이"].abs()
        out = out.sort_values("_abs", ascending=False).drop(columns="_abs")
    return out

def quartile_feature_performance(df, feature):
    q = df[[feature, "순수익률(%)"]].dropna().copy()
    if q.empty:
        return pd.DataFrame()
    try:
        q["구간"] = pd.qcut(q[feature], 4, duplicates="drop")
    except Exception:
        q["구간"] = pd.cut(q[feature], 4, duplicates="drop")
    g = q.groupby("구간", observed=True).agg(
        신호=(feature, "size"),
        변수평균=(feature, "mean"),
        평균수익률=("순수익률(%)", "mean"),
        승률=("순수익률(%)", lambda s: (s > 0).mean()*100),
        대박률=("순수익률(%)", lambda s: (s >= 10).mean()*100),
    ).reset_index()
    g["구간"] = g["구간"].astype(str)
    for c in ["변수평균","평균수익률","승률","대박률"]:
        g[c] = g[c].round(2)
    return g

def winsorize_series(s, lower=0.01, upper=0.99):
    q = s.dropna()
    if q.empty:
        return s
    lo = q.quantile(lower)
    hi = q.quantile(upper)
    return s.clip(lo, hi)

def filter_rule_mask(df, rule_name):
    """v10.3에서 발견된 상위 4개 후보를 과도하게 세밀 조정하지 않고
    단순/해석 가능한 컷으로 검증한다.
    """
    if rule_name == "F1 돌파강도":
        # 돌파종가수익률이 강한 모멘텀 구간
        return df["돌파종가수익률(%)"] >= 5.0

    if rule_name == "F2 전고점20":
        # 최근 20일 전고점 근처 또는 돌파
        return df["전고점20이격(%)"] >= -2.0

    if rule_name == "F3 5일선이격":
        # 단기 모멘텀 유지
        return df["진입5일선이격(%)"] >= 5.0

    if rule_name == "F4 전고점60":
        # 60일 전고점에서 크게 멀지 않은 종목
        return df["전고점60이격(%)"] >= -5.0

    return pd.Series(True, index=df.index)

def evaluate_subset(df, ret_col="순수익률(%)"):
    if df.empty:
        return {
            "신호": 0, "승률(%)": 0.0, "평균수익률(%)": 0.0, "중앙수익률(%)": 0.0,
            "MDD(%)": 0.0, "최대연속손실": 0, "ProfitFactor": None,
            "누적복리수익률(%)": 0.0
        }
    return performance_stats(df, ret_col)

def build_filter_comparison(df):
    rules = ["F1 돌파강도", "F2 전고점20", "F3 5일선이격", "F4 전고점60"]
    rows = []

    base = evaluate_subset(df)
    rows.append({"필터": "기본전략", "구성": "없음", **base})

    for r in rules:
        q = df[filter_rule_mask(df, r)].copy()
        rows.append({"필터": r, "구성": r, **evaluate_subset(q)})

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            r1, r2 = rules[i], rules[j]
            mask = filter_rule_mask(df, r1) & filter_rule_mask(df, r2)
            q = df[mask].copy()
            rows.append({
                "필터": f"{r1}+{r2}",
                "구성": f"{r1} & {r2}",
                **evaluate_subset(q)
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["평균수익률(%)", "ProfitFactor", "MDD(%)", "신호"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        out.insert(0, "순위", range(1, len(out) + 1))
    return out

def yearly_filter_compare(df, filter_name):
    if filter_name == "기본전략":
        q = df.copy()
    elif "+" in filter_name:
        a, b = filter_name.split("+", 1)
        q = df[filter_rule_mask(df, a) & filter_rule_mask(df, b)].copy()
    else:
        q = df[filter_rule_mask(df, filter_name)].copy()

    q["연도"] = pd.to_datetime(q["진입일"]).dt.year
    rows = []
    for y, g in q.groupby("연도"):
        rows.append({"연도": int(y), **evaluate_subset(g)})
    return pd.DataFrame(rows).sort_values("연도") if rows else pd.DataFrame()

def holdout_filter_compare(df, filter_name, train_ratio=0.60):
    if filter_name == "기본전략":
        q = df.copy()
    elif "+" in filter_name:
        a, b = filter_name.split("+", 1)
        q = df[filter_rule_mask(df, a) & filter_rule_mask(df, b)].copy()
    else:
        q = df[filter_rule_mask(df, filter_name)].copy()

    train, test, split_date = chronological_oos(q, train_ratio)
    rows = []
    if split_date is not None:
        rows.append({"구간": "IS", **evaluate_subset(train)})
        rows.append({"구간": "OOS", **evaluate_subset(test)})
    return pd.DataFrame(rows), split_date

def choose_universe(listing, n):
    if n >= len(listing):
        return listing.copy()
    # 코드순 표본. 향후 v10에서 거래대금/시총 기반 표본으로 개선 가능.
    return listing.sort_values(["Market", "Code"]).head(n).copy()

ENTRY_MODE = "E1 돌파당일"
EXIT_MODE = "X3 +3%후트레일"

# v10.1에서 500종목·5년 표본의 1위였던 값을 고정한다.
FROZEN_ACTIVATION = 2.0
FROZEN_TRAIL = 2.0

with st.sidebar:
    st.header("v10.4 검증 설정")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    years = st.selectbox("검증 기간", [3, 4, 5], index=2, format_func=lambda x: f"{x}년")
    universe_n = st.selectbox("검증 종목 수", [300, 500, 700], index=1)

    st.divider()
    st.subheader("전략 조건 · 동결")
    st.write("E1 돌파당일")
    st.write("X3 활성 +2% / 트레일 2%")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("기준봉 최소 거래대금(억)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)
    breakout_vol_cut = st.number_input("돌파 거래량 / 눌림 거래량(배)", value=1.8, step=0.1)
    holding_days = st.number_input("최대 보유 거래일", 3, 15, 5)

    st.divider()
    st.subheader("실전 비용")
    buy_slip = st.number_input("매수 슬리피지(%)", value=0.10, step=0.05, format="%.2f")
    sell_slip = st.number_input("매도 슬리피지(%)", value=0.10, step=0.05, format="%.2f")
    fees_tax = st.number_input("왕복 수수료·세금 합계(%)", value=0.20, step=0.05, format="%.2f")

    st.divider()
    st.subheader("OOS")
    train_ratio = st.slider("앞 구간(IS) 비율(%)", 50, 80, 60, 5)

    run = st.button("▶ v10.4 진입필터 검증", type="primary", use_container_width=True)

st.info(
    "전략/청산/비용은 그대로 고정합니다. "
    "v10.3에서 구분력이 높았던 4개 진입특성만 단독 및 2개 조합으로 검증하고, "
    "OOS와 연도별 성과까지 비교합니다."
)

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(years=years)
    start_ts = cut - pd.Timedelta(days=90)
    fetch_end = end_ts + pd.Timedelta(days=45)

    p = {
        "base_rise": base_rise,
        "value_eok": value_eok,
        "volume_mult": volume_mult,
        "pullback_ratio": pullback_ratio / 100,
        "breakout_vol_cut": breakout_vol_cut,
        "retest_days": 3,
        "retest_touch_pct": 2.0,
        "holding_days": int(holding_days),
        "max_structural_risk": 7.0,
        "trail_pct": FROZEN_TRAIL,
        "activation_pct": FROZEN_ACTIVATION,
    }

    try:
        listing = stock_listing()
    except Exception as e:
        st.error(f"종목 목록 조회 실패: {type(e).__name__}")
        st.exception(e)
        st.stop()

    universe = choose_universe(listing, int(universe_n))
    total = len(universe)
    bar = st.progress(0)
    status = st.empty()
    setups, data_map, errors = [], {}, []

    for pos, (_, r) in enumerate(universe.iterrows(), 1):
        code = str(r["Code"]).zfill(6)
        name = r["Name"]
        market = r["Market"]
        status.write(f"1/2 데이터/신호 탐색 {pos}/{total} · {name}")

        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d")
            )
            if d is None or d.empty or len(d) < 80:
                continue

            found_rows, prepared = find_setups(
                d, code, name, market, p, cut, end_ts
            )
            data_map[code] = prepared
            if found_rows:
                setups.extend(found_rows)

        except Exception as e:
            errors.append(f"{name}({code}): {type(e).__name__}")

        bar.progress(pos / (total * 2))
        time.sleep(0.01)

    if not setups:
        bar.empty()
        status.warning("조건을 만족한 setup이 없습니다.")
        st.stop()

    setup_df = dedup_setups(pd.DataFrame(setups))

    rows = []
    for idx, (_, setup) in enumerate(setup_df.iterrows(), 1):
        code = str(setup["코드"]).zfill(6)
        d = data_map.get(code)
        if d is None:
            continue

        entry = make_entry(d, setup, ENTRY_MODE, p)
        if entry is None:
            continue

        ev = evaluate_trade(d, setup, entry, EXIT_MODE, p)
        if ev is None:
            continue

        row = setup.drop(
            labels=["_b", "_pull_i", "_breakout_i"],
            errors="ignore"
        ).to_dict()
        row.update(entry)
        row.pop("_entry_i", None)
        row.pop("_eval_i", None)
        row.update(ev)
        row.update(entry_features(d, setup, entry))
        rows.append(row)

        if len(setup_df):
            bar.progress(0.5 + idx / (len(setup_df) * 2))

    bar.empty()

    gross = pd.DataFrame(rows)
    if gross.empty:
        status.warning("실제 매매가 없습니다.")
        st.stop()

    t = apply_costs(gross, buy_slip, sell_slip, fees_tax)
    status.success(
        f"완료 · {total}종목 / {years}년 · setup {len(setup_df)}건 · 실제매매 {len(t)}건"
    )

    gross_stats = performance_stats(
        t.assign(**{"총수익률임시": t["최종수익률(%)"]}),
        "총수익률임시"
    )
    net_stats = performance_stats(t, "순수익률(%)")

    st.subheader("① 비용 차감 전/후")
    compare = pd.DataFrame([
        {"구분": "비용 차감 전", **gross_stats},
        {"구분": "비용 차감 후", **net_stats},
    ])
    st.dataframe(compare, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("순평균수익률", f'{net_stats["평균수익률(%)"]:.2f}%')
    c2.metric("순승률", f'{net_stats["승률(%)"]:.1f}%')
    c3.metric("MDD", f'{net_stats["MDD(%)"]:.2f}%')
    c4.metric("최대 연속손실", f'{net_stats["최대연속손실"]}회')

    total_cost = float(buy_slip + sell_slip + fees_tax)
    st.caption(
        f"적용 총 비용: 거래당 {total_cost:.2f}% "
        f"(매수 슬리피지 {buy_slip:.2f}% + 매도 슬리피지 {sell_slip:.2f}% + 수수료·세금 {fees_tax:.2f}%)."
    )

    st.subheader("② 시간순 OOS Holdout")
    train, test, split_date = chronological_oos(t, train_ratio / 100.0)

    if split_date is None:
        st.warning("OOS 분할에 필요한 표본이 부족합니다.")
    else:
        oos_table = pd.DataFrame([
            {"구간": f"IS 앞 {train_ratio}% · {len(train)}건", **performance_stats(train, "순수익률(%)")},
            {"구간": f"OOS 뒤 {100-train_ratio}% · {len(test)}건", **performance_stats(test, "순수익률(%)")},
        ])
        st.write(f"**OOS 시작일:** {split_date}")
        st.dataframe(oos_table, use_container_width=True, hide_index=True)

        oos_stats = performance_stats(test, "순수익률(%)")
        if oos_stats["평균수익률(%)"] > 0:
            st.success(f'OOS 평균수익률 +{oos_stats["평균수익률(%)"]:.2f}% · 플러스 유지')
        else:
            st.error(f'OOS 평균수익률 {oos_stats["평균수익률(%)"]:.2f}% · 전략 엣지 재검토 필요')

    st.subheader("③ 연도별 Walk-Forward")
    wf = yearly_walkforward(t, "순수익률(%)")
    st.caption(
        "각 행의 OOS연도 성과는 그 연도의 실제 거래만 사용합니다. "
        "전략 파라미터는 모든 연도에서 +2% 활성/2% 트레일로 고정되어 있습니다."
    )
    st.dataframe(wf, use_container_width=True, hide_index=True)

    st.subheader("④ MDD / 누적수익 진단")
    eq = equity_curve_table(t, "순수익률(%)")
    eq_show = eq[
        ["거래순번","진입일","종목명","코드","순수익률(%)","누적수익률(%)","Drawdown(%)"]
    ].copy()
    st.dataframe(
        eq_show.sort_values("거래순번", ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )
    st.caption(
        "누적복리/MDD는 모든 신호를 진입일 순서로 동일 비중 순차 체결한 진단값입니다. "
        "동시 보유·자금배분을 반영한 포트폴리오 시뮬레이션은 아닙니다."
    )

    st.subheader("⑤ 아웃라이어 제거 · 비용 차감 후")
    stress = []
    ordered = t.sort_values("순수익률(%)", ascending=False)
    for n in [0, 1, 3, 5, 10]:
        q = ordered.iloc[n:].copy() if len(ordered) > n else pd.DataFrame()
        if q.empty:
            continue
        stress.append({
            "최고수익 제거": "제거 없음" if n == 0 else f"상위 {n}건",
            **performance_stats(q, "순수익률(%)")
        })
    st.dataframe(pd.DataFrame(stress), use_container_width=True, hide_index=True)

    st.subheader("⑥ 연도별 순성과")
    t["연도"] = pd.to_datetime(t["진입일"]).dt.year
    yr = []
    for y, q in t.groupby("연도"):
        yr.append({"연도": int(y), **performance_stats(q, "순수익률(%)")})
    st.dataframe(
        pd.DataFrame(yr).sort_values("연도"),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("⑦ 종목별 의존도")
    stock = []
    for (code, name), q in t.groupby(["코드", "종목명"]):
        s = performance_stats(q, "순수익률(%)")
        stock.append({
            "코드": code,
            "종목명": name,
            **s,
            "순수익률합(%)": round(q["순수익률(%)"].sum(), 2),
        })
    stock_df = pd.DataFrame(stock).sort_values("순수익률합(%)", ascending=False)
    st.dataframe(stock_df, use_container_width=True, hide_index=True)

    st.subheader("⑧ 실제 거래")
    cols = [
        "시장","종목명","코드","기준봉일","눌림일","돌파일","진입일","진입가",
        "초기손절가","청산일","청산가","청산사유",
        "최종수익률(%)","총비용(%)","순수익률(%)",
        "MFE(%)","MAE(%)","+3%","+5%","+7%","+10%"
    ]
    st.dataframe(
        t[[c for c in cols if c in t.columns]].sort_values("진입일", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "v10.2 OOS/WFA 요약 CSV",
        wf.to_csv(index=False).encode("utf-8-sig"),
        "swing_v10_2_walkforward.csv",
        "text/csv",
        use_container_width=True
    )
    st.download_button(
        "v10.2 전체 거래 CSV",
        t.to_csv(index=False).encode("utf-8-sig"),
        "swing_v10_2_all_trades.csv",
        "text/csv",
        use_container_width=True
    )

    st.subheader("⑨ v10.3 · 진입특성 비교")
    t["결과군"] = t["순수익률(%)"].apply(classify_trade)
    features = [
        "진입거래대금(억)", "진입거래량배수20", "진입5일선이격(%)", "진입20일선이격(%)", "진입60일선이격(%)",
        "20일선5일기울기(%)", "전고점20이격(%)", "전고점60이격(%)", "돌파봉등락률(%)", "돌파봉종가위치(%)",
        "돌파봉거래대금(억)", "돌파봉거래량배수20", "돌파거래량vs눌림(배)", "돌파종가수익률(%)",
        "기준봉상승률(%)", "기준봉거래량배수", "기준봉거래대금(억)", "눌림깊이(%)"
    ]
    features = [f for f in features if f in t.columns]

    group_counts = t["결과군"].value_counts().reindex(["대박(+10%↑)", "수익(0~10%)", "손절/손실"]).fillna(0).astype(int).reset_index()
    group_counts.columns = ["결과군", "건수"]
    st.dataframe(group_counts, use_container_width=True, hide_index=True)

    comp = standardized_group_compare(t, features)
    st.caption("표준화차이의 절대값이 클수록 대박군과 손실군을 구분할 가능성이 큽니다. 양수면 대박군이 높고, 음수면 대박군이 낮습니다.")
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.subheader("⑩ 변수별 4구간 성과")
    selected_feature = st.selectbox("세부 분석 변수", features)
    band = quartile_feature_performance(t, selected_feature)
    st.dataframe(band, use_container_width=True, hide_index=True)

    st.subheader("⑪ 상위 구분 변수 자동 요약")
    top_vars = comp.dropna(subset=["대박-손실 표준화차이"]).head(5).copy()
    if top_vars.empty:
        st.warning("구분력이 계산된 변수가 없습니다.")
    else:
        st.dataframe(top_vars[["변수","대박평균","수익평균","손실평균","대박-손실 표준화차이"]], use_container_width=True, hide_index=True)

    st.subheader("⑫ 거래별 진입특성 원자료")
    raw_cols = ["진입일","시장","종목명","코드","결과군","순수익률(%)","MFE(%)","MAE(%)"] + features
    st.dataframe(t[[c for c in raw_cols if c in t.columns]].sort_values("순수익률(%)", ascending=False), use_container_width=True, hide_index=True)

    st.download_button("v10.3 진입특성 비교 CSV", comp.to_csv(index=False).encode("utf-8-sig"), "swing_v10_3_entry_feature_compare.csv", "text/csv", use_container_width=True)
    st.download_button("v10.3 거래별 진입특성 CSV", t.to_csv(index=False).encode("utf-8-sig"), "swing_v10_3_entry_feature_trades.csv", "text/csv", use_container_width=True)


    st.subheader("⑬ v10.4 · 진입필터 성과 비교")

    # 이상치 보정용 진단 컬럼: 필터 기준 자체에는 사용하지 않고 참고용으로만 제공
    if "돌파거래량vs눌림(배)" in t.columns:
        t["돌파거래량vs눌림_윈저"] = winsorize_series(
            pd.to_numeric(t["돌파거래량vs눌림(배)"], errors="coerce")
        )

    filter_comp = build_filter_comparison(t)
    st.caption(
        "필터 컷은 v10.3 결과를 바탕으로 단순하게 고정했습니다. "
        "F1=돌파종가수익률≥5%, F2=20일 전고점 이격≥-2%, "
        "F3=5일선 이격≥5%, F4=60일 전고점 이격≥-5%."
    )
    st.dataframe(filter_comp, use_container_width=True, hide_index=True)

    if not filter_comp.empty:
        top_filter = filter_comp.iloc[0]["필터"]
        st.subheader("⑭ 현재 1위 필터")
        top_row = filter_comp.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("신호", int(top_row["신호"]))
        c2.metric("승률", f'{top_row["승률(%)"]:.1f}%')
        c3.metric("평균수익률", f'{top_row["평균수익률(%)"]:.2f}%')
        pf_val = top_row["ProfitFactor"]
        c4.metric("Profit Factor", "-" if pd.isna(pf_val) else f"{pf_val:.2f}")

        st.write(f"**1위:** {top_filter}")

        st.subheader("⑮ 1위 필터 · 연도별 성과")
        yrf = yearly_filter_compare(t, top_filter)
        st.dataframe(yrf, use_container_width=True, hide_index=True)

        st.subheader("⑯ 1위 필터 · 시간순 OOS")
        hdf, split_date2 = holdout_filter_compare(t, top_filter, train_ratio / 100.0)
        if split_date2 is not None:
            st.write(f"**OOS 시작일:** {split_date2}")
            st.dataframe(hdf, use_container_width=True, hide_index=True)

        st.subheader("⑰ 상위 3개 필터 OOS 비교")
        oos_rows = []
        for _, rr in filter_comp.head(3).iterrows():
            fname = rr["필터"]
            h, sp = holdout_filter_compare(t, fname, train_ratio / 100.0)
            if not h.empty:
                oos = h[h["구간"] == "OOS"]
                if not oos.empty:
                    r = oos.iloc[0].to_dict()
                    oos_rows.append({
                        "필터": fname,
                        "OOS시작일": sp,
                        **{k: v for k, v in r.items() if k != "구간"}
                    })
        st.dataframe(pd.DataFrame(oos_rows), use_container_width=True, hide_index=True)

        st.subheader("⑱ 거래량비 이상치 보정 진단")
        if "돌파거래량vs눌림(배)" in t.columns:
            raw_s = pd.to_numeric(t["돌파거래량vs눌림(배)"], errors="coerce")
            win_s = pd.to_numeric(t["돌파거래량vs눌림_윈저"], errors="coerce")
            diag = pd.DataFrame([
                {
                    "구분": "원본",
                    "평균": round(raw_s.mean(), 2),
                    "중앙값": round(raw_s.median(), 2),
                    "95%": round(raw_s.quantile(0.95), 2),
                    "99%": round(raw_s.quantile(0.99), 2),
                    "최대": round(raw_s.max(), 2),
                },
                {
                    "구분": "1~99% 윈저",
                    "평균": round(win_s.mean(), 2),
                    "중앙값": round(win_s.median(), 2),
                    "95%": round(win_s.quantile(0.95), 2),
                    "99%": round(win_s.quantile(0.99), 2),
                    "최대": round(win_s.max(), 2),
                }
            ])
            st.dataframe(diag, use_container_width=True, hide_index=True)

        st.download_button(
            "v10.4 필터 비교 CSV",
            filter_comp.to_csv(index=False).encode("utf-8-sig"),
            "swing_v10_4_filter_compare.csv",
            "text/csv",
            use_container_width=True
        )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption(
    "v10.4는 진입필터 검증판입니다. "
    "청산/비용/기본전략은 고정하고, v10.3에서 발견한 4개 진입특성만 검증합니다. "
    "단독 및 2개 조합의 전체/OOS/연도별 성과를 비교해 과최적화 위험을 줄입니다."
)
