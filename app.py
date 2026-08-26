import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date
from io import BytesIO

st.set_page_config(page_title="단기스윙 v11.2 KOSPI 고승률 연구", layout="wide")
st.title("🎯 단기스윙 v11.2 · KOSPI 고승률 집중 분석")
st.caption("F1+F2 유지 · KOSPI 안에서 승률을 더 높이는 단일 필터 탐색 · KOSDAQ은 비교군으로 보존")

# ============================================================
# 완전 동결된 전략값
# ============================================================
TARGETS = [3.0, 5.0, 7.0, 10.0]

ENTRY_MODE = "E1 돌파당일"
EXIT_MODE = "X3"

BASE_RISE = 10.0
BASE_VALUE_EOK = 1000
BASE_VOL_MULT = 2.0
PULLBACK_RATIO = 0.50
BREAKOUT_VOL_CUT = 1.8
HOLDING_DAYS = 5

INITIAL_STOP_PCT = -3.0
ACTIVATION_PCT = 2.0
TRAIL_PCT = 2.0

# v10.4에서 확정한 필터. v10.5에서는 절대 최적화하지 않는다.
F1_BREAKOUT_CLOSE_RET = 5.0       # 돌파종가수익률 >= +5%
F2_PRIOR20_GAP = -2.0             # 20일 전고점 이격 >= -2%

# 실전비용도 동결
BUY_SLIP = 0.10
SELL_SLIP = 0.10
FEES_TAX = 0.20
TOTAL_COST = BUY_SLIP + SELL_SLIP + FEES_TAX

# 이전 v10.x에서 사용했던 universe 생성 방식과 동일해야
# 정확히 "기존 500종목"을 제외할 수 있다.
OLD_SAMPLE_SIZE = 500


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

        out["Code"] = out["Code"].astype(str).str.zfill(6)
        out["Name"] = out["Name"].astype(str)
        out["Market"] = market
        return out[["Code", "Name", "Market"]]

    return pd.concat(
        [normalize(kospi, "KOSPI"), normalize(kosdaq, "KOSDAQ")],
        ignore_index=True
    ).drop_duplicates(subset=["Code"]).reset_index(drop=True)


def old_universe_codes(listing):
    """v10.1~10.4의 기존 500종목 선택 방식 재현."""
    old = (
        listing.sort_values(["Market", "Code"])
        .head(OLD_SAMPLE_SIZE)
        .copy()
    )
    return set(old["Code"].astype(str))


def independent_balanced_universe(listing, total_n):
    """기존 500종목을 완전히 제외하고 KOSPI/KOSDAQ 균형 표본 생성."""
    old_codes = old_universe_codes(listing)
    remain = listing[~listing["Code"].isin(old_codes)].copy()

    half = total_n // 2
    kp = remain[remain["Market"] == "KOSPI"].sort_values("Code").head(half)
    kq = remain[remain["Market"] == "KOSDAQ"].sort_values("Code").head(half)

    # 한 시장의 남은 종목 수가 부족할 경우 다른 시장으로 보충
    picked = pd.concat([kp, kq], ignore_index=True)
    need = total_n - len(picked)

    if need > 0:
        picked_codes = set(picked["Code"])
        extra = (
            remain[~remain["Code"].isin(picked_codes)]
            .sort_values(["Market", "Code"])
            .head(need)
        )
        picked = pd.concat([picked, extra], ignore_index=True)

    return picked.head(total_n).reset_index(drop=True), old_codes


@st.cache_data(ttl=86400, show_spinner=False)
def load_data(code, start, end):
    return fdr.DataReader("NAVER:" + str(code).zfill(6), start, end)


@st.cache_data(ttl=86400, show_spinner=False)
def load_benchmark(symbol, start, end):
    # KS11: KOSPI, KQ11: KOSDAQ
    return fdr.DataReader(symbol, start, end)


def add_indicators(d):
    x = d.copy()
    x.index = pd.to_datetime(x.index)

    x["거래대금"] = x["Close"] * x["Volume"]
    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA60"] = x["Close"].rolling(60).mean()
    x["AVG_VOL20"] = x["Volume"].shift(1).rolling(20).mean()
    x["등락률"] = x["Close"].pct_change() * 100
    x["전고점20"] = x["High"].shift(1).rolling(20).max()
    x["전고점60"] = x["High"].shift(1).rolling(60).max()
    x["MA20기울기5일(%)"] = (x["MA20"] / x["MA20"].shift(5) - 1) * 100
    return x


def base_ok(d, b):
    x = d.iloc[b]

    return (
        pd.notna(x["AVG_VOL20"])
        and x["등락률"] >= BASE_RISE
        and x["거래대금"] >= BASE_VALUE_EOK * 1e8
        and x["Volume"] >= x["AVG_VOL20"] * BASE_VOL_MULT
        and x["Close"] > x["MA5"]
        and x["MA5"] > d["MA5"].iloc[b - 1]
        and x["MA20"] > d["MA20"].iloc[b - 1]
    )


def find_setups(d, code, name, market, cut, end):
    d = add_indicators(d)
    rows = []

    for b in range(60, len(d)):
        if d.index[b] < cut or d.index[b] > end:
            continue

        if not base_ok(d, b):
            continue

        x = d.iloc[b]

        # 기준봉 이후 2~5거래일 눌림
        for k in range(2, 6):
            pull_i = b + k
            breakout_i = pull_i + 1

            if breakout_i >= len(d):
                break

            pull = d.iloc[pull_i]
            br = d.iloc[breakout_i]

            pull_ok = (
                pull["Volume"] <= x["Volume"] * PULLBACK_RATIO
                and pull["Close"] > x["Low"]
            )

            if not pull_ok:
                continue

            breakout_level = float(pull["High"])

            if float(br["High"]) < breakout_level:
                continue

            breakout_vol_ratio = (
                float(br["Volume"]) /
                max(float(pull["Volume"]), 1.0)
            )

            if breakout_vol_ratio < BREAKOUT_VOL_CUT:
                continue

            breakout_close_ret = (
                float(br["Close"]) / breakout_level - 1
            ) * 100

            prior20 = br["전고점20"]

            if pd.isna(prior20) or float(prior20) == 0:
                continue

            prior20_gap = (
                float(br["Close"]) / float(prior20) - 1
            ) * 100

            rows.append({
                "시장": market,
                "종목명": name,
                "코드": str(code).zfill(6),

                "기준봉일": d.index[b].date(),
                "기준봉상승률(%)": round(float(x["등락률"]), 2),
                "기준봉거래대금(억)": round(float(x["거래대금"]) / 1e8, 0),
                "기준봉거래량배수": round(
                    float(x["Volume"] / x["AVG_VOL20"]), 2
                ),

                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(breakout_level),
                "눌림저가": round(float(pull["Low"])),
                "눌림종가": round(float(pull["Close"])),

                "돌파일": d.index[breakout_i].date(),
                "돌파시가": round(float(br["Open"])),
                "돌파고가": round(float(br["High"])),
                "돌파저가": round(float(br["Low"])),
                "돌파종가": round(float(br["Close"])),

                "돌파종가수익률(%)": round(
                    breakout_close_ret, 2
                ),
                "전고점20이격(%)": round(
                    prior20_gap, 2
                ),
                "돌파거래량vs눌림(배)": round(
                    breakout_vol_ratio, 2
                ),

                "_b": b,
                "_pull_i": pull_i,
                "_breakout_i": breakout_i,
            })

            break

    return rows, d



def setup_features(d, setup):
    """진입 당일까지 알 수 있는 정보만 사용."""
    bi = int(setup["_b"])
    pi = int(setup["_pull_i"])
    bri = int(setup["_breakout_i"])
    base = d.iloc[bi]
    pull = d.iloc[pi]
    br = d.iloc[bri]

    def pct(a, b):
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return float("nan")
        return (float(a) / float(b) - 1) * 100

    rng = max(float(br["High"]) - float(br["Low"]), 1.0)
    body_top = max(float(br["Open"]), float(br["Close"]))
    upper_wick = max(0.0, float(br["High"]) - body_top)

    return {
        "돌파봉등락률(%)": round(float(br["등락률"]), 2),
        "돌파봉종가위치(%)": round((float(br["Close"]) - float(br["Low"])) / rng * 100, 2),
        "돌파봉윗꼬리비율(%)": round(upper_wick / rng * 100, 2),
        "돌파갭(%)": round(pct(br["Open"], float(pull["Close"])), 2),
        "돌파거래대금(억)": round(float(br["거래대금"]) / 1e8, 2),
        "돌파거래량배수20": round(float(br["Volume"] / br["AVG_VOL20"]), 2) if pd.notna(br["AVG_VOL20"]) and br["AVG_VOL20"] != 0 else None,
        "진입5일선이격(%)": round(pct(br["Close"], br["MA5"]), 2),
        "진입20일선이격(%)": round(pct(br["Close"], br["MA20"]), 2),
        "진입60일선이격(%)": round(pct(br["Close"], br["MA60"]), 2),
        "20일선5일기울기(%)": round(float(br["MA20기울기5일(%)"]), 2) if pd.notna(br["MA20기울기5일(%)"]) else None,
        "전고점60이격(%)": round(pct(br["Close"], br["전고점60"]), 2),
        "눌림깊이(%)": round(pct(float(pull["Close"]), float(base["Close"])), 2),
        "눌림기간": int(pi - bi),
        "기준봉후유지율(%)": round(pct(float(pull["Low"]), float(base["Low"])), 2),
    }


def candidate_rules():
    """KOSPI 내부에서 단일 추가필터만 탐색한다."""
    return {
        "C1 돌파종가≥7%": lambda d: d["돌파종가수익률(%)"] >= 7.0,
        "C2 돌파종가≥10%": lambda d: d["돌파종가수익률(%)"] >= 10.0,
        "C3 돌파봉상승≥8%": lambda d: d["돌파봉등락률(%)"] >= 8.0,
        "C4 종가위치≥70%": lambda d: d["돌파봉종가위치(%)"] >= 70.0,
        "C5 종가위치50~85%": lambda d: d["돌파봉종가위치(%)"].between(50.0, 85.0),
        "C6 윗꼬리≤20%": lambda d: d["돌파봉윗꼬리비율(%)"] <= 20.0,
        "C7 갭≤3%": lambda d: d["돌파갭(%)"] <= 3.0,
        "C8 거래대금≥2000억": lambda d: d["돌파거래대금(억)"] >= 2000.0,
        "C9 거래량20≥2배": lambda d: d["돌파거래량배수20"] >= 2.0,
        "C10 5일이격≥7%": lambda d: d["진입5일선이격(%)"] >= 7.0,
        "C11 5일이격5~15%": lambda d: d["진입5일선이격(%)"].between(5.0, 15.0),
        "C12 MA20기울기>0": lambda d: d["20일선5일기울기(%)"] > 0.0,
        "C13 전고점60≥-2%": lambda d: d["전고점60이격(%)"] >= -2.0,
        "C14 눌림깊이≥-5%": lambda d: d["눌림깊이(%)"] >= -5.0,
        "C15 눌림깊이0~8%": lambda d: d["눌림깊이(%)"].between(0.0, 8.0),
        "C16 눌림기간≤3일": lambda d: d["눌림기간"] <= 3,
        "C17 기준봉저가유지": lambda d: d["기준봉후유지율(%)"] >= 0.0,
    }


def score_candidate(stats, min_trades):
    """승률 최우선. 표본/수익/PF/MDD가 나쁜 고승률 착시는 강하게 감점."""
    n = stats["신호"]
    wr = stats["승률(%)"]
    avg = stats["평균수익률(%)"]
    pf = stats["ProfitFactor"] if stats["ProfitFactor"] is not None else 0.0
    mdd = stats["MDD(%)"]

    if n < min_trades or avg <= 0 or pf < 1.5:
        return -9999.0

    sample_bonus = min(n, 80) * 0.03
    return round(wr * 1.0 + avg * 1.5 + min(pf, 8) * 1.5 + mdd * 0.15 + sample_bonus, 3)


def apply_named_rules(df, names, rules):
    mask = pd.Series(True, index=df.index)
    for name in names:
        mask &= rules[name](df).fillna(False)
    return df[mask].copy()


def search_high_winrate(df, min_trades=20):
    rules = candidate_rules()
    rows = []

    s = performance_stats(df)
    rows.append({
        "조건": "기준 F1+F2 KOSPI",
        "추가조건수": 0,
        **s,
        "점수": score_candidate(s, min_trades)
    })

    for name in rules:
        q = apply_named_rules(df, [name], rules)
        s = performance_stats(q)
        rows.append({
            "조건": name,
            "추가조건수": 1,
            **s,
            "점수": score_candidate(s, min_trades)
        })

    out = pd.DataFrame(rows).sort_values(
        ["점수", "승률(%)", "평균수익률(%)", "신호"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)
    out.insert(0, "순위", range(1, len(out)+1))
    return out, rules


def parse_condition_name(label, rules):
    if label in ["기준 F1+F2", "기준 F1+F2 KOSPI"]:
        return []
    return [x.strip() for x in label.split(" + ") if x.strip() in rules]


def fixed_time_split(df, ratio=0.60):
    """조건 선택 전에 원자료 자체를 시간순으로 나눔."""
    q = df.sort_values(["진입일", "코드"]).reset_index(drop=True)
    if len(q) < 20:
        return q, pd.DataFrame(), None
    split_i = max(1, min(len(q)-1, int(len(q)*ratio)))
    return q.iloc[:split_i].copy(), q.iloc[split_i:].copy(), q.iloc[split_i]["진입일"]


def frozen_filter(setup):
    """v10.4 1위 F1+F2. v10.5에서는 고정."""
    return (
        float(setup["돌파종가수익률(%)"]) >= F1_BREAKOUT_CLOSE_RET
        and float(setup["전고점20이격(%)"]) >= F2_PRIOR20_GAP
    )


def make_entry(d, setup):
    breakout_i = int(setup["_breakout_i"])
    level = float(setup["눌림고가"])

    return {
        "진입일": d.index[breakout_i].date(),
        "진입가": level,
        "_entry_i": breakout_i,
        "진입설명": "E1 눌림고가 돌파 당일 진입",
    }


def evaluate_trade(d, setup, entry):
    entry_i = int(entry["_entry_i"])
    entry_price = float(entry["진입가"])

    initial_stop = entry_price * (
        1 + INITIAL_STOP_PCT / 100
    )

    hit = {t: False for t in TARGETS}
    mfe = 0.0
    mae = 0.0

    last_i = min(
        len(d) - 1,
        entry_i + HOLDING_DAYS - 1
    )

    exit_i = None
    exit_price = None
    reason = None

    activated = False
    highest_before_today = entry_price

    for i in range(entry_i, last_i + 1):
        r = d.iloc[i]
        high = float(r["High"])
        low = float(r["Low"])

        # +2%가 전 거래일까지 활성화된 경우
        # 전일까지의 최고가만 사용하여 2% 트레일 계산
        if activated:
            trailing = highest_before_today * (
                1 - TRAIL_PCT / 100
            )
            active_stop = max(entry_price, trailing)
        else:
            active_stop = initial_stop

        high_ret = (
            high / entry_price - 1
        ) * 100
        low_ret = (
            low / entry_price - 1
        ) * 100

        # 같은 날 stop/target이 모두 보이면 stop 우선
        if low <= active_stop:
            mae = min(
                mae,
                (active_stop / entry_price - 1) * 100
            )

            # +2%를 장중 찍었다 하더라도 같은 날 stop이면
            # 일봉상 순서를 모르므로 활성화 이전 stop으로 간주
            if high < entry_price * (
                1 + ACTIVATION_PCT / 100
            ):
                mfe = max(
                    mfe,
                    max(0.0, high_ret)
                )

            exit_i = i
            exit_price = active_stop

            if activated and active_stop >= entry_price:
                reason = "본전/트레일"
            else:
                reason = "손절"

            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)

        for t in TARGETS:
            if high >= entry_price * (
                1 + t / 100
            ):
                hit[t] = True

        # 활성화는 그 거래일이 끝난 뒤부터 적용
        if high >= entry_price * (
            1 + ACTIVATION_PCT / 100
        ):
            activated = True

        highest_before_today = max(
            highest_before_today,
            high
        )

    if exit_i is None:
        exit_i = last_i
        exit_price = float(
            d.iloc[last_i]["Close"]
        )
        reason = "기간종료"

    gross_ret = (
        exit_price / entry_price - 1
    ) * 100

    net_ret = gross_ret - TOTAL_COST

    return {
        "초기손절가": round(initial_stop),
        "청산일": d.index[exit_i].date(),
        "청산가": round(exit_price),
        "청산사유": reason,

        "총수익률(%)": round(gross_ret, 2),
        "거래비용(%)": round(TOTAL_COST, 2),
        "순수익률(%)": round(net_ret, 2),

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
        df.sort_values(
            ["코드", "돌파일", "기준봉일"]
        )
        .drop_duplicates(
            subset=["코드", "돌파일"],
            keep="last"
        )
        .reset_index(drop=True)
    )


def performance_stats(df):
    if df.empty:
        return {
            "신호": 0,
            "승률(%)": 0.0,
            "평균수익률(%)": 0.0,
            "중앙수익률(%)": 0.0,
            "MDD(%)": 0.0,
            "최대연속손실": 0,
            "ProfitFactor": None,
            "누적복리수익률(%)": 0.0,
        }

    q = df.sort_values(
        ["진입일", "코드"]
    ).copy()

    r = q["순수익률(%)"].astype(float) / 100

    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (
        equity / peak - 1
    ) * 100

    streak = 0
    max_streak = 0

    for x in q["순수익률(%)"].astype(float):
        if x <= 0:
            streak += 1
            max_streak = max(
                max_streak,
                streak
            )
        else:
            streak = 0

    gains = q.loc[
        q["순수익률(%)"] > 0,
        "순수익률(%)"
    ].sum()

    losses = q.loc[
        q["순수익률(%)"] <= 0,
        "순수익률(%)"
    ].sum()

    pf = (
        gains / abs(losses)
        if losses < 0
        else None
    )

    return {
        "신호": len(q),
        "승률(%)": round(
            (q["순수익률(%)"] > 0).mean() * 100,
            1
        ),
        "평균수익률(%)": round(
            q["순수익률(%)"].mean(),
            2
        ),
        "중앙수익률(%)": round(
            q["순수익률(%)"].median(),
            2
        ),
        "MDD(%)": round(
            float(dd.min()),
            2
        ),
        "최대연속손실": int(
            max_streak
        ),
        "ProfitFactor": (
            None if pf is None
            else round(float(pf), 2)
        ),
        "누적복리수익률(%)": round(
            (equity.iloc[-1] - 1) * 100,
            2
        ),
    }


def yearly_stats(df):
    if df.empty:
        return pd.DataFrame()

    q = df.copy()
    q["연도"] = pd.to_datetime(
        q["진입일"]
    ).dt.year

    rows = []

    for y, g in q.groupby("연도"):
        rows.append({
            "연도": int(y),
            **performance_stats(g)
        })

    return pd.DataFrame(rows).sort_values(
        "연도"
    )


def market_stats(df):
    rows = []

    for market, g in df.groupby("시장"):
        rows.append({
            "시장": market,
            **performance_stats(g)
        })

    return pd.DataFrame(rows)


def chronological_holdout(df, train_ratio=0.60):
    q = df.sort_values(
        ["진입일", "코드"]
    ).reset_index(drop=True)

    if len(q) < 10:
        return pd.DataFrame(), None

    split_i = max(
        1,
        min(
            len(q) - 1,
            int(len(q) * train_ratio)
        )
    )

    train = q.iloc[:split_i]
    test = q.iloc[split_i:]

    table = pd.DataFrame([
        {
            "구간": "독립군 앞 60%",
            **performance_stats(train)
        },
        {
            "구간": "독립군 뒤 40%",
            **performance_stats(test)
        }
    ])

    return table, test.iloc[0]["진입일"]


def prepare_benchmark(df):
    x = df.copy()
    x.index = pd.to_datetime(x.index)
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA60"] = x["Close"].rolling(60).mean()

    conditions = []

    for _, r in x.iterrows():
        if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
            conditions.append(None)
        elif (
            r["Close"] > r["MA60"]
            and r["MA20"] > r["MA60"]
        ):
            conditions.append("상승장")
        elif (
            r["Close"] < r["MA60"]
            and r["MA20"] < r["MA60"]
        ):
            conditions.append("하락장")
        else:
            conditions.append("중립")

    x["시장국면"] = conditions

    return x[["시장국면"]]


def attach_market_regime(
    trades,
    start,
    end
):
    q = trades.copy()
    result = []

    symbols = {
        "KOSPI": "KS11",
        "KOSDAQ": "KQ11"
    }

    bench = {}

    for market, symbol in symbols.items():
        try:
            b = load_benchmark(
                symbol,
                start,
                end
            )
            if b is not None and not b.empty:
                bench[market] = prepare_benchmark(b)
        except Exception:
            pass

    for _, row in q.iterrows():
        market = row["시장"]
        d = pd.Timestamp(row["진입일"])

        regime = None

        if market in bench:
            b = bench[market]
            eligible = b[b.index <= d]

            if not eligible.empty:
                regime = eligible.iloc[-1][
                    "시장국면"
                ]

        r = row.to_dict()
        r["시장국면"] = (
            regime if regime
            else "국면데이터없음"
        )
        result.append(r)

    return pd.DataFrame(result)


def regime_stats(df):
    rows = []

    for regime, g in df.groupby("시장국면"):
        rows.append({
            "시장국면": regime,
            **performance_stats(g)
        })

    return pd.DataFrame(rows)


def stress_outliers(df):
    ordered = df.sort_values(
        "순수익률(%)",
        ascending=False
    )

    rows = []

    for n in [0, 1, 3, 5]:
        q = (
            ordered.iloc[n:].copy()
            if len(ordered) > n
            else pd.DataFrame()
        )

        if q.empty:
            continue

        rows.append({
            "최고수익 제거": (
                "제거 없음"
                if n == 0
                else f"상위 {n}건"
            ),
            **performance_stats(q)
        })

    return pd.DataFrame(rows)


def build_excel(sheets):
    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        for name, df in sheets.items():
            if df is None:
                continue

            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)

            safe_name = str(name)[:31]

            df.to_excel(
                writer,
                sheet_name=safe_name,
                index=False
            )

    output.seek(0)

    return output.getvalue()



def balanced_development_universe(listing, total_n):
    """KOSPI/KOSDAQ을 가능한 한 50:50으로 구성."""
    half = total_n // 2

    kp = (
        listing[listing["Market"] == "KOSPI"]
        .sort_values("Code")
        .head(half)
        .copy()
    )

    kq = (
        listing[listing["Market"] == "KOSDAQ"]
        .sort_values("Code")
        .head(half)
        .copy()
    )

    picked = pd.concat([kp, kq], ignore_index=True)

    # 홀수 또는 한 시장 종목 부족 시 남은 시장에서 보충
    need = total_n - len(picked)
    if need > 0:
        used = set(picked["Code"])
        extra = (
            listing[~listing["Code"].isin(used)]
            .sort_values(["Market", "Code"])
            .head(need)
            .copy()
        )
        picked = pd.concat([picked, extra], ignore_index=True)

    return picked.head(total_n).reset_index(drop=True)


# ============================================================
# UI
# ============================================================
with st.sidebar:
    st.header("v11.2 연구 설정")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    years = st.selectbox("개발 기간", [3, 4, 5], index=2, format_func=lambda x: f"{x}년")
    universe_n = st.selectbox("개발 종목 수", [300, 500], index=1)
    min_trades = st.number_input("후보 최소 거래수", 20, 80, 30, 5)
    train_ratio = st.slider("조건탐색 IS 비율(%)", 50, 75, 60, 5)
    run = st.button("▶ v11.2 KOSPI 조건 탐색", type="primary", use_container_width=True)

st.info(
    "v10.5와 v11.1 결과는 그대로 보존합니다. "
    "v11.2에서는 F1+F2 거래 중 KOSPI만 대상으로 승률을 더 높이는 단일 추가필터를 찾습니다. "
    "필터 선택은 KOSPI IS에서만 하고, KOSPI OOS에 그대로 적용합니다."
)

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(years=int(years))
    start_ts = cut - pd.Timedelta(days=100)
    fetch_end = end_ts + pd.Timedelta(days=45)

    try:
        listing = stock_listing()
    except Exception as e:
        st.error(f"종목 목록 조회 실패: {type(e).__name__}")
        st.exception(e)
        st.stop()

    # v11.1: KOSPI/KOSDAQ 균형 개발표본
    universe = balanced_development_universe(listing, int(universe_n))
    total = len(universe)

    kp_n = int((universe["Market"] == "KOSPI").sum())
    kq_n = int((universe["Market"] == "KOSDAQ").sum())

    st.write(
        f"**개발표본 구성:** {total}종목 "
        f"(KOSPI {kp_n} / KOSDAQ {kq_n})"
    )

    progress = st.progress(0)
    status = st.empty()
    setups, data_map, errors = [], {}, []

    for pos, (_, r) in enumerate(universe.iterrows(), 1):
        code0 = str(r["Code"]).zfill(6)
        name = r["Name"]
        market = r["Market"]
        status.info(f"1/2 개발표본 데이터/신호 {pos}/{total} · {market} {name}")

        try:
            d = load_data(code0, start_ts.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d"))
            if d is None or d.empty or len(d) < 100:
                continue
            found, prepared = find_setups(d, code0, name, market, cut, end_ts)
            data_map[code0] = prepared
            if found:
                setups.extend(found)
        except Exception as e:
            errors.append(f"{name}({code0}): {type(e).__name__}")

        progress.progress(pos / (total * 2))
        time.sleep(0.01)

    if not setups:
        progress.empty()
        status.warning("setup이 없습니다.")
        st.stop()

    setup_df = dedup_setups(pd.DataFrame(setups))
    selected = setup_df[setup_df.apply(frozen_filter, axis=1)].reset_index(drop=True)

    rows = []
    for idx, (_, setup) in enumerate(selected.iterrows(), 1):
        code0 = str(setup["코드"]).zfill(6)
        d = data_map.get(code0)
        if d is None:
            continue

        entry = make_entry(d, setup)
        ev = evaluate_trade(d, setup, entry)

        row = setup.drop(labels=["_b","_pull_i","_breakout_i"], errors="ignore").to_dict()
        row.update({k:v for k,v in entry.items() if not k.startswith("_")})
        row.update(ev)
        row.update(setup_features(d, setup))
        rows.append(row)

        if len(selected):
            progress.progress(0.5 + idx / (len(selected) * 2))

    progress.empty()
    trades = pd.DataFrame(rows)

    if trades.empty:
        status.warning("F1+F2 거래가 없습니다.")
        st.stop()

    trades = attach_market_regime(
        trades,
        start_ts.strftime("%Y-%m-%d"),
        fetch_end.strftime("%Y-%m-%d")
    )

    status.success(f"완료 · 개발군 {total}종목 · F1+F2 거래 {len(trades)}건")

    kospi_trades = trades[trades["시장"] == "KOSPI"].copy()
    kosdaq_trades = trades[trades["시장"] == "KOSDAQ"].copy()

    if len(kospi_trades) < 20:
        st.warning("KOSPI 거래 표본이 너무 적습니다.")
        st.stop()

    # KOSPI만 시간순 분할 후 IS에서 조건 선택
    is_df, oos_df, split_date = fixed_time_split(kospi_trades, train_ratio/100.0)
    if oos_df.empty:
        st.warning("KOSPI OOS 분할 표본이 부족합니다.")
        st.stop()

    st.subheader("① 기준 F1+F2 · 전체/KOSPI/KOSDAQ")
    base_all = performance_stats(trades)
    base_kp = performance_stats(kospi_trades)
    base_kq = performance_stats(kosdaq_trades)
    base_is = performance_stats(is_df)
    base_oos = performance_stats(oos_df)
    st.dataframe(pd.DataFrame([
        {"구간":"전체", **base_all},
        {"구간":"KOSPI 전체", **base_kp},
        {"구간":"KOSDAQ 전체", **base_kq},
        {"구간":"KOSPI IS", **base_is},
        {"구간":"KOSPI OOS", **base_oos},
    ]), use_container_width=True, hide_index=True)
    st.write(f"**KOSPI OOS 시작일:** {split_date}")

    st.subheader("② 기준전략 시장별 성과")
    market_base_rows = []
    for market in ["KOSPI", "KOSDAQ"]:
        q = trades[trades["시장"] == market].copy()
        if q.empty:
            continue
        market_base_rows.append({"시장": market, **performance_stats(q)})
    market_base_df = pd.DataFrame(market_base_rows)
    st.dataframe(market_base_df, use_container_width=True, hide_index=True)

    st.subheader("③ KOSPI IS에서 단일 고승률 필터 탐색")
    search_df, rules = search_high_winrate(is_df, int(min_trades))
    st.caption(
        "점수는 승률을 가장 크게 반영하되 평균수익률·PF·MDD·표본수를 함께 봅니다. "
        "평균수익률≤0 또는 PF<1.5 또는 최소 거래수 미달 후보는 자동 탈락 처리합니다."
    )
    st.dataframe(search_df.head(30), use_container_width=True, hide_index=True)

    valid = search_df[search_df["점수"] > -9000].copy()
    if valid.empty:
        st.error("IS 합격 후보가 없습니다. 조건을 억지로 완화하지 말고 현재 전략을 유지하는 것이 맞습니다.")
        st.stop()

    winner = valid.iloc[0]
    winner_name = winner["조건"]
    chosen_names = parse_condition_name(winner_name, rules)

    is_winner = apply_named_rules(is_df, chosen_names, rules)
    oos_winner = apply_named_rules(oos_df, chosen_names, rules)
    all_winner = apply_named_rules(kospi_trades, chosen_names, rules)

    st.subheader("④ KOSPI IS 1위 → KOSPI OOS 고정 검증")
    st.write(f"**선택 조건:** {winner_name}")

    compare = pd.DataFrame([
        {"구간":"IS", **performance_stats(is_winner)},
        {"구간":"OOS", **performance_stats(oos_winner)},
        {"구간":"전체 참고", **performance_stats(all_winner)},
    ])
    st.dataframe(compare, use_container_width=True, hide_index=True)

    oos_s = performance_stats(oos_winner)
    pass_oos = (
        oos_s["신호"] >= 10
        and oos_s["승률(%)"] >= 60.0
        and oos_s["평균수익률(%)"] >= 3.0
        and oos_s["ProfitFactor"] is not None
        and oos_s["ProfitFactor"] >= 2.0
    )

    if pass_oos:
        st.success(
            f'1차 목표 통과 · OOS 승률 {oos_s["승률(%)"]:.1f}% / '
            f'평균 {oos_s["평균수익률(%)"]:.2f}% / PF {oos_s["ProfitFactor"]:.2f}'
        )
    else:
        st.warning(
            f'OOS 결과: 승률 {oos_s["승률(%)"]:.1f}% / '
            f'평균 {oos_s["평균수익률(%)"]:.2f}% / '
            f'PF {("-" if oos_s["ProfitFactor"] is None else f"{oos_s["ProfitFactor"]:.2f}")}. '
            "60% 승률 목표를 충족하지 못하면 이 조건을 최종전략으로 채택하지 않습니다."
        )

    st.subheader("⑤ 상위 후보의 KOSPI OOS 실제 성과")
    oos_rows = []
    for _, rr in valid.head(10).iterrows():
        nm = rr["조건"]
        nms = parse_condition_name(nm, rules)
        q = apply_named_rules(oos_df, nms, rules)
        ss = performance_stats(q)
        oos_rows.append({
            "IS순위": int(rr["순위"]),
            "조건": nm,
            "IS신호": int(rr["신호"]),
            "IS승률(%)": rr["승률(%)"],
            "IS평균(%)": rr["평균수익률(%)"],
            "OOS신호": ss["신호"],
            "OOS승률(%)": ss["승률(%)"],
            "OOS평균(%)": ss["평균수익률(%)"],
            "OOS_PF": ss["ProfitFactor"],
            "OOS_MDD(%)": ss["MDD(%)"],
        })
    oos_compare = pd.DataFrame(oos_rows)
    st.dataframe(oos_compare, use_container_width=True, hide_index=True)

    st.subheader("⑥ 1위 조건 연도별")
    yr = yearly_stats(all_winner)
    st.dataframe(yr, use_container_width=True, hide_index=True)

    st.subheader("⑦ KOSPI 1위 조건 vs KOSDAQ 기준전략")
    mk = pd.DataFrame([
        {"구분":"KOSPI 1위 조건", **performance_stats(all_winner)},
        {"구분":"KOSDAQ F1+F2 기준", **performance_stats(kosdaq_trades)},
    ])
    st.dataframe(mk, use_container_width=True, hide_index=True)

    st.subheader("⑧ KOSPI 1위 조건 시장국면별")
    rg = regime_stats(all_winner)
    st.dataframe(rg, use_container_width=True, hide_index=True)

    st.subheader("⑨ 1위 조건 아웃라이어 제거")
    stress = stress_outliers(all_winner)
    st.dataframe(stress, use_container_width=True, hide_index=True)

    st.subheader("⑩ 승/패 진입특성")
    feature_cols = [
        "돌파종가수익률(%)","전고점20이격(%)","돌파봉등락률(%)",
        "돌파봉종가위치(%)","돌파봉윗꼬리비율(%)","돌파갭(%)",
        "돌파거래대금(억)","돌파거래량배수20","진입5일선이격(%)",
        "진입20일선이격(%)","진입60일선이격(%)","20일선5일기울기(%)",
        "전고점60이격(%)","눌림깊이(%)","눌림기간","기준봉후유지율(%)"
    ]
    feat_rows = []
    for f in feature_cols:
        if f not in trades.columns:
            continue
        win = trades.loc[trades["순수익률(%)"] > 0, f].dropna()
        loss = trades.loc[trades["순수익률(%)"] <= 0, f].dropna()
        feat_rows.append({
            "변수": f,
            "승리평균": round(win.mean(),2) if len(win) else None,
            "승리중앙": round(win.median(),2) if len(win) else None,
            "손실평균": round(loss.mean(),2) if len(loss) else None,
            "손실중앙": round(loss.median(),2) if len(loss) else None,
        })
    feat_df = pd.DataFrame(feat_rows)
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.subheader("⑪ 전체 연구결과 Excel")
    frozen_df = pd.DataFrame([
        {"항목":"기준전략","값":"v10.5 F1+F2"},
        {"항목":"목적함수","값":"승률 최우선 + 평균수익률/PF/MDD/표본수"},
        {"항목":"탐색범위","값":"KOSPI 단일 추가조건만"},
        {"항목":"OOS 원칙","값":"시간분할 후 IS에서만 조건 선택"},
        {"항목":"최소거래수","값":int(min_trades)},
        {"항목":"IS비율","값":f"{train_ratio}%"},
    ])
    excel_bytes = build_excel({
        "00_연구설정": frozen_df,
        "01_기준성과": pd.DataFrame([
            {"구간":"전체",**base_all},
            {"구간":"KOSPI전체",**base_kp},
            {"구간":"KOSDAQ전체",**base_kq},
            {"구간":"KOSPI_IS",**base_is},
            {"구간":"KOSPI_OOS",**base_oos}
        ]),
        "02_IS후보전체": search_df,
        "03_1위_IS_OOS": compare,
        "04_상위10_OOS": oos_compare,
        "05_기준시장별": market_base_df,
        "06_1위연도별": yr,
        "07_1위시장별": mk,
                "09_1위시장국면": rg,
        "10_1위아웃라이어": stress,
        "11_승패특성": feat_df,
        "12_전체F1F2거래": trades,
        "12A_KOSPI기준거래": kospi_trades,
        "12B_KOSDAQ기준거래": kosdaq_trades,
        "13_1위조건거래": all_winner,
    })
    st.download_button(
        "📦 v11.2 전체 연구결과 Excel 다운로드",
        data=excel_bytes,
        file_name="swing_v11_2_kospi_high_winrate_research.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    if errors:
        with st.expander(f"조회 실패/데이터 부족 {len(errors)}건"):
            st.write(errors)

st.caption(
    "v11.2는 KOSPI 고승률 집중 연구 브랜치입니다. v10.5/v11.1 결과는 변경하지 않습니다. "
    "여기서 발견된 조건도 곧바로 실전 채택하지 않고, 다음 단계에서 다시 완전히 독립된 종목군으로 검증해야 합니다."
)
