import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date
from io import BytesIO

st.set_page_config(page_title="단기스윙 v14.2 최종 독립검증", layout="wide")
st.title("🧪 단기스윙 v14.2 · A+B 복합형 최종 독립검증")
st.caption("A+B 복합형 조건 완전 동결 · 미사용 KOSPI + KOSDAQ 별도 독립검증 · 조건 재조정 금지")

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



def feature_catalog():
    return [
        "돌파종가수익률(%)",
        "전고점20이격(%)",
        "돌파봉등락률(%)",
        "돌파봉종가위치(%)",
        "돌파봉윗꼬리비율(%)",
        "돌파갭(%)",
        "돌파거래대금(억)",
        "돌파거래량배수20",
        "진입5일선이격(%)",
        "진입20일선이격(%)",
        "진입60일선이격(%)",
        "20일선5일기울기(%)",
        "전고점60이격(%)",
        "눌림깊이(%)",
        "눌림기간",
        "기준봉후유지율(%)",
    ]


def win_loss_distribution(df):
    """승리/손절 분포 요약."""
    rows = []
    for f in feature_catalog():
        if f not in df.columns:
            continue

        win = pd.to_numeric(
            df.loc[df["순수익률(%)"] > 0, f],
            errors="coerce"
        ).dropna()

        loss = pd.to_numeric(
            df.loc[df["순수익률(%)"] <= 0, f],
            errors="coerce"
        ).dropna()

        allv = pd.to_numeric(df[f], errors="coerce").dropna()
        std = allv.std()

        effect = None
        if len(win) and len(loss) and pd.notna(std) and std != 0:
            effect = (win.mean() - loss.mean()) / std

        rows.append({
            "변수": f,
            "승리N": len(win),
            "손실N": len(loss),
            "승리평균": round(win.mean(), 2) if len(win) else None,
            "승리중앙": round(win.median(), 2) if len(win) else None,
            "손실평균": round(loss.mean(), 2) if len(loss) else None,
            "손실중앙": round(loss.median(), 2) if len(loss) else None,
            "표준화차이": round(effect, 2) if effect is not None else None,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_abs"] = out["표준화차이"].abs()
        out = out.sort_values("_abs", ascending=False).drop(columns="_abs")
    return out


def candidate_intervals_from_quantiles(s):
    """데이터 분포에서 자동으로 단측/양측 구간 후보 생성."""
    q = pd.to_numeric(s, errors="coerce").dropna()
    if len(q) < 12:
        return []

    quantiles = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    vals = {p: float(q.quantile(p)) for p in quantiles}

    candidates = []

    # lower bound candidates: x >= q
    for p in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        candidates.append({
            "형태": "이상",
            "하한": vals[p],
            "상한": None,
            "설명": f">= Q{int(p*100)} ({vals[p]:.2f})"
        })

    # upper bound candidates: x <= q
    for p in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        candidates.append({
            "형태": "이하",
            "하한": None,
            "상한": vals[p],
            "설명": f"<= Q{int(p*100)} ({vals[p]:.2f})"
        })

    # central / upper bands
    bands = [
        (0.20, 0.60),
        (0.30, 0.70),
        (0.40, 0.80),
        (0.50, 0.80),
        (0.30, 0.80),
    ]
    for a, b in bands:
        candidates.append({
            "형태": "구간",
            "하한": vals[a],
            "상한": vals[b],
            "설명": f"Q{int(a*100)}~Q{int(b*100)} ({vals[a]:.2f}~{vals[b]:.2f})"
        })

    return candidates


def interval_mask(df, feature, cand):
    s = pd.to_numeric(df[feature], errors="coerce")

    if cand["형태"] == "이상":
        return s >= cand["하한"]

    if cand["형태"] == "이하":
        return s <= cand["상한"]

    return (s >= cand["하한"]) & (s <= cand["상한"])


def interval_search_is(df, min_trades=12):
    """IS 데이터 자체의 quantile로 후보를 만든다."""
    rows = []

    base = performance_stats(df)
    rows.append({
        "변수": "기준전략",
        "조건": "없음",
        "형태": "없음",
        "하한": None,
        "상한": None,
        **base,
    })

    for f in feature_catalog():
        if f not in df.columns:
            continue

        cands = candidate_intervals_from_quantiles(df[f])

        for cand in cands:
            mask = interval_mask(df, f, cand).fillna(False)
            q = df[mask].copy()

            if len(q) < min_trades:
                continue

            s = performance_stats(q)

            # 승률 우선, 단 평균/PF가 무너지면 제외
            if s["평균수익률(%)"] <= 0:
                continue
            if s["ProfitFactor"] is None or s["ProfitFactor"] < 1.5:
                continue

            rows.append({
                "변수": f,
                "조건": cand["설명"],
                "형태": cand["형태"],
                "하한": cand["하한"],
                "상한": cand["상한"],
                **s,
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["승률(%)", "평균수익률(%)", "ProfitFactor", "신호"],
            ascending=[False, False, False, False]
        ).reset_index(drop=True)
        out.insert(0, "순위", range(1, len(out) + 1))

    return out


def apply_interval_rule(df, row):
    if row["변수"] == "기준전략":
        return df.copy()

    feature = row["변수"]
    cand = {
        "형태": row["형태"],
        "하한": row["하한"],
        "상한": row["상한"],
    }
    return df[interval_mask(df, feature, cand).fillna(False)].copy()


def evaluate_top_is_rules_on_oos(is_ranked, oos_df, top_n=10):
    rows = []

    if is_ranked.empty:
        return pd.DataFrame()

    for _, r in is_ranked.head(top_n).iterrows():
        q = apply_interval_rule(oos_df, r)
        s = performance_stats(q)

        rows.append({
            "IS순위": int(r["순위"]),
            "변수": r["변수"],
            "조건": r["조건"],
            "IS신호": int(r["신호"]),
            "IS승률(%)": r["승률(%)"],
            "IS평균(%)": r["평균수익률(%)"],
            "IS_PF": r["ProfitFactor"],
            "OOS신호": s["신호"],
            "OOS승률(%)": s["승률(%)"],
            "OOS평균(%)": s["평균수익률(%)"],
            "OOS_PF": s["ProfitFactor"],
            "OOS_MDD(%)": s["MDD(%)"],
        })

    return pd.DataFrame(rows)


def market_research(market_df, train_ratio, min_is_trades):
    is_df, oos_df, split_date = fixed_time_split(
        market_df,
        train_ratio / 100.0
    )

    dist = win_loss_distribution(market_df)

    if oos_df.empty:
        return {
            "is": is_df,
            "oos": oos_df,
            "split_date": split_date,
            "dist": dist,
            "ranked": pd.DataFrame(),
            "oos_top": pd.DataFrame(),
        }

    ranked = interval_search_is(
        is_df,
        min_trades=min_is_trades
    )

    oos_top = evaluate_top_is_rules_on_oos(
        ranked,
        oos_df,
        top_n=10
    )

    return {
        "is": is_df,
        "oos": oos_df,
        "split_date": split_date,
        "dist": dist,
        "ranked": ranked,
        "oos_top": oos_top,
    }



V114_PULLBACK_DEPTH_CUT = -3.77
V113_USED_KOSPI_N = 250

def independent_kospi_universe(listing, n):
    kp = listing[listing["Market"]=="KOSPI"].sort_values("Code").reset_index(drop=True)
    used = set(kp.head(V113_USED_KOSPI_N)["Code"].astype(str))
    remain = kp[~kp["Code"].astype(str).isin(used)].copy()
    picked = remain.head(int(n)).reset_index(drop=True)
    overlap = set(picked["Code"].astype(str)) & used
    return picked, used, overlap

def fixed_v114_filter(df):
    return pd.to_numeric(df["눌림깊이(%)"], errors="coerce") >= V114_PULLBACK_DEPTH_CUT

def compare_baseline_vs_v114(df):
    return pd.DataFrame([
        {"전략":"KOSPI F1+F2 기준", **performance_stats(df)},
        {"전략":f"F1+F2 + 눌림깊이≥{V114_PULLBACK_DEPTH_CUT}%",
         **performance_stats(df[fixed_v114_filter(df)].copy())},
    ])

def time_split_compare(df, ratio=0.60):
    train, test, split_date = fixed_time_split(df, ratio)
    rows=[]
    for label, part in [("앞 60%",train),("뒤 40%",test)]:
        if part.empty: continue
        rows.append({"전략":"기준","구간":label,**performance_stats(part)})
        rows.append({"전략":"v11.4 필터","구간":label,
                     **performance_stats(part[fixed_v114_filter(part)].copy())})
    return pd.DataFrame(rows), split_date

def yearly_compare(df):
    q=df.copy()
    q["연도"]=pd.to_datetime(q["진입일"]).dt.year
    rows=[]
    for y,g in q.groupby("연도"):
        rows.append({"연도":int(y),"전략":"기준",**performance_stats(g)})
        gf=g[fixed_v114_filter(g)].copy()
        if not gf.empty:
            rows.append({"연도":int(y),"전략":"v11.4 필터",**performance_stats(gf)})
    return pd.DataFrame(rows)

def stress_compare(df):
    rows=[]
    for label,q in [("기준",df.copy()),("v11.4 필터",df[fixed_v114_filter(df)].copy())]:
        ordered=q.sort_values("순수익률(%)",ascending=False)
        for n in [0,1,3,5]:
            z=ordered.iloc[n:].copy() if len(ordered)>n else pd.DataFrame()
            if z.empty: continue
            rows.append({"전략":label,"최고수익 제거":"제거 없음" if n==0 else f"상위 {n}건",
                         **performance_stats(z)})
    return pd.DataFrame(rows)



# ============================================================
# v11.5 market environment research
# ============================================================
def prepare_market_features(df):
    x = df.copy()
    x.index = pd.to_datetime(x.index)
    x["RET1"] = x["Close"].pct_change() * 100
    x["MA5"] = x["Close"].rolling(5).mean()
    x["MA20"] = x["Close"].rolling(20).mean()
    x["MA60"] = x["Close"].rolling(60).mean()
    x["MA120"] = x["Close"].rolling(120).mean()

    x["지수5일수익률(%)"] = (x["Close"] / x["Close"].shift(5) - 1) * 100
    x["지수20일수익률(%)"] = (x["Close"] / x["Close"].shift(20) - 1) * 100
    x["지수60일수익률(%)"] = (x["Close"] / x["Close"].shift(60) - 1) * 100

    x["지수MA20이격(%)"] = (x["Close"] / x["MA20"] - 1) * 100
    x["지수MA60이격(%)"] = (x["Close"] / x["MA60"] - 1) * 100
    x["지수MA120이격(%)"] = (x["Close"] / x["MA120"] - 1) * 100

    x["MA20_5일기울기(%)"] = (x["MA20"] / x["MA20"].shift(5) - 1) * 100
    x["MA60_10일기울기(%)"] = (x["MA60"] / x["MA60"].shift(10) - 1) * 100

    x["고점60"] = x["High"].shift(1).rolling(60).max()
    x["고점120"] = x["High"].shift(1).rolling(120).max()
    x["지수60일고점이격(%)"] = (x["Close"] / x["고점60"] - 1) * 100
    x["지수120일고점이격(%)"] = (x["Close"] / x["고점120"] - 1) * 100

    # 최근 20일 일간수익률 표준편차. 단위 %
    x["지수20일변동성(%)"] = x["RET1"].rolling(20).std()

    # 단순 추세 구조
    x["시장배열"] = "혼조"
    bull = (x["Close"] > x["MA20"]) & (x["MA20"] > x["MA60"]) & (x["MA60"] > x["MA120"])
    bear = (x["Close"] < x["MA20"]) & (x["MA20"] < x["MA60"]) & (x["MA60"] < x["MA120"])
    x.loc[bull, "시장배열"] = "정배열"
    x.loc[bear, "시장배열"] = "역배열"

    return x


def attach_kospi_market_features(trades, market_df):
    q = trades.copy()
    m = prepare_market_features(market_df)

    feature_cols = [
        "지수5일수익률(%)",
        "지수20일수익률(%)",
        "지수60일수익률(%)",
        "지수MA20이격(%)",
        "지수MA60이격(%)",
        "지수MA120이격(%)",
        "MA20_5일기울기(%)",
        "MA60_10일기울기(%)",
        "지수60일고점이격(%)",
        "지수120일고점이격(%)",
        "지수20일변동성(%)",
        "시장배열",
    ]

    rows = []
    for _, r in q.iterrows():
        d = pd.Timestamp(r["진입일"])
        avail = m[m.index <= d]
        row = r.to_dict()

        if avail.empty:
            for c in feature_cols:
                row[c] = None
        else:
            last = avail.iloc[-1]
            for c in feature_cols:
                v = last[c]
                if c == "시장배열":
                    row[c] = v
                else:
                    row[c] = None if pd.isna(v) else round(float(v), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def market_feature_catalog():
    return [
        "지수5일수익률(%)",
        "지수20일수익률(%)",
        "지수60일수익률(%)",
        "지수MA20이격(%)",
        "지수MA60이격(%)",
        "지수MA120이격(%)",
        "MA20_5일기울기(%)",
        "MA60_10일기울기(%)",
        "지수60일고점이격(%)",
        "지수120일고점이격(%)",
        "지수20일변동성(%)",
    ]


def market_win_loss_compare(df):
    rows = []
    for f in market_feature_catalog():
        win = pd.to_numeric(df.loc[df["순수익률(%)"] > 0, f], errors="coerce").dropna()
        loss = pd.to_numeric(df.loc[df["순수익률(%)"] <= 0, f], errors="coerce").dropna()
        allv = pd.to_numeric(df[f], errors="coerce").dropna()
        std = allv.std()

        effect = None
        if len(win) and len(loss) and pd.notna(std) and std != 0:
            effect = (win.mean() - loss.mean()) / std

        rows.append({
            "변수": f,
            "승리평균": round(win.mean(), 3) if len(win) else None,
            "승리중앙": round(win.median(), 3) if len(win) else None,
            "손실평균": round(loss.mean(), 3) if len(loss) else None,
            "손실중앙": round(loss.median(), 3) if len(loss) else None,
            "표준화차이": round(effect, 3) if effect is not None else None,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_abs"] = out["표준화차이"].abs()
        out = out.sort_values("_abs", ascending=False).drop(columns="_abs")
    return out


def market_quantile_candidates(s):
    q = pd.to_numeric(s, errors="coerce").dropna()
    if len(q) < 12:
        return []

    ps = [0.20,0.30,0.40,0.50,0.60,0.70,0.80]
    vals = {p: float(q.quantile(p)) for p in ps}
    cands = []

    for p in [0.20,0.30,0.40,0.50,0.60,0.70]:
        cands.append({
            "형태":"이상","하한":vals[p],"상한":None,
            "조건":f">=Q{int(p*100)} ({vals[p]:.3f})"
        })

    for p in [0.30,0.40,0.50,0.60,0.70,0.80]:
        cands.append({
            "형태":"이하","하한":None,"상한":vals[p],
            "조건":f"<=Q{int(p*100)} ({vals[p]:.3f})"
        })

    for a,b in [(0.20,0.60),(0.30,0.70),(0.40,0.80),(0.50,0.80),(0.30,0.80)]:
        cands.append({
            "형태":"구간","하한":vals[a],"상한":vals[b],
            "조건":f"Q{int(a*100)}~Q{int(b*100)} ({vals[a]:.3f}~{vals[b]:.3f})"
        })

    return cands


def market_rule_mask(df, feature, shape, lower, upper):
    s = pd.to_numeric(df[feature], errors="coerce")
    if shape == "이상":
        return s >= float(lower)
    if shape == "이하":
        return s <= float(upper)
    return (s >= float(lower)) & (s <= float(upper))


def search_market_rules_is(df, min_trades=15):
    rows = []

    # baseline
    bs = performance_stats(df)
    rows.append({
        "변수":"기준전략","조건":"없음","형태":"없음","하한":None,"상한":None,
        **bs
    })

    for f in market_feature_catalog():
        if f not in df.columns:
            continue

        for cand in market_quantile_candidates(df[f]):
            mask = market_rule_mask(
                df, f, cand["형태"], cand["하한"], cand["상한"]
            ).fillna(False)

            q = df[mask].copy()
            if len(q) < int(min_trades):
                continue

            s = performance_stats(q)
            if s["평균수익률(%)"] <= 0:
                continue
            if s["ProfitFactor"] is None or s["ProfitFactor"] < 1.5:
                continue

            rows.append({
                "변수":f,
                "조건":cand["조건"],
                "형태":cand["형태"],
                "하한":cand["하한"],
                "상한":cand["상한"],
                **s
            })

    # categorical alignment candidates
    for cat in ["정배열","혼조","역배열"]:
        q = df[df["시장배열"] == cat].copy()
        if len(q) >= int(min_trades):
            s = performance_stats(q)
            if s["평균수익률(%)"] > 0 and s["ProfitFactor"] is not None and s["ProfitFactor"] >= 1.5:
                rows.append({
                    "변수":"시장배열","조건":cat,"형태":"범주","하한":None,"상한":None,
                    **s
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["승률(%)","평균수익률(%)","ProfitFactor","신호"],
            ascending=[False,False,False,False]
        ).reset_index(drop=True)
        out.insert(0,"순위",range(1,len(out)+1))
    return out


def apply_market_rule(df, row):
    if row["변수"] == "기준전략":
        return df.copy()

    if row["변수"] == "시장배열":
        return df[df["시장배열"] == row["조건"]].copy()

    mask = market_rule_mask(
        df,
        row["변수"],
        row["형태"],
        row["하한"],
        row["상한"]
    ).fillna(False)
    return df[mask].copy()


def evaluate_market_top_oos(ranked, oos_df, top_n=12):
    rows=[]
    if ranked.empty:
        return pd.DataFrame()

    for _,r in ranked.head(top_n).iterrows():
        q = apply_market_rule(oos_df, r)
        s = performance_stats(q)
        rows.append({
            "IS순위":int(r["순위"]),
            "변수":r["변수"],
            "조건":r["조건"],
            "IS신호":int(r["신호"]),
            "IS승률(%)":r["승률(%)"],
            "IS평균(%)":r["평균수익률(%)"],
            "IS_PF":r["ProfitFactor"],
            "OOS신호":s["신호"],
            "OOS승률(%)":s["승률(%)"],
            "OOS평균(%)":s["평균수익률(%)"],
            "OOS_PF":s["ProfitFactor"],
            "OOS_MDD(%)":s["MDD(%)"],
        })
    return pd.DataFrame(rows)


# ============================================================
# v11.6 frozen final hypothesis
# ============================================================
V116_MA60_SLOPE_CUT = 0.328
V115_USED_KOSPI_START = 250
V115_USED_KOSPI_N = 300

def v116_independent_kospi_universe(listing, n):
    """v11.5에서 사용한 KOSPI 구간(개발 250 제외 후 다음 300)을 제외한 그 이후 종목."""
    kp = (
        listing[listing["Market"] == "KOSPI"]
        .sort_values("Code")
        .reset_index(drop=True)
    )

    # v11.3 개발 250 + v11.5 연구 300 = 앞 550개를 모두 제외
    start = V115_USED_KOSPI_START + V115_USED_KOSPI_N
    picked = kp.iloc[start:start + int(n)].copy().reset_index(drop=True)

    used = set(kp.iloc[:start]["Code"].astype(str))
    overlap = set(picked["Code"].astype(str)) & used

    return picked, used, overlap


def fixed_v116_market_filter(df):
    return (
        pd.to_numeric(
            df["MA60_10일기울기(%)"],
            errors="coerce"
        ) >= V116_MA60_SLOPE_CUT
    )


def compare_v116(df):
    base = performance_stats(df)
    filt = performance_stats(
        df[fixed_v116_market_filter(df)].copy()
    )

    return pd.DataFrame([
        {"전략":"KOSPI F1+F2 기준", **base},
        {
            "전략":f"F1+F2 + MA60 10일기울기≥{V116_MA60_SLOPE_CUT}%",
            **filt
        },
    ])


def v116_time_split_compare(df, ratio=0.60):
    train, test, split_date = fixed_time_split(df, ratio)
    rows = []

    for label, part in [("앞 60%", train), ("뒤 40%", test)]:
        if part.empty:
            continue

        rows.append({
            "전략":"기준",
            "구간":label,
            **performance_stats(part)
        })

        fp = part[fixed_v116_market_filter(part)].copy()

        rows.append({
            "전략":"v11.6 필터",
            "구간":label,
            **performance_stats(fp)
        })

    return pd.DataFrame(rows), split_date


def v116_yearly_compare(df):
    q = df.copy()
    q["연도"] = pd.to_datetime(q["진입일"]).dt.year

    rows = []
    for y, g in q.groupby("연도"):
        rows.append({
            "연도":int(y),
            "전략":"기준",
            **performance_stats(g)
        })

        gf = g[fixed_v116_market_filter(g)].copy()
        if not gf.empty:
            rows.append({
                "연도":int(y),
                "전략":"v11.6 필터",
                **performance_stats(gf)
            })

    return pd.DataFrame(rows)


def v116_stress_compare(df):
    rows = []
    for label, q in [
        ("기준", df.copy()),
        ("v11.6 필터", df[fixed_v116_market_filter(df)].copy())
    ]:
        ordered = q.sort_values("순수익률(%)", ascending=False)

        for n in [0,1,3,5]:
            z = ordered.iloc[n:].copy() if len(ordered) > n else pd.DataFrame()
            if z.empty:
                continue

            rows.append({
                "전략":label,
                "최고수익 제거":"제거 없음" if n == 0 else f"상위 {n}건",
                **performance_stats(z)
            })

    return pd.DataFrame(rows)



# ============================================================
# v12 score model
# ============================================================
import numpy as np

V12_DEV_KOSPI_N = 550       # v11.3 + v11.5에서 이미 연구에 사용한 구간
V12_PRIOR_USED_N = 750      # v11.6까지 사용한 KOSPI 범위
V12_DEFAULT_FINAL_N = 150   # 완전히 새로운 최종 검증군

V12_FEATURES = [
    "돌파종가수익률(%)",
    "전고점20이격(%)",
    "돌파봉등락률(%)",
    "돌파봉종가위치(%)",
    "돌파봉윗꼬리비율(%)",
    "돌파갭(%)",
    "돌파거래대금(억)",
    "돌파거래량배수20",
    "진입5일선이격(%)",
    "진입20일선이격(%)",
    "진입60일선이격(%)",
    "20일선5일기울기(%)",
    "전고점60이격(%)",
    "눌림깊이(%)",
    "눌림기간",
    "기준봉후유지율(%)",
    "지수5일수익률(%)",
    "지수20일수익률(%)",
    "지수60일수익률(%)",
    "지수MA20이격(%)",
    "지수MA60이격(%)",
    "지수MA120이격(%)",
    "MA20_5일기울기(%)",
    "MA60_10일기울기(%)",
    "지수60일고점이격(%)",
    "지수120일고점이격(%)",
    "지수20일변동성(%)",
]

def v12_dev_universe(listing):
    kp = (
        listing[listing["Market"] == "KOSPI"]
        .sort_values("Code")
        .reset_index(drop=True)
    )
    return kp.head(V12_DEV_KOSPI_N).copy().reset_index(drop=True)

def v12_final_universe(listing, n):
    """v11.6까지 사용한 앞 750개를 전부 제외한 이후 종목."""
    kp = (
        listing[listing["Market"] == "KOSPI"]
        .sort_values("Code")
        .reset_index(drop=True)
    )
    picked = kp.iloc[V12_PRIOR_USED_N:V12_PRIOR_USED_N + int(n)].copy().reset_index(drop=True)
    used = set(kp.iloc[:V12_PRIOR_USED_N]["Code"].astype(str))
    overlap = set(picked["Code"].astype(str)) & used
    return picked, overlap

def feature_effect_table(df):
    rows = []

    for f in V12_FEATURES:
        if f not in df.columns:
            continue

        allv = pd.to_numeric(df[f], errors="coerce")
        valid = allv.notna().mean()

        win = pd.to_numeric(
            df.loc[df["순수익률(%)"] > 0, f],
            errors="coerce"
        ).dropna()

        loss = pd.to_numeric(
            df.loc[df["순수익률(%)"] <= 0, f],
            errors="coerce"
        ).dropna()

        std = allv.std()

        if (
            valid < 0.70
            or len(win) < 5
            or len(loss) < 5
            or pd.isna(std)
            or std == 0
        ):
            continue

        effect = (win.mean() - loss.mean()) / std

        rows.append({
            "변수": f,
            "승리평균": round(win.mean(), 3),
            "손실평균": round(loss.mean(), 3),
            "표준화차이": round(effect, 4),
            "방향": "높을수록 유리" if effect > 0 else "낮을수록 유리",
            "가용률(%)": round(valid * 100, 1),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["절대차이"] = out["표준화차이"].abs()
        out = out.sort_values("절대차이", ascending=False).reset_index(drop=True)
    return out

def build_score_model(train_df, top_n=8):
    effects = feature_effect_table(train_df)

    if effects.empty:
        return None, pd.DataFrame()

    selected = effects.head(int(top_n)).copy()

    refs = {}
    total_weight = 0.0

    for _, r in selected.iterrows():
        f = r["변수"]
        s = pd.to_numeric(train_df[f], errors="coerce").dropna().sort_values()

        if len(s) < 10:
            continue

        weight = abs(float(r["표준화차이"]))
        if weight <= 0:
            continue

        refs[f] = {
            "values": s.to_numpy(dtype=float),
            "direction": 1 if float(r["표준화차이"]) > 0 else -1,
            "weight": weight,
        }
        total_weight += weight

    model = {
        "refs": refs,
        "total_weight": total_weight,
        "features": list(refs.keys()),
    }

    return model, selected[selected["변수"].isin(model["features"])].copy()

def percentile_from_reference(values, ref_values):
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(vals), np.nan)

    for i, x in enumerate(vals):
        if np.isnan(x):
            continue
        pos = np.searchsorted(ref_values, x, side="right")
        result[i] = pos / len(ref_values)

    return result

def apply_score_model(df, model):
    q = df.copy()

    if model is None or not model["refs"]:
        q["승리점수"] = np.nan
        return q

    weighted_sum = np.zeros(len(q), dtype=float)
    weight_sum = np.zeros(len(q), dtype=float)

    for f, meta in model["refs"].items():
        if f not in q.columns:
            continue

        pct = percentile_from_reference(q[f], meta["values"])

        if meta["direction"] < 0:
            pct = 1.0 - pct

        valid = ~np.isnan(pct)
        w = float(meta["weight"])

        weighted_sum[valid] += pct[valid] * w
        weight_sum[valid] += w

    score = np.full(len(q), np.nan)
    valid = weight_sum > 0
    score[valid] = weighted_sum[valid] / weight_sum[valid] * 100.0

    q["승리점수"] = np.round(score, 2)
    return q

def score_threshold_table(df, thresholds):
    rows = []

    for th in thresholds:
        q = df[
            pd.to_numeric(df["승리점수"], errors="coerce") >= float(th)
        ].copy()

        if q.empty:
            continue

        rows.append({
            "점수기준": f"{th}점 이상",
            "기준점수": th,
            **performance_stats(q)
        })

    return pd.DataFrame(rows)

def choose_score_threshold(oos_df, min_signals=10):
    thresholds = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    table = score_threshold_table(oos_df, thresholds)

    if table.empty:
        return None, table

    eligible = table[
        (table["신호"] >= int(min_signals))
        & (table["평균수익률(%)"] > 0)
        & (table["ProfitFactor"].fillna(0) >= 1.5)
    ].copy()

    if eligible.empty:
        return None, table

    # 승률 최우선, 그 다음 평균수익률/PF/표본
    best = eligible.sort_values(
        ["승률(%)", "평균수익률(%)", "ProfitFactor", "신호"],
        ascending=[False, False, False, False]
    ).iloc[0]

    return float(best["기준점수"]), table

def score_band_table(df):
    q = df.copy()
    bins = [-0.01, 50, 60, 70, 80, 90, 100.01]
    labels = ["<50", "50~59", "60~69", "70~79", "80~89", "90~100"]

    q["점수구간"] = pd.cut(
        pd.to_numeric(q["승리점수"], errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    rows = []
    for band, g in q.groupby("점수구간", observed=True):
        rows.append({
            "점수구간": str(band),
            **performance_stats(g)
        })

    return pd.DataFrame(rows)

def yearly_scored(df, threshold):
    q = df[
        pd.to_numeric(df["승리점수"], errors="coerce") >= float(threshold)
    ].copy()

    if q.empty:
        return pd.DataFrame()

    return yearly_stats(q)

def stress_scored(df, threshold):
    q = df[
        pd.to_numeric(df["승리점수"], errors="coerce") >= float(threshold)
    ].copy()

    if q.empty:
        return pd.DataFrame()

    return stress_outliers(q)


# ============================================================
# v12.1 completely frozen model from v12.0 workbook
# ============================================================
V121_PRIOR_USED_N = 900   # v11.6까지 750 + v12.0 최종군 150
V121_THRESHOLDS = [50, 55, 60, 65, 70]

V121_FROZEN_REFS = {'돌파갭(%)': {'weight': 0.9968, 'direction': 1, 'values': [-2.96, -1.87, -1.68, -1.52, -1.4, -1.39, -1.34, -1.21, -1.08, -0.89, -0.71, -0.67, -0.47, -0.44, -0.36, -0.24, -0.06, -0.01, 0.0, 0.0, 0.18, 0.19, 0.22, 0.22, 0.27, 0.41, 0.41, 0.6, 0.68, 0.7, 0.98, 1.09, 1.35, 1.48, 1.57, 1.65, 1.87, 1.91, 1.93, 2.03, 2.05, 2.18, 2.75, 2.8, 2.87, 2.88, 3.2, 4.9, 5.96, 6.24, 6.63, 6.75, 7.6, 7.97, 12.53, 12.9, 14.89, 15.12]}, '전고점20이격(%)': {'weight': 0.4904, 'direction': 1, 'values': [-1.88, -1.87, -1.73, -1.71, -1.34, -1.2, -1.04, -0.71, -0.46, -0.4, -0.19, 0.32, 0.88, 1.13, 1.46, 1.46, 1.57, 1.74, 1.85, 2.03, 2.42, 2.7, 2.74, 2.95, 2.99, 3.15, 3.2, 3.34, 3.69, 3.75, 3.84, 3.93, 4.05, 4.2, 5.12, 5.26, 5.66, 6.32, 6.97, 8.06, 8.57, 9.51, 9.78, 10.65, 11.23, 11.29, 11.49, 12.11, 12.73, 13.42, 13.46, 15.03, 15.07, 17.35, 17.84, 18.1, 18.44, 27.99]}, '눌림깊이(%)': {'weight': 0.473, 'direction': 1, 'values': [-10.41, -10.13, -8.37, -7.02, -6.66, -6.41, -5.4, -4.91, -4.69, -4.6, -4.42, -4.32, -4.08, -4.08, -3.84, -3.72, -3.48, -3.17, -2.61, -1.65, -1.63, -1.47, -1.09, -0.94, -0.93, -0.92, -0.41, -0.3, -0.06, 0.36, 0.47, 0.68, 0.7, 1.04, 1.23, 1.9, 3.09, 3.49, 3.67, 3.86, 4.14, 4.27, 4.64, 5.48, 5.53, 5.94, 6.52, 7.14, 7.16, 7.5, 7.95, 8.78, 9.07, 9.18, 9.44, 14.52, 17.52, 20.74]}, '지수60일수익률(%)': {'weight': 0.4293, 'direction': 1, 'values': [-15.123, -9.7955, -8.4151, -8.2272, -6.4892, -5.8202, -5.4536, -4.5719, -4.4559, -4.3483, -3.9096, -3.6858, -2.2683, -2.2344, -1.9898, -1.8673, -1.7433, -1.6251, -1.4154, -1.2565, -0.7015, -0.4788, -0.3907, -0.0617, 0.2725, 0.6275, 0.746, 1.1906, 1.2889, 1.3225, 1.3289, 1.4578, 2.0323, 2.1942, 2.3087, 2.4638, 2.5642, 2.5655, 2.5925, 3.2763, 3.2763, 3.38, 3.434, 3.7256, 3.9815, 4.3267, 5.356, 5.5204, 5.605, 5.625, 5.9159, 6.1691, 6.1952, 6.1952, 6.523, 8.1233, 8.7653, 11.6366]}, '돌파종가수익률(%)': {'weight': 0.4145, 'direction': 1, 'values': [5.11, 5.15, 5.19, 5.26, 5.5, 5.6, 5.76, 5.82, 5.85, 6.05, 6.34, 6.47, 6.52, 6.65, 6.7, 6.97, 7.07, 7.11, 7.37, 7.57, 7.99, 8.0, 8.06, 8.08, 8.24, 8.41, 8.48, 8.57, 8.87, 9.61, 9.75, 10.08, 10.55, 10.65, 10.67, 11.39, 11.69, 13.07, 13.12, 13.41, 13.51, 14.44, 14.48, 14.82, 15.03, 15.75, 16.53, 16.82, 17.31, 17.35, 17.82, 18.44, 18.97, 20.24, 20.84, 20.9, 28.24, 28.99]}, '전고점60이격(%)': {'weight': 0.4041, 'direction': 1, 'values': [-66.73, -29.58, -15.33, -10.67, -8.5, -2.65, -2.34, -1.88, -1.87, -1.73, -1.71, -1.34, -1.2, -1.04, -0.71, -0.46, -0.4, 0.32, 0.88, 1.13, 1.46, 1.46, 1.74, 1.85, 2.03, 2.7, 2.74, 2.95, 2.99, 3.15, 3.2, 3.69, 3.75, 3.93, 4.05, 4.2, 5.12, 5.66, 6.32, 6.97, 8.06, 8.57, 9.51, 9.78, 10.65, 11.23, 11.29, 12.11, 12.73, 13.42, 13.46, 14.9, 15.03, 15.07, 17.35, 17.84, 18.44, 27.99]}, '돌파거래대금(억)': {'weight': 0.2829, 'direction': 1, 'values': [173.15, 179.45, 182.67, 579.26, 645.69, 757.19, 1038.55, 1268.35, 1357.87, 1558.14, 1584.15, 1669.34, 1701.4, 1746.68, 1823.3, 1849.93, 1913.64, 1954.82, 2013.88, 2020.6, 2034.12, 2048.0, 2215.83, 2256.38, 2278.52, 2296.49, 2298.95, 2300.85, 2406.15, 2521.82, 2681.6, 2799.3, 2913.56, 3071.84, 3118.78, 3159.5, 3324.42, 3751.12, 3772.43, 3837.58, 3966.88, 3995.98, 4525.03, 4618.16, 4809.65, 4856.37, 5102.14, 5340.07, 5362.47, 5530.89, 6049.9, 6754.95, 6810.7, 7580.9, 9095.01, 13828.7, 14724.39, 26590.15]}, 'MA60_10일기울기(%)': {'weight': 0.2751, 'direction': 1, 'values': [-2.47, -1.7663, -1.5039, -1.3124, -1.2514, -1.1855, -1.1834, -1.0908, -1.0434, -0.8451, -0.8208, -0.817, -0.7434, -0.7421, -0.4178, -0.4032, -0.334, -0.3336, -0.3268, -0.1826, -0.0919, -0.0456, -0.0209, 0.0264, 0.0518, 0.0565, 0.1415, 0.1733, 0.2024, 0.3243, 0.3286, 0.332, 0.3344, 0.3344, 0.3506, 0.3511, 0.4813, 0.4851, 0.4888, 0.569, 0.6536, 0.6825, 0.6848, 0.6964, 0.7319, 0.8493, 0.8745, 0.9035, 0.933, 0.933, 0.9691, 0.9718, 1.0091, 1.1918, 1.2261, 1.254, 1.2561, 1.5031]}}

def v121_frozen_model():
    return {
        "refs": {
            f: {
                "values": np.array(meta["values"], dtype=float),
                "direction": int(meta["direction"]),
                "weight": float(meta["weight"]),
            }
            for f, meta in V121_FROZEN_REFS.items()
        },
        "total_weight": sum(float(meta["weight"]) for meta in V121_FROZEN_REFS.values()),
        "features": list(V121_FROZEN_REFS.keys()),
    }

def v121_independent_universe(listing, n):
    kp = (
        listing[listing["Market"] == "KOSPI"]
        .sort_values("Code")
        .reset_index(drop=True)
    )

    picked = kp.iloc[
        V121_PRIOR_USED_N:
        V121_PRIOR_USED_N + int(n)
    ].copy().reset_index(drop=True)

    used = set(
        kp.iloc[:V121_PRIOR_USED_N]["Code"]
        .astype(str)
    )

    overlap = (
        set(picked["Code"].astype(str))
        & used
    )

    return picked, overlap, len(kp)

def wilson_interval(wins, n, z=1.96):
    if n <= 0:
        return (None, None)

    p = wins / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom

    return (
        max(0.0, center-margin) * 100,
        min(1.0, center+margin) * 100
    )

def threshold_comparison_with_ci(df):
    rows = []

    # baseline
    s = performance_stats(df)
    wins = int((df["순수익률(%)"] > 0).sum())
    lo, hi = wilson_interval(wins, len(df))

    rows.append({
        "점수기준":"F1+F2 전체",
        "기준점수":0,
        **s,
        "승률95%CI하한":None if lo is None else round(lo,1),
        "승률95%CI상한":None if hi is None else round(hi,1),
    })

    for th in V121_THRESHOLDS:
        q = df[
            pd.to_numeric(
                df["승리점수"],
                errors="coerce"
            ) >= th
        ].copy()

        if q.empty:
            continue

        s = performance_stats(q)
        wins = int((q["순수익률(%)"] > 0).sum())
        lo, hi = wilson_interval(wins, len(q))

        rows.append({
            "점수기준":f"{th}점 이상",
            "기준점수":th,
            **s,
            "승률95%CI하한":round(lo,1),
            "승률95%CI상한":round(hi,1),
        })

    return pd.DataFrame(rows)

def low_vs_high_score(df, cut=50):
    low = df[
        pd.to_numeric(df["승리점수"], errors="coerce") < cut
    ].copy()
    high = df[
        pd.to_numeric(df["승리점수"], errors="coerce") >= cut
    ].copy()

    rows = []

    for label, q in [
        (f"{cut}점 미만", low),
        (f"{cut}점 이상", high),
    ]:
        s = performance_stats(q)
        wins = int((q["순수익률(%)"] > 0).sum()) if len(q) else 0
        lo, hi = wilson_interval(wins, len(q))

        rows.append({
            "그룹":label,
            **s,
            "승률95%CI하한":None if lo is None else round(lo,1),
            "승률95%CI상한":None if hi is None else round(hi,1),
        })

    return pd.DataFrame(rows)

def threshold_yearly(df):
    q = df.copy()
    q["연도"] = pd.to_datetime(q["진입일"]).dt.year
    rows = []

    for th in V121_THRESHOLDS:
        z = q[
            pd.to_numeric(q["승리점수"], errors="coerce") >= th
        ].copy()

        for y, g in z.groupby("연도"):
            rows.append({
                "기준점수":th,
                "연도":int(y),
                **performance_stats(g)
            })

    return pd.DataFrame(rows)


# ============================================================
# v13 entry-style comparison
# ============================================================
V13_ENTRY_MODES = [
    "A 현재 돌파진입",
    "B 돌파일 종가베팅",
    "C 종가강도 종가베팅",
]

def v13_make_entry(d, setup, mode):
    breakout_i = int(setup["_breakout_i"])
    br = d.iloc[breakout_i]
    level = float(setup["눌림고가"])

    if mode == "A 현재 돌파진입":
        return {
            "진입전략": mode,
            "진입일": d.index[breakout_i].date(),
            "진입가": level,
            "_entry_i": breakout_i,
            "진입설명": "눌림고가 돌파 당일 돌파가 진입",
        }

    if mode == "B 돌파일 종가베팅":
        return {
            "진입전략": mode,
            "진입일": d.index[breakout_i].date(),
            "진입가": float(br["Close"]),
            "_entry_i": breakout_i,
            "진입설명": "F1+F2 충족 돌파일 종가 진입",
        }

    if mode == "C 종가강도 종가베팅":
        # 공개된 '강한 종목의 종가베팅' 구조를 정량화한 비교용 가설.
        # 1) 양봉
        # 2) 종가 > 5일선
        # 3) 5일선 상승
        # 4) 종가가 당일 Range 상단 70% 이상
        prev_ma5 = d["MA5"].iloc[breakout_i - 1] if breakout_i > 0 else None
        rng = float(br["High"]) - float(br["Low"])

        bullish = float(br["Close"]) > float(br["Open"])
        above_ma5 = pd.notna(br["MA5"]) and float(br["Close"]) > float(br["MA5"])
        rising_ma5 = (
            pd.notna(br["MA5"])
            and pd.notna(prev_ma5)
            and float(br["MA5"]) > float(prev_ma5)
        )
        close_pos = (
            (float(br["Close"]) - float(br["Low"])) / rng * 100.0
            if rng > 0 else 0.0
        )
        strong_close = close_pos >= 70.0

        if not (bullish and above_ma5 and rising_ma5 and strong_close):
            return None

        return {
            "진입전략": mode,
            "진입일": d.index[breakout_i].date(),
            "진입가": float(br["Close"]),
            "_entry_i": breakout_i,
            "진입설명": "돌파일 종가 + 양봉 + 5일선 위/상승 + 종가위치70%↑",
            "종가위치(%)": round(close_pos, 2),
        }

    return None


def v13_evaluate_trade(d, setup, entry):
    """
    v13.1 수정 핵심
    ----------------
    1) B/C는 돌파일 '종가' 진입이므로 진입 당일 High/Low를
       매수 이후 가격처럼 사용하지 않는다.
    2) A는 장중 돌파가 진입이라 일봉만으로 High/Low의 선후관계를
       알 수 없다. 따라서 진입 당일 Low로 임의 손절하지 않는다.
       단, 진입 당일 종가가 초기손절가 이하라면 돌파 진입 후
       종가까지 내려온 것이 확실하므로 당일 손절로 처리한다.
    3) 정상적인 손절/트레일/MFE/MAE 평가는 다음 거래일부터 시작한다.
    4) 최대 보유기간은 진입 후 '완전한 5거래일'로 통일한다.
    """
    entry_i = int(entry["_entry_i"])
    entry_price = float(entry["진입가"])
    mode = str(entry["진입전략"])

    initial_stop = entry_price * (1 - 0.03)

    hit = {t: False for t in TARGETS}
    mfe = 0.0
    mae = 0.0

    exit_i = None
    exit_price = None
    reason = None

    # --------------------------------------------------------
    # A만: 진입 당일 '확실한' 손절 여부만 판정
    # --------------------------------------------------------
    if mode == "A 현재 돌파진입":
        entry_close = float(d.iloc[entry_i]["Close"])

        # 돌파가에 실제 진입했다면, 종가가 손절선 이하인 경우
        # 진입 이후 손절선을 통과한 사실이 확실하다.
        if entry_close <= initial_stop:
            exit_i = entry_i
            exit_price = initial_stop
            reason = "손절(당일확정)"
            mae = -3.0

    # 이미 당일 확정 손절이면 즉시 반환
    if exit_i is not None:
        gross_ret = (exit_price / entry_price - 1) * 100
        net_ret = gross_ret - TOTAL_COST

        return {
            "청산평가시작": "진입당일 확정손절",
            "초기손절가": round(initial_stop),
            "청산일": d.index[exit_i].date(),
            "청산가": round(exit_price),
            "청산사유": reason,
            "손절": True,
            "총수익률(%)": round(gross_ret, 2),
            "거래비용(%)": round(TOTAL_COST, 2),
            "순수익률(%)": round(net_ret, 2),
            "MFE(%)": round(mfe, 2),
            "MAE(%)": round(mae, 2),
            "+3%": False,
            "+5%": False,
            "+7%": False,
            "+10%": False,
        }

    # --------------------------------------------------------
    # 모든 전략: 다음 거래일부터 정상 평가
    # --------------------------------------------------------
    start_i = entry_i + 1

    if start_i >= len(d):
        # 다음 거래일 데이터가 없으면 진입일 종가로 종료
        exit_i = entry_i
        exit_price = float(d.iloc[entry_i]["Close"])
        reason = "데이터종료"

        gross_ret = (exit_price / entry_price - 1) * 100
        net_ret = gross_ret - TOTAL_COST

        return {
            "청산평가시작": "다음거래일",
            "초기손절가": round(initial_stop),
            "청산일": d.index[exit_i].date(),
            "청산가": round(exit_price),
            "청산사유": reason,
            "손절": False,
            "총수익률(%)": round(gross_ret, 2),
            "거래비용(%)": round(TOTAL_COST, 2),
            "순수익률(%)": round(net_ret, 2),
            "MFE(%)": 0.0,
            "MAE(%)": 0.0,
            "+3%": False,
            "+5%": False,
            "+7%": False,
            "+10%": False,
        }

    # 진입 후 완전한 HOLDING_DAYS 거래일을 평가
    last_i = min(
        len(d) - 1,
        entry_i + HOLDING_DAYS
    )

    activated = False
    highest_before_today = entry_price

    for i in range(start_i, last_i + 1):
        r = d.iloc[i]
        high = float(r["High"])
        low = float(r["Low"])

        if activated:
            trailing = highest_before_today * (1 - 0.02)
            active_stop = max(entry_price, trailing)
        else:
            active_stop = initial_stop

        high_ret = (high / entry_price - 1) * 100
        low_ret = (low / entry_price - 1) * 100

        # 일봉 내 stop/high 선후관계가 불명확한 경우
        # 기존 전략과 동일하게 stop 우선으로 보수 처리
        if low <= active_stop:
            mae = min(
                mae,
                (active_stop / entry_price - 1) * 100
            )

            if high < entry_price * 1.02:
                mfe = max(mfe, max(0.0, high_ret))

            exit_i = i
            exit_price = active_stop
            reason = (
                "본전/트레일"
                if activated and active_stop >= entry_price
                else "손절"
            )
            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)

        for t in TARGETS:
            if high >= entry_price * (1 + t / 100):
                hit[t] = True

        if high >= entry_price * 1.02:
            activated = True

        highest_before_today = max(
            highest_before_today,
            high
        )

    if exit_i is None:
        exit_i = last_i
        exit_price = float(d.iloc[last_i]["Close"])
        reason = "기간종료"

    gross_ret = (exit_price / entry_price - 1) * 100
    net_ret = gross_ret - TOTAL_COST

    return {
        "청산평가시작": "다음거래일",
        "초기손절가": round(initial_stop),
        "청산일": d.index[exit_i].date(),
        "청산가": round(exit_price),
        "청산사유": reason,
        "손절": str(reason).startswith("손절"),
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


def v13_stats(df):
    s = performance_stats(df)
    if df.empty:
        s["손절률(%)"] = 0.0
    else:
        s["손절률(%)"] = round(
            df["청산사유"].astype(str).str.startswith("손절").mean() * 100,
            1
        )
    return s


def v13_rank_table(result):
    rows = []
    for mode in V13_ENTRY_MODES:
        q = result[result["진입전략"] == mode].copy()
        s = v13_stats(q)
        rows.append({
            "진입전략": mode,
            **s
        })

    out = pd.DataFrame(rows)

    # 승률 > 표본수 > 평균수익률 > PF > 낮은 손절률
    out = out.sort_values(
        [
            "승률(%)",
            "신호",
            "평균수익률(%)",
            "ProfitFactor",
            "손절률(%)"
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    out.insert(0, "순위", range(1, len(out)+1))
    return out


def v13_yearly(df):
    q = df.copy()
    q["연도"] = pd.to_datetime(q["진입일"]).dt.year

    rows = []

    for (mode, y), g in q.groupby(["진입전략","연도"]):
        rows.append({
            "진입전략": mode,
            "연도": int(y),
            **v13_stats(g)
        })

    return pd.DataFrame(rows)


def v13_time_split(df, ratio=0.60):
    rows = []

    # 각 전략을 자신의 신호 시간순으로 60/40
    for mode in V13_ENTRY_MODES:
        q = df[df["진입전략"] == mode].copy()
        train, test, split_date = fixed_time_split(q, ratio)

        if not train.empty:
            rows.append({
                "진입전략": mode,
                "구간": "앞60%",
                "시작일": None,
                **v13_stats(train)
            })

        if not test.empty:
            rows.append({
                "진입전략": mode,
                "구간": "뒤40%",
                "시작일": split_date,
                **v13_stats(test)
            })

    return pd.DataFrame(rows)


# ============================================================
# v14.1 filters
# ============================================================
def v141_base_filter(df):
    vol = pd.to_numeric(df["돌파거래량배수20"], errors="coerce")
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    return (vol <= 6.0) & (ma5 >= 12.0)

def v141_filter_a(df):
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    return v141_base_filter(df) | (ma5 <= 7.0)

def v141_filter_b(df):
    value = pd.to_numeric(df["기준봉거래대금(억)"], errors="coerce")
    return v141_base_filter(df) | (value >= 6000.0)

def v141_filter_ab(df):
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    value = pd.to_numeric(df["기준봉거래대금(억)"], errors="coerce")
    return v141_base_filter(df) | (ma5 <= 7.0) | (value >= 6000.0)

def v141_stats(df):
    s = performance_stats(df)
    s["손절률(%)"] = 0.0 if df.empty else round(
        df["청산사유"].astype(str).str.startswith("손절").mean() * 100, 1
    )
    return s

def v141_apply(df, name):
    if name == "C 원본":
        return df.copy()
    if name == "v14 1차":
        return df[v141_base_filter(df).fillna(False)].copy()
    if name == "A 승률형":
        return df[v141_filter_a(df).fillna(False)].copy()
    if name == "B 세력형":
        return df[v141_filter_b(df).fillna(False)].copy()
    if name == "A+B 복합형":
        return df[v141_filter_ab(df).fillna(False)].copy()
    return pd.DataFrame()

def v141_compare(df):
    rows = []
    for name in ["C 원본","v14 1차","A 승률형","B 세력형","A+B 복합형"]:
        rows.append({"전략":name, **v141_stats(v141_apply(df,name))})
    out = pd.DataFrame(rows).sort_values(
        ["승률(%)","신호","평균수익률(%)","ProfitFactor","손절률(%)"],
        ascending=[False,False,False,False,True]
    ).reset_index(drop=True)
    out.insert(0,"순위",range(1,len(out)+1))
    return out

def v141_time_split(df):
    rows=[]
    for name in ["C 원본","v14 1차","A 승률형","B 세력형","A+B 복합형"]:
        q=v141_apply(df,name).sort_values("진입일").reset_index(drop=True)
        if q.empty:
            continue
        n=len(q)
        cut=max(1,min(n-1,int(n*0.60))) if n>1 else 1
        for label,g in [("앞60%",q.iloc[:cut]),("뒤40%",q.iloc[cut:])]:
            if not g.empty:
                rows.append({"전략":name,"구간":label,**v141_stats(g)})
    return pd.DataFrame(rows)

def v141_yearly(df):
    rows=[]
    for name in ["C 원본","v14 1차","A 승률형","B 세력형","A+B 복합형"]:
        q=v141_apply(df,name).copy()
        if q.empty:
            continue
        q["연도"]=pd.to_datetime(q["진입일"]).dt.year
        for y,g in q.groupby("연도"):
            rows.append({"전략":name,"연도":int(y),**v141_stats(g)})
    return pd.DataFrame(rows)

def v141_rescue_audit(df):
    base=v141_apply(df,"v14 1차")
    base_keys=set(zip(base["코드"].astype(str),base["돌파일"].astype(str)))
    rows=[]
    for name in ["A 승률형","B 세력형","A+B 복합형"]:
        q=v141_apply(df,name)
        extra=q[~q.apply(lambda r:(str(r["코드"]),str(r["돌파일"])) in base_keys,axis=1)].copy()
        s=v141_stats(extra)
        rows.append({
            "구제유형":name,
            "추가구제건수":len(extra),
            "추가구제승리":int((extra["순수익률(%)"]>0).sum()) if len(extra) else 0,
            "추가구제손실":int((extra["순수익률(%)"]<=0).sum()) if len(extra) else 0,
            "추가구제승률(%)":s["승률(%)"],
            "추가구제평균(%)":s["평균수익률(%)"],
            "추가구제PF":s["ProfitFactor"],
        })
    return pd.DataFrame(rows)

def v141_removed_analysis(df):
    removed=df[~v141_base_filter(df).fillna(False)].copy()
    if removed.empty:
        return removed,pd.DataFrame()
    removed["결과그룹"]=removed["순수익률(%)"].apply(lambda x:"제거승리" if x>0 else "제거손실")
    feats=[
        "기준봉거래대금(억)","돌파거래량배수20","진입5일선이격(%)",
        "돌파종가수익률(%)","전고점20이격(%)","전고점60이격(%)",
        "눌림깊이(%)","돌파봉종가위치(%)"
    ]
    rows=[]
    for f in feats:
        if f not in removed.columns:
            continue
        w=pd.to_numeric(removed.loc[removed["결과그룹"]=="제거승리",f],errors="coerce").dropna()
        l=pd.to_numeric(removed.loc[removed["결과그룹"]=="제거손실",f],errors="coerce").dropna()
        rows.append({
            "변수":f,
            "제거승리평균":round(w.mean(),2) if len(w) else None,
            "제거손실평균":round(l.mean(),2) if len(l) else None,
            "차이":round(w.mean()-l.mean(),2) if len(w) and len(l) else None,
        })
    return removed,pd.DataFrame(rows)

# ============================================================
# v14.1 filters
# ============================================================
def v141_base_filter(df):
    vol = pd.to_numeric(df["돌파거래량배수20"], errors="coerce")
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    return (vol <= 6.0) & (ma5 >= 12.0)

def v141_filter_a(df):
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    return v141_base_filter(df) | (ma5 <= 7.0)

def v141_filter_b(df):
    value = pd.to_numeric(df["기준봉거래대금(억)"], errors="coerce")
    return v141_base_filter(df) | (value >= 6000.0)

def v141_filter_ab(df):
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    value = pd.to_numeric(df["기준봉거래대금(억)"], errors="coerce")
    return v141_base_filter(df) | (ma5 <= 7.0) | (value >= 6000.0)

def v141_stats(df):
    s = performance_stats(df)
    s["손절률(%)"] = 0.0 if df.empty else round(
        df["청산사유"].astype(str).str.startswith("손절").mean() * 100, 1
    )
    return s

def v141_apply(df, name):
    if name == "C 원본":
        return df.copy()
    if name == "v14 1차":
        return df[v141_base_filter(df).fillna(False)].copy()
    if name == "A 승률형":
        return df[v141_filter_a(df).fillna(False)].copy()
    if name == "B 세력형":
        return df[v141_filter_b(df).fillna(False)].copy()
    if name == "A+B 복합형":
        return df[v141_filter_ab(df).fillna(False)].copy()
    return pd.DataFrame()

def v141_compare(df):
    rows = []
    for name in ["C 원본","v14 1차","A 승률형","B 세력형","A+B 복합형"]:
        rows.append({"전략":name, **v141_stats(v141_apply(df,name))})
    out = pd.DataFrame(rows).sort_values(
        ["승률(%)","신호","평균수익률(%)","ProfitFactor","손절률(%)"],
        ascending=[False,False,False,False,True]
    ).reset_index(drop=True)
    out.insert(0,"순위",range(1,len(out)+1))
    return out

def v141_time_split(df):
    rows=[]
    for name in ["C 원본","v14 1차","A 승률형","B 세력형","A+B 복합형"]:
        q=v141_apply(df,name).sort_values("진입일").reset_index(drop=True)
        if q.empty:
            continue
        n=len(q)
        cut=max(1,min(n-1,int(n*0.60))) if n>1 else 1
        for label,g in [("앞60%",q.iloc[:cut]),("뒤40%",q.iloc[cut:])]:
            if not g.empty:
                rows.append({"전략":name,"구간":label,**v141_stats(g)})
    return pd.DataFrame(rows)

def v141_yearly(df):
    rows=[]
    for name in ["C 원본","v14 1차","A 승률형","B 세력형","A+B 복합형"]:
        q=v141_apply(df,name).copy()
        if q.empty:
            continue
        q["연도"]=pd.to_datetime(q["진입일"]).dt.year
        for y,g in q.groupby("연도"):
            rows.append({"전략":name,"연도":int(y),**v141_stats(g)})
    return pd.DataFrame(rows)

def v141_rescue_audit(df):
    base=v141_apply(df,"v14 1차")
    base_keys=set(zip(base["코드"].astype(str),base["돌파일"].astype(str)))
    rows=[]
    for name in ["A 승률형","B 세력형","A+B 복합형"]:
        q=v141_apply(df,name)
        extra=q[~q.apply(lambda r:(str(r["코드"]),str(r["돌파일"])) in base_keys,axis=1)].copy()
        s=v141_stats(extra)
        rows.append({
            "구제유형":name,
            "추가구제건수":len(extra),
            "추가구제승리":int((extra["순수익률(%)"]>0).sum()) if len(extra) else 0,
            "추가구제손실":int((extra["순수익률(%)"]<=0).sum()) if len(extra) else 0,
            "추가구제승률(%)":s["승률(%)"],
            "추가구제평균(%)":s["평균수익률(%)"],
            "추가구제PF":s["ProfitFactor"],
        })
    return pd.DataFrame(rows)

def v141_removed_analysis(df):
    removed=df[~v141_base_filter(df).fillna(False)].copy()
    if removed.empty:
        return removed,pd.DataFrame()
    removed["결과그룹"]=removed["순수익률(%)"].apply(lambda x:"제거승리" if x>0 else "제거손실")
    feats=[
        "기준봉거래대금(억)","돌파거래량배수20","진입5일선이격(%)",
        "돌파종가수익률(%)","전고점20이격(%)","전고점60이격(%)",
        "눌림깊이(%)","돌파봉종가위치(%)"
    ]
    rows=[]
    for f in feats:
        if f not in removed.columns:
            continue
        w=pd.to_numeric(removed.loc[removed["결과그룹"]=="제거승리",f],errors="coerce").dropna()
        l=pd.to_numeric(removed.loc[removed["결과그룹"]=="제거손실",f],errors="coerce").dropna()
        rows.append({
            "변수":f,
            "제거승리평균":round(w.mean(),2) if len(w) else None,
            "제거손실평균":round(l.mean(),2) if len(l) else None,
            "차이":round(w.mean()-l.mean(),2) if len(w) and len(l) else None,
        })
    return removed,pd.DataFrame(rows)


# ============================================================
# v14.2 frozen independent validation
# ============================================================
V142_KOSPI_USED_N = 500

def v142_final_filter(df):
    """
    v14.1 최종 후보 완전 동결:
    (돌파거래량배수20 <= 6 AND 진입5일선이격 >= 12)
      OR 진입5일선이격 <= 7
      OR 기준봉거래대금 >= 6000억
    """
    vol = pd.to_numeric(df["돌파거래량배수20"], errors="coerce")
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    value = pd.to_numeric(df["기준봉거래대금(억)"], errors="coerce")

    return (
        ((vol <= 6.0) & (ma5 >= 12.0))
        | (ma5 <= 7.0)
        | (value >= 6000.0)
    )


def v142_stats(df):
    s = performance_stats(df)

    if df.empty:
        s["손절률(%)"] = 0.0
    else:
        s["손절률(%)"] = round(
            df["청산사유"].astype(str).str.startswith("손절").mean() * 100,
            1
        )

    return s


def v142_market_universe(listing, market, n):
    q = (
        listing[listing["Market"] == market]
        .sort_values("Code")
        .reset_index(drop=True)
    )

    if market == "KOSPI":
        picked = q.iloc[
            V142_KOSPI_USED_N:
            V142_KOSPI_USED_N + int(n)
        ].copy().reset_index(drop=True)

        used = set(
            q.iloc[:V142_KOSPI_USED_N]["Code"].astype(str)
        )

        overlap = set(picked["Code"].astype(str)) & used

        return picked, overlap, len(q)

    # v14.1 filter discovery did not use KOSDAQ.
    picked = q.head(int(n)).copy().reset_index(drop=True)
    return picked, set(), len(q)


def v142_time_split(df):
    q = df.sort_values("진입일").reset_index(drop=True)

    if len(q) < 2:
        return pd.DataFrame()

    cut = max(
        1,
        min(
            len(q) - 1,
            int(len(q) * 0.60)
        )
    )

    rows = []

    for label, g in [
        ("앞60%", q.iloc[:cut].copy()),
        ("뒤40%", q.iloc[cut:].copy())
    ]:
        if not g.empty:
            rows.append({
                "구간": label,
                **v142_stats(g)
            })

    return pd.DataFrame(rows)


def v142_yearly(df):
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
            **v142_stats(g)
        })

    return pd.DataFrame(rows)


def v142_outlier_stress(df):
    if df.empty:
        return pd.DataFrame()

    ordered = df.sort_values(
        "순수익률(%)",
        ascending=False
    ).reset_index(drop=True)

    rows = []

    for n in [0,1,3,5]:
        if len(ordered) <= n:
            continue

        q = ordered.iloc[n:].copy()

        rows.append({
            "최고수익 제거": (
                "제거 없음"
                if n == 0
                else f"상위 {n}건"
            ),
            **v142_stats(q)
        })

    return pd.DataFrame(rows)


def v142_pass_judgement(stats):
    return (
        stats["신호"] >= 20
        and stats["승률(%)"] >= 50.0
        and stats["평균수익률(%)"] > 0
        and stats["ProfitFactor"] is not None
        and stats["ProfitFactor"] >= 1.5
        and stats["MDD(%)"] > -25.0
    )


# ============================================================
# v14.2 frozen independent validation
# ============================================================
V142_KOSPI_USED_N = 500

def v142_final_filter(df):
    """
    v14.1 최종 후보 완전 동결:
    (돌파거래량배수20 <= 6 AND 진입5일선이격 >= 12)
      OR 진입5일선이격 <= 7
      OR 기준봉거래대금 >= 6000억
    """
    vol = pd.to_numeric(df["돌파거래량배수20"], errors="coerce")
    ma5 = pd.to_numeric(df["진입5일선이격(%)"], errors="coerce")
    value = pd.to_numeric(df["기준봉거래대금(억)"], errors="coerce")

    return (
        ((vol <= 6.0) & (ma5 >= 12.0))
        | (ma5 <= 7.0)
        | (value >= 6000.0)
    )


def v142_stats(df):
    s = performance_stats(df)

    if df.empty:
        s["손절률(%)"] = 0.0
    else:
        s["손절률(%)"] = round(
            df["청산사유"].astype(str).str.startswith("손절").mean() * 100,
            1
        )

    return s


def v142_market_universe(listing, market, n):
    q = (
        listing[listing["Market"] == market]
        .sort_values("Code")
        .reset_index(drop=True)
    )

    if market == "KOSPI":
        picked = q.iloc[
            V142_KOSPI_USED_N:
            V142_KOSPI_USED_N + int(n)
        ].copy().reset_index(drop=True)

        used = set(
            q.iloc[:V142_KOSPI_USED_N]["Code"].astype(str)
        )

        overlap = set(picked["Code"].astype(str)) & used

        return picked, overlap, len(q)

    # v14.1 filter discovery did not use KOSDAQ.
    picked = q.head(int(n)).copy().reset_index(drop=True)
    return picked, set(), len(q)


def v142_time_split(df):
    q = df.sort_values("진입일").reset_index(drop=True)

    if len(q) < 2:
        return pd.DataFrame()

    cut = max(
        1,
        min(
            len(q) - 1,
            int(len(q) * 0.60)
        )
    )

    rows = []

    for label, g in [
        ("앞60%", q.iloc[:cut].copy()),
        ("뒤40%", q.iloc[cut:].copy())
    ]:
        if not g.empty:
            rows.append({
                "구간": label,
                **v142_stats(g)
            })

    return pd.DataFrame(rows)


def v142_yearly(df):
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
            **v142_stats(g)
        })

    return pd.DataFrame(rows)


def v142_outlier_stress(df):
    if df.empty:
        return pd.DataFrame()

    ordered = df.sort_values(
        "순수익률(%)",
        ascending=False
    ).reset_index(drop=True)

    rows = []

    for n in [0,1,3,5]:
        if len(ordered) <= n:
            continue

        q = ordered.iloc[n:].copy()

        rows.append({
            "최고수익 제거": (
                "제거 없음"
                if n == 0
                else f"상위 {n}건"
            ),
            **v142_stats(q)
        })

    return pd.DataFrame(rows)


def v142_pass_judgement(stats):
    return (
        stats["신호"] >= 20
        and stats["승률(%)"] >= 50.0
        and stats["평균수익률(%)"] > 0
        and stats["ProfitFactor"] is not None
        and stats["ProfitFactor"] >= 1.5
        and stats["MDD(%)"] > -25.0
    )


# ============================================================
# UI
# ============================================================
with st.sidebar:
    st.header("v14.2 독립검증 설정")

    end_date = st.date_input(
        "종료일",
        date(2026,7,31),
        disabled=True
    )

    years = st.number_input(
        "검증 기간(년)",
        min_value=5,
        max_value=5,
        value=5,
        disabled=True
    )

    kospi_n = st.selectbox(
        "미사용 KOSPI 종목 수",
        [100,150,200,250],
        index=2
    )

    kosdaq_n = st.selectbox(
        "KOSDAQ 검증 종목 수",
        [200,300,500],
        index=1
    )

    run = st.button(
        "▶ v14.2 최종 독립검증",
        type="primary",
        use_container_width=True
    )

st.info(
    "v14.2에서는 어떤 조건도 다시 찾거나 수정하지 않습니다. "
    "C 종가강도 전략 + A+B 복합형을 완전히 고정하고, "
    "v14.1에서 사용한 KOSPI 앞 500종목을 제외한 미사용 KOSPI와 "
    "별도의 KOSDAQ 표본에서 독립검증합니다."
)

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(years=int(years))
    start_ts = cut - pd.Timedelta(days=120)
    fetch_end = end_ts + pd.Timedelta(days=45)

    try:
        listing = stock_listing()
    except Exception as e:
        st.error(
            f"종목 목록 조회 실패: {type(e).__name__}"
        )
        st.exception(e)
        st.stop()

    kospi_u, kospi_overlap, kospi_total = (
        v142_market_universe(
            listing,
            "KOSPI",
            int(kospi_n)
        )
    )

    kosdaq_u, _, kosdaq_total = (
        v142_market_universe(
            listing,
            "KOSDAQ",
            int(kosdaq_n)
        )
    )

    if kospi_overlap:
        st.error(
            f"KOSPI 독립표본 오류: "
            f"기존 v14.1 사용군과 "
            f"{len(kospi_overlap)}종목 중복"
        )
        st.stop()

    if kospi_u.empty:
        st.error(
            f"KOSPI 전체 {kospi_total}종목 중 "
            f"앞 {V142_KOSPI_USED_N}종목 제외 후 "
            "검증 가능한 종목이 없습니다."
        )
        st.stop()

    if kosdaq_u.empty:
        st.error("KOSDAQ 검증 종목이 없습니다.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    errors = []

    def collect_market_trades(
        universe,
        market_name,
        progress_start,
        progress_end
    ):
        setups = []
        data_map = {}
        total = len(universe)

        # ---------------------------
        # data / setup
        # ---------------------------
        for pos, (_, r) in enumerate(
            universe.iterrows(),
            1
        ):
            code0 = str(r["Code"]).zfill(6)
            name = r["Name"]

            status.info(
                f"{market_name} 데이터/신호 "
                f"{pos}/{total} · {name}"
            )

            try:
                d = load_data(
                    code0,
                    start_ts.strftime("%Y-%m-%d"),
                    fetch_end.strftime("%Y-%m-%d")
                )

                if (
                    d is None
                    or d.empty
                    or len(d) < 100
                ):
                    continue

                found, prepared = find_setups(
                    d,
                    code0,
                    name,
                    market_name,
                    cut,
                    end_ts
                )

                data_map[code0] = prepared

                if found:
                    setups.extend(found)

            except Exception as e:
                errors.append(
                    f"{market_name} {name}({code0}): "
                    f"{type(e).__name__}"
                )

            frac = (
                progress_start
                + (progress_end-progress_start)
                * 0.50
                * (pos/max(total,1))
            )
            progress.progress(
                min(frac,1.0)
            )
            time.sleep(0.005)

        if not setups:
            return (
                pd.DataFrame(),
                pd.DataFrame()
            )

        setup_df = dedup_setups(
            pd.DataFrame(setups)
        )

        selected = setup_df[
            setup_df.apply(
                frozen_filter,
                axis=1
            )
        ].reset_index(drop=True)

        # ---------------------------
        # C strategy
        # ---------------------------
        rows = []

        for idx, (_, setup) in enumerate(
            selected.iterrows(),
            1
        ):
            code0 = str(
                setup["코드"]
            ).zfill(6)

            d = data_map.get(code0)

            if d is None:
                continue

            entry = v13_make_entry(
                d,
                setup,
                "C 종가강도 종가베팅"
            )

            if entry is None:
                continue

            ev = v13_evaluate_trade(
                d,
                setup,
                entry
            )

            row = setup.drop(
                labels=[
                    "_b",
                    "_pull_i",
                    "_breakout_i"
                ],
                errors="ignore"
            ).to_dict()

            row.update({
                k:v
                for k,v in entry.items()
                if not k.startswith("_")
            })

            row.update(ev)
            row.update(
                setup_features(
                    d,
                    setup
                )
            )

            rows.append(row)

            frac = (
                progress_start
                + (progress_end-progress_start)
                * (
                    0.50
                    + 0.50
                    * idx/max(len(selected),1)
                )
            )

            progress.progress(
                min(frac,1.0)
            )

        c_df = pd.DataFrame(rows)

        if c_df.empty:
            return c_df, c_df

        filtered = c_df[
            v142_final_filter(
                c_df
            ).fillna(False)
        ].copy()

        return c_df, filtered

    # KOSPI first half / KOSDAQ second half
    kospi_c, kospi_f = collect_market_trades(
        kospi_u,
        "KOSPI",
        0.0,
        0.50
    )

    kosdaq_c, kosdaq_f = collect_market_trades(
        kosdaq_u,
        "KOSDAQ",
        0.50,
        1.00
    )

    progress.empty()

    if kospi_c.empty:
        st.warning(
            "미사용 KOSPI에서 C전략 거래가 없습니다."
        )

    if kosdaq_c.empty:
        st.warning(
            "KOSDAQ에서 C전략 거래가 없습니다."
        )

    if kospi_c.empty and kosdaq_c.empty:
        st.stop()

    status.success(
        f"완료 · KOSPI {len(kospi_u)}종목 / "
        f"KOSDAQ {len(kosdaq_u)}종목"
    )

    # ========================================================
    # ① Independence audit
    # ========================================================
    st.subheader(
        "① 독립표본 감사"
    )

    audit_df = pd.DataFrame([
        {
            "시장":"KOSPI",
            "v14.1 사용범위":
                f"Code정렬 앞 {V142_KOSPI_USED_N}종목",
            "v14.2 검증종목":len(kospi_u),
            "기존군 중복":len(kospi_overlap),
        },
        {
            "시장":"KOSDAQ",
            "v14.1 사용범위":"미사용",
            "v14.2 검증종목":len(kosdaq_u),
            "기존군 중복":0,
        }
    ])

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ② Headline market comparison
    # ========================================================
    st.subheader(
        "② 시장별 C원본 vs A+B 동결필터"
    )

    compare_rows = []

    for market_name, cdf, fdf in [
        ("KOSPI", kospi_c, kospi_f),
        ("KOSDAQ", kosdaq_c, kosdaq_f),
    ]:
        if cdf.empty:
            continue

        compare_rows.append({
            "시장":market_name,
            "전략":"C 원본",
            **v142_stats(cdf)
        })

        compare_rows.append({
            "시장":market_name,
            "전략":"A+B 동결필터",
            **v142_stats(fdf)
        })

    compare_df = pd.DataFrame(
        compare_rows
    )

    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ③ Pass / fail
    # ========================================================
    st.subheader(
        "③ 사전 합격기준 판정"
    )

    judgement_rows = []

    for market_name, fdf in [
        ("KOSPI",kospi_f),
        ("KOSDAQ",kosdaq_f),
    ]:
        if fdf.empty:
            continue

        s = v142_stats(fdf)

        judgement_rows.append({
            "시장":market_name,
            "신호":s["신호"],
            "승률(%)":s["승률(%)"],
            "평균수익률(%)":
                s["평균수익률(%)"],
            "PF":s["ProfitFactor"],
            "MDD(%)":s["MDD(%)"],
            "손절률(%)":s["손절률(%)"],
            "합격":v142_pass_judgement(s),
        })

    judgement_df = pd.DataFrame(
        judgement_rows
    )

    st.dataframe(
        judgement_df,
        use_container_width=True,
        hide_index=True
    )

    if not judgement_df.empty:
        passes = int(
            judgement_df["합격"].sum()
        )

        if passes == len(judgement_df):
            st.success(
                "✅ 독립검증 통과: "
                "검증 가능한 모든 시장에서 "
                "신호≥20 / 승률≥50% / "
                "평균수익률>0 / PF≥1.5 / "
                "MDD>-25%를 충족했습니다."
            )
        elif passes > 0:
            st.warning(
                "⚠️ 부분 통과: 시장에 따라 "
                "재현성이 다릅니다. "
                "통과 시장과 탈락 시장을 분리해서 봐야 합니다."
            )
        else:
            st.warning(
                "❌ 독립검증 탈락: "
                "A+B 필터의 v14.1 성과가 "
                "새 표본에서 충분히 재현되지 않았습니다."
            )

    # ========================================================
    # ④ Time 60/40
    # ========================================================
    st.subheader(
        "④ 시장별 시간순 60/40"
    )

    time_rows = []

    for market_name, fdf in [
        ("KOSPI",kospi_f),
        ("KOSDAQ",kosdaq_f),
    ]:
        if fdf.empty:
            continue

        t = v142_time_split(fdf)

        if not t.empty:
            t.insert(
                0,
                "시장",
                market_name
            )
            time_rows.append(t)

    time_df = (
        pd.concat(
            time_rows,
            ignore_index=True
        )
        if time_rows
        else pd.DataFrame()
    )

    st.dataframe(
        time_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑤ Yearly
    # ========================================================
    st.subheader(
        "⑤ 시장별 연도성과"
    )

    year_rows = []

    for market_name, fdf in [
        ("KOSPI",kospi_f),
        ("KOSDAQ",kosdaq_f),
    ]:
        if fdf.empty:
            continue

        y = v142_yearly(fdf)

        if not y.empty:
            y.insert(
                0,
                "시장",
                market_name
            )
            year_rows.append(y)

    year_df = (
        pd.concat(
            year_rows,
            ignore_index=True
        )
        if year_rows
        else pd.DataFrame()
    )

    st.dataframe(
        year_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑥ Outlier stress
    # ========================================================
    st.subheader(
        "⑥ 아웃라이어 스트레스"
    )

    stress_rows = []

    for market_name, fdf in [
        ("KOSPI",kospi_f),
        ("KOSDAQ",kosdaq_f),
    ]:
        if fdf.empty:
            continue

        z = v142_outlier_stress(
            fdf
        )

        if not z.empty:
            z.insert(
                0,
                "시장",
                market_name
            )
            stress_rows.append(z)

    stress_df = (
        pd.concat(
            stress_rows,
            ignore_index=True
        )
        if stress_rows
        else pd.DataFrame()
    )

    st.dataframe(
        stress_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑦ Actual filtered trades
    # ========================================================
    st.subheader(
        "⑦ 독립검증 실제 거래"
    )

    trade_frames = []

    for market_name, fdf in [
        ("KOSPI",kospi_f),
        ("KOSDAQ",kosdaq_f),
    ]:
        if fdf.empty:
            continue

        q = fdf.copy()
        q["검증시장"] = market_name
        trade_frames.append(q)

    filtered_all = (
        pd.concat(
            trade_frames,
            ignore_index=True
        )
        if trade_frames
        else pd.DataFrame()
    )

    cols = [
        "검증시장",
        "종목명",
        "코드",
        "진입일",
        "기준봉거래대금(억)",
        "돌파거래량배수20",
        "진입5일선이격(%)",
        "돌파종가수익률(%)",
        "전고점20이격(%)",
        "눌림깊이(%)",
        "순수익률(%)",
        "MFE(%)",
        "MAE(%)",
        "청산사유",
    ]

    if not filtered_all.empty:
        st.dataframe(
            filtered_all[
                [
                    c for c in cols
                    if c in filtered_all.columns
                ]
            ].sort_values(
                ["검증시장","진입일"],
                ascending=[True,False]
            ),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # ⑧ Excel
    # ========================================================
    st.subheader(
        "⑧ v14.2 전체 독립검증 Excel"
    )

    settings_df = pd.DataFrame([
        {
            "항목":"기준전략",
            "값":"C 종가강도 종가베팅"
        },
        {
            "항목":"최종필터",
            "값":"(거래량배수20<=6 AND 5일선이격>=12%) OR 5일선이격<=7% OR 기준봉거래대금>=6000억"
        },
        {
            "항목":"조건재조정",
            "값":"금지"
        },
        {
            "항목":"KOSPI 독립성",
            "값":"v14.1 앞500종목 제외"
        },
        {
            "항목":"KOSDAQ",
            "값":"v14.1 필터개발 미사용 시장 별도 검증"
        },
        {
            "항목":"사전합격선",
            "값":"시장별 신호>=20 / 승률>=50% / 평균>0 / PF>=1.5 / MDD>-25%"
        },
    ])

    excel_bytes = build_excel({
        "00_설정":settings_df,
        "01_표본감사":audit_df,
        "02_시장별비교":compare_df,
        "03_합격판정":judgement_df,
        "04_시간60_40":time_df,
        "05_연도별":year_df,
        "06_아웃라이어":stress_df,
        "07_KOSPI_C원본":kospi_c,
        "08_KOSPI_필터":kospi_f,
        "09_KOSDAQ_C원본":kosdaq_c,
        "10_KOSDAQ_필터":kosdaq_f,
        "11_필터전체거래":filtered_all,
    })

    st.download_button(
        "📦 v14.2 최종 독립검증 Excel 다운로드",
        data=excel_bytes,
        file_name="swing_v14_2_final_independent_validation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    if errors:
        with st.expander(
            f"조회 실패/데이터 부족 "
            f"{len(errors)}건"
        ):
            st.write(errors)

st.caption(
    "v14.2는 최종 독립검증판입니다. "
    "A+B 복합형 조건은 완전히 동결하며, "
    "결과가 나쁘더라도 이 버전에서 6배/7%/12%/6000억 값을 수정하지 않습니다."
)
