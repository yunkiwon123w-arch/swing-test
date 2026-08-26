import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date
from io import BytesIO

st.set_page_config(page_title="단기스윙 백테스트 v10.5", layout="wide")
st.title("🧪 단기스윙 v10.5 · 독립 표본 최종검증")
st.caption("기존 500종목과 겹치지 않는 독립군 · F1+F2/진입/청산/비용 전부 동결 · KOSPI/KOSDAQ 및 시장국면 분리")

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


# ============================================================
# UI
# ============================================================
with st.sidebar:
    st.header("v10.5 독립검증 설정")

    end_date = st.date_input(
        "종료일",
        date(2026, 7, 31),
        disabled=True
    )

    years = st.number_input(
        "검증 기간(년)",
        min_value=5,
        max_value=5,
        value=5,
        disabled=True
    )

    universe_n = st.selectbox(
        "독립 검증 종목 수",
        [300, 500, 700],
        index=1
    )

    st.divider()

    st.subheader("전략 · 완전 동결")

    st.write("E1 돌파당일")
    st.write("F1 돌파종가수익률 ≥ +5%")
    st.write("F2 20일 전고점 이격 ≥ -2%")
    st.write("돌파 거래량 ≥ 눌림의 1.8배")
    st.write("초기 손절 -3%")
    st.write("+2% 활성 / 2% 트레일")
    st.write("최대 5거래일")
    st.write("거래비용 0.40%")

    run = st.button(
        "▶ 독립 표본 검증 실행",
        type="primary",
        use_container_width=True
    )


st.info(
    "v10.5에서는 조건을 하나도 바꾸지 않습니다. "
    "이전 v10.x의 첫 500종목을 코드로 재현해 완전히 제외한 뒤, "
    "남은 종목에서 KOSPI/KOSDAQ 균형 독립군을 만들어 같은 전략을 시험합니다."
)


if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(
        years=int(years)
    )

    start_ts = (
        cut - pd.Timedelta(days=100)
    )

    fetch_end = (
        end_ts + pd.Timedelta(days=45)
    )

    # -----------------------------
    # listing / independent universe
    # -----------------------------
    try:
        listing = stock_listing()
    except Exception as e:
        st.error(
            f"종목 목록 조회 실패: "
            f"{type(e).__name__}"
        )
        st.exception(e)
        st.stop()

    universe, old_codes = (
        independent_balanced_universe(
            listing,
            int(universe_n)
        )
    )

    overlap = (
        set(universe["Code"]) &
        old_codes
    )

    if overlap:
        st.error(
            f"독립표본 오류: 기존 500종목과 "
            f"{len(overlap)}종목 중복"
        )
        st.stop()

    kp_n = int(
        (universe["Market"] == "KOSPI").sum()
    )
    kq_n = int(
        (universe["Market"] == "KOSDAQ").sum()
    )

    st.write(
        f"**독립군 구성:** "
        f"{len(universe)}종목 "
        f"(KOSPI {kp_n} / KOSDAQ {kq_n})"
    )

    st.write(
        "**기존 500종목과 중복:** 0종목"
    )

    # -----------------------------
    # data & setup
    # -----------------------------
    progress = st.progress(0)
    status = st.empty()

    setups = []
    data_map = {}
    errors = []

    total = len(universe)

    for pos, (_, r) in enumerate(
        universe.iterrows(),
        1
    ):
        code = str(r["Code"]).zfill(6)
        name = r["Name"]
        market = r["Market"]

        status.info(
            f"1/2 독립군 데이터/신호 "
            f"{pos}/{total} · "
            f"{market} {name}"
        )

        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d")
            )

            if (
                d is None
                or d.empty
                or len(d) < 100
            ):
                errors.append(
                    f"{name}: 데이터 부족"
                )
                continue

            found, prepared = find_setups(
                d,
                code,
                name,
                market,
                cut,
                end_ts
            )

            data_map[code] = prepared

            if found:
                setups.extend(found)

        except Exception as e:
            errors.append(
                f"{name}({code}): "
                f"{type(e).__name__}"
            )

        progress.progress(
            pos / (total * 2)
        )

        time.sleep(0.01)

    if not setups:
        progress.empty()
        status.warning(
            "독립군에서 setup이 없습니다."
        )
        st.stop()

    setup_df = dedup_setups(
        pd.DataFrame(setups)
    )

    # -----------------------------
    # apply frozen F1+F2 & trade
    # -----------------------------
    selected = setup_df[
        setup_df.apply(
            frozen_filter,
            axis=1
        )
    ].reset_index(drop=True)

    rows = []

    for idx, (_, setup) in enumerate(
        selected.iterrows(),
        1
    ):
        code = str(
            setup["코드"]
        ).zfill(6)

        d = data_map.get(code)

        if d is None:
            continue

        entry = make_entry(
            d,
            setup
        )

        ev = evaluate_trade(
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
            k: v
            for k, v in entry.items()
            if not k.startswith("_")
        })

        row.update(ev)

        rows.append(row)

        if len(selected):
            progress.progress(
                0.5 +
                idx / (len(selected) * 2)
            )

    progress.empty()

    trades = pd.DataFrame(rows)

    if trades.empty:
        status.warning(
            "F1+F2 독립검증 거래가 없습니다."
        )
        st.stop()

    # 시장국면 부착
    trades = attach_market_regime(
        trades,
        start_ts.strftime("%Y-%m-%d"),
        fetch_end.strftime("%Y-%m-%d")
    )

    status.success(
        f"완료 · 독립군 {len(universe)}종목 · "
        f"기본 setup {len(setup_df)}건 · "
        f"F1+F2 실제매매 {len(trades)}건"
    )

    # ========================================================
    # ① independent test headline
    # ========================================================
    st.subheader(
        "① 독립 표본 전체 성과"
    )

    overall = performance_stats(
        trades
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "실제매매",
        overall["신호"]
    )

    c2.metric(
        "승률",
        f'{overall["승률(%)"]:.1f}%'
    )

    c3.metric(
        "평균 순수익률",
        f'{overall["평균수익률(%)"]:.2f}%'
    )

    pf = overall["ProfitFactor"]

    c4.metric(
        "Profit Factor",
        "-" if pf is None
        else f"{pf:.2f}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "중앙 수익률",
        f'{overall["중앙수익률(%)"]:.2f}%'
    )

    c2.metric(
        "MDD",
        f'{overall["MDD(%)"]:.2f}%'
    )

    c3.metric(
        "최대 연속손실",
        f'{overall["최대연속손실"]}회'
    )

    c4.metric(
        "누적복리 진단",
        f'{overall["누적복리수익률(%)"]:.1f}%'
    )

    # 사전 합격 기준
    passed = (
        overall["신호"] >= 30
        and overall["평균수익률(%)"] >= 2.0
        and (
            overall["ProfitFactor"] is not None
            and overall["ProfitFactor"] >= 1.5
        )
        and overall["MDD(%)"] > -35.0
    )

    if passed:
        st.success(
            "독립검증 1차 합격: "
            "신호≥30, 평균수익률≥+2%, "
            "PF≥1.5, MDD>-35% 기준을 통과했습니다."
        )
    else:
        st.warning(
            "독립검증 합격 기준을 모두 충족하지 못했습니다. "
            "조건을 수정하지 말고 결과 자체를 먼저 분석하세요."
        )

    # ========================================================
    # ② market split
    # ========================================================
    st.subheader(
        "② KOSPI / KOSDAQ 분리 성과"
    )

    market_df = market_stats(
        trades
    )

    st.dataframe(
        market_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ③ yearly
    # ========================================================
    st.subheader(
        "③ 연도별 독립 성과"
    )

    year_df = yearly_stats(
        trades
    )

    st.dataframe(
        year_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ④ regime
    # ========================================================
    st.subheader(
        "④ 시장국면별 성과"
    )

    regime_df = regime_stats(
        trades
    )

    st.caption(
        "시장국면은 해당 시장지수(KOSPI/KOSDAQ)의 "
        "종가·20일선·60일선으로 분류합니다. "
        "종가>60일선 & 20일선>60일선=상승장, "
        "반대=하락장, 나머지=중립."
    )

    st.dataframe(
        regime_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑤ second holdout inside unseen universe
    # ========================================================
    st.subheader(
        "⑤ 독립군 내부 시간순 60/40 재검증"
    )

    hold_df, hold_date = (
        chronological_holdout(
            trades,
            0.60
        )
    )

    if hold_date is not None:
        st.write(
            f"**뒤 40% 시작일:** "
            f"{hold_date}"
        )

        st.dataframe(
            hold_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "시간순 재검증 표본이 부족합니다."
        )

    # ========================================================
    # ⑥ outlier
    # ========================================================
    st.subheader(
        "⑥ 독립군 아웃라이어 스트레스"
    )

    stress_df = stress_outliers(
        trades
    )

    st.dataframe(
        stress_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑦ annual x market
    # ========================================================
    st.subheader(
        "⑦ 시장 × 연도 교차 성과"
    )

    cross_rows = []

    temp = trades.copy()
    temp["연도"] = pd.to_datetime(
        temp["진입일"]
    ).dt.year

    for (market, year), g in temp.groupby(
        ["시장", "연도"]
    ):
        cross_rows.append({
            "시장": market,
            "연도": int(year),
            **performance_stats(g)
        })

    cross_df = pd.DataFrame(
        cross_rows
    ).sort_values(
        ["시장", "연도"]
    )

    st.dataframe(
        cross_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑧ trades
    # ========================================================
    st.subheader(
        "⑧ 독립군 실제 거래"
    )

    cols = [
        "시장",
        "시장국면",
        "종목명",
        "코드",
        "기준봉일",
        "눌림일",
        "돌파일",
        "진입일",
        "진입가",
        "돌파종가수익률(%)",
        "전고점20이격(%)",
        "돌파거래량vs눌림(배)",
        "초기손절가",
        "청산일",
        "청산가",
        "청산사유",
        "총수익률(%)",
        "거래비용(%)",
        "순수익률(%)",
        "MFE(%)",
        "MAE(%)",
        "+3%",
        "+5%",
        "+7%",
        "+10%",
    ]

    st.dataframe(
        trades[
            [c for c in cols
             if c in trades.columns]
        ].sort_values(
            "진입일",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # ⑨ universe audit
    # ========================================================
    st.subheader(
        "⑨ 독립표본 감사"
    )

    audit_df = pd.DataFrame([
        {
            "항목": "기존 v10.x 표본",
            "종목수": OLD_SAMPLE_SIZE,
            "설명": "기존 Market+Code 정렬 후 첫 500종목"
        },
        {
            "항목": "v10.5 독립군",
            "종목수": len(universe),
            "설명": (
                f"KOSPI {kp_n} / KOSDAQ {kq_n}, "
                "기존 500종목 완전 제외"
            )
        },
        {
            "항목": "중복",
            "종목수": len(overlap),
            "설명": "반드시 0이어야 함"
        }
    ])

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # Excel
    # ========================================================
    st.subheader(
        "⑩ 전체 독립검증 Excel"
    )

    overview_df = pd.DataFrame([
        {
            "구분": "v10.5 독립검증",
            **overall
        }
    ])

    frozen_df = pd.DataFrame([
        {"항목": "진입", "값": ENTRY_MODE},
        {"항목": "필터 F1", "값": f"돌파종가수익률 >= {F1_BREAKOUT_CLOSE_RET}%"},
        {"항목": "필터 F2", "값": f"전고점20이격 >= {F2_PRIOR20_GAP}%"},
        {"항목": "돌파거래량", "값": f">= {BREAKOUT_VOL_CUT}배"},
        {"항목": "초기손절", "값": f"{INITIAL_STOP_PCT}%"},
        {"항목": "활성화", "값": f"+{ACTIVATION_PCT}%"},
        {"항목": "트레일", "값": f"{TRAIL_PCT}%"},
        {"항목": "최대보유", "값": f"{HOLDING_DAYS}거래일"},
        {"항목": "비용", "값": f"{TOTAL_COST:.2f}%"},
    ])

    excel_bytes = build_excel({
        "00_전략동결값": frozen_df,
        "01_전체성과": overview_df,
        "02_시장별": market_df,
        "03_연도별": year_df,
        "04_시장국면": regime_df,
        "05_시간순60_40": hold_df,
        "06_아웃라이어": stress_df,
        "07_시장연도교차": cross_df,
        "08_실제거래": trades,
        "09_독립군종목": universe,
        "10_독립표본감사": audit_df,
    })

    st.download_button(
        "📦 v10.5 전체 독립검증 Excel 다운로드",
        data=excel_bytes,
        file_name="swing_v10_5_independent_validation.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
        use_container_width=True
    )

    if errors:
        with st.expander(
            f"조회 실패/데이터 부족 "
            f"{len(errors)}건"
        ):
            st.write(errors)


st.caption(
    "v10.5는 조건 탐색판이 아닙니다. "
    "F1+F2/E1/X3/비용을 모두 동결한 상태에서 "
    "기존 500종목과 겹치지 않는 독립 종목군의 재현성만 검사합니다. "
    "결과가 나쁘더라도 이 버전에서 조건을 수정하지 않습니다."
)
