import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v8.1", layout="wide")
st.title("🧪 단기스윙 v8.1 · 대규모 표본 검증")
st.caption("B 진입 + 손절 -3% + 최대 5거래일 고정 · 돌파 거래량 필터의 지속성 검증")

TARGETS = [3.0, 5.0, 7.0, 10.0]
STOP = -3.0
HOLD = 5

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
    x["AVG_VOL20"] = x["Volume"].shift(1).rolling(20).mean()
    x["등락률"] = x["Close"].pct_change() * 100
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

def find_b_signals(d, code, name, market, p, cut, end):
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

            entry_price = float(pull["High"])
            if float(br["High"]) < entry_price:
                continue

            base_close = float(x["Close"])
            br_range = max(float(br["High"]) - float(br["Low"]), 1.0)

            rows.append({
                "시장": market,
                "종목명": name,
                "코드": str(code).zfill(6),
                "기준봉일": d.index[b].date(),
                "기준봉상승률(%)": round(float(x["등락률"]), 2),
                "거래대금(억)": round(float(x["거래대금"]) / 1e8, 0),
                "기준봉거래량배수": round(float(x["Volume"] / x["AVG_VOL20"]), 2),
                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(entry_price),
                "눌림깊이(%)": round((float(pull["Close"]) / base_close - 1) * 100, 2),
                "눌림거래량비율(%)": round(float(pull["Volume"]) / float(x["Volume"]) * 100, 2),
                "돌파일": d.index[breakout_i].date(),
                "돌파종가": round(float(br["Close"])),
                "돌파종가위치(%)": round(
                    (float(br["Close"]) - float(br["Low"])) / br_range * 100, 2
                ),
                "돌파종가수익률(%)": round(
                    (float(br["Close"]) / entry_price - 1) * 100, 2
                ),
                "돌파거래량vs눌림(배)": round(
                    float(br["Volume"]) / max(float(pull["Volume"]), 1.0), 2
                ),
                "진입가": round(entry_price),
                "_breakout_i": breakout_i,
            })
            break

    return rows, d

def evaluate_b(d, breakout_i, entry_price, stop_pct=-3.0, hold_days=5):
    stop_price = entry_price * (1 + stop_pct / 100.0)
    hit = {t: False for t in TARGETS}
    mfe, mae = 0.0, 0.0

    last_i = min(len(d) - 1, breakout_i + hold_days - 1)
    exit_i = None
    exit_price = None
    reason = None

    for i in range(breakout_i, last_i + 1):
        r = d.iloc[i]
        high = float(r["High"])
        low = float(r["Low"])

        high_ret = (high / entry_price - 1) * 100
        low_ret = (low / entry_price - 1) * 100

        if low <= stop_price:
            mae = min(mae, stop_pct)
            if high < entry_price * 1.03:
                mfe = max(mfe, max(0.0, high_ret))
            exit_i = i
            exit_price = stop_price
            reason = "손절"
            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)

        for t in TARGETS:
            if high >= entry_price * (1 + t / 100.0):
                hit[t] = True

    if exit_i is None:
        exit_i = last_i
        exit_price = float(d.iloc[last_i]["Close"])
        reason = "기간종료"

    ret = (exit_price / entry_price - 1) * 100

    return {
        "손절가": round(stop_price),
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

def dedup(df):
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
            "신호": 0, "승률(%)": 0.0, "평균수익률(%)": 0.0,
            "손절률(%)": 0.0, "+3%(%)": 0.0, "+5%(%)": 0.0,
            "+7%(%)": 0.0, "+10%(%)": 0.0,
            "평균MFE(%)": 0.0, "평균MAE(%)": 0.0
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

def choose_universe(listing, n):
    # 데이터 조회 부담을 줄이기 위해 상장 종목을 코드순으로 제한.
    # 전체시장 검증 전 단계에서 100~300종목 표본 비교용.
    if n >= len(listing):
        return listing.copy()
    return listing.sort_values(["Market", "Code"]).head(n).copy()

with st.sidebar:
    st.header("대규모 검증 설정")

    end_date = st.date_input("종료일", date(2026, 7, 31))
    years = st.selectbox("검증 기간", [1, 2, 3], index=1, format_func=lambda x: f"{x}년")
    universe_n = st.selectbox("검증 종목 수", [50, 100, 200, 300], index=2)

    st.divider()
    st.subheader("기준 신호 조건")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("기준봉 최소 거래대금(억)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)

    st.divider()
    st.subheader("검증할 돌파 거래량 컷")
    selected_cuts = st.multiselect(
        "눌림 대비 돌파 거래량",
        [0.0, 1.0, 1.2, 1.5, 1.8, 2.0],
        default=[0.0, 1.2, 1.5, 1.8]
    )

    run = st.button("▶ 대규모 검증 실행", type="primary", use_container_width=True)

st.info(
    "고정 전략: B(눌림고가 돌파 당일 진입) · 손절 -3% · 최대 5거래일. "
    "v8에서 유망했던 '돌파 거래량 증가'가 종목/기간을 늘려도 유지되는지 검증합니다."
)

if run:
    if not selected_cuts:
        st.warning("돌파 거래량 컷을 하나 이상 선택하세요.")
        st.stop()

    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(years=years)
    start_ts = cut - pd.Timedelta(days=70)
    fetch_end = end_ts + pd.Timedelta(days=30)

    try:
        listing = stock_listing()
    except Exception as e:
        st.error(f"종목 목록 조회 실패: {type(e).__name__}")
        st.exception(e)
        st.stop()

    universe = choose_universe(listing, int(universe_n))

    p = {
        "base_rise": base_rise,
        "value_eok": value_eok,
        "volume_mult": volume_mult,
        "pullback_ratio": pullback_ratio / 100,
    }

    status = st.empty()
    bar = st.progress(0)
    raw_rows = []
    data_map = {}
    errors = []

    total = len(universe)

    for i, r in universe.iterrows():
        idx = list(universe.index).index(i) + 1
        code = str(r["Code"]).zfill(6)
        name = r["Name"]
        market = r["Market"]

        status.info(f"{idx}/{total} · {market} {name} 조회/검증 중")
        bar.progress(idx / total)

        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d")
            )

            if d is None or d.empty or len(d) < 30:
                errors.append(f"{name}: 데이터 부족")
                continue

            rows, prepared = find_b_signals(
                d, code, name, market, p, cut, end_ts
            )
            raw_rows.extend(rows)
            data_map[code] = prepared

        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")

        time.sleep(0.02)

    bar.empty()

    raw = pd.DataFrame(raw_rows)
    signals = dedup(raw)

    if signals.empty:
        status.warning("현재 조건에서는 신호가 없습니다.")
        st.stop()

    trades = []

    for _, sig in signals.iterrows():
        code = sig["코드"]
        if code not in data_map:
            continue

        ev = evaluate_b(
            data_map[code],
            int(sig["_breakout_i"]),
            float(sig["진입가"]),
            STOP,
            HOLD,
        )

        row = sig.drop(labels=["_breakout_i"]).to_dict()
        row.update(ev)
        trades.append(row)

    t = pd.DataFrame(trades)

    status.success(
        f"완료 · {total}종목 / {years}년 · "
        f"원신호 {len(raw)}건 → 실제매매 {len(t)}건"
    )

    st.subheader("전체 기준 전략 성과")
    base = summarize(t)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("실제매매", base["신호"])
    c2.metric("승률", f'{base["승률(%)"]:.1f}%')
    c3.metric("평균수익률", f'{base["평균수익률(%)"]:.2f}%')
    c4.metric("손절률", f'{base["손절률(%)"]:.1f}%')

    comparison = []

    for cut_v in sorted(selected_cuts):
        q = t[t["돌파거래량vs눌림(배)"] >= cut_v].copy()
        if q.empty:
            continue

        s = summarize(q)

        comparison.append({
            "돌파거래량 컷(배)": cut_v,
            "신호": s["신호"],
            "신호유지율(%)": round(s["신호"] / len(t) * 100, 1),
            "승률(%)": s["승률(%)"],
            "평균수익률(%)": s["평균수익률(%)"],
            "손절률(%)": s["손절률(%)"],
            "+3%(%)": s["+3%(%)"],
            "+5%(%)": s["+5%(%)"],
            "+7%(%)": s["+7%(%)"],
            "+10%(%)": s["+10%(%)"],
            "평균MFE(%)": s["평균MFE(%)"],
            "평균MAE(%)": s["평균MAE(%)"],
        })

    comp = pd.DataFrame(comparison)

    if comp.empty:
        st.warning("선택한 거래량 컷에서 유효한 신호가 없습니다.")
        st.stop()

    st.subheader("돌파 거래량 컷 비교")
    st.dataframe(
        comp.sort_values("돌파거래량 컷(배)"),
        use_container_width=True,
        hide_index=True
    )

    # 안정성 관점의 추천:
    # 최소 20건 이상이면서 평균수익률이 가장 높은 컷.
    enough = comp[comp["신호"] >= 20].copy()
    if not enough.empty:
        recommended = enough.sort_values(
            ["평균수익률(%)", "손절률(%)", "+5%(%)"],
            ascending=[False, True, False]
        ).iloc[0]

        st.subheader("현재 안정성 우선 추천 컷")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("돌파 거래량", f'≥ {recommended["돌파거래량 컷(배)"]:.1f}배')
        c2.metric("신호", int(recommended["신호"]))
        c3.metric("평균수익률", f'{recommended["평균수익률(%)"]:.2f}%')
        c4.metric("손절률", f'{recommended["손절률(%)"]:.1f}%')

        rec_trades = t[
            t["돌파거래량vs눌림(배)"] >= recommended["돌파거래량 컷(배)"]
        ].copy()

        st.subheader("추천 컷 실제 거래")
        cols = [
            "시장","종목명","코드","기준봉일","돌파일","진입가",
            "기준봉상승률(%)","거래대금(억)",
            "돌파종가수익률(%)","돌파거래량vs눌림(배)",
            "최종수익률(%)","손절","+3%","+5%","+7%","+10%",
            "MFE(%)","MAE(%)"
        ]
        st.dataframe(
            rec_trades[cols].sort_values("최종수익률(%)", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning(
            "현재 선택 컷 중 신호 20건 이상인 조합이 없습니다. "
            "종목 수나 기간을 늘려 표본을 확보하세요."
        )

    st.subheader("시장별 성과")
    market_rows = []
    for market in sorted(t["시장"].unique()):
        q = t[t["시장"] == market]
        s = summarize(q)
        market_rows.append({"시장": market, **s})

    st.dataframe(
        pd.DataFrame(market_rows),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "거래량 컷 비교 CSV",
        comp.to_csv(index=False).encode("utf-8-sig"),
        "swing_v8_1_volume_cut_compare.csv",
        "text/csv",
        use_container_width=True
    )

    st.download_button(
        "전체 거래 CSV",
        t.to_csv(index=False).encode("utf-8-sig"),
        "swing_v8_1_all_trades.csv",
        "text/csv",
        use_container_width=True
    )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption(
    "주의: 돌파 당일 일봉에서 손절가와 목표가를 모두 터치한 경우에는 "
    "장중 순서를 알 수 없어 손절 우선으로 보수 처리합니다. "
    "v8.1은 전략 확정판이 아니라 표본 확대 검증판입니다."
)
