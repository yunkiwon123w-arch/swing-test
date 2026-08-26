import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v9", layout="wide")
st.title("🧪 단기스윙 v9 · 진입/청산 구조 비교")
st.caption("대규모 표본 · 돌파 거래량 필터 유지 · 진입 3종 × 청산 3종 = 9개 조합")

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
            if high < entry_price * 1.03:
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
        if exit_mode == "X3 +3%후트레일" and high >= entry_price * 1.03:
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

def choose_universe(listing, n):
    if n >= len(listing):
        return listing.copy()
    # 코드순 표본. 향후 v10에서 거래대금/시총 기반 표본으로 개선 가능.
    return listing.sort_values(["Market", "Code"]).head(n).copy()

ENTRY_MODES = ["E1 돌파당일", "E2 종가확인", "E3 재눌림"]
EXIT_MODES = ["X1 고정-3%", "X2 눌림저점", "X3 +3%후트레일"]

with st.sidebar:
    st.header("대규모 검증 설정")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    years = st.selectbox("검증 기간", [1, 2, 3], index=1, format_func=lambda x: f"{x}년")
    universe_n = st.selectbox("검증 종목 수", [50, 100, 200, 300], index=2)

    st.divider()
    st.subheader("기준 신호")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("기준봉 최소 거래대금(억)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)
    breakout_vol_cut = st.number_input(
        "돌파 거래량 / 눌림 거래량(배)",
        value=1.8, step=0.1
    )

    st.divider()
    st.subheader("재눌림/청산")
    retest_days = st.number_input("재눌림 대기 거래일", 1, 5, 3)
    retest_touch_pct = st.number_input("돌파선 위 재눌림 허용폭(%)", value=2.0, step=0.5)
    holding_days = st.number_input("최대 보유 거래일", 3, 15, 5)
    max_structural_risk = st.number_input("눌림저점 손절 최대 위험폭(%)", value=7.0, step=0.5)
    trail_pct = st.number_input("+3% 이후 트레일 폭(%)", value=3.0, step=0.5)

    run = st.button("▶ 9조합 대규모 검증", type="primary", use_container_width=True)

st.info(
    "기본값: 200종목·2년·돌파거래량 1.8배. "
    "E1=돌파당일 / E2=돌파종가 확인 다음날 / E3=돌파 후 재눌림. "
    "X1=-3% 고정 / X2=눌림저점 손절 / X3=+3% 확인 후 본전·트레일."
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
        "retest_days": int(retest_days),
        "retest_touch_pct": retest_touch_pct,
        "holding_days": int(holding_days),
        "max_structural_risk": max_structural_risk,
        "trail_pct": trail_pct,
    }

    try:
        listing = stock_listing()
    except Exception as e:
        st.error(f"종목 목록 조회 실패: {type(e).__name__}")
        st.exception(e)
        st.stop()

    universe = choose_universe(listing, int(universe_n))
    total = len(universe)

    status = st.empty()
    bar = st.progress(0)
    raw_rows, data_map, errors = [], {}, []

    for pos, (_, r) in enumerate(universe.iterrows(), 1):
        code = str(r["Code"]).zfill(6)
        name = r["Name"]
        market = r["Market"]

        status.info(f"1/2 신호수집 {pos}/{total} · {market} {name}")
        bar.progress(pos / (total * 2))

        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d")
            )

            if d is None or d.empty or len(d) < 30:
                errors.append(f"{name}: 데이터 부족")
                continue

            rows, prepared = find_setups(d, code, name, market, p, cut, end_ts)
            raw_rows.extend(rows)
            data_map[code] = prepared

        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")

        time.sleep(0.02)

    raw = pd.DataFrame(raw_rows)
    setups = dedup_setups(raw)

    if setups.empty:
        bar.empty()
        status.warning("현재 조건에서는 setup이 없습니다.")
        st.stop()

    results = []
    combos = [(e, x) for e in ENTRY_MODES for x in EXIT_MODES]

    for ci, (entry_mode, exit_mode) in enumerate(combos, 1):
        status.info(f"2/2 조합 계산 {ci}/9 · {entry_mode} × {exit_mode}")
        bar.progress(0.5 + ci / 18)

        for _, setup in setups.iterrows():
            code = setup["코드"]
            if code not in data_map:
                continue

            d = data_map[code]
            entry = make_entry(d, setup, entry_mode, p)
            if entry is None:
                continue

            ev = evaluate_trade(d, setup, entry, exit_mode, p)
            if ev is None:
                continue

            row = setup.drop(
                labels=["_b", "_pull_i", "_breakout_i"]
            ).to_dict()
            row.update(entry)
            row.pop("_entry_i", None)
            row.pop("_eval_i", None)
            row.update(ev)
            row["진입전략"] = entry_mode
            row["청산전략"] = exit_mode
            results.append(row)

    bar.empty()
    status.success(
        f"완료 · {total}종목 / {years}년 · setup {len(setups)}건 · 9조합 계산"
    )

    result = pd.DataFrame(results)

    if result.empty:
        st.warning("9개 조합 모두에서 실제 매매가 생성되지 않았습니다.")
        st.stop()

    summary_rows = []
    for entry_mode in ENTRY_MODES:
        for exit_mode in EXIT_MODES:
            q = result[
                (result["진입전략"] == entry_mode)
                & (result["청산전략"] == exit_mode)
            ]
            if q.empty:
                continue
            s = summarize(q)
            summary_rows.append({
                "진입": entry_mode,
                "청산": exit_mode,
                **s
            })

    comp = pd.DataFrame(summary_rows)

    # 최소 20건 우선, 부족하면 전체에서 순위.
    eligible = comp[comp["신호"] >= 20].copy()
    if eligible.empty:
        eligible = comp.copy()

    ranked = eligible.sort_values(
        ["평균수익률(%)", "손절률(%)", "+5%(%)", "신호"],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)
    ranked.insert(0, "순위", range(1, len(ranked) + 1))

    st.subheader("9조합 성과 비교")
    st.dataframe(ranked, use_container_width=True, hide_index=True)

    best = ranked.iloc[0]

    st.subheader("현재 1위")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("신호", int(best["신호"]))
    c2.metric("승률", f'{best["승률(%)"]:.1f}%')
    c3.metric("평균수익률", f'{best["평균수익률(%)"]:.2f}%')
    c4.metric("손절률", f'{best["손절률(%)"]:.1f}%')

    st.markdown(f'**진입:** {best["진입"]}  \n**청산:** {best["청산"]}')

    best_trades = result[
        (result["진입전략"] == best["진입"])
        & (result["청산전략"] == best["청산"])
    ].copy()

    st.subheader("1위 조합 실제 거래")
    cols = [
        "시장","종목명","코드","기준봉일","눌림일","돌파일",
        "진입일","진입가","진입설명","초기손절가",
        "청산일","청산가","청산사유","최종수익률(%)",
        "돌파거래량vs눌림(배)","돌파종가수익률(%)",
        "MFE(%)","MAE(%)","+3%","+5%","+7%","+10%"
    ]
    available_cols = [c for c in cols if c in best_trades.columns]
    st.dataframe(
        best_trades[available_cols].sort_values("최종수익률(%)", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("진입 방식별 평균")
    entry_rows = []
    for e in ENTRY_MODES:
        q = result[result["진입전략"] == e]
        s = summarize(q)
        entry_rows.append({"진입": e, **s})
    st.dataframe(pd.DataFrame(entry_rows), use_container_width=True, hide_index=True)

    st.subheader("청산 방식별 평균")
    exit_rows = []
    for x in EXIT_MODES:
        q = result[result["청산전략"] == x]
        s = summarize(q)
        exit_rows.append({"청산": x, **s})
    st.dataframe(pd.DataFrame(exit_rows), use_container_width=True, hide_index=True)

    st.download_button(
        "9조합 요약 CSV",
        comp.to_csv(index=False).encode("utf-8-sig"),
        "swing_v9_combo_summary.csv",
        "text/csv",
        use_container_width=True
    )

    st.download_button(
        "전체 거래 CSV",
        result.to_csv(index=False).encode("utf-8-sig"),
        "swing_v9_all_trades.csv",
        "text/csv",
        use_container_width=True
    )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption(
    "주의: 일봉에서는 장중 순서를 정확히 알 수 없습니다. "
    "같은 거래일에 손절가와 상승 목표가가 모두 관측되면 손절 우선으로 보수 처리합니다. "
    "X3 트레일링은 당일 최고가가 아니라 전일까지 확인된 최고가만 사용합니다."
)
