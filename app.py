import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v7.1", layout="wide")
st.title("📈 단기스윙 백테스트 v7.1")
st.caption("B 전략 집중검증 · 돌파 당일 보수 체결 + 손절폭/보유기간 16조합 비교")

U = {
"005930":"삼성전자","000660":"SK하이닉스","005380":"현대차","000270":"기아",
"035420":"NAVER","035720":"카카오","051910":"LG화학","373220":"LG에너지솔루션",
"006400":"삼성SDI","207940":"삼성바이오로직스","068270":"셀트리온","105560":"KB금융",
"055550":"신한지주","086790":"하나금융지주","066570":"LG전자","034020":"두산에너빌리티",
"042700":"한미반도체","010140":"삼성중공업","329180":"HD현대중공업","012450":"한화에어로스페이스",
"272210":"한화시스템","298040":"효성중공업","267260":"HD현대일렉트릭","000100":"유한양행",
"196170":"알테오젠","009150":"삼성전기","005490":"POSCO홀딩스","028260":"삼성물산",
"012330":"현대모비스","009540":"HD한국조선해양"
}

STOP_LIST = [-2.0, -3.0, -4.0, -5.0]
HOLD_LIST = [5, 10, 15, 20]
TARGETS = [3.0, 5.0, 7.0, 10.0]

@st.cache_data(ttl=86400, show_spinner=False)
def load_data(code, start, end):
    return fdr.DataReader("NAVER:" + code, start, end)

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
        and x["MA5"] > d["MA5"].iloc[b-1]
        and x["MA20"] > d["MA20"].iloc[b-1]
    )

def find_b_signals(d, code, name, p, cut, end):
    """B 전략 원신호 생성:
    기준봉 -> 2~5일 눌림 -> 다음 거래일에 눌림고가 돌파 -> 돌파가 진입
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

            entry_price = float(pull["High"])
            breakout_ok = float(br["High"]) >= entry_price
            if not breakout_ok:
                continue

            rows.append({
                "종목명": name,
                "코드": code,
                "기준봉일": d.index[b].date(),
                "기준봉(%)": round(float(x["등락률"]), 2),
                "거래대금(억)": round(float(x["거래대금"]) / 1e8),
                "거래량배수": round(float(x["Volume"] / x["AVG_VOL20"]), 2),
                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(entry_price),
                "눌림종가": round(float(pull["Close"])),
                "돌파일": d.index[breakout_i].date(),
                "돌파시가": round(float(br["Open"])),
                "돌파고가": round(float(br["High"])),
                "돌파저가": round(float(br["Low"])),
                "돌파종가": round(float(br["Close"])),
                "진입가": round(entry_price),
                "_breakout_i": breakout_i,
            })
            break

    return rows, d

def evaluate_b(d, breakout_i, entry_price, stop_pct, hold_days):
    """보수적 B 체결 모델.
    - 돌파 당일 눌림고가(entry_price)에서 진입했다고 가정
    - 일봉상 장중 순서를 알 수 없으므로 돌파 당일 저가가 손절가 이하이면 당일 손절
    - 돌파 당일 손절이 없으면 당일 목표가 터치는 인정
    - 이후에도 같은 날 손절/목표 동시 터치면 손절 우선
    """
    stop_price = entry_price * (1 + stop_pct / 100.0)
    hit = {t: False for t in TARGETS}
    mfe, mae = 0.0, 0.0

    last_i = min(len(d)-1, breakout_i + hold_days - 1)
    exit_i = None
    exit_price = None
    reason = None

    for i in range(breakout_i, last_i + 1):
        r = d.iloc[i]
        high = float(r["High"])
        low = float(r["Low"])

        high_ret = (high / entry_price - 1) * 100
        low_ret = (low / entry_price - 1) * 100

        stop_hit = low <= stop_price

        if stop_hit:
            # 보수적: 같은 날 손절/목표가 모두 존재할 수 있어도 손절 우선
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
        "손절폭(%)": stop_pct,
        "보유기간": hold_days,
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

def dedup_signals(df):
    if df.empty:
        return df
    # 같은 종목·같은 돌파일은 실제로 하나의 매매로 간주.
    return (
        df.sort_values(["코드", "돌파일", "기준봉일"])
          .drop_duplicates(subset=["코드", "돌파일"], keep="last")
          .reset_index(drop=True)
    )

def summarize(df):
    if df.empty:
        return {}
    wins = df[df["최종수익률(%)"] > 0]
    losses = df[df["최종수익률(%)"] <= 0]
    avg_win = wins["최종수익률(%)"].mean() if not wins.empty else float("nan")
    avg_loss = losses["최종수익률(%)"].mean() if not losses.empty else float("nan")
    payoff = abs(avg_win / avg_loss) if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0 else float("nan")
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
        "평균이익(%)": None if pd.isna(avg_win) else round(avg_win, 2),
        "평균손실(%)": None if pd.isna(avg_loss) else round(avg_loss, 2),
        "손익비": None if pd.isna(payoff) else round(payoff, 2),
    }

with st.sidebar:
    st.header("기준 신호 조건")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    months = st.selectbox("기간", [3, 6, 12], index=2, format_func=lambda x: f"{x}개월")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("거래대금(억원)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)

    run = st.button("▶ 16조합 백테스트 실행", type="primary", use_container_width=True)

st.info(
    "B 전략 고정: 눌림고가 돌파 당일 진입. "
    "돌파 당일 저가가 손절가 이하이면 보수적으로 당일 손절 처리합니다."
)
st.write("비교 조합: 손절 **-2 / -3 / -4 / -5%** × 보유기간 **5 / 10 / 15 / 20거래일**")

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(months=months)
    start_ts = cut - pd.Timedelta(days=70)
    fetch_end = end_ts + pd.Timedelta(days=60)

    p = {
        "base_rise": base_rise,
        "value_eok": value_eok,
        "volume_mult": volume_mult,
        "pullback_ratio": pullback_ratio / 100,
    }

    bar = st.progress(0)
    status = st.empty()
    signal_rows = []
    data_map = {}
    errors = []

    for i, (code, name) in enumerate(U.items(), 1):
        status.info(f"1/2 신호 수집 {i}/30 · {name}")
        bar.progress(i / 60)

        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d")
            )
            if d is None or d.empty:
                errors.append(name + " 데이터 없음")
                continue

            rows, prepared = find_b_signals(d, code, name, p, cut, end_ts)
            signal_rows.extend(rows)
            data_map[code] = prepared

        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")

        time.sleep(0.05)

    raw = pd.DataFrame(signal_rows)
    signals = dedup_signals(raw)

    if signals.empty:
        bar.empty()
        status.warning("신호 없음")
        st.stop()

    result_rows = []
    combos = [(s, h) for s in STOP_LIST for h in HOLD_LIST]

    for ci, (stop_pct, hold_days) in enumerate(combos, 1):
        status.info(f"2/2 조합 계산 {ci}/16 · 손절 {stop_pct:.0f}% / {hold_days}일")
        bar.progress((30 + ci * (30/16)) / 60)

        for _, sig in signals.iterrows():
            d = data_map[sig["코드"]]
            ev = evaluate_b(
                d,
                int(sig["_breakout_i"]),
                float(sig["진입가"]),
                stop_pct,
                hold_days
            )
            row = sig.drop(labels=["_breakout_i"]).to_dict()
            row.update(ev)
            result_rows.append(row)

    bar.progress(1.0)
    bar.empty()
    status.success(
        f"완료 · 원신호 {len(raw)}건 → 중복 제거 실제신호 {len(signals)}건 · 16조합 계산"
    )

    result = pd.DataFrame(result_rows)

    summary_rows = []
    for stop_pct in STOP_LIST:
        for hold_days in HOLD_LIST:
            sub = result[
                (result["손절폭(%)"] == stop_pct) &
                (result["보유기간"] == hold_days)
            ]
            sm = summarize(sub)
            summary_rows.append({
                "손절폭": f"{stop_pct:.0f}%",
                "보유기간": f"{hold_days}일",
                **sm
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["평균수익률(%)", "손절률(%)", "+5%(%)"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    st.subheader("16조합 성과 순위")
    st.caption("기본 정렬: 평균수익률 높은 순 → 손절률 낮은 순 → +5% 도달률 높은 순")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    best = summary_df.iloc[0]
    st.subheader("현재 1위 조합")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("손절폭", best["손절폭"])
    c2.metric("보유기간", best["보유기간"])
    c3.metric("평균수익률", f'{best["평균수익률(%)"]:.2f}%')
    c4.metric("손절률", f'{best["손절률(%)"]:.1f}%')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("승률", f'{best["승률(%)"]:.1f}%')
    c2.metric("+5% 도달", f'{best["+5%(%)"]:.1f}%')
    c3.metric("평균 MFE", f'{best["평균MFE(%)"]:.2f}%')
    c4.metric("평균 MAE", f'{best["평균MAE(%)"]:.2f}%')

    selected_stop = st.selectbox("상세 손절폭", STOP_LIST, index=1, format_func=lambda x: f"{x:.0f}%")
    selected_hold = st.selectbox("상세 보유기간", HOLD_LIST, index=1, format_func=lambda x: f"{x}일")

    detail = result[
        (result["손절폭(%)"] == selected_stop) &
        (result["보유기간"] == selected_hold)
    ].copy()

    st.subheader("선택 조합 거래별 결과")
    cols = [
        "종목명","코드","기준봉일","기준봉(%)","거래대금(억)","거래량배수",
        "눌림일","눌림고가","눌림종가",
        "돌파일","돌파시가","돌파고가","돌파저가","돌파종가","진입가",
        "손절폭(%)","보유기간","손절가","청산일","청산가","청산사유","손절",
        "최종수익률(%)","MFE(%)","MAE(%)","+3%","+5%","+7%","+10%"
    ]
    st.dataframe(
        detail[cols].sort_values(["돌파일","종목명"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "16조합 요약 CSV 다운로드",
        summary_df.to_csv(index=False).encode("utf-8-sig"),
        "swing_v7_1_summary.csv",
        "text/csv",
        use_container_width=True
    )

    st.download_button(
        "전체 거래 결과 CSV 다운로드",
        result.drop(columns=[], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
        "swing_v7_1_all_trades.csv",
        "text/csv",
        use_container_width=True
    )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption(
    "주의: B 전략의 돌파 당일에는 일봉만으로 장중 순서를 알 수 없으므로 "
    "돌파 당일 저가가 손절가 이하이면 손절을 먼저 맞은 것으로 보수 처리합니다."
)
