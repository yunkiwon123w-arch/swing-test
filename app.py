import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v5", layout="wide")
st.title("📈 단기스윙 백테스트 v5")
st.caption("FDR/NAVER 실제 일봉 · 중복 신호 제거 및 진입/청산 검산 강화")

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

TARGETS = [3, 5, 7, 10]

@st.cache_data(ttl=86400, show_spinner=False)
def load_data(code, start, end):
    return fdr.DataReader("NAVER:" + code, start, end)

def evaluate_trade(d, entry_i, entry_price, stop_loss_pct, holding_days):
    stop_price = entry_price * (1 + stop_loss_pct / 100.0)
    targets = [3.0, 5.0, 7.0, 10.0]
    target_hit = {t: False for t in targets}

    mfe = 0.0
    mae = 0.0
    exit_price = None
    exit_i = None
    exit_reason = None

    last_i = min(len(d) - 1, entry_i + holding_days - 1)

    for i in range(entry_i, last_i + 1):
        day = d.iloc[i]
        high = float(day["High"])
        low = float(day["Low"])

        high_ret = (high / entry_price - 1) * 100.0
        low_ret = (low / entry_price - 1) * 100.0

        stop_hit = low <= stop_price

        if stop_hit:
            # 손절일에는 손절가 이후 저가는 MAE에서 제외
            mae = min(mae, stop_loss_pct)
            # 같은 날 +3% 이상과 손절이 모두 터치되면 보수적으로 손절 우선
            if high < entry_price * 1.03:
                mfe = max(mfe, max(0.0, high_ret))
            exit_price = stop_price
            exit_i = i
            exit_reason = "손절"
            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)

        for t in targets:
            if high >= entry_price * (1 + t / 100.0):
                target_hit[t] = True

    if exit_i is None:
        exit_i = last_i
        exit_price = float(d.iloc[last_i]["Close"])
        exit_reason = "기간종료"

    ret = (exit_price / entry_price - 1) * 100.0

    return {
        "청산일": d.index[exit_i].date(),
        "청산가": round(exit_price),
        "청산사유": exit_reason,
        "손절": exit_reason == "손절",
        "최종수익률(%)": round(ret, 2),
        "MFE(%)": round(mfe, 2),
        "MAE(%)": round(mae, 2),
        "+3%": target_hit[3.0],
        "+5%": target_hit[5.0],
        "+7%": target_hit[7.0],
        "+10%": target_hit[10.0],
        "손절가": round(stop_price),
    }

def find_signals(d, code, name, p, cut, end):
    d = d.copy()
    d["거래대금"] = d["Close"] * d["Volume"]
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["AVG_VOL20"] = d["Volume"].shift(1).rolling(20).mean()
    d["등락률"] = d["Close"].pct_change() * 100
    rows = []

    for b in range(21, len(d)):
        x = d.iloc[b]
        if d.index[b] < cut or d.index[b] > end:
            continue
        if pd.isna(x["AVG_VOL20"]):
            continue

        base_ok = (
            x["등락률"] >= p["base_rise"]
            and x["거래대금"] >= p["value_eok"] * 1e8
            and x["Volume"] >= x["AVG_VOL20"] * p["volume_mult"]
            and x["Close"] > x["MA5"]
            and x["MA5"] > d["MA5"].iloc[b - 1]
            and x["MA20"] > d["MA20"].iloc[b - 1]
        )
        if not base_ok:
            continue

        for k in range(2, 6):
            pull_i = b + k
            breakout_i = pull_i + 1
            entry_i = breakout_i + 1
            if entry_i >= len(d):
                break

            pull = d.iloc[pull_i]
            breakout_day = d.iloc[breakout_i]
            ent = d.iloc[entry_i]

            pull_ok = (
                pull["Volume"] <= x["Volume"] * p["pullback_ratio"]
                and pull["Close"] > x["Low"]
            )
            breakout = breakout_day["High"] >= pull["High"]

            if not (pull_ok and breakout):
                continue

            entry_price = float(ent["Open"])
            ev = evaluate_trade(
                d, entry_i, entry_price, p["stop_loss"], p["holding_days"]
            )

            row = {
                "종목명": name,
                "코드": code,
                "기준봉일": d.index[b].date(),
                "기준봉(%)": round(float(x["등락률"]), 2),
                "거래대금(억)": round(float(x["거래대금"]) / 1e8),
                "거래량배수": round(float(x["Volume"] / x["AVG_VOL20"]), 2),

                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(float(pull["High"])),
                "눌림종가": round(float(pull["Close"])),

                "재돌파확인일": d.index[breakout_i].date(),
                "재돌파확인가": round(float(pull["High"])),
                "재돌파일고가": round(float(breakout_day["High"])),
                "재돌파일종가": round(float(breakout_day["Close"])),

                "진입일": d.index[entry_i].date(),
                "진입시가": round(float(ent["Open"])),
                "진입고가": round(float(ent["High"])),
                "진입저가": round(float(ent["Low"])),
                "진입종가": round(float(ent["Close"])),
                "진입가": round(entry_price),
            }
            row.update(ev)
            rows.append(row)
            break

    return rows

with st.sidebar:
    st.header("검증 조건")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    months = st.selectbox("기간", [1, 3, 6, 12], index=1,
                          format_func=lambda x: f"{x}개월")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("거래대금(억원)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)
    stop_loss = st.number_input("손절(%)", value=-3.0, step=0.5)
    holding_days = st.number_input("관찰 거래일", 3, 30, 10)
    run = st.button("▶ 백테스트 실행", type="primary", use_container_width=True)

st.info("v5: 동일 종목+동일 진입일 중복 신호를 제거하고, 재돌파 가격 및 진입일 OHLC를 모두 표시합니다.")

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(months=months)
    start_ts = cut - pd.Timedelta(days=70)
    fetch_end = end_ts + pd.Timedelta(days=45)

    p = {
        "base_rise": base_rise,
        "value_eok": value_eok,
        "volume_mult": volume_mult,
        "pullback_ratio": pullback_ratio / 100,
        "stop_loss": stop_loss,
        "holding_days": int(holding_days),
    }

    progress = st.progress(0)
    status = st.empty()
    rows, errors = [], []

    for i, (code, name) in enumerate(U.items(), 1):
        status.info(f"{i}/30 · {name} 조회/검증 중")
        progress.progress(i / 30)
        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d"),
            )
            if d is not None and not d.empty:
                rows.extend(find_signals(d, code, name, p, cut, end_ts))
            else:
                errors.append(name + " 데이터 없음")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")

        time.sleep(0.05)

    progress.empty()

    raw_count = len(rows)
    t = pd.DataFrame(rows)

    if not t.empty:
        # 동일 종목 + 동일 진입일이면 하나의 실제 매매로 간주.
        # 여러 기준봉이 같은 거래를 만든 경우 가장 최근 기준봉만 남김.
        t = (
            t.sort_values(["코드", "진입일", "기준봉일"])
             .drop_duplicates(subset=["코드", "진입일"], keep="last")
             .reset_index(drop=True)
        )

    dedup_count = len(t)
    status.success(f"완료 · 원신호 {raw_count}건 → 중복 제거 후 실제매매 {dedup_count}건")

    c1, c2 = st.columns(2)
    c1.metric("원신호", raw_count)
    c2.metric("중복 제거 후 실제매매", dedup_count)

    if t.empty:
        st.warning("현재 조건에서는 매매신호가 없습니다.")
    else:
        win_rate = (t["최종수익률(%)"] > 0).mean() * 100
        avg_ret = t["최종수익률(%)"].mean()
        avg_win = t.loc[t["최종수익률(%)"] > 0, "최종수익률(%)"].mean()
        avg_loss = t.loc[t["최종수익률(%)"] <= 0, "최종수익률(%)"].mean()
        payoff = (
            abs(avg_win / avg_loss)
            if pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0
            else float("nan")
        )

        a, b, c, dcol = st.columns(4)
        a.metric("실제매매", len(t))
        b.metric("승률", f"{win_rate:.1f}%")
        c.metric("평균수익률", f"{avg_ret:.2f}%")
        dcol.metric("손절률", f'{t["손절"].mean()*100:.1f}%')

        a, b, c, dcol = st.columns(4)
        a.metric("+3% 도달", f'{t["+3%"].mean()*100:.1f}%')
        b.metric("+5% 도달", f'{t["+5%"].mean()*100:.1f}%')
        c.metric("+7% 도달", f'{t["+7%"].mean()*100:.1f}%')
        dcol.metric("+10% 도달", f'{t["+10%"].mean()*100:.1f}%')

        a, b, c, dcol = st.columns(4)
        a.metric("평균 MFE", f'{t["MFE(%)"].mean():.2f}%')
        b.metric("평균 MAE", f'{t["MAE(%)"].mean():.2f}%')
        c.metric("평균 이익", "-" if pd.isna(avg_win) else f"{avg_win:.2f}%")
        dcol.metric("손익비", "-" if pd.isna(payoff) else f"{payoff:.2f}")

        st.subheader("거래별 검산 결과")
        st.caption(
            "기준봉 → 눌림일/눌림고가 → 재돌파확인일/재돌파가격 "
            "→ 진입일 OHLC/진입가 → 손절가 → 청산일/청산가/사유 → MFE/MAE"
        )

        display_cols = [
            "종목명","코드","기준봉일","기준봉(%)","거래대금(억)","거래량배수",
            "눌림일","눌림고가","재돌파확인일","재돌파확인가",
            "재돌파일고가","재돌파일종가",
            "진입일","진입시가","진입고가","진입저가","진입종가","진입가",
            "손절가","청산일","청산가","청산사유","손절","최종수익률(%)",
            "MFE(%)","MAE(%)","+3%","+5%","+7%","+10%"
        ]

        st.dataframe(
            t[display_cols].sort_values(["진입일", "종목명"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "CSV 다운로드",
            t[display_cols].to_csv(index=False).encode("utf-8-sig"),
            "swing_backtest_v5.csv",
            "text/csv",
            use_container_width=True,
        )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption("주의: 진입 후 같은 거래일에 손절가와 목표가를 모두 터치한 경우 일봉만으로 장중 순서를 알 수 없어 손절 우선으로 보수 처리합니다.")
