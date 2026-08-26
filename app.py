import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v3", layout="wide")
st.title("📈 단기스윙 백테스트 v3")
st.caption("FDR/NAVER 실제 일봉 · 재돌파 확인 다음 거래일 시가 진입 방식")

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

def evaluate_trade(d, entry_i, entry_price, stop_pct, holding_days):
    stop_price = entry_price * (1 + stop_pct / 100)
    last_i = min(len(d) - 1, entry_i + holding_days - 1)
    window = d.iloc[entry_i:last_i + 1]

    mfe = 0.0
    mae = 0.0
    hit = {t: False for t in TARGETS}
    exit_reason = "기간종료"
    exit_price = float(window.iloc[-1]["Close"])
    exit_date = window.index[-1]
    stopped = False

    for idx, r in window.iterrows():
        low_ret = (float(r["Low"]) / entry_price - 1) * 100
        high_ret = (float(r["High"]) / entry_price - 1) * 100
        mae = min(mae, low_ret)
        mfe = max(mfe, high_ret)

        # 일봉 데이터에서는 장중 순서를 알 수 없으므로
        # 같은 날 손절가와 목표가가 모두 닿으면 보수적으로 손절 우선.
        if float(r["Low"]) <= stop_price:
            stopped = True
            exit_reason = "손절"
            exit_price = stop_price
            exit_date = idx
            break

        for t in TARGETS:
            if not hit[t] and float(r["High"]) >= entry_price * (1 + t / 100):
                hit[t] = True

    final_ret = (exit_price / entry_price - 1) * 100
    return {
        "손절": stopped,
        "+3%": hit[3], "+5%": hit[5], "+7%": hit[7], "+10%": hit[10],
        "MFE(%)": round(mfe, 2),
        "MAE(%)": round(mae, 2),
        "청산일": exit_date.date(),
        "청산사유": exit_reason,
        "최종수익률(%)": round(final_ret, 2),
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

        # 기준봉 후 2~5거래일 중 눌림을 찾는다.
        # 눌림 다음 거래일의 고가가 눌림 고가를 돌파하면 '재돌파 확인일'로 기록한다.
        # 실제 매수는 그 다음 거래일 시가에서 한다.
        # 이렇게 하면 재돌파 확인일의 저가가 진입 전인지 후인지 알 수 없는 일봉 한계를 피할 수 있다.
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

            # 재돌파 확인 다음 거래일 시가 매수
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
                "재돌파확인일": d.index[breakout_i].date(),
                "진입일": d.index[entry_i].date(),
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

st.info("v3: 눌림 고가 재돌파는 확인 신호로만 사용하고, 실제 진입은 다음 거래일 시가로 계산합니다.")

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
    status.success(f"완료 · 매매신호 {len(rows)}건")
    t = pd.DataFrame(rows)

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
        a.metric("신호", len(t))
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

        st.subheader("거래별 결과")
        st.dataframe(
            t.sort_values(["진입일", "종목명"], ascending=[False, True]),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "CSV 다운로드",
            t.to_csv(index=False).encode("utf-8-sig"),
            "swing_backtest_v2.csv",
            "text/csv",
            use_container_width=True,
        )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption("주의: 진입 후 같은 거래일에 손절가와 목표가를 모두 터치한 경우에는 일봉만으로 장중 순서를 알 수 없어 손절 우선으로 보수 처리합니다.")
