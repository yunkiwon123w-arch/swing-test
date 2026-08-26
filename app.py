import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v7", layout="wide")
st.title("📈 단기스윙 백테스트 v7")
st.caption("FDR/NAVER 실제 일봉 · 진입방식 A/B/C 비교")

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

def evaluate_trade(d, start_i, entry_price, stop_loss_pct, holding_days):
    stop_price = entry_price * (1 + stop_loss_pct / 100.0)
    targets = [3.0, 5.0, 7.0, 10.0]
    hit = {t: False for t in targets}

    mfe, mae = 0.0, 0.0
    last_i = min(len(d) - 1, start_i + holding_days - 1)
    exit_i, exit_price, reason = None, None, None

    for i in range(start_i, last_i + 1):
        r = d.iloc[i]
        high = float(r["High"])
        low = float(r["Low"])

        high_ret = (high / entry_price - 1) * 100
        low_ret = (low / entry_price - 1) * 100

        if low <= stop_price:
            mae = min(mae, stop_loss_pct)
            if high < entry_price * 1.03:
                mfe = max(mfe, max(0.0, high_ret))
            exit_i = i
            exit_price = stop_price
            reason = "손절"
            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)

        for t in targets:
            if high >= entry_price * (1 + t / 100):
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

def common_row(d, code, name, b, pull_i, signal_i, entry_i, entry_price, strategy, note):
    x = d.iloc[b]
    pull = d.iloc[pull_i]
    sig = d.iloc[signal_i]
    ent = d.iloc[entry_i] if entry_i < len(d) else sig

    return {
        "전략": strategy,
        "종목명": name,
        "코드": code,
        "기준봉일": d.index[b].date(),
        "기준봉(%)": round(float(x["등락률"]), 2),
        "거래대금(억)": round(float(x["거래대금"]) / 1e8),
        "거래량배수": round(float(x["Volume"] / x["AVG_VOL20"]), 2),
        "눌림일": d.index[pull_i].date(),
        "눌림고가": round(float(pull["High"])),
        "눌림종가": round(float(pull["Close"])),
        "신호일": d.index[signal_i].date(),
        "신호종가": round(float(sig["Close"])),
        "진입일": d.index[entry_i].date(),
        "진입가": round(entry_price),
        "진입시가": round(float(ent["Open"])),
        "진입고가": round(float(ent["High"])),
        "진입저가": round(float(ent["Low"])),
        "진입종가": round(float(ent["Close"])),
        "진입설명": note,
    }

def find_entries(d, code, name, p, cut, end):
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
            if pull_i + 2 >= len(d):
                break

            pull = d.iloc[pull_i]
            pull_ok = (
                pull["Volume"] <= x["Volume"] * p["pullback_ratio"]
                and pull["Close"] > x["Low"]
            )
            if not pull_ok:
                continue

            # -----------------------------------------
            # A: 재돌파 확인 다음날 시가 진입
            # -----------------------------------------
            breakout_i = pull_i + 1
            entry_i_A = breakout_i + 1
            br = d.iloc[breakout_i]

            if br["High"] >= pull["High"] and entry_i_A < len(d):
                entry_price_A = float(d.iloc[entry_i_A]["Open"])
                row = common_row(
                    d, code, name, b, pull_i, breakout_i, entry_i_A,
                    entry_price_A, "A", "재돌파 확인 다음날 시가"
                )
                row.update(
                    evaluate_trade(
                        d, entry_i_A, entry_price_A,
                        p["stop_loss"], p["holding_days"]
                    )
                )
                rows.append(row)

            # -----------------------------------------
            # B: 눌림고가 돌파 당일 진입
            # 일봉 한계 때문에 돌파 당일 저가/고가는 성과평가에서 제외.
            # 실제 성과 평가는 다음 거래일부터 시작.
            # -----------------------------------------
            if br["High"] >= pull["High"]:
                entry_price_B = float(pull["High"])
                eval_start_B = breakout_i + 1
                if eval_start_B < len(d):
                    row = common_row(
                        d, code, name, b, pull_i, breakout_i, breakout_i,
                        entry_price_B, "B", "눌림고가 돌파 당일 진입(성과평가는 다음날부터)"
                    )
                    row.update(
                        evaluate_trade(
                            d, eval_start_B, entry_price_B,
                            p["stop_loss"], p["holding_days"]
                        )
                    )
                    rows.append(row)

            # -----------------------------------------
            # C: 5일선 부근 눌림 → 반등 확인 → 다음날 시가 진입
            # -----------------------------------------
            ma5 = float(pull["MA5"]) if pd.notna(pull["MA5"]) else None
            if ma5:
                near_ma5 = (
                    float(pull["Low"]) <= ma5 * (1 + p["ma5_touch_pct"] / 100)
                    and float(pull["Close"]) >= ma5 * (1 - p["ma5_below_pct"] / 100)
                )
            else:
                near_ma5 = False

            if near_ma5:
                # 눌림 후 최대 3거래일 내 반등 확인
                for j in range(pull_i + 1, min(len(d) - 1, pull_i + 4)):
                    sig = d.iloc[j]
                    prev = d.iloc[j - 1]

                    rebound = (
                        sig["Close"] > sig["MA5"]
                        and sig["Close"] > sig["Open"]
                        and sig["Close"] > prev["Close"]
                        and sig["MA5"] >= d["MA5"].iloc[j - 1] * 0.995
                    )
                    if not rebound:
                        continue

                    entry_i_C = j + 1
                    if entry_i_C >= len(d):
                        break

                    entry_price_C = float(d.iloc[entry_i_C]["Open"])
                    row = common_row(
                        d, code, name, b, pull_i, j, entry_i_C,
                        entry_price_C, "C", "5일선 부근 눌림 후 반등 확인 다음날 시가"
                    )
                    row.update(
                        evaluate_trade(
                            d, entry_i_C, entry_price_C,
                            p["stop_loss"], p["holding_days"]
                        )
                    )
                    rows.append(row)
                    break

            break

    return rows

def dedup(df):
    if df.empty:
        return df
    return (
        df.sort_values(["전략", "코드", "진입일", "기준봉일"])
          .drop_duplicates(subset=["전략", "코드", "진입일"], keep="last")
          .reset_index(drop=True)
    )

def summary(df):
    if df.empty:
        return {
            "신호": 0, "승률(%)": 0.0, "평균수익률(%)": 0.0,
            "손절률(%)": 0.0, "+3%(%)": 0.0, "+5%(%)": 0.0,
            "+7%(%)": 0.0, "+10%(%)": 0.0, "평균MFE(%)": 0.0,
            "평균MAE(%)": 0.0
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

with st.sidebar:
    st.header("검증 조건")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    months = st.selectbox(
        "기간", [3, 6, 12], index=2,
        format_func=lambda x: f"{x}개월"
    )
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("거래대금(억원)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)
    stop_loss = st.number_input("손절(%)", value=-3.0, step=0.5)
    holding_days = st.number_input("관찰 거래일", 3, 30, 10)

    st.divider()
    st.subheader("C 전략 조건")
    ma5_touch_pct = st.number_input(
        "5일선 위쪽 허용폭(%)", value=2.0, step=0.5
    )
    ma5_below_pct = st.number_input(
        "5일선 아래 허용폭(%)", value=3.0, step=0.5
    )

    run = st.button(
        "▶ A/B/C 백테스트 실행",
        type="primary",
        use_container_width=True
    )

st.info(
    "A=재돌파 다음날 시가 / "
    "B=눌림고가 돌파 당일 진입(일봉 한계로 성과평가는 다음날부터) / "
    "C=5일선 부근 반등 확인 다음날 시가"
)

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
        "ma5_touch_pct": ma5_touch_pct,
        "ma5_below_pct": ma5_below_pct,
    }

    bar = st.progress(0)
    status = st.empty()
    rows, errors = [], []

    for i, (code, name) in enumerate(U.items(), 1):
        status.info(f"{i}/30 · {name} 조회/검증 중")
        bar.progress(i / 30)

        try:
            d = load_data(
                code,
                start_ts.strftime("%Y-%m-%d"),
                fetch_end.strftime("%Y-%m-%d")
            )
            if d is None or d.empty:
                errors.append(name + " 데이터 없음")
                continue

            rows.extend(find_entries(d, code, name, p, cut, end_ts))

        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")

        time.sleep(0.05)

    bar.empty()
    status.success("A/B/C 비교 완료")

    t = dedup(pd.DataFrame(rows))

    A = t[t["전략"] == "A"].copy() if not t.empty else pd.DataFrame()
    B = t[t["전략"] == "B"].copy() if not t.empty else pd.DataFrame()
    C = t[t["전략"] == "C"].copy() if not t.empty else pd.DataFrame()

    comp = pd.DataFrame([
        {"전략": "A 재돌파 다음날 시가", **summary(A)},
        {"전략": "B 돌파 당일", **summary(B)},
        {"전략": "C 5일선 반등", **summary(C)},
    ])

    st.subheader("A/B/C 비교")
    st.dataframe(comp, use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("A 신호", len(A))
    c2.metric("B 신호", len(B))
    c3.metric("C 신호", len(C))

    for label, df in [
        ("A 재돌파 다음날 시가", A),
        ("B 돌파 당일", B),
        ("C 5일선 반등", C)
    ]:
        with st.expander(f"{label} 거래별 결과 ({len(df)}건)"):
            if df.empty:
                st.write("신호 없음")
            else:
                cols = [
                    "종목명","코드","기준봉일","기준봉(%)","거래대금(억)",
                    "거래량배수","눌림일","눌림고가","눌림종가",
                    "신호일","신호종가","진입일","진입가","진입설명",
                    "손절가","청산일","청산가","청산사유","손절",
                    "최종수익률(%)","MFE(%)","MAE(%)",
                    "+3%","+5%","+7%","+10%"
                ]
                st.dataframe(
                    df[cols].sort_values(
                        ["진입일","종목명"],
                        ascending=[False, True]
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    if not t.empty:
        st.download_button(
            "전체 A/B/C 결과 CSV 다운로드",
            t.to_csv(index=False).encode("utf-8-sig"),
            "swing_backtest_v7_abc.csv",
            "text/csv",
            use_container_width=True
        )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption(
    "주의: B 전략은 일봉 데이터만으로 돌파 당일의 장중 순서를 알 수 없어 "
    "돌파 당일 손익은 계산하지 않고 다음 거래일부터 성과를 평가합니다."
)
