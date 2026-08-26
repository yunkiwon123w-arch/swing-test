import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v6", layout="wide")
st.title("📈 단기스윙 백테스트 v6")
st.caption("FDR/NAVER 실제 일봉 · 재돌파 품질·눌림 구조·추격매수 제한 A/B 검증")

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

def evaluate_trade(d, entry_i, entry_price, stop_loss_pct, holding_days):
    stop_price = entry_price * (1 + stop_loss_pct / 100.0)
    targets = [3.0, 5.0, 7.0, 10.0]
    hit = {t: False for t in targets}
    mfe, mae = 0.0, 0.0
    last_i = min(len(d)-1, entry_i + holding_days - 1)
    exit_i = None
    exit_price = None
    reason = None

    for i in range(entry_i, last_i+1):
        r = d.iloc[i]
        high = float(r["High"]); low = float(r["Low"])
        high_ret = (high/entry_price - 1)*100
        low_ret = (low/entry_price - 1)*100

        if low <= stop_price:
            mae = min(mae, stop_loss_pct)
            if high < entry_price*1.03:
                mfe = max(mfe, max(0.0, high_ret))
            exit_i = i
            exit_price = stop_price
            reason = "손절"
            break

        mfe = max(mfe, high_ret)
        mae = min(mae, low_ret)
        for t in targets:
            if high >= entry_price*(1+t/100):
                hit[t] = True

    if exit_i is None:
        exit_i = last_i
        exit_price = float(d.iloc[last_i]["Close"])
        reason = "기간종료"

    ret = (exit_price/entry_price - 1)*100
    return {
        "손절가": round(stop_price),
        "청산일": d.index[exit_i].date(),
        "청산가": round(exit_price),
        "청산사유": reason,
        "손절": reason=="손절",
        "최종수익률(%)": round(ret,2),
        "MFE(%)": round(mfe,2),
        "MAE(%)": round(mae,2),
        "+3%": hit[3.0], "+5%": hit[5.0],
        "+7%": hit[7.0], "+10%": hit[10.0],
    }

def base_ok(d, b, p):
    x = d.iloc[b]
    return (
        pd.notna(x["AVG_VOL20"])
        and x["등락률"] >= p["base_rise"]
        and x["거래대금"] >= p["value_eok"]*1e8
        and x["Volume"] >= x["AVG_VOL20"]*p["volume_mult"]
        and x["Close"] > x["MA5"]
        and x["MA5"] > d["MA5"].iloc[b-1]
        and x["MA20"] > d["MA20"].iloc[b-1]
    )

def find_signals(d, code, name, p, cut, end, mode):
    d = add_indicators(d)
    rows = []

    for b in range(21, len(d)):
        x = d.iloc[b]
        if d.index[b] < cut or d.index[b] > end:
            continue
        if not base_ok(d,b,p):
            continue

        base_low = float(x["Low"])
        base_close = float(x["Close"])
        base_high = float(x["High"])

        for k in range(2,6):
            pull_i = b+k
            breakout_i = pull_i+1
            entry_i = breakout_i+1
            if entry_i >= len(d):
                break

            pull = d.iloc[pull_i]
            br = d.iloc[breakout_i]
            ent = d.iloc[entry_i]

            # 공통 기본 눌림 조건
            pull_basic = (
                pull["Volume"] <= x["Volume"]*p["pullback_ratio"]
                and pull["Close"] > base_low
            )
            if not pull_basic:
                continue

            if mode == "A":
                # 기존 v5 방식
                breakout = br["High"] >= pull["High"]
                if not breakout:
                    continue
                entry_price = float(ent["Open"])

            else:
                # 개선 v6 방식
                # 1) 눌림이 지나치게 깊지 않게: 기준봉 종가 대비 최대 -8%
                pull_depth = (float(pull["Close"])/base_close - 1)*100
                if pull_depth < -p["max_pullback_depth"]:
                    continue

                # 2) 눌림 시 5일선 유지 또는 근접
                if float(pull["Close"]) < float(pull["MA5"])*0.98:
                    continue

                # 3) 재돌파는 장중 터치가 아니라 종가 기준
                breakout = br["Close"] >= pull["High"]
                if not breakout:
                    continue

                # 4) 재돌파 당일 거래량 회복
                if br["Volume"] < pull["Volume"]*p["breakout_vol_mult"]:
                    continue

                # 5) 재돌파 시점에도 5/20일선 상승 유지
                if not (
                    br["Close"] > br["MA5"]
                    and br["MA5"] > d["MA5"].iloc[breakout_i-1]
                    and br["MA20"] > d["MA20"].iloc[breakout_i-1]
                ):
                    continue

                # 6) 다음날 시가 추격 제한
                breakout_ref = float(pull["High"])
                next_open = float(ent["Open"])
                chase_pct = (next_open/breakout_ref - 1)*100
                if chase_pct > p["max_chase_pct"]:
                    continue

                # 7) 다음날 갭 제한: 재돌파일 종가 대비 +3% 초과면 제외
                gap_pct = (next_open/float(br["Close"]) - 1)*100
                if gap_pct > p["max_gap_pct"]:
                    continue

                entry_price = next_open

            ev = evaluate_trade(d, entry_i, entry_price, p["stop_loss"], p["holding_days"])

            row = {
                "전략": mode,
                "종목명": name,
                "코드": code,
                "기준봉일": d.index[b].date(),
                "기준봉(%)": round(float(x["등락률"]),2),
                "거래대금(억)": round(float(x["거래대금"])/1e8),
                "거래량배수": round(float(x["Volume"]/x["AVG_VOL20"]),2),
                "기준봉고가": round(base_high),
                "기준봉종가": round(base_close),
                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(float(pull["High"])),
                "눌림종가": round(float(pull["Close"])),
                "눌림깊이(%)": round((float(pull["Close"])/base_close-1)*100,2),
                "재돌파확인일": d.index[breakout_i].date(),
                "재돌파일종가": round(float(br["Close"])),
                "재돌파일거래량배수": round(float(br["Volume"]/max(float(pull["Volume"]),1)),2),
                "진입일": d.index[entry_i].date(),
                "진입시가": round(float(ent["Open"])),
                "진입고가": round(float(ent["High"])),
                "진입저가": round(float(ent["Low"])),
                "진입종가": round(float(ent["Close"])),
                "진입가": round(entry_price),
                "추격폭(%)": round((entry_price/float(pull["High"])-1)*100,2),
            }
            row.update(ev)
            rows.append(row)
            break

    return rows

def dedup(df):
    if df.empty:
        return df
    return (
        df.sort_values(["전략","코드","진입일","기준봉일"])
          .drop_duplicates(subset=["전략","코드","진입일"], keep="last")
          .reset_index(drop=True)
    )

def summary_metrics(df):
    if df.empty:
        return {
            "신호":0,"승률":0.0,"평균수익률":0.0,"손절률":0.0,
            "+3%":0.0,"+5%":0.0,"+7%":0.0,"+10%":0.0,
            "MFE":0.0,"MAE":0.0
        }
    return {
        "신호":len(df),
        "승률":(df["최종수익률(%)"]>0).mean()*100,
        "평균수익률":df["최종수익률(%)"].mean(),
        "손절률":df["손절"].mean()*100,
        "+3%":df["+3%"].mean()*100,
        "+5%":df["+5%"].mean()*100,
        "+7%":df["+7%"].mean()*100,
        "+10%":df["+10%"].mean()*100,
        "MFE":df["MFE(%)"].mean(),
        "MAE":df["MAE(%)"].mean(),
    }

with st.sidebar:
    st.header("검증 조건")
    end_date = st.date_input("종료일", date(2026,7,31))
    months = st.selectbox("기간",[1,3,6,12], index=1, format_func=lambda x:f"{x}개월")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("거래대금(억원)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)",10,100,50,5)
    stop_loss = st.number_input("손절(%)", value=-3.0, step=0.5)
    holding_days = st.number_input("관찰 거래일",3,30,10)

    st.divider()
    st.subheader("v6 개선조건")
    max_pullback_depth = st.number_input("최대 눌림 깊이(%)", value=8.0, step=1.0)
    breakout_vol_mult = st.number_input("재돌파 거래량 / 눌림 거래량", value=1.2, step=0.1)
    max_chase_pct = st.number_input("최대 추격폭(%)", value=5.0, step=0.5)
    max_gap_pct = st.number_input("재돌파일 종가 대비 다음날 최대 갭(%)", value=3.0, step=0.5)

    run = st.button("▶ A/B 백테스트 실행", type="primary", use_container_width=True)

st.info("A = 기존 v5 방식 / B = v6 개선 방식. 같은 30종목·같은 기간에서 직접 비교합니다.")

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(months=months)
    start_ts = cut - pd.Timedelta(days=70)
    fetch_end = end_ts + pd.Timedelta(days=45)

    p = {
        "base_rise":base_rise,
        "value_eok":value_eok,
        "volume_mult":volume_mult,
        "pullback_ratio":pullback_ratio/100,
        "stop_loss":stop_loss,
        "holding_days":int(holding_days),
        "max_pullback_depth":max_pullback_depth,
        "breakout_vol_mult":breakout_vol_mult,
        "max_chase_pct":max_chase_pct,
        "max_gap_pct":max_gap_pct,
    }

    bar = st.progress(0)
    status = st.empty()
    rowsA, rowsB, errors = [], [], []

    for i,(code,name) in enumerate(U.items(),1):
        status.info(f"{i}/30 · {name} 조회/검증 중")
        bar.progress(i/30)
        try:
            d = load_data(code,start_ts.strftime("%Y-%m-%d"),fetch_end.strftime("%Y-%m-%d"))
            if d is None or d.empty:
                errors.append(name+" 데이터 없음")
                continue
            rowsA.extend(find_signals(d,code,name,p,cut,end_ts,"A"))
            rowsB.extend(find_signals(d,code,name,p,cut,end_ts,"B"))
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
        time.sleep(0.05)

    bar.empty()

    A = dedup(pd.DataFrame(rowsA))
    B = dedup(pd.DataFrame(rowsB))
    status.success(f"완료 · A {len(A)}건 / B {len(B)}건")

    ma = summary_metrics(A)
    mb = summary_metrics(B)

    st.subheader("A/B 비교")
    comp = pd.DataFrame([
        {"전략":"A 기존 v5", **ma},
        {"전략":"B 개선 v6", **mb},
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("A 신호", ma["신호"])
    c2.metric("B 신호", mb["신호"])
    c3.metric("A 손절률", f'{ma["손절률"]:.1f}%')
    c4.metric("B 손절률", f'{mb["손절률"]:.1f}%')

    st.subheader("B 개선전략 거래별 결과")
    if B.empty:
        st.warning("개선 조건에서는 신호가 없습니다. 기간을 6개월/12개월로 늘리거나 개선조건을 완화해 비교하세요.")
    else:
        cols = [
            "종목명","코드","기준봉일","기준봉(%)","거래대금(억)","거래량배수",
            "눌림일","눌림고가","눌림종가","눌림깊이(%)",
            "재돌파확인일","재돌파일종가","재돌파일거래량배수",
            "진입일","진입시가","진입고가","진입저가","진입종가","진입가","추격폭(%)",
            "손절가","청산일","청산가","청산사유","손절","최종수익률(%)",
            "MFE(%)","MAE(%)","+3%","+5%","+7%","+10%"
        ]
        st.dataframe(B[cols].sort_values(["진입일","종목명"],ascending=[False,True]),
                     use_container_width=True, hide_index=True)
        st.download_button(
            "B 전략 CSV 다운로드",
            B[cols].to_csv(index=False).encode("utf-8-sig"),
            "swing_backtest_v6_B.csv",
            "text/csv",
            use_container_width=True
        )

    with st.expander("A 기존전략 거래별 결과"):
        if A.empty:
            st.write("신호 없음")
        else:
            st.dataframe(A, use_container_width=True, hide_index=True)

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption("주의: 일봉 데이터 특성상 같은 날 손절가와 목표가를 모두 터치한 경우 손절 우선으로 보수 처리합니다.")
