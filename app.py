import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v7.2", layout="wide")
st.title("📊 단기스윙 v7.2 · 성공/실패 패턴 분석기")
st.caption("B 진입 + 손절 -3% + 최대 5거래일 고정 · 성공군/실패군의 차이를 데이터로 분석")

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
            base_high = float(x["High"])
            base_low = float(x["Low"])
            base_range = max(base_high - base_low, 1.0)
            br_range = max(float(br["High"]) - float(br["Low"]), 1.0)

            rows.append({
                "종목명": name,
                "코드": code,
                "기준봉일": d.index[b].date(),
                "기준봉상승률(%)": round(float(x["등락률"]), 2),
                "거래대금(억)": round(float(x["거래대금"]) / 1e8, 0),
                "기준봉거래량배수": round(float(x["Volume"] / x["AVG_VOL20"]), 2),
                "기준봉종가위치(%)": round((base_close - base_low) / base_range * 100, 2),
                "기준봉5일선이격(%)": round((base_close / float(x["MA5"]) - 1) * 100, 2),
                "기준봉20일선이격(%)": round((base_close / float(x["MA20"]) - 1) * 100, 2),

                "눌림일": d.index[pull_i].date(),
                "눌림고가": round(entry_price),
                "눌림깊이(%)": round((float(pull["Close"]) / base_close - 1) * 100, 2),
                "눌림거래량비율(%)": round(float(pull["Volume"]) / float(x["Volume"]) * 100, 2),
                "눌림5일선이격(%)": round((float(pull["Close"]) / float(pull["MA5"]) - 1) * 100, 2),

                "돌파일": d.index[breakout_i].date(),
                "돌파시가": round(float(br["Open"])),
                "돌파고가": round(float(br["High"])),
                "돌파저가": round(float(br["Low"])),
                "돌파종가": round(float(br["Close"])),
                "돌파종가위치(%)": round((float(br["Close"]) - float(br["Low"])) / br_range * 100, 2),
                "돌파종가수익률(%)": round((float(br["Close"]) / entry_price - 1) * 100, 2),
                "돌파거래량vs눌림(배)": round(
                    float(br["Volume"]) / max(float(pull["Volume"]), 1.0), 2
                ),
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
    st.header("신호 조건")
    end_date = st.date_input("종료일", date(2026, 7, 31))
    months = st.selectbox("기간", [3, 6, 12], index=2, format_func=lambda x: f"{x}개월")
    base_rise = st.number_input("기준봉 상승률(%)", value=10.0, step=0.5)
    value_eok = st.number_input("거래대금(억원)", value=1000, step=100)
    volume_mult = st.number_input("20일 거래량 배수", value=2.0, step=0.5)
    pullback_ratio = st.slider("눌림 거래량 비율(%)", 10, 100, 50, 5)
    run = st.button("▶ 패턴 분석 실행", type="primary", use_container_width=True)

STOP = -3.0
HOLD = 5

st.info("전략 고정: B(눌림고가 돌파 당일 진입) · 손절 -3% · 최대 5거래일")
st.caption("목적: 조건을 더 넣기 전에 성공군과 실패군에서 실제 차이가 나는 변수를 찾습니다.")

if run:
    end_ts = pd.Timestamp(end_date)
    cut = end_ts - pd.DateOffset(months=months)
    start_ts = cut - pd.Timedelta(days=70)
    fetch_end = end_ts + pd.Timedelta(days=30)

    p = {
        "base_rise": base_rise,
        "value_eok": value_eok,
        "volume_mult": volume_mult,
        "pullback_ratio": pullback_ratio / 100,
    }

    progress = st.progress(0)
    status = st.empty()
    raw_rows, errors, data_map = [], [], {}

    for i, (code, name) in enumerate(U.items(), 1):
        status.info(f"{i}/30 · {name} 조회/분석 중")
        progress.progress(i / 30)
        try:
            d = load_data(code, start_ts.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d"))
            if d is None or d.empty:
                errors.append(name + " 데이터 없음")
                continue
            rows, prepared = find_b_signals(d, code, name, p, cut, end_ts)
            raw_rows.extend(rows)
            data_map[code] = prepared
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}")
        time.sleep(0.05)

    raw = pd.DataFrame(raw_rows)
    signals = dedup_signals(raw)

    if signals.empty:
        progress.empty()
        status.warning("현재 조건에서는 신호가 없습니다.")
        st.stop()

    trades = []
    for _, sig in signals.iterrows():
        ev = evaluate_b(
            data_map[sig["코드"]],
            int(sig["_breakout_i"]),
            float(sig["진입가"]),
            STOP,
            HOLD,
        )
        row = sig.drop(labels=["_breakout_i"]).to_dict()
        row.update(ev)
        trades.append(row)

    t = pd.DataFrame(trades)
    t["결과군"] = t["최종수익률(%)"].apply(lambda x: "성공" if x > 0 else "실패")

    progress.empty()
    status.success(f"완료 · 원신호 {len(raw)}건 → 중복 제거 실제매매 {len(t)}건")

    st.subheader("전체 성과")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("신호", len(t))
    c2.metric("승률", f'{(t["최종수익률(%)"] > 0).mean()*100:.1f}%')
    c3.metric("평균수익률", f'{t["최종수익률(%)"].mean():.2f}%')
    c4.metric("손절률", f'{t["손절"].mean()*100:.1f}%')

    features = [
        "기준봉상승률(%)", "거래대금(억)", "기준봉거래량배수",
        "기준봉종가위치(%)", "기준봉5일선이격(%)", "기준봉20일선이격(%)",
        "눌림깊이(%)", "눌림거래량비율(%)", "눌림5일선이격(%)",
        "돌파종가위치(%)", "돌파종가수익률(%)", "돌파거래량vs눌림(배)"
    ]

    success = t[t["결과군"] == "성공"]
    fail = t[t["결과군"] == "실패"]

    comp = []
    for f in features:
        s = success[f].mean() if not success.empty else float("nan")
        q = fail[f].mean() if not fail.empty else float("nan")
        all_std = t[f].std()
        effect = (s - q) / all_std if pd.notna(all_std) and all_std != 0 else float("nan")
        comp.append({
            "변수": f,
            "성공군평균": round(s, 2) if pd.notna(s) else None,
            "실패군평균": round(q, 2) if pd.notna(q) else None,
            "차이(성공-실패)": round(s-q, 2) if pd.notna(s) and pd.notna(q) else None,
            "표준화차이": round(effect, 2) if pd.notna(effect) else None,
        })

    comp_df = pd.DataFrame(comp)
    comp_df["차이절대값"] = comp_df["표준화차이"].abs()
    comp_df = comp_df.sort_values("차이절대값", ascending=False).drop(columns="차이절대값")

    st.subheader("① 성공군 vs 실패군")
    st.caption("표준화차이의 절대값이 클수록 두 집단을 구분할 가능성이 큰 변수입니다. 표본이 작으므로 확정 필터가 아니라 후보 탐색용입니다.")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("② 변수별 4구간 성과")
    selected = st.selectbox("분석 변수", features)

    work = t[[selected, "최종수익률(%)", "손절", "+3%", "+5%", "+7%", "+10%"]].dropna().copy()
    try:
        work["구간"] = pd.qcut(work[selected], q=4, duplicates="drop")
    except Exception:
        work["구간"] = pd.cut(work[selected], bins=4, duplicates="drop")

    band = work.groupby("구간", observed=True).agg(
        신호=(selected, "size"),
        평균값=(selected, "mean"),
        평균수익률=("최종수익률(%)", "mean"),
        손절률=("손절", "mean"),
        도달3=("+3%", "mean"),
        도달5=("+5%", "mean"),
        도달7=("+7%", "mean"),
        도달10=("+10%", "mean"),
    ).reset_index()

    band["구간"] = band["구간"].astype(str)
    for col in ["손절률", "도달3", "도달5", "도달7", "도달10"]:
        band[col] = (band[col] * 100).round(1)
    band["평균값"] = band["평균값"].round(2)
    band["평균수익률"] = band["평균수익률"].round(2)

    st.dataframe(band, use_container_width=True, hide_index=True)

    st.subheader("③ 모든 변수 구간별 요약")
    all_bands = []
    for f in features:
        w = t[[f, "최종수익률(%)", "손절", "+5%"]].dropna().copy()
        try:
            w["구간"] = pd.qcut(w[f], q=4, duplicates="drop")
        except Exception:
            continue
        g = w.groupby("구간", observed=True).agg(
            신호=(f, "size"),
            평균값=(f, "mean"),
            평균수익률=("최종수익률(%)", "mean"),
            손절률=("손절", "mean"),
            도달5=("+5%", "mean"),
        ).reset_index()
        for _, r in g.iterrows():
            all_bands.append({
                "변수": f,
                "구간": str(r["구간"]),
                "신호": int(r["신호"]),
                "평균값": round(r["평균값"], 2),
                "평균수익률(%)": round(r["평균수익률"], 2),
                "손절률(%)": round(r["손절률"] * 100, 1),
                "+5%도달(%)": round(r["도달5"] * 100, 1),
            })

    bands_df = pd.DataFrame(all_bands)
    st.dataframe(bands_df, use_container_width=True, hide_index=True)

    st.subheader("④ 거래별 원자료")
    show_cols = [
        "종목명","코드","기준봉일","돌파일","진입가","결과군",
        "최종수익률(%)","손절","MFE(%)","MAE(%)"
    ] + features
    st.dataframe(
        t[show_cols].sort_values(["결과군","최종수익률(%)"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "성공/실패 비교 CSV",
        comp_df.to_csv(index=False).encode("utf-8-sig"),
        "swing_v7_2_success_fail_compare.csv",
        "text/csv",
        use_container_width=True,
    )
    st.download_button(
        "거래별 분석자료 CSV",
        t.drop(columns=[], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
        "swing_v7_2_trades.csv",
        "text/csv",
        use_container_width=True,
    )

    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"):
            st.write(errors)

st.caption("주의: 돌파 당일 일봉에서 손절가와 상승 목표가가 모두 관측되면 장중 순서를 알 수 없어 손절 우선으로 처리합니다.")
