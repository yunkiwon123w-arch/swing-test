import streamlit as st
import pandas as pd
import numpy as np
import time
from dataclasses import dataclass
from pathlib import Path
from pykrx import stock

st.set_page_config(page_title="한국주식 스윙 백테스트", layout="wide")

@dataclass
class Params:
    base_return_pct: float
    base_value_krw: int
    volume_mult_20d: float
    pullback_start: int
    pullback_end: int
    pullback_volume_ratio: float
    stop_loss_pct: float
    forward_days: int
    sleep_sec: float = 0.25

TARGETS = [3.0, 5.0, 7.0, 10.0]

def ymd(x):
    return pd.Timestamp(x).strftime("%Y%m%d")

@st.cache_data(show_spinner=False)
def get_market_snapshot(date: str, market: str):
    df = stock.get_market_ohlcv_by_ticker(date, market=market)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = df.index.astype(str)
    return df

@st.cache_data(show_spinner=False)
def get_ticker_history(ticker: str, start: str, end: str):
    df = stock.get_market_ohlcv_by_date(start, end, ticker, adjusted=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if "등락률" not in df.columns:
        df["등락률"] = df["종가"].pct_change() * 100
    return df.sort_index()

@st.cache_data(show_spinner=False)
def trading_days_between(start_date: str, end_date: str):
    cal = stock.get_market_ohlcv_by_date(start_date, end_date, "005930", adjusted=True)
    return list(pd.to_datetime(cal.index))

def enrich(df):
    x = df.copy()
    x["MA5"] = x["종가"].rolling(5).mean()
    x["MA20"] = x["종가"].rolling(20).mean()
    x["VOL20"] = x["거래량"].shift(1).rolling(20).mean()
    x["MA5_RISING"] = x["MA5"] > x["MA5"].shift(1)
    x["MA20_RISING"] = x["MA20"] > x["MA20"].shift(1)
    return x

def qualifies_base(row, p):
    if pd.isna(row.get("MA5")) or pd.isna(row.get("MA20")) or pd.isna(row.get("VOL20")):
        return False
    return (
        float(row["등락률"]) >= p.base_return_pct
        and float(row["거래대금"]) >= p.base_value_krw
        and float(row["거래량"]) >= float(row["VOL20"]) * p.volume_mult_20d
        and float(row["종가"]) > float(row["MA5"])
        and bool(row["MA5_RISING"])
        and bool(row["MA20_RISING"])
    )

def find_entry(df, base_idx, p):
    base = df.iloc[base_idx]
    base_vol = float(base["거래량"])
    base_low = float(base["저가"])

    for k in range(p.pullback_start, p.pullback_end + 1):
        sig_idx = base_idx + k
        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            break

        sig = df.iloc[sig_idx]
        ent = df.iloc[entry_idx]

        volume_ok = float(sig["거래량"]) <= base_vol * p.pullback_volume_ratio
        structure_ok = float(sig["종가"]) > base_low
        if not (volume_ok and structure_ok):
            continue

        trigger = float(sig["고가"])
        if float(ent["고가"]) < trigger:
            continue

        entry_price = max(float(ent["시가"]), trigger)
        return {
            "signal_idx": sig_idx,
            "entry_idx": entry_idx,
            "signal_date": df.index[sig_idx],
            "entry_date": df.index[entry_idx],
            "entry_price": entry_price,
            "signal_volume_ratio": float(sig["거래량"]) / base_vol if base_vol else np.nan,
        }
    return None

def evaluate_trade(df, entry_idx, entry_price, p):
    stop_price = entry_price * (1 + p.stop_loss_pct / 100)
    targets = {t: entry_price * (1 + t / 100) for t in TARGETS}
    out = {
        "stop_hit": False,
        "stop_date": pd.NaT,
        "mfe_pct": np.nan,
        "mae_pct": np.nan,
        "ret_5d_pct": np.nan,
        "ret_10d_pct": np.nan,
    }
    for t in TARGETS:
        out[f"target_{int(t)}_hit_before_stop"] = False
        out[f"target_{int(t)}_date"] = pd.NaT

    end_idx = min(len(df)-1, entry_idx + p.forward_days)
    window = df.iloc[entry_idx:end_idx+1].copy()

    out["mfe_pct"] = (float(window["고가"].max()) / entry_price - 1) * 100
    out["mae_pct"] = (float(window["저가"].min()) / entry_price - 1) * 100

    alive = True
    hit_targets = set()

    for idx, row in window.iterrows():
        low, high = float(row["저가"]), float(row["고가"])
        if alive and low <= stop_price:
            out["stop_hit"] = True
            out["stop_date"] = idx
            alive = False
            continue
        if alive:
            for t, tp in targets.items():
                if t not in hit_targets and high >= tp:
                    hit_targets.add(t)
                    out[f"target_{int(t)}_hit_before_stop"] = True
                    out[f"target_{int(t)}_date"] = idx

    for n in (5, 10):
        idx2 = min(len(df)-1, entry_idx + n)
        if idx2 > entry_idx:
            out[f"ret_{n}d_pct"] = (float(df.iloc[idx2]["종가"]) / entry_price - 1) * 100

    out["primary_win"] = bool(out["target_5_hit_before_stop"])
    return out

def run_backtest(start_date, end_date, p, progress):
    days = trading_days_between(start_date, end_date)
    candidates = []

    total_steps = max(1, len(days)*2)
    done = 0

    for d in days:
        ds = ymd(d)
        for market in ("KOSPI", "KOSDAQ"):
            snap = get_market_snapshot(ds, market)
            done += 1
            progress.progress(min(done/total_steps, 0.45), text=f"1차 후보 검색: {ds} {market}")

            if snap.empty:
                continue
            ret = pd.to_numeric(snap["등락률"], errors="coerce")
            val = pd.to_numeric(snap["거래대금"], errors="coerce")
            hit = snap.loc[(ret >= p.base_return_pct) & (val >= p.base_value_krw)]
            for ticker, r in hit.iterrows():
                candidates.append({
                    "base_date": pd.Timestamp(d),
                    "market": market,
                    "ticker": str(ticker).zfill(6),
                    "snap_return_pct": float(r["등락률"]),
                    "snap_value_krw": float(r["거래대금"]),
                })
            time.sleep(p.sleep_sec)

    results = []
    for i, c in enumerate(candidates):
        ticker = c["ticker"]
        base_date = c["base_date"]
        hist_start = (base_date - pd.Timedelta(days=70)).strftime("%Y%m%d")
        hist_end = (base_date + pd.Timedelta(days=45)).strftime("%Y%m%d")

        df = get_ticker_history(ticker, hist_start, hist_end)
        if df.empty or base_date not in df.index:
            continue

        df = enrich(df)
        base_idx = df.index.get_loc(base_date)
        base = df.iloc[base_idx]
        if not qualifies_base(base, p):
            continue

        entry = find_entry(df, base_idx, p)
        if entry is None:
            continue

        ev = evaluate_trade(df, entry["entry_idx"], entry["entry_price"], p)
        try:
            name = stock.get_market_ticker_name(ticker)
        except Exception:
            name = ticker

        rec = {
            "종목코드": ticker,
            "종목명": name,
            "시장": c["market"],
            "기준봉일": base_date.date(),
            "기준봉등락률(%)": round(float(base["등락률"]), 2),
            "기준봉거래대금(억)": round(float(base["거래대금"]) / 1e8, 1),
            "거래량배수": round(float(base["거래량"]) / float(base["VOL20"]), 2),
            "눌림일": entry["signal_date"].date(),
            "진입일": entry["entry_date"].date(),
            "진입가": round(entry["entry_price"], 0),
            "눌림거래량비율": round(entry["signal_volume_ratio"], 3),
            "손절여부": ev["stop_hit"],
            "+5%선도달": ev["target_5_hit_before_stop"],
            "+7%선도달": ev["target_7_hit_before_stop"],
            "+10%선도달": ev["target_10_hit_before_stop"],
            "MFE(%)": round(ev["mfe_pct"], 2),
            "MAE(%)": round(ev["mae_pct"], 2),
            "5일수익률(%)": round(ev["ret_5d_pct"], 2),
            "10일수익률(%)": round(ev["ret_10d_pct"], 2),
        }
        results.append(rec)
        progress.progress(
            0.45 + 0.55 * ((i+1)/max(1, len(candidates))),
            text=f"상세 검증: {i+1}/{len(candidates)}"
        )
        time.sleep(p.sleep_sec)

    return pd.DataFrame(results), len(candidates)

st.title("한국주식 단기스윙 백테스트")
st.caption("KOSPI·KOSDAQ 전체에서 기준봉 → 첫 눌림 → 재돌파 조건을 과거 데이터로 검증합니다.")

with st.sidebar:
    st.header("검증 조건")
    start = st.date_input("시작일", pd.Timestamp("2025-01-01"))
    end = st.date_input("종료일", pd.Timestamp("2026-07-31"))

    base_return = st.number_input("기준봉 상승률 이상 (%)", 1.0, 30.0, 10.0, 1.0)
    base_value_eok = st.number_input("거래대금 이상 (억원)", 100, 10000, 1000, 100)
    vol_mult = st.number_input("20일 평균 거래량 대비 배수", 1.0, 10.0, 2.0, 0.5)

    pull_start = st.number_input("눌림 시작 거래일", 1, 10, 2)
    pull_end = st.number_input("눌림 종료 거래일", 2, 20, 5)
    pull_vol_pct = st.slider("눌림 거래량 / 기준봉 거래량 (%)", 10, 100, 50, 5)

    stop_loss = st.number_input("손절 (%)", -10.0, -0.5, -3.0, 0.5)
    forward_days = st.number_input("진입 후 관찰 거래일", 3, 30, 10)

run = st.button("백테스트 실행", type="primary", use_container_width=True)

if run:
    if start >= end:
        st.error("종료일은 시작일보다 뒤여야 합니다.")
        st.stop()
    if pull_start > pull_end:
        st.error("눌림 시작일이 종료일보다 클 수 없습니다.")
        st.stop()

    p = Params(
        base_return_pct=float(base_return),
        base_value_krw=int(base_value_eok * 1e8),
        volume_mult_20d=float(vol_mult),
        pullback_start=int(pull_start),
        pullback_end=int(pull_end),
        pullback_volume_ratio=float(pull_vol_pct/100),
        stop_loss_pct=float(stop_loss),
        forward_days=int(forward_days),
    )

    progress = st.progress(0, text="준비 중...")
    try:
        trades, first_candidates = run_backtest(
            pd.Timestamp(start).strftime("%Y%m%d"),
            pd.Timestamp(end).strftime("%Y%m%d"),
            p,
            progress
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    progress.empty()

    st.subheader("결과 요약")
    if trades.empty:
        st.warning(f"1차 후보 {first_candidates}건 중 최종 진입 조건을 만족한 거래가 없습니다.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("최종 거래 수", len(trades))
        c2.metric("+5% 선도달 승률", f'{trades["+5%선도달"].mean()*100:.1f}%')
        c3.metric("평균 MFE", f'{trades["MFE(%)"].mean():.2f}%')
        c4.metric("평균 MAE", f'{trades["MAE(%)"].mean():.2f}%')

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("손절 발생률", f'{trades["손절여부"].mean()*100:.1f}%')
        c6.metric("+7% 선도달", f'{trades["+7%선도달"].mean()*100:.1f}%')
        c7.metric("+10% 선도달", f'{trades["+10%선도달"].mean()*100:.1f}%')
        c8.metric("평균 10일 수익률", f'{trades["10일수익률(%)"].mean():.2f}%')

        st.subheader("개별 거래")
        st.dataframe(trades, use_container_width=True, hide_index=True)

        csv = trades.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "결과 CSV 다운로드",
            csv,
            "swing_backtest_results.csv",
            "text/csv",
            use_container_width=True
        )

        st.info(
            "일봉에서는 같은 날 손절가와 목표가의 발생 순서를 알 수 없어, "
            "같은 날 둘 다 터치하면 손절 우선으로 처리합니다."
        )
