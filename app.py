import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import time
from datetime import date

st.set_page_config(page_title="단기스윙 백테스트 v1", layout="wide")
st.title("📈 단기스윙 백테스트 v1")
st.caption("FDR/NAVER 실제 일봉 · 30종목 엔진 검증")

U={"005930":"삼성전자","000660":"SK하이닉스","005380":"현대차","000270":"기아",
"035420":"NAVER","035720":"카카오","051910":"LG화학","373220":"LG에너지솔루션",
"006400":"삼성SDI","207940":"삼성바이오로직스","068270":"셀트리온","105560":"KB금융",
"055550":"신한지주","086790":"하나금융지주","066570":"LG전자","034020":"두산에너빌리티",
"042700":"한미반도체","010140":"삼성중공업","329180":"HD현대중공업","012450":"한화에어로스페이스",
"272210":"한화시스템","298040":"효성중공업","267260":"HD현대일렉트릭","000100":"유한양행",
"196170":"알테오젠","009150":"삼성전기","005490":"POSCO홀딩스","028260":"삼성물산",
"012330":"현대모비스","009540":"HD한국조선해양"}

@st.cache_data(ttl=86400,show_spinner=False)
def load(c,s,e):
    return fdr.DataReader("NAVER:"+c,s,e)

def signals(d,c,n,p,cut,end):
    d=d.copy()
    d["V"]=d.Close*d.Volume
    d["M5"]=d.Close.rolling(5).mean()
    d["M20"]=d.Close.rolling(20).mean()
    d["AV"]=d.Volume.shift(1).rolling(20).mean()
    d["R"]=d.Close.pct_change()*100
    out=[]
    for b in range(21,len(d)):
        x=d.iloc[b]
        if d.index[b]<cut or d.index[b]>end: continue
        if pd.isna(x.AV): continue
        if not (x.R>=p["br"] and x.V>=p["val"]*1e8 and x.Volume>=x.AV*p["vm"]): continue
        if not (x.Close>x.M5 and x.M5>d.M5.iloc[b-1] and x.M20>d.M20.iloc[b-1]): continue
        for k in range(2,6):
            si,ei=b+k,b+k+1
            if ei>=len(d): break
            s,e=d.iloc[si],d.iloc[ei]
            if s.Volume>x.Volume*p["pr"] or s.Close<=x.Low or e.High<s.High: continue
            ep=max(float(e.Open),float(s.High)); stop=ep*(1+p["sl"]/100)
            w=d.iloc[ei:min(len(d),ei+p["days"]+1)]
            alive=True; hit={3:False,5:False,7:False,10:False}; stopped=False
            for _,r in w.iterrows():
                if alive and r.Low<=stop:
                    stopped=True; alive=False; continue
                if alive:
                    for t in hit:
                        if r.High>=ep*(1+t/100): hit[t]=True
            out.append({"종목명":n,"코드":c,"기준봉일":d.index[b].date(),
            "기준봉(%)":round(x.R,2),"거래대금(억)":round(x.V/1e8),
            "거래량배수":round(x.Volume/x.AV,2),"진입일":d.index[ei].date(),
            "진입가":round(ep),"손절":stopped,"+3%":hit[3],"+5%":hit[5],
            "+7%":hit[7],"+10%":hit[10],
            "MFE(%)":round((w.High.max()/ep-1)*100,2),
            "MAE(%)":round((w.Low.min()/ep-1)*100,2)})
            break
    return out

with st.sidebar:
    st.header("검증 조건")
    end=st.date_input("종료일",date(2026,7,31))
    months=st.selectbox("기간",[1,3,6,12],index=1,format_func=lambda x:f"{x}개월")
    br=st.number_input("기준봉 상승률(%)",value=10.0)
    val=st.number_input("거래대금(억원)",value=1000,step=100)
    vm=st.number_input("20일 거래량 배수",value=2.0,step=.5)
    pr=st.slider("눌림 거래량 비율(%)",10,100,50,5)
    sl=st.number_input("손절(%)",value=-3.0,step=.5)
    days=st.number_input("관찰 거래일",3,30,10)
    go=st.button("▶ 백테스트 실행",type="primary",use_container_width=True)

st.info("먼저 대표 30종목으로 규칙을 검증합니다. 정상 작동 확인 후 KOSPI·KOSDAQ 전체로 확대합니다.")

if go:
    end=pd.Timestamp(end); cut=end-pd.DateOffset(months=months)
    start=cut-pd.Timedelta(days=70); fetch_end=end+pd.Timedelta(days=45)
    p={"br":br,"val":val,"vm":vm,"pr":pr/100,"sl":sl,"days":days}
    bar=st.progress(0); msg=st.empty(); rows=[]; errors=[]
    for i,(c,n) in enumerate(U.items(),1):
        msg.info(f"{i}/30 · {n} 데이터 조회/검증 중")
        bar.progress(i/30)
        try:
            d=load(c,start.strftime("%Y-%m-%d"),fetch_end.strftime("%Y-%m-%d"))
            if d is not None and not d.empty: rows+=signals(d,c,n,p,cut,end)
            else: errors.append(n+" 데이터 없음")
        except Exception as e: errors.append(n+" "+type(e).__name__)
        time.sleep(.05)
    bar.empty(); msg.success(f"완료 · 매매신호 {len(rows)}건")
    t=pd.DataFrame(rows)
    if t.empty: st.warning("현재 조건에서 신호가 없습니다.")
    else:
        a,b,c,d=st.columns(4)
        a.metric("신호",len(t)); b.metric("+5% 선도달",f'{t["+5%"].mean()*100:.1f}%')
        c.metric("손절률",f'{t["손절"].mean()*100:.1f}%'); d.metric("평균 MFE",f'{t["MFE(%)"].mean():.2f}%')
        a,b,c,d=st.columns(4)
        a.metric("+3%",f'{t["+3%"].mean()*100:.1f}%'); b.metric("+7%",f'{t["+7%"].mean()*100:.1f}%')
        c.metric("+10%",f'{t["+10%"].mean()*100:.1f}%'); d.metric("평균 MAE",f'{t["MAE(%)"].mean():.2f}%')
        st.dataframe(t.sort_values("진입일",ascending=False),use_container_width=True,hide_index=True)
        st.download_button("CSV 다운로드",t.to_csv(index=False).encode("utf-8-sig"),
                           "swing_v1.csv","text/csv",use_container_width=True)
    if errors:
        with st.expander(f"조회 실패 {len(errors)}건"): st.write(errors)

st.caption("같은 일봉에서 손절가와 목표가를 모두 터치하면 손절 우선 처리합니다.")
