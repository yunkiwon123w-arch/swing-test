import streamlit as st
import time
st.set_page_config(page_title="KRX 연결 진단")
st.title("스윙 백테스트 연결 진단")
st.success("1/4 Streamlit 실행 정상")
try:
    from pykrx import stock
    st.success("2/4 pykrx 로딩 정상")
except Exception as e:
    st.error("2/4 pykrx 로딩 실패")
    st.exception(e)
    st.stop()
if st.button("KRX 연결 테스트", type="primary", use_container_width=True):
    st.info("3/4 KRX 접속 테스트 중...")
    t=time.time()
    try:
        df=stock.get_market_ohlcv_by_date("20260701","20260710","005930",adjusted=False)
        sec=time.time()-t
        if df is None or df.empty:
            st.error(f"4/4 데이터 없음 ({sec:.1f}초)")
        else:
            st.success(f"4/4 KRX 데이터 수신 성공 ({sec:.1f}초)")
            st.dataframe(df,use_container_width=True)
    except Exception as e:
        st.error(f"4/4 KRX 조회 실패 ({time.time()-t:.1f}초)")
        st.exception(e)
st.caption("진단용 파일 · 실제 주문/계좌와 연결되지 않습니다.")
