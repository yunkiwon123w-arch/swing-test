import streamlit as st
import time

st.set_page_config(page_title="FDR/NAVER 연결 진단")

st.title("스윙 백테스트 연결 진단 v2")
st.success("1/4 Streamlit 실행 정상")

try:
    import FinanceDataReader as fdr
    st.success("2/4 FinanceDataReader 로딩 정상")
except Exception as e:
    st.error("2/4 FinanceDataReader 로딩 실패")
    st.exception(e)
    st.stop()

if st.button("NAVER 주가 연결 테스트", type="primary", use_container_width=True):
    st.info("3/4 NAVER 삼성전자(005930) 조회 중...")
    t = time.time()

    try:
        df = fdr.DataReader(
            "NAVER:005930",
            "2026-07-01",
            "2026-07-10"
        )

        sec = time.time() - t

        if df is None or df.empty:
            st.error(f"4/4 데이터 없음 ({sec:.1f}초)")
        else:
            st.success(f"4/4 NAVER 데이터 수신 성공 ({sec:.1f}초)")
            st.write(f"수신 데이터: {len(df)}일")
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"4/4 NAVER 조회 실패 ({time.time()-t:.1f}초)")
        st.exception(e)

st.caption("진단 v2 · 실제 주문/계좌와 연결되지 않습니다.")
