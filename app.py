import streamlit as st
import numpy as np
import joblib
import time

# =========================
# 페이지 설정
# =========================

st.set_page_config(
    page_title="Electronic Nose AI",
    page_icon="🧪",
    layout="centered"
)

# =========================
# 모델 불러오기
# =========================

model_co = joblib.load("model_co.pkl")
model_eth = joblib.load("model_eth.pkl")

# =========================
# 제목
# =========================

st.title("🧪 전자코 AI 시스템")

st.subheader("실시간 가스 누출 감지 및 농도 예측")

st.write("---")

# =========================
# 시작 버튼
# =========================

if st.button("🚨 Start Detection"):

    st.info("센서 데이터 수집 중...")

    time.sleep(1)

    # =========================
    # 랜덤 센서 데이터 생성
    # =========================

    random_sensor = np.random.randn(1,16) * np.random.randint(10,100)

    # =========================
    # AI 예측
    # =========================

    pred_co = model_co.predict(random_sensor)[0]
    pred_eth = model_eth.predict(random_sensor)[0]

    # 음수 제거
    pred_co = max(pred_co, 0)
    pred_eth = max(pred_eth, 0)

    st.write("---")

    st.subheader("Prediction Results")

    # =========================
    # 결과 카드
    # =========================

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="CO Concentration",
            value=f"{pred_co:.2f} ppm"
        )

    with col2:
        st.metric(
            label="Ethylene Concentration",
            value=f"{pred_eth:.2f} ppm"
        )

    st.write("---")

    # =========================
    # 위험도 판단
    # =========================

    if pred_co > 200:

        st.error("🚨 HIGH RISK")
        st.error("위험한 가스 누출 감지")
        st.write("📱 긴급 알림 전송 완료")

    elif pred_co > 100:

        st.warning("⚠ MEDIUM RISK")
        st.warning("비정상 가스 농도 감지")

    else:

        st.success("✅ LOW RISK")
        st.success("안전 상태 유지 중")

    st.write("---")

    # =========================
    # 센서 로그
    # =========================

    st.subheader("System Log")

    st.code(f"""
[INFO] Sensor array activated
[INFO] Gas pattern analyzed
[INFO] CO concentration = {pred_co:.2f} ppm
[INFO] Ethylene concentration = {pred_eth:.2f} ppm
[INFO] Monitoring complete
""")

    # =========================
    # 현재 센서값 표시
    # =========================

    st.subheader("Current Sensor Values")

    st.write(random_sensor)
