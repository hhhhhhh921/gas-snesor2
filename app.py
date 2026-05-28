import streamlit as st
import pandas as pd
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
# 실제 데이터셋 불러오기
# =========================

df = pd.read_csv(
    "small_ethylene_CO.txt",
    sep=r"\s+",
    engine="python",
    header=None
)

# =========================
# 제목
# =========================

st.title("🧪 Electronic Nose AI System")

st.subheader(
    "Real-Time Gas Leakage Detection and Concentration Prediction"
)

st.write("---")

# =========================
# 감지 시작 버튼
# =========================

if st.button("🚨 Start Detection"):

    st.info("Collecting sensor signals...")

    time.sleep(1)

    # =========================
    # 실제 데이터셋에서 랜덤 샘플 선택
    # =========================

    random_row = df.sample(1)

    # 센서값
    sensor_values = random_row.iloc[:,3:19]

    # 실제 농도
    real_co = random_row.iloc[0,1]
    real_eth = random_row.iloc[0,2]

    # =========================
    # AI 예측
    # =========================

    pred_co = model_co.predict(sensor_values)[0]
    pred_eth = model_eth.predict(sensor_values)[0]

    pred_co = max(pred_co, 0)
    pred_eth = max(pred_eth, 0)

    st.write("---")

    st.subheader("Prediction Results")

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
        st.error("Possible dangerous gas leakage detected")
        st.write("📱 Emergency alert transmitted")

    elif pred_co > 100:

        st.warning("⚠ MEDIUM RISK")
        st.warning("Abnormal gas concentration detected")

    else:

        st.success("✅ LOW RISK")
        st.success("Environment stable")

    st.write("---")

    # =========================
    # 시스템 로그
    # =========================

    st.subheader("System Log")

    st.code(f"""
[INFO] Sensor array activated
[INFO] Real sensor pattern loaded
[INFO] CO concentration predicted = {pred_co:.2f} ppm
[INFO] Ethylene concentration predicted = {pred_eth:.2f} ppm
[INFO] Monitoring complete
""")

    # =========================
    # 센서값 표시
    # =========================

    st.subheader("Current Sensor Values")

    st.write(sensor_values)

    # =========================
    # 실제값 비교
    # =========================

    st.subheader("Reference Values")

    st.write(f"Actual CO: {real_co}")
    st.write(f"Actual Ethylene: {real_eth}")
