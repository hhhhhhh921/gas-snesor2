```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
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

st.title("🧪 Electronic Nose AI System")

st.subheader("Real-time Gas Leakage Detection")

st.write("---")

# =========================
# 랜덤 감지 버튼
# =========================

if st.button("🚨 Start Detection"):

    st.write("### Sensor Status")
    st.info("Collecting sensor signals...")

    time.sleep(1)

    # 랜덤 센서값 생성
    random_sensor = np.random.uniform(
        low=-50,
        high=50,
        size=(1,16)
    )

    # 예측
    pred_co = model_co.predict(random_sensor)[0]
    pred_eth = model_eth.predict(random_sensor)[0]

    # 음수 제거
    pred_co = max(pred_co, 0)
    pred_eth = max(pred_eth, 0)

    st.write("---")

    st.write("## Prediction Results")

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

    # 위험도 판단

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

    # 로그 느낌

    st.write("### System Log")

    st.code(f"""
[INFO] Sensor array activated
[INFO] Gas pattern analyzed
[INFO] CO concentration = {pred_co:.2f} ppm
[INFO] Ethylene concentration = {pred_eth:.2f} ppm
[INFO] Monitoring complete
""")
```
