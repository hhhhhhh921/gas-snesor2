import streamlit as st
import pandas as pd
import joblib

# =========================
# 모델 불러오기
# =========================

model_co = joblib.load("model_co.pkl")
model_eth = joblib.load("model_eth.pkl")

# =========================
# 제목
# =========================

st.title("Electronic Nose AI System")

st.write("Gas Leakage Detection and Concentration Prediction")

# =========================
# 파일 업로드
# =========================

uploaded_file = st.file_uploader(
    "Upload Sensor Data File",
    type=["csv", "txt"]
)

# =========================
# 예측
# =========================

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file,
        sep=r"\s+",
        engine="python",
        header=None,
        skiprows=1
    )

    X = df.iloc[:, 3:19]

    pred_co = model_co.predict(X)
    pred_eth = model_eth.predict(X)

    avg_co = pred_co.mean()
    avg_eth = pred_eth.mean()

    st.subheader("Prediction Results")

    st.write(f"CO Concentration: {avg_co:.2f} ppm")
    st.write(f"Ethylene Concentration: {avg_eth:.2f} ppm")

    # 위험 판단

    if avg_co > 200:

        st.error("⚠ HIGH RISK: Possible Gas Leakage")
        st.write("📱 Alert sent to mobile device")

    elif avg_co > 100:

        st.warning("⚠ MEDIUM RISK")

    else:

        st.success("✅ LOW RISK")
