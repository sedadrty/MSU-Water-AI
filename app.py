import streamlit as st
import joblib
from pathlib import Path
from train_model import recommend

st.set_page_config(
    page_title="MSU Water AI",
    page_icon="💧",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "water_ai_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("💧 MSU Water AI")
st.subheader("ระบบ AI แนะนำปริมาณสารเคมีสำหรับผลิตน้ำประปา")

st.info("ระบบต้นแบบสำหรับงานวิจัย โดยใช้ข้อมูล Jar Test และ AI")

st.markdown("### 1. กรอกข้อมูลคุณภาพน้ำดิบ")

ph = st.number_input(
    "ค่า pH",
    min_value=0.0,
    max_value=14.0,
    value=7.3,
    step=0.1
)

turbidity = st.number_input(
    "ความขุ่นน้ำดิบ (NTU)",
    min_value=0.0,
    value=373.0,
    step=1.0
)

temperature = st.number_input(
    "อุณหภูมิน้ำ (°C)",
    value=30.0,
    step=0.5
)

if st.button("🔍 วิเคราะห์และแนะนำปริมาณสารเคมี", type="primary"):

    try:
        bundle = load_model()

        result = recommend(
            bundle,
            ph,
            turbidity,
            temperature,
            5.0,   # เป้าหมายความขุ่น
            1.0,   # ยอมให้สูงกว่าเป้าหมาย 1 NTU
            1.0,   # ความบริสุทธิ์สารส้ม
            1.0,   # ความบริสุทธิ์ปูนขาว
            1.0    # ยอมให้สูงกว่าความขุ่นต่ำสุด 1 NTU
        )

        if result.get("status") == "EXPLORATORY_ONLY":

            st.success("วิเคราะห์ข้อมูลสำเร็จ")

            st.markdown("### 2. ผลการแนะนำ")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "สารส้มที่แนะนำ",
                    f"{result['alum_dose_mg_L']:g} mg/L"
                )

            with col2:
                st.metric(
                    "ปูนขาวตามทฤษฎี",
                    f"{result['theoretical_lime_CaOH2_mg_L']:g} mg/L"
                )

            st.metric(
                "ความขุ่นสุดท้าย",
                f"{result['final_turbidity_NTU']:g} NTU"
            )

            best = result["lowest_turbidity_dose"]

            st.markdown("### 3. เปรียบเทียบกับค่าที่ดีที่สุด")

            st.write(
                f"สารส้มที่ให้ความขุ่นต่ำสุด: "
                f"{best['alum_mg_L']:g} mg/L"
            )

            st.write(
                f"ความขุ่นต่ำสุด: "
                f"{best['final_turbidity_NTU']:g} NTU"
            )

            dose_range = result["near_optimal_dose_range_mg_L"]

            st.write(
                f"ช่วงปริมาณสารส้มที่เหมาะสม: "
                f"{dose_range[0]:g}–{dose_range[1]:g} mg/L"
            )

            source = result["data_source"]

            if source == "MEASURED_JAR_TEST":
                st.info("ผลจากข้อมูล Jar Test ที่วัดจริง")
            else:
                st.warning("ผลจาก AI ทำนาย ต้องทดสอบ Jar Test ยืนยัน")

            st.caption(
                "ปูนขาวเป็นค่าคำนวณตามทฤษฎี "
                "ไม่ใช่ปริมาณที่ยืนยันสำหรับเติมจริง"
            )

        else:
            st.warning(
                f"ไม่สามารถแนะนำปริมาณได้: "
                f"{result.get('status')}"
            )
            st.write(result.get("message", ""))

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")