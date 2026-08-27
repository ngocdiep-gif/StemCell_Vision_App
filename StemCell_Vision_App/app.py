import os
import urllib.request
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Corneal & StemCell Vision AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Corneal & StemCell Vision AI")
st.caption("Hệ thống AI phân loại & định lượng tế bào dựa trên hình thái học y sinh")

MODEL_PATH = "best.1.pt"
MODEL_URL = "https://github.com/ngocdiep-gif/StemCell_Vision_App/releases/download/v1.0/best.1.pt"

# ---------------------------------------------------------
# NẠP MÔ HÌNH (CHUẨN HÓA THỤT LÙI DÒNG)
# ---------------------------------------------------------
@st.cache_resource
def load_pure_torch_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Đang tải tệp trọng số mô hình từ GitHub Release...")
        req = urllib.request.Request(
            MODEL_URL, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())
    
    # Đọc checkpoint PyTorch bằng weights_only=False
    model_ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    return model_ckpt

model = None
try:
    model = load_pure_torch_model()
    st.sidebar.success("✅ Mô hình PyTorch đã sẵn sàng!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi nạp mô hình: {e}")

# ---------------------------------------------------------
# GIAO DIỆN & XỬ LÝ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình nhận diện")
conf_threshold = st.sidebar.slider("Ngưỡng độ tin cậy (Confidence)", 0.05, 1.0, 0.20, 0.05)

uploaded_file = st.file_uploader("Tải ảnh soi kính hiển vi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Vi trường ảnh gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Phân tích & Định lượng Tế bào"):
        st.info("✅ Đang chuẩn bị phân tích dữ liệu...")
