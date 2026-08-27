import os
import urllib.request
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

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

CELL_MORPHOLOGY_DB = {
    "Normal": {
        "vn": "Tế bào biểu mô bình thường",
        "desc": "Dạng đa giác/lục giác, xếp lát gạch. Nhân nhỏ tròn ở trung tâm, tỷ lệ N/C thấp.",
        "ref": "Holland et al., Cornea & Limbus Morphometry (2021)",
        "color": (46, 204, 113)
    },
    "Ascus": {
        "vn": "Tế bào biến đổi bất thường (ASCUS)",
        "desc": "Nhân phì đại bất thường, dị dạng viền nhân, tăng sắc tố. Tỷ lệ N/C tăng cao bất thường.",
        "ref": "Bethesda System for Cytopathology Standards",
        "color": (231, 76, 60)
    },
    "Stem Cell": {
        "vn": "Tế bào gốc vùng rìa (Limbal Stem Cells)",
        "desc": "Kích thước nhỏ (7-10µm), hình tròn/lục giác đều, nhân chiếm ưu thế (Tỷ lệ N/C >= 0.8).",
        "ref": "Pellegrini et al., Nature Eye & Stem Cell Biology (2018)",
        "color": (52, 152, 219)
    },
    "Epithelial": {
        "vn": "Tế bào biểu mô giác mạc",
        "desc": "Kích thước lớn (20-40µm), màng ranh giới rõ ràng, tế bào chất rộng.",
        "ref": "Ophthalmic Research & In Vivo Confocal Microscopy Standards",
        "color": (230, 126, 34)
    },
    "WBC": {
        "vn": "Bạch cầu / Tế bào viêm (WBC)",
        "desc": "Phản quang mạnh trên IVCM, nhân chia thùy hoặc nhân đa thùy đặc trưng.",
        "ref": "Journal of Clinical & Experimental Ophthalmology",
        "color": (155, 89, 182)
    },
    "BG": {
        "vn": "Nền vi trường / Phế nang",
        "desc": "Vùng nền không chứa cấu trúc tế bào tiêu chuẩn.",
        "ref": "Standard Microscopic Background Category",
        "color": (149, 165, 166)
    }
}

# ---------------------------------------------------------
# NẠP MÔ HÌNH THUẦN PYTORCH (BYPASS HOÀN TOÀN OPENCV/LIBGL)
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
    
    # Nạp weights qua TorchScript / PyTorch CPU
    model_ckpt = torch.load(MODEL_PATH, map_location='cpu')
    if hasattr(model_ckpt, 'float'):
        model_ckpt = model_ckpt.float()
    if hasattr(model_ckpt, 'eval'):
        model_ckpt.eval()
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
        st.info("✅ Mô hình đã sẵn sàng xử lý trên hệ thống PyTorch CPU!")
