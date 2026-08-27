import os
# Ép Ultralytics và OpenCV chạy ở chế độ Headless không dùng libGL
os.environ["OPENCV_HEADLESS"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys
import urllib.request
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="StemCell Vision App",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 StemCell Vision - Phân Loại & Nhận Diện Tế Bào")
st.write("Ứng dụng AI phát hiện và phân loại tế bào độ chính xác cao.")

# ---------------------------------------------------------
# 2. LOAD MÔ HÌNH YOLO
# ---------------------------------------------------------
MODEL_PATH = "best.1.pt"
MODEL_URL = "https://github.com/ngocdiep-gif/StemCell_Vision_App/releases/download/v1.0/best.1.pt"

CLASS_NAME_VIETNAMESE = {
    "Normal": "Tế bào bình thường",
    "Ascus": "Tế bào bất thường (ASCUS)",
    "BG": "Nền ảnh / Phế nang",
    "Stem Cell": "Tế bào gốc",
    "Epithelial": "Tế bào biểu mô",
    "WBC": "Tế bào bạch cầu"
}

CLASS_COLORS = {
    "Normal": (0, 255, 0),
    "Ascus": (255, 0, 0),
    "BG": (180, 180, 180),
    "Stem Cell": (0, 102, 255),
    "Epithelial": (255, 165, 0),
    "WBC": (204, 0, 204)
}

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Đang tải weights mô hình YOLOv8m..."):
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
    
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)

model = None
try:
    model = load_yolo_model()
    st.sidebar.success("✅ Mô hình YOLOv8m đã sẵn sàng!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải mô hình: {e}")

# ---------------------------------------------------------
# 3. NHẬN DIỆN VÀ VẼ KHUNG THUẦN PIL
# ---------------------------------------------------------
conf_threshold = st.sidebar.slider("Độ tin cậy (Confidence Threshold)", 0.1, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("Tải ảnh tế bào lên để phân tích...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Ảnh tế bào gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Phân tích tế bào"):
        if model is None:
            st.error("Mô hình chưa được nạp. Vui lòng kiểm tra lại log hệ thống.")
        else:
            with st.spinner("🔍 Đang phân loại tế bào..."):
                import numpy as np
                img_np = np.array(image)
                results = model.predict(source=img_np, conf=conf_threshold)
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    img_draw = image.copy()
                    draw = ImageDraw.Draw(img_draw)
                    font = ImageFont.load_default()
                    counts = {}
                    
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        orig_name = model.names[cls_id]
                        vn_name = CLASS_NAME_VIETNAMESE.get(orig_name, orig_name)
                        counts[vn_name] = counts.get(vn_name, 0) + 1
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        color = CLASS_COLORS.get(orig_name, (0, 255, 0))
                        
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        draw.rectangle([x1, y1 - 18, x1 + 170, y1], fill=color)
                        draw.text((x1 + 4, y1 - 15), f"{vn_name} {conf:.2f}", fill=(0, 0, 0), font=font)
                    
                    with col2:
                        st.subheader("🎯 Kết quả nhận diện")
                        st.image(img_draw, caption="Hoàn tất", use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📊 Thống kê chi tiết:")
                    for cell_type, count in counts.items():
                        st.write(f"- **{cell_type}**: {count} tế bào")
                else:
                    with col2:
                        st.warning("Không tìm thấy tế bào nào. Hãy giảm thanh slider Độ tin cậy.")
