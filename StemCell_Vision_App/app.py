ython


import streamlit as st
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. CẤU HÌNH TRANG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="StemCell Vision App",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 StemCell Vision - Phân Loại & Nhận Diện Tế Bào")
st.write("Ứng dụng AI phân biệt các loại tế bào bằng mô hình YOLOv8m.")

# ---------------------------------------------------------
# 2. KHỞI TẠO VÀ TẢI MÔ HÌNH YOLOV8M TỪ GITHUB RELEASE
# ---------------------------------------------------------
MODEL_PATH = "best.1.pt"
MODEL_URL = "https://github.com/ngocdiep-gif/StemCell_Vision_App/releases/download/v1.0/best.1.pt"

# Từ điển Việt hóa tên nhãn tế bào
CLASS_NAME_VIETNAMESE = {
    "Normal": "Tế bào bình thường",
    "Ascus": "Tế bào bất thường (ASCUS)",
    "BG": "Nền ảnh / Phế nang",
    "Stem Cell": "Tế bào gốc",
    "Epithelial": "Tế bào biểu mô",
    "WBC": "Tế bào bạch cầu"
}

# Màu sắc RGB cho từng nhãn
CLASS_COLORS = {
    "Normal": (255, 255, 255),      # Trắng
    "Ascus": (255, 0, 0),           # Đỏ
    "BG": (255, 255, 0),            # Vàng
    "Stem Cell": (0, 102, 255),     # Xanh dương
    "Epithelial": (0, 204, 0),      # Xanh lá
    "WBC": (204, 0, 204)            # Tím
}

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Đang tải weights mô hình YOLOv8m từ Release... Vui lòng chờ vài giây!"):
            req = urllib.request.Request(
                MODEL_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
    
    model = YOLO(MODEL_PATH)
    return model

try:
    model = load_yolo_model()
    st.sidebar.success("✅ Mô hình YOLOv8m đã sẵn sàng!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi tải mô hình: {e}")

# ---------------------------------------------------------
# 3. THANH ĐIỀU CHỈNH THAM SỐ (SIDEBAR)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình nhận diện")
conf_threshold = st.sidebar.slider("Độ tin cậy (Confidence Threshold)", 0.1, 1.0, 0.25, 0.05)

# ---------------------------------------------------------
# 4. HÀM VẼ KHUNG CHÚ THÍCH TIẾNG VIỆT (DÙNG PIL THUẦN)
# ---------------------------------------------------------
def draw_vietnamese_boxes(img_pil, boxes, orig_names):
    img_draw = img_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        
    counts = {}
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        original_name = orig_names[cls_id]
        
        vn_name = CLASS_NAME_VIETNAMESE.get(original_name, original_name)
        counts[vn_name] = counts.get(vn_name, 0) + 1
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(original_name, (0, 255, 0))
        
        # Vẽ viền khung
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Vẽ thẻ tên Tiếng Việt
        label_text = f"{vn_name} {conf:.2f}"
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([x1, y1 - text_height - 6, x1 + text_width + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - text_height - 4), label_text, fill=(0, 0, 0), font=font)
        
    return img_draw, counts

# ---------------------------------------------------------
# 5. GIAO DIỆN TẢI ANH VÀ DỰ ĐOÁN
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tải ảnh tế bào lên để phân tích (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("🖼️ Ảnh tế bào gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Phân tích tế bào"):
        with st.spinner("🔍 Đang phát hiện và phân loại các loại tế bào..."):
            results = model.predict(source=image, conf=conf_threshold)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                res_plotted, counts = draw_vietnamese_boxes(image, boxes, model.names)
                
                with col2:
                    st.subheader("🎯 Kết quả phân loại (Tiếng Việt)")
                    st.image(res_plotted, caption="Các tế bào đã được phát hiện", use_container_width=True)
                    
                st.markdown("---")
                st.subheader("📊 Bảng thống kê số lượng phát hiện:")
                for cell_type, count in counts.items():
                    if "ASCUS" in cell_type or "bất thường" in cell_type:
                        st.error(f"- **{cell_type}**: {count} tế bào (⚠️ Cần chú ý)")
                    else:
                        st.write(f"- **{cell_type}**: {count} tế bào")
            else:
                with col2:
                    st.warning("Chưa phát hiện thấy tế bào nào với ngưỡng độ tin cậy
