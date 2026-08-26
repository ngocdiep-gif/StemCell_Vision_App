import streamlit as st
import os
import urllib.request
from PIL import Image
import cv2
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
st.write("Ứng dụng AI phát hiện và phân loại tế bào độ chính xác cao bằng YOLOv8m.")

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

# Màu sắc cho từng nhãn (BGR Format)
CLASS_COLORS = {
    "Normal": (255, 255, 255),      # Trắng
    "Ascus": (0, 0, 255),          # Đỏ
    "BG": (0, 255, 255),           # Vàng
    "Stem Cell": (255, 0, 0),      # Xanh dương
    "Epithelial": (0, 255, 0),     # Xanh lá
    "WBC": (255, 0, 255)           # Tím
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
# 4. HÀM VẼ KHUNG CHÚ THÍCH TIẾNG VIỆT
# ---------------------------------------------------------
def draw_vietnamese_boxes(img_pil, boxes, orig_names):
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    counts = {}
    
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        original_name = orig_names[cls_id]
        
        # Chuyển đổi nhãn sang tiếng Việt
        vn_name = CLASS_NAME_VIETNAMESE.get(original_name, original_name)
        counts[vn_name] = counts.get(vn_name, 0) + 1
        
        # Lấy tọa độ khung
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = CLASS_COLORS.get(original_name, (0, 255, 0))
        
        # Vẽ khung viền
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # Vẽ nhãn chữ Tiếng Việt + Độ tin cậy
        label_text = f"{vn_name} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_cv, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img_cv, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), counts

# ---------------------------------------------------------
# 5. GIAO DIỆN TẢI ẢNH VÀ DỰ ĐOÁN
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tải ảnh tế bào lên để phân tích (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
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
                    st.warning("Chưa phát hiện thấy tế bào nào với ngưỡng độ tin cậy hiện tại. Hãy thử giảm thanh Slider bên trái!")
