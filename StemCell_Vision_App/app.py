import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import os

# ---------------------------------------------------------
# 1. Cấu hình giao diện Trang web
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hệ Thống Phân Tích & Phân Loại Tế Bào",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Phân Tích & Phát Hiện Tế Bào Độ Chính Xác Cao")

# Bảng ánh xạ tên class từ YOLO sang Tiếng Việt chuẩn y tế
CLASS_MAPPING = {
    'Te_Bao_Goc': 'Tế Bào Gốc',
    'Te_Bao_Niem_Mac': 'Tế Bào Niêm Mạc',
    'Te_Bao_Phi_Dai': 'Tế Bào Phì Đại',
    'WBC': 'Bạch Cầu (WBC)',
    'stem_cell': 'Tế Bào Gốc',
    'epithelial': 'Tế Bào Niêm Mạc',
    'hypertrophic': 'Tế Bào Phì Đại',
    'wbc': 'Bạch Cầu (WBC)'
}

# ---------------------------------------------------------
# 2. Thanh điều khiển (Sidebar Options)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Thiết lập Phân tích Advanced")

conf_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy (Confidence)",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05,
    help="Hạ thấp để bắt các tế bào mờ, mỏng hoặc nhỏ."
)

iou_threshold = st.sidebar.slider(
    "Ngưỡng tách tế bào dính nhau (IoU)",
    min_value=0.10,
    max_value=0.80,
    value=0.40,
    step=0.05,
    help="Giúp nhận diện chính xác các tế bào nằm đè/chồng lên nhau."
)

enable_contrast = st.sidebar.checkbox(
    "Tăng cường tương phản (Cân bằng Histogram CLAHE)",
    value=True,
    help="Giúp làm rõ viền tế bào bị nhạt màu trước khi đưa vào AI."
)

# ---------------------------------------------------------
# 3. Nạp Mô Hình YOLOv8 (Tự động tìm đường dẫn file best.pt)
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    
    alt_path = os.path.join("StemCell_Vision_App", "best.pt")
    if os.path.exists(alt_path):
        return YOLO(alt_path)
    
    for root, dirs, files in os.walk("."):
        if "best.pt" in files:
            return YOLO(os.path.join(root, "best.pt"))
            
    raise FileNotFoundError("Không tìm thấy file best.pt trong dự án.")

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"❌ Không tìm thấy file `best.pt`. Lỗi: {e}")
    st.stop()

# ---------------------------------------------------------
# 4. Tải Ảnh & Xử Lý Nhận Diện
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tải ảnh hiển vi tế bào lên...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(image_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if enable_contrast:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        processed_img_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        input_for_ai = cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_for_ai = img_rgb

    results = model.predict(
        source=input_for_ai,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=640
    )[0]

    boxes = results.boxes
    annotated_img = input_for_ai.copy()

    counts = {v: 0 for v in set(CLASS_MAPPING.values())}

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            raw_label = model.names[cls_id]
            
            label_vn = CLASS_MAPPING.get(raw_label, raw_label)
            
            if label_vn in counts:
                counts[label_vn] += 1
            else:
                counts[label_vn] = 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 127), 2)
            caption_text = f"{label_vn} {conf:.2f}"
            cv2.putText(annotated_img, caption_text, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Ảnh Đầu Vào (Gốc)", use_container_width=True)
    with col2:
        st.image(annotated_img, caption="Ảnh AI Nhận Diện & Đếm Tất Cả Tế Bào", use_container_width=True)

    # ---------------------------------------------------------
    # 5. Bảng Báo Cáo Thống Kê Chi Tiết
    # ---------------------------------------------------------
    st.subheader("📊 Bảng Báo Cáo Thống Kê Phân Loại Tế Bào")

    total_cells = len(boxes)
    report_data = {
        "Tên File": [uploaded_file.name],
        "Tổng Số Tế Bào": [total_cells]
    }
    
    for cell_type, count in counts.items():
        report_data[cell_type] = [count]

    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True)

    if total_cells == 0:
        st.warning("⚠️ Không tìm thấy tế bào nào. Hãy thử kéo thanh 'Ngưỡng tin cậy (Confidence)' xuống mức 0.10 ở cột bên trái.")
