import streamlit as st
import os
import urllib.request
from PIL import Image
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
st.write("Ứng dụng AI phân biệt Stem Cells, Epithelial Cells và White Blood Cells sử dụng YOLOv8m.")

# ---------------------------------------------------------
# 2. KHỞI TẠO VÀ TẢI MÔ HÌNH YOLOV8M TỪ GITHUB RELEASE
# ---------------------------------------------------------
MODEL_PATH = "best.pt"

# ⚠️ Julie hãy thay link dưới đây bằng Link Direct Copy từ Releases của Julie nếu có thay đổi phiên bản:
MODEL_URL = "https://github.com/ngocdiep-gif/StemCell_Vision_App/releases/download/v1.0/best.pt"

@st.cache_resource
def load_yolo_model():
    # Tự động tải file best.pt về máy chủ Streamlit nếu chưa có
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Đang tải weights mô hình YOLOv8m từ Release... Vui lòng chờ vài giây!"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
    # Load mô hình YOLO
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
# 4. GIAO DIỆN TẢI ANH VÀ DỰ ĐOÁN
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tải ảnh tế bào lên để phân tích (JPG, PNG, JPEG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Hiển thị 2 cột: Ảnh gốc và Kết quả
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("🖼️ Ảnh tế bào gốc")
        st.image(image, use_column_width=True)
    
    # Tiến hành nhận diện khi bấm nút
    if st.button("🚀 Phân tích tế bào"):
        with st.spinner("🔍 Đang phát hiện và phân loại các loại tế bào..."):
            # Chạy dự đoán với mô hình YOLOv8m
            results = model.predict(source=image, conf=conf_threshold)
            
            # Lấy ảnh kết quả chứa bounding box
            res_plotted = results[0].plot()
            
            with col2:
                st.subheader("🎯 Kết quả phân loại")
                st.image(res_plotted, caption="Các tế bào đã được khoanh vùng", use_column_width=True)
                
            # Thống kê chi tiết từng loại tế bào tìm được
            st.markdown("---")
            st.subheader("📊 Thống kê số lượng tế bào phát hiện:")
            boxes = results[0].boxes
            if len(boxes) > 0:
                class_names = model.names
                counts = {}
                for box in boxes:
                    cls_id = int(box.cls[0])
                    name = class_names[cls_id]
                    counts[name] = counts.get(name, 0) + 1
                
                for cell_type, count in counts.items():
                    st.write(f"- **{cell_type}**: {count} tế bào")
            else:
