import os
import urllib.request
import numpy as np
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
st.write("Ứng dụng AI phát hiện và phân loại tế bào trên toàn bộ vùng ảnh.")

# ---------------------------------------------------------
# 2. LOAD MÔ HÌNH YOLO & CẤU HÌNH NHÃN TIẾNG VIỆT
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
    "Normal": (0, 255, 0),        # Xanh lá
    "Ascus": (255, 0, 0),         # Đỏ
    "BG": (180, 180, 180),       # Xám
    "Stem Cell": (0, 102, 255),   # Xanh dương
    "Epithelial": (255, 165, 0), # Cam
    "WBC": (204, 0, 204)          # Tím
}

@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⏳ Đang tải weights mô hình YOLOv8m... Vui lòng chờ vài giây!"):
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
# 3. THANH ĐIỀU CHỈNH THAM SỐ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình nhận diện")
conf_threshold = st.sidebar.slider("Độ tin cậy (Confidence Threshold)", 0.05, 1.0, 0.20, 0.05)

# ---------------------------------------------------------
# 4. HÀM VẼ TOÀN BỘ KHUNG TRÊN ẢNH (BẮT ALL NGUỒN)
# ---------------------------------------------------------
def draw_boxes_pil(img_pil, boxes, orig_names):
    img_draw = img_pil.copy()
    draw = ImageDraw.Draw(img_draw)
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
        
        # Vẽ khung nét rõ
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Vẽ nhãn tên tế bào
        label_text = f"{vn_name} {conf:.2f}"
        draw.rectangle([x1, max(0, y1 - 18), x1 + 170, max(18, y1)], fill=color)
        draw.text((x1 + 4, max(0, y1 - 15)), label_text, fill=(0, 0, 0), font=font)
        
    return img_draw, counts

# ---------------------------------------------------------
# 5. XỬ LÝ PHÂN TÍCH ALL ẢNH ĐẦU VÀO
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tải ảnh tế bào lên để phân tích...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Ảnh tế bào gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Phân tích ALL tế bào"):
        if model is None:
            st.error("Mô hình chưa được nạp. Vui lòng kiểm tra lại log hệ thống.")
        else:
            with st.spinner("🔍 Đang quét toàn bộ ảnh và phân loại tế bào..."):
                # Ép kiểu mảng NumPy chuẩn định dạng RGB cho YOLO quét ALL vùng ảnh
                img_input = np.array(image)
                
                # Quét và lấy toàn bộ kết quả phát hiện
                results = model(img_input, conf=conf_threshold, verbose=False)
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    res_plotted, counts = draw_boxes_pil(image, boxes, model.names)
                    
                    with col2:
                        st.subheader("🎯 Kết quả nhận diện toàn bộ")
                        st.image(res_plotted, caption=f"Đã phân tích xong - Tổng {len(boxes)} đối tượng phát hiện", use_container_width=True)
                        
                    st.markdown("---")
                    st.subheader("📊 Thống kê chi tiết từng loại tế bào:")
                    for cell_type, count in counts.items():
                        if "ASCUS" in cell_type or "bất thường" in cell_type:
                            st.error(f"- **{cell_type}**: {count} tế bào (⚠️ Cảnh báo bất thường)")
                        else:
                            st.write(f"- **{cell_type}**: {count} tế bào")
                else:
                    with col2:
                        st.warning("Không phát hiện tế bào nào. Hãy kéo thanh 'Độ tin cậy' ở bên trái xuống mức thấp hơn (ví dụ 0.1) để bắt nhiều tế bào hơn.")
