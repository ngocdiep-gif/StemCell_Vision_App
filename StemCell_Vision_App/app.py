import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO
import io

# 1. Cấu hình giao diện Web Chuyên nghiệp
st.set_page_config(
    page_title="StemCell & Epithelial Analyzer Lab",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Hệ Thống AI Phân Tích & Đo Đạc Tế Bào Chuyên Sinh")
st.markdown("*Công cụ hỗ trợ phân loại Tế bào gốc, Tế bào niêm mạc và tự động chuẩn hóa mọi định dạng/kênh màu ảnh.*")
st.divider()

# 2. Thanh Cấu hình & Tải Mô hình AI
st.sidebar.title("⚙️ Thiết lập Phân tích")
conf_thresh = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.1, 1.0, 0.25, 0.05)
pixel_per_micron = st.sidebar.number_input("Tỷ lệ Pixel / μm", min_value=0.1, value=2.0, step=0.1)
enable_contrast = st.sidebar.checkbox("Tăng cường tương phản (Ảnh huỳnh quang / tối)", value=False)

@st.cache_resource
def load_model():
    try:
        return YOLO("StemCell_Vision_App/best.pt")
    except Exception:
        return YOLO("best.pt")
    # Sử dụng yolov8n.pt làm fallback nếu chưa có best.pt
    try:
        return YOLO("best.pt")
    except Exception:
        return YOLO("yolov8n.pt")

model = load_model()

# 3. Hàm Tiền Xử Lý Ảnh Chuẩn Hóa RGB
def process_and_convert_to_rgb(uploaded_file, enhance_contrast=False):
    """
    Tự động xử lý mọi định dạng (PNG, JPG, TIF 8/16-bit) 
    và chuyển đổi mọi kiểu kênh màu (Grayscale, RGBA, CMYK) về đúng định dạng RGB 8-bit chuẩn cho YOLO.
    """
    file_bytes = uploaded_file.read()
    
    # Sử dụng PIL để mở đa dạng định dạng (bao gồm cả TIF chuyên dụng)
    pil_img = Image.open(io.BytesIO(file_bytes))
    
    # Tự động xoay ảnh theo exif orientation nếu có
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # Chuyển đổi mọi không gian màu sang RGB chuẩn
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
        
    img_np = np.array(pil_img)
    
    # Ép kiểu dữ liệu về uint8 (8-bit chuẩn) nếu là ảnh 16-bit
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # Tăng cường tương phản bằng CLAHE cho ảnh huỳnh quang / ảnh bị tối nền
    if enhance_contrast:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        
    return img_np

# 4. Giao diện Tải Ảnh & Phân Tích
uploaded_files = st.file_uploader(
    "📥 Tải ảnh tế bào lên (Hỗ trợ JPG, PNG, TIF, BMP - chọn nhiều ảnh cùng lúc):",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    summary_list = []
    
    st.subheader("🖼️ Kết Quả Nhận Diện & Phân Tích")
    
    for file in uploaded_files:
        try:
            # Tiền xử lý tự động về chuẩn RGB
            img_rgb = process_and_convert_to_rgb(file, enhance_contrast=enable_contrast)
            
            # Đưa qua mô hình YOLO phân tích
            results = model.predict(source=img_rgb, conf=conf_thresh)
            res = results[0]
            
            stem_count = 0
            epithelial_count = 0
            
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = res.names[cls_id].lower()
                    
                    if "stem" in cls_name:
                        stem_count += 1
                    elif "epithelial" in cls_name or "mucosal" in cls_name:
                        epithelial_count += 1
                    else:
                        stem_count += 1 # Mặc định phân loại thử nghiệm
                        
            total = stem_count + epithelial_count
            
            summary_list.append({
                "Tên File": file.name,
                "Tổng Số Tế Bào": total,
                "Tế Bào Gốc": stem_count,
                "Tế Bào Niêm Mạc": epithelial_count
            })
            
            # Hiển thị kết quả từng ảnh
            with st.expander(f"🔍 Xem chi tiết: {file.name}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption="Ảnh Đầu Vào (Đã Chuẩn Hóa RGB)", use_container_width=True)
                with col2:
                    annotated_img = res.plot()
                    st.image(annotated_img, caption="Ảnh AI Nhận Diện & Đếm Tế Bào", use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Không thể xử lý file {file.name}: {str(e)}")

    # 5. Xuất Báo Cáo Thống Kê
    if summary_list:
        st.divider()
        st.subheader("📊 Bảng Báo Cáo Tổng Hợp (Batch Report)")
        df = pd.DataFrame(summary_list)
        st.dataframe(df, use_container_width=True)
        
        # Tải báo cáo CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải Báo Cáo CSV",
            data=csv_data,
            file_name="Bao_Cao_Phan_Tich_Te_Bao.csv",
            mime="text/csv"
        )
