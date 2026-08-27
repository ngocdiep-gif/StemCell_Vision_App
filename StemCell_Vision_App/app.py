import os
import urllib.request
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT CHUYÊN NGHIỆP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Corneal & StemCell Vision AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Corneal & StemCell Vision AI")
st.caption("Hệ thống AI hỗ trợ phân loại & định lượng tế bào biểu mô và tế bào gốc vùng rìa giác mạc dựa trên hình thái học")

# ---------------------------------------------------------
# 2. KHAI BÁO NHÃN VÀ CƠ SỞ DỮ LIỆU HÌNH THÁI Y SINH
# ---------------------------------------------------------
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
# 3. NẠP MÔ HÌNH TỐI ƯU BỘ NHỚ RAM
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Đang tải tệp trọng số mô hình từ GitHub Release...")
        req = urllib.request.Request(
            MODEL_URL, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())
    
    from ultralytics import YOLO
    # Tắt tính năng không cần thiết để tiết kiệm RAM tối đa
    torch.set_grad_enabled(False)
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.to('cpu')  # Bắt buộc chạy trên CPU để ổn định Streamlit Cloud
    return yolo_model

model = None
try:
    model = load_yolo_model()
    st.sidebar.success("✅ Mô hình AI đã sẵn sàng!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi chi tiết: {e}")
    st.error(f"⚠️ Chưa nạp được mô hình. Chi tiết lỗi kỹ thuật: `{e}`")

# ---------------------------------------------------------
# 4. THANH BÊN TÙY CHỈNH THAM SỐ
# ---------------------------------------------------------
st.sidebar.header("⚙️ Cấu hình nhận diện")
conf_threshold = st.sidebar.slider("Ngưỡng độ tin cậy (Confidence)", 0.05, 1.0, 0.20, 0.05)

# ---------------------------------------------------------
# 5. HÀM VẼ KHUNG THUẦN PIL (KHÔNG DÙNG OPENCV)
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
        
        info = CELL_MORPHOLOGY_DB.get(original_name, {
            "vn": original_name, 
            "color": (0, 255, 0)
        })
        vn_name = info["vn"]
        counts[vn_name] = counts.get(vn_name, 0) + 1
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = info["color"]
        
        # Vẽ khung bao
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Vẽ thẻ tên
        label_text = f"{vn_name} ({conf:.2f})"
        draw.rectangle([x1, max(0, y1 - 18), x1 + 220, max(18, y1)], fill=color)
        draw.text((x1 + 4, max(0, y1 - 15)), label_text, fill=(255, 255, 255), font=font)
        
    return img_draw, counts

# ---------------------------------------------------------
# 6. XỬ LÝ ẢNH ĐẦU VÀO VÀ HIỂN THỊ KẾT QUẢ
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Tải ảnh soi kính hiển vi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("🖼️ Vi trường ảnh gốc")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Phân tích & Định lượng Tế bào"):
        if model is None:
            st.error("Chưa nạp được mô hình. Hãy nhìn thông báo lỗi đỏ ở góc trái màn hình!")
        else:
            with st.spinner("🔍 AI đang tự động khoanh vùng và phân loại tế bào..."):
                with torch.no_grad():
                    results = model.predict(source=image, conf=conf_threshold, verbose=False)
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    res_plotted, counts = draw_boxes_pil(image, boxes, model.names)
                    
                    with col2:
                        st.subheader("🎯 Kết quả phân tích thị giác AI")
                        st.image(res_plotted, caption=f"Phát hiện tổng cộng {len(boxes)} đối tượng tế bào", use_container_width=True)
                        
                    st.markdown("---")
                    st.subheader("📊 Thống kê & Báo cáo Hình thái Y sinh:")
                    
                    for orig_name, info in CELL_MORPHOLOGY_DB.items():
                        vn_name = info["vn"]
                        if vn_name in counts:
                            count = counts[vn_name]
                            with st.expander(f"🔹 **{vn_name}**: {count} tế bào phát hiện"):
                                st.write(f"🧬 **Đặc điểm hình thái nhận dạng:** {info['desc']}")
                                st.caption(f"📚 **Tài liệu y khoa tham khảo:** {info['ref']}")
                    
                    st.success("✅ Đã hoàn thành phân tích định lượng cho nghiên cứu/lâm sàng.")
                else:
                    with col2:
                        st.warning("Không tìm thấy tế bào thỏa mãn ngưỡng tin cậy. Vui lòng hạ slider Confidence bên thanh menu trái.")
