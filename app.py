import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

st.set_page_config(page_title="نظام السلامة الذكي | Smart Safety", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #004c97;
    }
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background-color: #1e1e1e;
            border-left: 5px solid #4da6ff;
        }
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1973/1973805.png", width=80)
st.sidebar.title("⚙️ إعدادات النظام المعمارية")
st.sidebar.markdown("---")
st.sidebar.info("يستخدم هذا النظام معمارية (Dual-Model) وحسابات (IoA) الدقيقة لضمان أعلى درجات الرصد المتقاطع.")
person_conf = st.sidebar.slider("دقة رصد الأشخاص (البشر):", 0.1, 1.0, 0.30, 0.05)
ppe_conf = st.sidebar.slider("دقة رصد معدات السلامة:", 0.1, 1.0, 0.40, 0.05)
st.sidebar.markdown("---")

@st.cache_resource
def load_models():
    person_model = YOLO("yolov8n.pt")
    ppe_model = YOLO("best.pt")
    return person_model, ppe_model

person_model, ppe_model = load_models()

st.title("🛡️ نظام الرصد المزدوج للسلامة المهنية (Dual-AI)")
st.markdown("### تحليل هندسي متقاطع (IoA) لاكتشاف المخالفات المعقدة")
st.markdown("---")

uploaded_file = st.file_uploader("📂 قم برفع صورة العمال للتحليل (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📷 البث الأصلي")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 بدء الفحص المزدوج الدقيق"):
        with st.spinner('جاري إجراء العمليات الحسابية للتقاطعات الهندسية...'):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            persons_results = person_model.predict(img_cv2, classes=[0], conf=person_conf)
            ppe_results = ppe_model.predict(img_cv2, conf=ppe_conf)
            names = ppe_model.names
            
            helmet_count = 0
            no_helmet_count = 0
            vest_count = 0
            no_vest_count = 0
            
            vest_boxes = []

            for box in ppe_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label_name = names[cls]
                
                if label_name == "helmet":
                    color = (0, 255, 0)
                    text_color = (0, 0, 0)
                    helmet_count += 1
                    display_name = "Helmet"
                elif label_name == "reflective":
                    color = (0, 255, 0) 
                    text_color = (0, 0, 0)
                    vest_count += 1
                    display_name = "Vest"
                    vest_boxes.append((x1, y1, x2, y2))
                elif label_name == "not_helmet":
                    color = (0, 0, 255)
                    text_color = (255, 255, 255)
                    no_helmet_count += 1
                    display_name = "NO HELMET!"
                else:
                    continue
                    
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
                label = f"{display_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img_cv2, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 5, y1), color, -1)
                cv2.putText(img_cv2, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # الخوارزمية الهندسية الدقيقة (Intersection over Area)
            for p_box in persons_results[0].boxes:
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                
                has_vest = False
                for (vx1, vy1, vx2, vy2) in vest_boxes:
                    ix1 = max(px1, vx1)
                    iy1 = max(py1, vy1)
                    ix2 = min(px2, vx2)
                    iy2 = min(py2, vy2)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        vest_area = (vx2 - vx1) * (vy2 - vy1)
                        # الشرط الجديد: يجب أن يكون 50% من السترة داخل مربع الشخص
                        if vest_area > 0 and (inter_area / vest_area) > 0.5:
                            has_vest = True
                            break
                
                if not has_vest:
                    no_vest_count += 1
                    color = (0, 0, 255) 
                    cv2.rectangle(img_cv2, (px1, py1), (px2, py2), color, 3)
                    
                    label = "NO VEST!"
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(img_cv2, (px1, py1 - t_size[1] - 8), (px1 + t_size[0] + 5, py1), color, -1)
                    cv2.putText(img_cv2, label, (px1 + 2, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            res_plotted_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("#### 🎯 الرصد الذكي المتقاطع")
                st.image(res_plotted_rgb, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 تقرير الامتثال اللحظي (مُحدث)")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            stat_col1.metric(label="✅ خوذة مطابقة", value=helmet_count)
            stat_col2.metric(label="🦺 سترة مطابقة", value=vest_count)
            stat_col3.metric(label="❌ بدون خوذة", value=no_helmet_count)
            stat_col4.metric(label="❌ بدون سترة", value=no_vest_count)
            
            if no_helmet_count > 0 or no_vest_count > 0:
                st.error(f"⚠️ **حالة طوارئ:** تم رصد ({no_helmet_count}) بدون خوذة، و ({no_vest_count}) بدون سترة!")
            else:
                st.success("✅ **امتثال تام:** الموقع مطابق لاشتراطات السلامة.")
            
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR))
            io_buf = io.BytesIO(buffer)
            st.download_button(label="📥 تحميل التقرير", data=io_buf, file_name="safety_report.jpg", mime="image/jpeg")
