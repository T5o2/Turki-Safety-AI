import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

# 1. إعدادات الصفحة
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

# 2. تصميم الشريط الجانبي
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1973/1973805.png", width=80)
st.sidebar.title("⚙️ إعدادات النظام المعمارية")
st.sidebar.markdown("---")
st.sidebar.info("يستخدم هذا النظام معمارية (Dual-Model) لضمان أعلى درجات الدقة في الرصد المتقاطع.")
person_conf = st.sidebar.slider("دقة رصد الأشخاص (البشر):", 0.1, 1.0, 0.40, 0.05)
ppe_conf = st.sidebar.slider("دقة رصد معدات السلامة:", 0.1, 1.0, 0.40, 0.05)
st.sidebar.markdown("---")

# 3. تحميل العقلين (النماذج)
@st.cache_resource
def load_models():
    # العقل الأول: النموذج العالمي لرصد البشر (سيتم تحميله تلقائياً)
    person_model = YOLO("yolov8n.pt")
    # العقل الثاني: نموذجك المتخصص في السلامة
    ppe_model = YOLO("best.pt")
    return person_model, ppe_model

person_model, ppe_model = load_models()

st.title("🛡️ نظام الرصد المزدوج للسلامة المهنية (Dual-AI)")
st.markdown("### تحليل متقاطع لاكتشاف غياب السترات والخوذ بدقة متناهية")
st.markdown("---")

uploaded_file = st.file_uploader(" قم برفع صورتك للتحليل الآلي (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📷 ")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 بدء الفحص المزدوج (Cross-Validation)"):
        with st.spinner('جاري تشغيل معمارية النماذج المزدوجة...'):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # تشغيل العقل الأول (للبحث عن الأشخاص فقط - الكلاس رقم 0)
            persons_results = person_model.predict(img_cv2, classes=[0], conf=person_conf)
            
            # تشغيل العقل الثاني (للبحث عن المعدات)
            ppe_results = ppe_model.predict(img_cv2, conf=ppe_conf)
            names = ppe_model.names
            
            helmet_count = 0
            no_helmet_count = 0
            vest_count = 0
            no_vest_count = 0
            
            # قائمة لحفظ إحداثيات السترات عشان نقارنها بالأشخاص لاحقاً
            vest_boxes = []

            # رسم الخوذ والسترات من نموذجك
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
                    vest_boxes.append((x1, y1, x2, y2)) # حفظ إحداثيات السترة
                elif label_name == "not_helmet":
                    color = (0, 0, 255)
                    text_color = (255, 255, 255)
                    no_helmet_count += 1
                    display_name = "NO HELMET!"
                else:
                    continue
                    
                # رسم معدات السلامة
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
                label = f"{display_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img_cv2, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 5, y1), color, -1)
                cv2.putText(img_cv2, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # الخوارزمية الذكية: فحص الأشخاص لمعرفة من لا يرتدي سترة
            for p_box in persons_results[0].boxes:
                px1, py1, px2, py2 = map(int, p_box.xyxy[0])
                
                has_vest = False
                for (vx1, vy1, vx2, vy2) in vest_boxes:
                    # حساب التقاطع بين مربع الشخص ومربع السترة
                    ix1 = max(px1, vx1)
                    iy1 = max(py1, vy1)
                    ix2 = min(px2, vx2)
                    iy2 = min(py2, vy2)
                    
                    # إذا كان هناك تقاطع فعلي، يعني الشخص لابس سترة
                    if ix1 < ix2 and iy1 < iy2:
                        has_vest = True
                        break
                
                # إذا الشخص ما عليه سترة، ارسم مربع أحمر حوله واكتب NO VEST
                if not has_vest:
                    no_vest_count += 1
                    color = (0, 0, 255) # أحمر
                    cv2.rectangle(img_cv2, (px1, py1), (px2, py2), color, 3)
                    
                    label = "NO VEST!"
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(img_cv2, (px1, py1 - t_size[1] - 8), (px1 + t_size[0] + 5, py1), color, -1)
                    cv2.putText(img_cv2, label, (px1 + 2, py1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            res_plotted_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("#### 🎯 ")
                st.image(res_plotted_rgb, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 تقرير الامتثال اللحظي (مُحدث)")
            
            # عرض 4 إحصائيات بدلاً من 3
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
