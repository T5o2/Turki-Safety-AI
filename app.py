import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io

# 1. إعدادات الصفحة (layout="wide" يجعل التصميم يأخذ الشاشة كاملة كالمواقع الاحترافية)
st.set_page_config(page_title="نظام السلامة الذكي | Smart Safety", page_icon="🛡️", layout="wide")

# 2. حقن أكواد CSS لتحسين التصميم وإخفاء هوية Streamlit
st.markdown("""
<style>
    /* إخفاء القوائم العلوية والسفلية الافتراضية للمنصة */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* تصميم بطاقات الإحصائيات لتشبه غرف التحكم */
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #004c97; /* لون أزرق مؤسسي */
    }
    
    /* عند استخدام الوضع الليلي تتغير ألوان البطاقات تلقائياً */
    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background-color: #1e1e1e;
            border-left: 5px solid #4da6ff;
        }
    }
</style>
""", unsafe_allow_html=True)

# 3. تصميم الشريط الجانبي (Sidebar)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1973/1973805.png", width=80)
st.sidebar.title("⚙️ إعدادات النظام")
st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider("مستوى دقة الرصد (Confidence):", min_value=0.1, max_value=1.0, value=0.40, step=0.05)
st.sidebar.markdown("---")
st.sidebar.info("هذا النظام مدعوم بنموذج YOLOv8 المخصص للرصد اللحظي لمعدات السلامة.")

# 4. واجهة الموقع الرئيسية
st.title("🛡️ نظام الرصد الذكي للسلامة المهنية")
st.markdown("### حلول الذكاء الاصطناعي لمراقبة الامتثال لمعايير السلامة الصناعية (HSE)")
st.markdown("---")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("📂 قم برفع صورة العمال للتحليل (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("####             📷 ")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 بدء فحص السلامة الآلي"):
        with st.spinner('جاري معالجة الرؤية الحاسوبية...'):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            results = model.predict(img_cv2, conf=confidence_threshold)
            names = model.names
            
            helmet_count = 0
            no_helmet_count = 0
            vest_count = 0

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label_name = names[cls]
                
                text_color = (255, 255, 255)
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
                elif label_name == "not_helmet":
                    color = (0, 0, 255)
                    no_helmet_count += 1
                    display_name = "NO HELMET!"
                else:
                    color = (0, 255, 255)
                    display_name = label_name
                    
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 3)
                
                label = f"{display_name} {conf:.2f}"
                font_scale = 0.7
                thickness = 2
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                cv2.rectangle(img_cv2, (x1, y1 - t_size[1] - 10), (x1 + t_size[0] + 10, y1), color, -1)
                cv2.putText(img_cv2, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            res_plotted_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("####               🎯 ")
                st.image(res_plotted_rgb, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 تقرير الامتثال اللحظي")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric(label="✅ خوذة مطابقة", value=helmet_count)
            stat_col2.metric(label="🦺 سترة مطابقة", value=vest_count)
            stat_col3.metric(label="❌ مخالفات (بدون خوذة)", value=no_helmet_count)
            
            if no_helmet_count > 0:
                st.error(f"⚠️ **حالة طوارئ:** تم رصد عدد ({no_helmet_count}) مخالفات عدم ارتداء الخوذة في الموقع!")
            else:
                st.success("✅ **امتثال تام:** الموقع مطابق لاشتراطات السلامة.")
            
            # ميزة تحميل التقرير (الصورة بعد الفحص)
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR))
            io_buf = io.BytesIO(buffer)
            
            st.download_button(
                label="📥 تحميل صورة الرصد للتوثيق",
                data=io_buf,
                file_name="safety_report.jpg",
                mime="image/jpeg"
            )
