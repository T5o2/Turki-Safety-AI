import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 1. إعدادات الصفحة (يجب أن تكون أول سطر)
st.set_page_config(page_title="نظام السلامة الذكي | Smart Safety", page_icon="👷", layout="wide")

# 2. تصميم الشريط الجانبي (Sidebar)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1973/1973805.png", width=100)
st.sidebar.title("⚙️ لوحة التحكم")
st.sidebar.markdown("---")
# إتاحة التحكم بحساسية المودل للمستخدم
confidence_threshold = st.sidebar.slider("مستوى دقة الرصد (Confidence):", min_value=0.1, max_value=1.0, value=0.40, step=0.05)
st.sidebar.markdown("---")
st.sidebar.info("هذا النظام مصمم للتعرف على:\n- الخوذة (Helmet)\n- السترة العاكسة (Vest)\n- مخالفة السلامة (No Helmet)")

# 3. واجهة الموقع الرئيسية
st.title("👷 نظام الرصد الذكي للسلامة المهنية")
st.markdown("### مدعوم بالذكاء الاصطناعي لمطابقة معايير السلامة الصناعية ")
st.markdown("---")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("📂 قم برفع صورة العمال هنا (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # تقسيم الشاشة لعمودين لعرض احترافي
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 الصورة الأصلية")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 بدء فحص السلامة"):
        with st.spinner('جاري تحليل البيانات واستخراج النتائج...'):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # استخدام نسبة الثقة من الشريط الجانبي
            results = model.predict(img_cv2, conf=confidence_threshold)
            names = model.names
            
            # عدادات للإحصائيات الذكية
            helmet_count = 0
            no_helmet_count = 0
            vest_count = 0

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label_name = names[cls]
                
                # إعداد الألوان
                text_color = (255, 255, 255) # أبيض كافتراضي
                if label_name == "helmet":
                    color = (0, 255, 0) # أخضر
                    text_color = (0, 0, 0) # أسود ليكون واضح فوق الأخضر
                    helmet_count += 1
                    display_name = "Helmet"
                elif label_name == "reflective":
                    color = (0, 255, 0) 
                    text_color = (0, 0, 0) 
                    vest_count += 1
                    display_name = "Vest"
                elif label_name == "not_helmet":
                    color = (0, 0, 255) # أحمر
                    no_helmet_count += 1
                    display_name = "NO HELMET!"
                else:
                    color = (0, 255, 255) # أصفر
                    display_name = label_name
                    
                # رسم المربع
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 3)
                
                # تضبيط خلفية النص لتكون أوضح بكثير
                label = f"{display_name} {conf:.2f}"
                font_scale = 0.7
                thickness = 2
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # رسم خلفية النص (مستطيل ممتلئ متناسق الأبعاد)
                cv2.rectangle(img_cv2, (x1, y1 - t_size[1] - 10), (x1 + t_size[0] + 10, y1), color, -1)
                
                # كتابة النص
                cv2.putText(img_cv2, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            res_plotted_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown("#### 🎯 نتيجة الفحص")
                st.image(res_plotted_rgb, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 لوحة الإحصائيات اللحظية")
            
            # عرض أرقام الإحصائيات
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric(label="✅ خوذة مطابقة", value=helmet_count)
            stat_col2.metric(label="🦺 سترة مطابقة", value=vest_count)
            stat_col3.metric(label="❌ مخالفة (بدون خوذة)", value=no_helmet_count)
            
            # نظام التنبيه الذكي
            if no_helmet_count > 0:
                st.error(f"⚠️ تنبيه أمني: تم رصد عدد ({no_helmet_count}) عمال لا يرتدون الخوذة!")
            else:
                st.success("✅ ممتاز: جميع العمال يلتزمون بمعايير السلامة.")
