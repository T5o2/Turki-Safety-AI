import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# تصميم واجهة الموقع
st.set_page_config(page_title="نظام السلامة الذكي", page_icon="👷")
st.title("👷 نظام الرصد الذكي للسلامة المهنية")
st.write(" هذا النظام مدعوم بالذكاء الاصطناعي للتحقق من التزام العمال بارتداء معدات السلامة.")

# تحميل النموذج
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# زر رفع الصورة
uploaded_file = st.file_uploader("قم برفع صورة العامل هنا...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("فحص السلامة 🔍"):
        with st.spinner('جاري التحليل...'):
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # نسبة الثقة 0.40 لتجنب الأخطاء
            results = model.predict(img_cv2, conf=0.40)
            names = model.names
            
            # رسم المربعات بالألوان المخصصة
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label_name = names[cls]
                
                # تحديد اللون
                if label_name == "helmet" or label_name == "reflective":
                    color = (0, 255, 0) # أخضر فاقع
                elif label_name == "not_helmet":
                    color = (0, 0, 255) # أحمر
                else:
                    color = (0, 255, 255) # أصفر
                    
                # رسم المربع والنص
                cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 3)
                label = f"{label_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
                cv2.rectangle(img_cv2, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 3), color, -1)
                cv2.putText(img_cv2, label, (x1, y1 - 2), 0, 0.6, (255, 255, 255), 1)
            
            # عرض النتيجة
            res_plotted_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            st.success("تم الفحص بنجاح!")
            st.image(res_plotted_rgb, caption=" النتيجة النهائية بعد التحليل ", use_container_width=True)
