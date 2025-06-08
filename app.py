import streamlit as st
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# ======================
# Konfigurasi Streamlit UI
# ======================
st.set_page_config(page_title="Klasifikasi Wajah Selebriti", page_icon="🎭", layout="centered")

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
# Judul dan Deskripsi
# ======================
st.title("🎭 Klasifikasi Wajah Selebriti")
st.write("Upload gambar wajah selebriti untuk mengklasifikasikannya menggunakan model **ResNet50**.")

# ======================
# Load Dataset & Model
# ======================
data_dir = r'D:\semester 6\ui viskom\Celebrity Faces Dataset'

if not os.path.exists(data_dir):
    st.error("❌ Dataset tidak ditemukan! Periksa path dataset.")
    st.stop()
else:
    class_names = sorted(os.listdir(data_dir))
    st.info(f"📂 Jumlah kelas selebriti terdeteksi: {len(class_names)}")

# Load model
model_path = "resnet50_cfid_model.keras"
try:
    model = load_model(model_path)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ======================
# Upload Gambar & Prediksi
# ======================
uploaded_file = st.file_uploader("📤 Unggah gambar wajah selebriti", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="🖼️ Gambar yang diupload", use_column_width=True)

        # Preprocessing
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Prediksi
        pred = model.predict(x)
        class_idx = np.argmax(pred)
        class_label = class_names[class_idx] if class_idx < len(class_names) else "Unknown"
        confidence = np.max(pred) * 100

        with col2:
            st.markdown("### 📌 Hasil Prediksi")
            st.success(f"Wajah ini diprediksi sebagai: **{class_label}**")
            st.info(f"Tingkat keyakinan: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam memproses gambar: {e}")

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("💡 Aplikasi ini dibuat dengan Streamlit dan ResNet50 - oleh Muhammad Fakhrul Hizrian✨")
