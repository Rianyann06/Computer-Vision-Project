import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
import zipfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Klasifikasi Wajah Selebriti", page_icon="🎭", layout="centered")

st.title("🎭 Klasifikasi Wajah Selebriti")
st.write("Upload gambar wajah selebriti untuk mengklasifikasikannya menggunakan model **ResNet50**.")

# ==== PATH / ID Google Drive ====
dataset_zip_url = "https://drive.google.com/uc?id=1fE4YSCqulnwUtH5CFABKHh5jU4j5vbl0"
model_url = "https://drive.google.com/uc?id=1qRGcgJdI3XAJm33Tg7B2hs6t4VY1vR64"    

dataset_path = "data/Celebrity Faces Dataset"
model_path = "resnet50_cfid_model.keras"

# ==== Download Dataset ====
if not os.path.exists(dataset_path):
    st.warning("📥 Mengunduh dataset...")
    gdown.download(dataset_zip_url, output="dataset.zip", quiet=False)
    if not os.path.exists("dataset.zip"):
        st.error("❌ Gagal mengunduh dataset. Cek URL atau koneksi.")
        st.stop()
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall("data")
    os.remove("dataset.zip")

# ==== Download Model ====
if not os.path.exists(model_path):
    st.warning("📥 Mengunduh model...")
    gdown.download(model_url, output=model_path, quiet=False)

# ==== Load Model ====
try:
    model = load_model(model_path)
    st.success("✅ Model berhasil dimuat.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ==== Kelas ====
if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
    st.error("❌ Dataset tidak ditemukan atau kosong. Periksa struktur hasil ekstrak ZIP.")
    st.stop()

class_names = sorted(os.listdir(dataset_path))
st.write("📂 Kelas tersedia:", class_names)

# ==== Upload Gambar ====
uploaded_file = st.file_uploader("📤 Unggah gambar wajah selebriti", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.write("📏 Ukuran gambar:", img.size)

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
            st.markdown(f"""
                <div style="text-align:center">
                    <h2>🎯 Prediksi: <span style='color:green'>{class_label}</span></h2>
                    <p>✅ Keyakinan: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan dalam memproses gambar: {e}")
