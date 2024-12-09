import streamlit as st
import numpy as np
import pickle
import json
from tensorflow.keras.models import load_model  # Phù hợp với tensorflow==2.15.0
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import pad_sequences
from PIL import Image  # Pillow==9.5.0
import matplotlib.pyplot as plt  # matplotlib==3.7.2




# Hàm load các tài nguyên
@st.cache_resource
def load_resources():
    # Load mô hình
    model = load_model("mymodel.h5")

    # Load features
    with open("features.pkl", 'rb') as f:
        features = pickle.load(f)

    # Load mapping
    with open("mapping.json", 'r') as f:
        mapping = json.load(f)

    # Load tokenizer
    with open("tokenizer.pkl", 'rb') as f:
        tokenizer = pickle.load(f)

    # Max sequence length
    max_length = 34
    return model, features, mapping, tokenizer, max_length


# Giảm chiều từ 4096 xuống 1280 trước khi truyền vào mô hình
def reduce_feature_dimensions(feature):
    # Giảm chiều dữ liệu từ 4096 xuống 1280
    feature = np.dot(feature, np.random.rand(4096, 1280))  # Biến đổi chiều thông qua một ma trận ngẫu nhiên
    return feature


# Hàm chuyển đổi index thành từ
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Hàm dự đoán caption
def predict_caption(model, image_feature, tokenizer, max_length):
    # Giảm chiều feature về 1280
    image_feature = reduce_feature_dimensions(image_feature)

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length, padding='post')

        # Giữ nguyên shape dữ liệu phù hợp với mô hình
        image_feature = np.reshape(image_feature, (1, 1280))

        # Dự đoán từ tiếp theo
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Lấy index của từ có xác suất cao nhất
        word = idx_to_word(yhat, tokenizer)

        # Dừng nếu từ không hợp lệ
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


# Load tài nguyên
st.title("Image Captioning with Streamlit")
st.write("Ứng dụng tạo caption cho hình ảnh bằng mô hình học sâu.")

st.write("Đang tải tài nguyên...")
model, features, mapping, tokenizer, max_length = load_resources()
st.success("Tải xong tài nguyên!")

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Tải lên hình ảnh bạn muốn dự đoán caption:", type=["jpg", "png"])

if uploaded_file is not None:
    # Hiển thị ảnh tải lên
    image = Image.open(uploaded_file)
    st.image(image, caption="Hình ảnh đã tải lên", use_column_width=True)

    # Dự đoán caption
    image_id = uploaded_file.name.split('.')[0]

    feature = features[image_id]

    # Dự đoán caption
    predicted_caption = predict_caption(model, feature, tokenizer, max_length)


    # Hiển thị caption thực tế nếu có
    if image_id in mapping:
        st.subheader("Caption:")
        for caption in mapping[image_id]:
            st.write(f"- {caption}")
