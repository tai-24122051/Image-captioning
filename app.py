import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from tensorflow.image import resize

# Tải mô hình và tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Load MobileNetV2 model
    mobilenet_model = MobileNetV2(weights="imagenet")
    mobilenet_model = Model(inputs=mobilenet_model.inputs, outputs=mobilenet_model.layers[-2].output)

    # Load trained model
    model = tf.keras.models.load_model('mymodel.h5')

    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    return model, mobilenet_model, tokenizer


# Hàm tạo caption từ mô hình và ảnh
def generate_caption(model, mobilenet_model, tokenizer, image):
    # Tiền xử lý ảnh
    image = img_to_array(image)  # Chuyển ảnh về numpy array
    image = resize(image, (224, 224))  # Resize ảnh về kích thước phù hợp
    image = tf.reshape(image, (1, 224, 224, 3))  # Thêm batch dimension bằng tf.reshape
    image = preprocess_input(image)  # Tiền xử lý ảnh chuẩn MobileNetV2

    # Lấy các đặc trưng từ MobileNetV2
    image_features = mobilenet_model.predict(image, verbose=0)

    # Dự đoán caption
    max_caption_length = 34

    # Hàm dịch index thành từ
    def get_word_from_index(index, tokenizer):
        return next(
            (word for word, idx in tokenizer.word_index.items() if idx == index), None
        )

    # Dự đoán caption từ ảnh
    caption = "startseq"
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        predicted_index = np.argmax(yhat)
        predicted_word = get_word_from_index(predicted_index, tokenizer)

        if predicted_word is None or predicted_word == "endseq":
            break
        caption += " " + predicted_word

    # Xóa các ký hiệu dư thừa
    caption = caption.replace("startseq", "").replace("endseq", "")
    return caption


# Giao diện Streamlit
st.title("Image Caption Generator")
st.write("Tải ảnh lên và nhận mô tả tương ứng từ mô hình của bạn.")

# Load mô hình và tokenizer
model, mobilenet_model, tokenizer = load_model_and_tokenizer()

# Upload ảnh
uploaded_image = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Hiện ảnh đã tải lên
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Hiện caption trong vòng lặp spinner
    with st.spinner("Đang tạo mô tả..."):
        caption_result = generate_caption(model, mobilenet_model, tokenizer, image)

    # Hiện kết quả
    st.success(f"Mô tả ảnh: {caption_result}")
