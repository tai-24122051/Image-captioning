import streamlit as st
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests

# Cấu hình Streamlit
st.set_page_config(page_title="Image Captioner", layout="centered", page_icon="🖼️")

# Tải mô hình, feature extractor và tokenizer
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, feature_extractor, tokenizer

# Hàm dự đoán caption từ ảnh
def predict_caption(image, model, feature_extractor, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tiền xử lý ảnh
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Sinh caption
    output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Ứng dụng chính
def main():
    # Load mô hình và các công cụ
    model, feature_extractor, tokenizer = load_model()

    # Giao diện Streamlit
    st.title("🖼️ Image Captioner")
    st.write("### Tạo chú thích cho ảnh của bạn bằng mô hình AI hiện đại!")

    # Input: Dán URL hoặc upload ảnh
    st.markdown("### Chọn một ảnh để bắt đầu")
    image_url = st.text_input("Dán URL ảnh:")
    uploaded_file = st.file_uploader("Hoặc tải lên một ảnh từ máy tính", type=["jpg", "png", "jpeg"])

    # Xử lý ảnh
    if uploaded_file or image_url:
        try:
            # Load ảnh
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
            else:
                response = requests.get(image_url, stream=True)
                image = Image.open(response.raw).convert("RGB")

            # Hiển thị ảnh
            st.image(image, caption="Ảnh của bạn", use_column_width=True)

            # Dự đoán caption
            with st.spinner("🔄 Đang tạo caption..."):
                caption = predict_caption(image, model, feature_extractor, tokenizer)

            # Hiển thị kết quả
            st.success("🎉 Caption đã được tạo:")
            st.write(f"**{caption}**")

        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý ảnh: {e}")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by [GROUP 5 FROM 24TNT-HCMUS]")

# Chạy ứng dụng
if __name__ == "__main__":
    main()
