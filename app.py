# !pip install streamlit tensorflow keras if hadnt dowloaded the libraries
import streamlit as st
import requests
import numpy as np
from PIL import Image
from model import get_caption_model, generate_caption


@st.cache_resource
def get_model():
    return get_caption_model()

caption_model = get_model()

def main():

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
                image = Image.open(uploaded_file)
                img = np.array(image)
            else:
                image = Image.open(requests.get(image_url, stream=True).raw)
                img = np.array(image)

            # Hiển thị ảnh
            st.image(image, caption="Ảnh của bạn", use_container_width=True)

            # Dự đoán caption
            with st.spinner("🔄 Đang tạo caption..."):
                caption = generate_caption(img, caption_model)

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