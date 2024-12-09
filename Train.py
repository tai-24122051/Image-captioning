import os
import re
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D


def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
        # take one caption at a time
          caption = captions[i]
        # preprocessing steps
        # convert to lowercase
          caption = caption.lower()
        # delete digits, special chars, etc.,
          caption = caption.replace('[^A-Za-z]', '')
        # delete additional spaces
          caption = re.sub(r'\s+', ' ', caption)
        # add start and end tags to the caption
          caption = 'startseq ' + " ".join([word for word in         caption.split() if len(word)>1]) + ' endseq'
          captions[i] = caption


def data_generator(captions, images, w2i, max_length, batch_size):
    X_image, X_cap, y = [], [], []
    n = 0
    while True:
        for image_id, caps in captions.items():
            image = images[image_id]  # Lấy đặc trưng của ảnh
            for cap in caps:
                # Mã hóa caption thành chuỗi số
                seq = [w2i[word] for word in cap.split() if word in w2i]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # Pad chuỗi
                    in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
                    X_image.append(image)
                    X_cap.append(in_seq)
                    y.append(out_seq)
            n += 1
            if n == batch_size:
                # Trả về dữ liệu dạng tf.Tensor
                yield ({
                    "image": tf.convert_to_tensor(np.array(X_image), dtype=tf.float32),
                    "text": tf.convert_to_tensor(np.array(X_cap), dtype=tf.int32),
                }, tf.convert_to_tensor(np.array(y), dtype=tf.int32))
                X_image, X_cap, y = [], [], []
                n = 0



if __name__ == "__main__":
    BASE_DIR = 'train'
    WORKING_DIR = 'Project_2'

    # Load VGG16 model with pre-trained weights
    base_model = VGG16(weights='imagenet', include_top=False)

    # Thêm lớp GlobalAveragePooling2D để giảm kích thước đầu ra
    x = GlobalAveragePooling2D()(base_model.output)

    # Thêm lớp Dense để tạo vector kích thước 4096
    x = Dense(4096, activation='relu')(x)

    # Tạo mô hình mới với đầu ra là 4096
    model = Model(inputs=base_model.input, outputs=x)

    # Summarize the model to verify layers
    print(model.summary())

    # Trích xuất đặc trưng
    features = {}
    directory = os.path.join(BASE_DIR, 'Images')

    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = model.predict(image)
        feature = feature.flatten()  # Đảm bảo vector phẳng
        image_id = img_name.split('.')[0]
        features[image_id] = feature

    # Lưu đặc trưng
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)
    with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
        next(f)
        captions_doc = f.read()

    # create mapping of image to captions
    mapping = {}
    # process lines
    for line in tqdm(captions_doc.split('\n')):
        # split the line by comma(,)
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        # remove extension from image ID
        image_id = image_id.split('.')[0]
        # convert caption list to string
        caption = " ".join(caption)
        # create list if needed
        if image_id not in mapping:
            mapping[image_id] = []
        # store the caption
        mapping[image_id].append(caption)

    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)

    tokenizer = Tokenizer(num_words=5000)  # Giới hạn từ điển chỉ chứa 5000 từ phổ biến nhất
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1  # Vocab size đã giới hạn

    # get maximum length of the caption available
    max_length = max(len(caption.split()) for caption in all_captions)

    image_ids = list(mapping.keys())
    split = int(len(image_ids) * 0.90)
    train = image_ids[:split]
    test = image_ids[split:]

    # Encoder model
    inputs1 = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    # plot the model
    plot_model(model, show_shapes=True)


    batch_size = 16  # Giảm batch size

    steps = len(train) // batch_size


    for i in range(20):
        # Gọi generator mới
        generator = data_generator(mapping, features, tokenizer.word_index, max_length, batch_size)

        # Huấn luyện với generator
        checkpoint = ModelCheckpoint('model_best.keras', save_best_only=True, monitor='loss', mode='min')
        model.fit(x=generator,epochs=1,steps_per_epoch=steps,callbacks=[checkpoint],verbose=1)

    # save the model
    model.save('model.keras')
