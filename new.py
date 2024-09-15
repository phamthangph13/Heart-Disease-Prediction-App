import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Khởi tạo lemmatizer
lemmatizer = WordNetLemmatizer()

# Tải dữ liệu intents
with open('intents.json') as file:
    intents = json.load(file)

# Khởi tạo danh sách từ và lớp
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Xử lý dữ liệu intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize và loại bỏ các ký tự không cần thiết
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sắp xếp các lớp
classes = sorted(set(classes))

# Lưu trữ từ và lớp vào tệp
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

# Tạo dữ liệu huấn luyện
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Trộn dữ liệu huấn luyện và chuyển thành numpy array
random.shuffle(training)
training = np.array(training)

# Tạo các biến đầu vào và đầu ra
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Tạo mô hình TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Biên dịch mô hình
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện mô hình
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Lưu mô hình
model.save('chatbot_model.h5')
print('Done')
