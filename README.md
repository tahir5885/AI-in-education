# AI-in-education
import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# Load the education dataset
data = np.load('education_data.npy', allow_pickle=True)
X = data[:, 0]
y = data[:, 1]

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Tokenize and lemmatize the text data
corpus = []
for i in range(len(X)):
    words = nltk.word_tokenize(X[i].lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    corpus.append(' '.join(words))

# Convert the text data into a one-hot encoded matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
X_train = tokenizer.texts_to_sequences(corpus)
X_train = pad_sequences(X_train, padding='post', maxlen=100)
vocab_size = len(tokenizer.word_index) + 1

# Define the model
inputs = Input(shape=(100,))
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

# Save the model
model.save('education_chatbot_model.h5')

# Use the model to generate responses
def generate_response(text):
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words]
    sequences = tokenizer.texts_to_sequences([' '.join(words)])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)
    prediction = model.predict(padded_sequences)[0][0]
    if prediction > 0.5:
        return "Yes"
    else:
        return "No"
