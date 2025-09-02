import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/body_shape_model.h5')
classes = ['hourglass', 'rectangle', 'pear', 'apple', 'inverted_triangle']

def predict_body_shape(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

def predict_body_shape_from_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

def recommend_outfit(body_shape):
    suggestions = {
        'hourglass': 'static/dresses/hourglass.png',
        'rectangle': 'static/dresses/rectangle.png',
        'pear': 'static/dresses/pear.png',
        'apple': 'static/dresses/apple.png',
        'inverted_triangle': 'static/dresses/inverted_triangle.png'
    }
    return suggestions.get(body_shape, 'static/dresses/default.png')
