from flask import Flask, render_template, Response, request, jsonify
from camera import VideoCamera
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import os
import tempfile

app = Flask(__name__)
camera = VideoCamera()

# Load the trained body shape classification model
model = load_model("body_shape_model.h5")
print("Model input shape:", model.input_shape)  # Optional: for debugging

# List of body shape classes (must match your training order)
classes = ['hourglass', 'rectangle', 'pear', 'apple', 'inverted_triangle' , 'val', 'train']

# ✅ Updated preprocessing to match model's input shape
def preprocess_image(frame):
    frame = cv2.resize(frame, (224, 224))  # Resized to match model input
    frame = img_to_array(frame) / 255.0
    return np.expand_dims(frame, axis=0)  # Shape becomes (1, 224, 224, 3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = camera.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    frame = camera.get_raw_frame()
    if frame is None:
        return jsonify({"error": "Camera capture failed"})

    # Save the captured image temporarily (optional for debugging or future use)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, frame)

    # Preprocess and predict
    img = preprocess_image(frame)
    prediction = model.predict(img)

    # Debugging: Print the prediction to check if it's giving valid probabilities
    print("Prediction output:", prediction)

    body_shape_class = np.argmax(prediction[0])
    predicted_shape = classes[body_shape_class]

    # Debugging: Print the predicted shape to ensure it's correct
    print("Predicted Body Shape:", predicted_shape)

    # Recommend dress images based on predicted shape
    dress_folder = os.path.join('static', 'dresses', predicted_shape)
    if not os.path.exists(dress_folder):
        print(f"No folder found for {predicted_shape} at {dress_folder}")
        return jsonify({
            "body_shape": predicted_shape,
            "recommendation_images": []
        })

    dress_images = [
        f"/static/dresses/{predicted_shape}/{fname}"
        for fname in os.listdir(dress_folder)
        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    return jsonify({
        "body_shape": predicted_shape,
        "recommendation_images": dress_images
    })

if __name__ == '__main__':
    app.run(debug=True)     