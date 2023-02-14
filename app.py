from flask import Flask,jsonify,request
import cv2
from deepface import DeepFace
from PIL import Image

app = Flask(__name__)
app.url_map.strict_slashes = False

@app.route("/")
def home():
    return "<h1>Emotion Detection App</h1>"

@app.route("/process",methods=['POST'])
def process_img():
    file=request.files['image']
    img=Image.open(file.stream)
    full_img=cv2.imread(img)
    prediction=DeepFace.analyze(full_img,actions = ['emotion'])
    return jsonify({'emotion':prediction['dominant_emotion']})


# app.run(host="0.0.0.0",debug=True)