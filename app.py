from flask import Flask,jsonify,request
import cv2
from deepface import DeepFace

app = Flask(__name__)
app.url_map.strict_slashes = False

@app.route("/")
def home():
    return "<h1>Emotion Detection App</h1>"

@app.route("/process",methods=['POST'])
def process_img():
    file=request.files['image']
    img=cv2.imread(file.stream)
    prediction=DeepFace.analyze(img,actions = ['emotion'])
    return jsonify({'emotion':prediction['dominant_emotion']})


# app.run(host="0.0.0.0",debug=True)