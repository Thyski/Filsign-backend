from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


app = Flask(__name__)

model = tf.lite.Interpreter(model_path="model.tflite")

names = [
"A","B","Boss","C","D","E","F","Father","G","Good","H","I","J","L","M","Me","Mine","Mother","N","O","Onion","P","Q","Quiet","R","Responsible","S","Serious","T","Think","This","U","V","W","Wait","Water","X","Y","You","Z",
]

@app.route('/')
def index():
    return "Hello World"

@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files["image"].read()

    img = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,[640,640])
    image = img.copy()
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255

    model.allocate_tensors()

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], im)

    model.invoke()

    output_data = model.get_tensor(output_details[0]['index'])

    pred_classes = []
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
        pred_classes.append(names[int(cls_id)])
        break

    return pred_classes[0]
