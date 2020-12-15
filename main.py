import base64
from os import listdir
from os.path import join, isfile
import cv2
import time
from threading import Thread
import numpy as np
from tensorflow.keras.models import load_model
import json
from scipy.spatial import distance
from utils.logging import logger
from flask import Flask
import socketio
import logging


class Timekeeper:
    def __init__(self):
        self.begin = False
        self.image = None
        self.old_faces = []
        self.faces = []
        self.cascade = cv2.CascadeClassifier("models/haarcascade_frontalface.xml")

        # open webcam
        self.cam = cv2.VideoCapture(0)

        # open mp4
        # self.cam = cv2.VideoCapture("test.mp4")
        self.color_box = [0, 192, 0]
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.model = None
        self.sio = socketio.Server(async_mode="threading", cors_allowed_origins="*")
        self.app = Flask(__name__)
        self.app.wsgi_app = socketio.WSGIApp(self.sio, self.app.wsgi_app)
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        log.disabled = True

        # data.json save feature vectors of all employees
        try:
            with open("models/data.json") as json_file:
                self.data = json.load(json_file)["data"]
        except:
            pass

    @staticmethod
    def prewhiten(x):
        """
        normalize the input
        """
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError("Dimension should be 3 or 4")

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    @staticmethod
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        """
        normalize the output
        """
        return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

    @staticmethod
    def cal_distance(vector1, vector2):
        """
        cal distance between two vectors by euclid function
        """
        return distance.euclidean(vector1, vector2)

    def detect_faces(self, image):
        margin_half = 5

        logger.info("Start cascade")
        boxes = self.cascade.detectMultiScale(image, 1.1, 4)
        logger.info("Done cascade: " + str(boxes))
        if len(boxes) == 0:
            return None, None

        aligned_images = []
        for box in boxes:
            x, y, w, h = box
            cropped = image[y - margin_half:y + h + margin_half, x - margin_half:x + w + margin_half, :]
            # input of facenet is 160 x 160
            try:
                aligned = cv2.resize(cropped, (160, 160))
                aligned_images.append(aligned)
            except:
                logger.info("Crop error")

        if len(aligned_images) == 0:
            return None, None

        aligned_images = np.array(aligned_images)
        prewhiten_images = self.prewhiten(aligned_images)

        predicts = self.model.predict(prewhiten_images)

        vectors = [np.array(self.l2_normalize(predict)).tolist() for predict in predicts]
        return boxes, vectors

    def recognize_faces(self):
        boxes, predicts = self.detect_faces(self.image)
        self.faces = []
        if boxes is None:
            return
        for box, predict in zip(boxes, predicts):
            face_vector = np.array(self.l2_normalize(predict)).tolist()
            min_value = 9
            label = "unknown"

            # scan all vectors from file json, choose closest vector with this face
            for vectors in self.data:  # vectors are list people
                for vector in vectors["values"]:  # one person has several faces
                    dis = self.cal_distance(vector, face_vector)
                    if dis < min_value and dis < 1.0:  # only accept vector has distance < 1.0
                        min_value = dis
                        label = vectors["name"]
            self.faces.append({"box": box, "label": label})
        self.check()
        # print(self.faces)
        logger.info("Done facenet: " + str(self.faces))

    def paint(self):
        if len(self.faces) == 0:
            return
        for face in self.faces:
            x, y, w, h = face["box"]
            cv2.rectangle(self.image, (x, y), (x + w, y + h), self.color_box, 2)
            cv2.putText(self.image, face["label"], (x + 10, y + 25), self.font, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

    def predict(self):
        self.model = load_model("models/facenet_keras.h5")
        self.image = cv2.imread("images/test/test6.jpg")
        self.recognize_faces()
        self.image = None
        self.faces = []
        # time.sleep(1)
        self.begin = True
        while True:
            if self.image is not None:
                self.recognize_faces()

    def train(self):
        self.model = load_model("models/facenet_keras.h5")

        # select all folders in Train, each folder name is a label
        # each folder contains list of training image
        train_names = listdir('images/train')

        data = []

        for name in train_names:
            arr = []
            path = join('images/train', name)
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for file in files:
                filepath = join(path, file)

                # read image, get face and calculate vector output
                image = cv2.imread(filepath)
                _, vectors = self.detect_faces(image)

                # each image contains only one face => element 0
                arr.append(vectors[0])

            # each face contains name and list vectors
            data.append({'name': name, 'values': arr})

        data = {'data': data}

        with open('models/data.json', 'w') as file:
            json.dump(data, file)

    def check(self):
        if not self.begin:
            return

        old_faces = []
        if len(self.faces) > 0:
            for face in self.faces:
                label = face["label"]
                if label == "unknown":
                    continue
                if label not in self.old_faces:
                    self.sio.emit("status", label + " is detected")
                old_faces.append(label)
        self.old_faces = old_faces

    def get_image(self):
        while True:
            if self.begin:
                break
            time.sleep(0.2)

        while True:
            ret, self.image = self.cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            self.paint()
            data = cv2.imencode(".jpg", self.image)[1].tobytes()
            self.sio.emit("image", base64.b64encode(data))

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            time.sleep(0.04)
        self.cam.release()

    def run(self):
        Thread(target=self.predict, daemon=True).start()
        Thread(target=self.get_image, daemon=True).start()

        self.app.run(host="0.0.0.0", port=5012, threaded=True)


if __name__ == "__main__":
    timekeeper = Timekeeper()
    # timekeeper.train()
    timekeeper.run()
