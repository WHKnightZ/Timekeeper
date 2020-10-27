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


class Timekeeper:
    def __init__(self):
        self.begin = False

        self.image = None
        self.old_faces = []
        self.faces = []
        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
        self.cam = cv2.VideoCapture(0)
        self.color_box = [0, 192, 0]
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.model = None

        # data.json save feature vectors of all employees
        with open("data.json") as json_file:
            self.data = json.load(json_file)["data"]

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
            return

        aligned_images = []
        for box in boxes:
            x, y, w, h = box
            cropped = image[y - margin_half:y + h + margin_half, x - margin_half:x + w + margin_half, :]
            # input of facenet is 160 x 160
            aligned = cv2.resize(cropped, (160, 160))
            aligned_images.append(aligned)

        aligned_images = np.array(aligned_images)
        prewhiten_images = self.prewhiten(aligned_images)

        predicts = self.model.predict_on_batch(prewhiten_images)

        vectors = [np.array(self.l2_normalize(predict)).tolist() for predict in predicts]
        return boxes, vectors

    def recognize_faces(self):
        boxes, predicts = self.detect_faces(self.image)
        self.faces = []
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
            cv2.putText(self.image, face["label"], (x + 10, y + 25), self.font, 1.2, (255, 255, 255), 1, cv2.LINE_AA)

    def predict(self):
        self.model = load_model("facenet_keras.h5")
        image = cv2.imread("images/test/test0.jpg")
        image = cv2.resize(image, (160, 160))
        images = np.array([image])
        self.model.predict_on_batch(images)
        time.sleep(1)
        self.begin = True
        while True:
            if self.image is not None:
                self.recognize_faces()

    def train(self):
        self.model = load_model("facenet_keras.h5")

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

        with open('data.json', 'w') as file:
            json.dump(data, file)

    def check(self):
        old_faces = []
        if len(self.faces) > 0:
            for face in self.faces:
                label = face["label"]
                if label == "unknown":
                    continue
                if label not in self.old_faces:
                    print(label + " is detected")
                old_faces.append(label)
        self.old_faces = old_faces

    def run(self):
        Thread(target=self.predict, daemon=True).start()

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
            cv2.imshow("Camera", self.image)

            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            time.sleep(0.04)

        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    timekeeper = Timekeeper()
    # timekeeper.train()
    timekeeper.run()
