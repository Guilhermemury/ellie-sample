import cv2
import tensorflow as tf
from PIL import Image
import os
image_list = []

CATEGORIES = ["Dog", "Cat"]

middleX = 30
middleY = 50

def prepare(filepath):
    IMG_SIZE = 150  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return new_array

model = tf.keras.models.load_model("jovem.h5")
winname = "Test"
quantidade = 0
for filename in os.listdir("D:\ELLIE\Ellie"):
    if filename.endswith(".jpg"):
        quantidade += 1
    else:
        pass


while quantidade > 0:
    image = '{}.jpg'.format(quantidade)
    images = cv2.imread(image)
    quantidade -= 1
    prediction = model.predict([prepare(image)])
    if prediction == 1:
        result = "Jovem"

    else:
        result = "Adulto"

    percent = "{:.2f}%".format(prediction[0][0] * 100)
    name = "{}".format(result)

    cv2.putText(images, percent, (middleX, middleY),cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
    cv2.putText(images, name, (middleX, middleY + 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.moveWindow(winname, 40, 30)
    cv2.imshow(winname, images)
    cv2.waitKey(0)


