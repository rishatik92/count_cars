import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import cv2, io
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
from flask import Flask, flash, request, redirect, url_for
import telegram
from ruamel import yaml

with open('config.yaml') as cf:
    config = yaml.safe_load(cf.read())

bot = telegram.Bot(token=config['telegram_bot_tocken'])


def send_message(msg):
    bot.send_message(config['chat_id'], msg)


def send_image(image):
    bot.send_photo(config['chat_id'], image)


VIDEO_SOURCE = config['video_source']
# Загружаем видеофайл, для которого хотим запустить распознавание.
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Проходимся в цикле по каждому кадру.
max_i = 2000
i = 0
frames_computed = 0
while video_capture.isOpened():

    success, frame = video_capture.read()
    # Конвертируем изображение из цветовой модели BGR (используется OpenCV) в RGB.
    rgb_image = frame[:, :, ::-1]
    i +=1
    if not success:
        break
    bbox, label, conf = cv.detect_common_objects(rgb_image)
    frames_computed+=1
    if i >= max_i:
        i = 0
    if i !=100:
        continue

    output_image = draw_bbox(rgb_image, bbox, label, conf)
    plt.imshow(output_image)
    plt.show()
    img = io.BytesIO()              # create file-like object in memory to save image without using disk
    plt.savefig(img, format='png')  # save image in file-like object
    img.seek(0)
    send_image(img)
    send_message(f'''Number of cars in the image is {label.count('car')}
frames_computed = {frames_computed}''')

