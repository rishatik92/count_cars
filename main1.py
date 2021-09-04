import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import cv2, imutils, io, telegram
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
from cvlib.object_detection import draw_bbox
from flask import Flask, flash, request, redirect, url_for
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
max_i = 300
i = 0
every_print = 5
assert every_print < max_i
frames_computed = 0

class Box:
    def __init__(self, start_point, width_height):
        self.start_point = start_point
        self.end_point = (start_point[0] + width_height[0], start_point[1] + width_height[1])
        self.counter = 0
        self.frame_countdown = 0

    def overlap(self, start_point, end_point):
        if self.start_point[0] >= end_point[0] or self.end_point[0] <= start_point[0] or \
                self.start_point[1] >= end_point[1] or self.end_point[1] <= start_point[1]:
            return False
        else:
            return True



video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# We will keep the last frame in order to see if there has been any movement
last_frame = None

# To build a text string with counting status
text = ""

# The boxes we want to count moving objects in
boxes = []
boxes.append(Box((100, 200), (10, 80)))
boxes.append(Box((300, 350), (10, 80)))

while video_capture.isOpened():
    success, frame = video_capture.read()
    i += 1
    if not success:
        break


    # Processing of frames are done in gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # We blur it to minimize reaction to small details
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Need to check if we have a lasqt_frame, if not get it
    if last_frame is None or last_frame.shape != gray.shape:
        last_frame = gray
        continue

    # Get the difference from last_frame
    delta_frame = cv2.absdiff(last_frame, gray)
    last_frame = gray
    # Have some threshold on what is enough movement
    thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    # This dilates with two iterations
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Returns a list of objects
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Converts it
    contours = imutils.grab_contours(contours)

    # Loops over all objects found
    for contour in contours:
        # Skip if contour is small (can be adjusted)
        if cv2.contourArea(contour) < 500:
            continue

        # Get's a bounding box and puts it on the frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # The text string we will build up
        text = "Cars:"
        # Go through all the boxes
        for box in boxes:
            box.frame_countdown -= 1
            if box.overlap((x, y), (x + w, y + h)):
                if box.frame_countdown <= 0:
                    box.counter += 1
                # The number might be adjusted, it is just set based on my settings
                box.frame_countdown = 20
            text += " (" + str(box.counter) + " ," + str(box.frame_countdown) + ")"

    # Set the text string we build up
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Let's also insert the boxes
    for box in boxes:
        cv2.rectangle(frame, box.start_point, box.end_point, (255, 255, 255), 2)

    if i >= max_i:
        i = 0
        print(f'i = {i}, frames_computed{frames_computed}')
        image = cv2.imencode('.jpeg', frame)[1].tostring()
        send_image(image)

    if i != every_print:
        continue

video_capture.release()

