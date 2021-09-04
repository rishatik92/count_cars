import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import cv2, threading
import cvlib as cv
from cvlib.object_detection import draw_bbox
import telegram
from ruamel import yaml

with open('config.yaml') as cf:
    config = yaml.safe_load(cf.read())

bot = telegram.Bot(token=config['telegram_bot_tocken'])


def send_message(msg):
    bot.send_message(config['chat_id'], msg)


def send_image(image):
    bot.send_photo(config['chat_id'], image)
# Define the thread that will continuously pull frames from the camera

class CameraBufferCleanerThread(threading.Thread):
    # Камера поставляет фреймы быстрее чем мы их обрабатываем, поэтому будем дропать буфер
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()


VIDEO_SOURCE = config['video_source']
# Загружаем видеофайл, для которого хотим запустить распознавание.
video_capture = cv2.VideoCapture(VIDEO_SOURCE)
cam_cleaner = CameraBufferCleanerThread(video_capture)
# Проходимся в цикле по каждому кадру.
max_i = 300
i = 0
every_print = 5
assert every_print < max_i
frames_computed = 0
last_string_num = 0
while video_capture.isOpened():
    if cam_cleaner.last_frame is None:
        continue
    frame = cam_cleaner.last_frame

    i +=1
    bbox, label, conf = cv.detect_common_objects(frame)
    frames_computed+=1
    debug_str = f'frames_computed: {frames_computed}'
    print(last_string_num * '\r' + debug_str, end='')
    last_string_num = len(debug_str)
    if i >= max_i:
        i = 0
    if i != every_print:
        continue

    text = 'On the image: \n'
    set_label = set(label)
    summary_vehicle = 0
    for type_vehicle in set_label:
        text += f'{type_vehicle}s count: {label.count(type_vehicle)}\n'
        summary_vehicle += label.count(type_vehicle)
    text += f'Summary: {summary_vehicle}'

    output_image = draw_bbox(frame, bbox, label, conf)
    send_image(cv2.imencode('.jpeg', frame)[1].tostring())
    send_message(f"{text} frames_computed: {frames_computed}")


