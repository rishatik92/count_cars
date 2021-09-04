import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import numpy as np
import cv2, threading
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
from ruamel import yaml
from cvlib.object_detection import draw_bbox
CAR_BOXES  = [3, 8, 6]
with open('config.yaml') as cf:
    config = yaml.safe_load(cf.read())

import telegram

bot = telegram.Bot(token=config['telegram_bot_tocken'])


def send_message(msg):
    bot.send_message(config['chat_id'], msg)


def send_image(image):
    bot.send_photo(config['chat_id'], image)


# Конфигурация, которую будет использовать библиотека Mask-RCNN.
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # в датасете COCO находится 80 классов + 1 фоновый класс.
    DETECTION_MIN_CONFIDENCE = 0.6


# Фильтруем список результатов распознавания, чтобы остались только автомобили.
def get_boxes(boxes, class_ids, boxes_filter):
    boxes = []

    for i, box in enumerate(boxes):
        # Если найденный объект не входит в фильтр, то пропускаем его.
        if not boxes_filter:
            car_boxes.append(box)
        else:
            if class_ids[i] in boxes_filter:
                car_boxes.append(box)

    return np.array(boxes)


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


# Корневая директория проекта.
ROOT_DIR = Path(".")

# Директория для сохранения логов и обученной модели.
MODEL_DIR = ROOT_DIR / "logs"

# Локальный путь к файлу с обученными весами.
COCO_MODEL_PATH = ROOT_DIR / "mask_rcnn_coco.h5"

# Загружаем датасет COCO при необходимости.
if not COCO_MODEL_PATH.exists():
    mrcnn.utils.download_trained_weights(f'{COCO_MODEL_PATH}')

# Директория с изображениями для обработки.
IMAGE_DIR = ROOT_DIR / "images"

# Видеофайл или камера для обработки — вставьте значение 0, если нужно использовать камеру, а не видеофайл.
VIDEO_SOURCE = config['video_source']

# Создаём модель Mask-RCNN в режиме вывода.
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Загружаем предобученную модель.
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Местоположение парковочных мест.
parked_car_boxes = None

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
font = cv2.FONT_HERSHEY_DUPLEX
while video_capture.isOpened():

    if cam_cleaner.last_frame is None:
        continue
    frame = cam_cleaner.last_frame

    i += 1
    rgb_image = frame[:, :, ::-1]
    results = model.detect([rgb_image], verbose=0)
    # Mask R-CNN предполагает, что мы распознаём объекты на множественных изображениях.
    # Мы передали только одно изображение, поэтому извлекаем только первый результат.
    r = results[0]
    # Переменная r теперь содержит результаты распознавания:
    # - r['rois'] — ограничивающая рамка для каждого распознанного объекта;
    # - r['class_ids'] — идентификатор (тип) объекта;
    # - r['scores'] — степень уверенности;
    # - r['masks'] — маски объектов (что даёт вам их контур).

    # Фильтруем результат для получения рамок автомобилей.
    car_boxes = get_boxes(r['rois'], r['class_ids'], [])

    # Отображаем каждую рамку на кадре.
    for box in car_boxes:
        y1, x1, y2, x2 = box
        # Рисуем рамку.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 30)
        cv2.putText(frame, f"SPACE AVAILABLE!", (10, 150), font, 3.0, (0, 255, 0), 20, cv2.FILLED)
    # Подаём изображение модели Mask R-CNN для получения результата.

    frames_computed += 1
    debug_str = f'frames_computed: {frames_computed}'
    print(last_string_num * '\r' + debug_str, end='')
    last_string_num = len(debug_str)
    if i >= max_i:
        i = 0
    if len(car_boxes) == 0:
        continue

    # Показываем кадр в чат.
    send_message(f'car_boxes = {len(car_boxes)} frames_computed={frames_computed}')
    image = cv2.imencode('.jpeg', frame)[1].tostring()
    send_image(image)

# Очищаем всё после завершения.
video_capture.release()
