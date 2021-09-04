import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import cv2
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


VIDEO_SOURCE = config['video_source']
# Загружаем видеофайл, для которого хотим запустить распознавание.
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# Проходимся в цикле по каждому кадру.
max_i = 300
i = 0
every_print = 5
assert every_print < max_i
frames_computed = 0
last_string_num = 0
while video_capture.isOpened():

    success, frame = video_capture.read()

    i +=1
    if not success:
        break
    bbox, label, conf = cv.detect_common_objects(frame)
    frames_computed+=1
    debug_str = f'frames_computed: {frames_computed}, video_capture = {video_capture} , dir {dir(video_capture)}'
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
        text += f'{type_vehicle}s count = {label.count(type_vehicle)}\n'
        summary_vehicle += label.count(type_vehicle)
    text += f'Summary: {summary_vehicle}'

    output_image = draw_bbox(frame, bbox, label, conf)
    send_image(cv2.imencode('.jpeg', frame)[1].tostring())
    send_message(f"{text} frames_computed = {frames_computed}")


