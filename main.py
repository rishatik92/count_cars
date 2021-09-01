import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from flask import Flask, flash, request, redirect, url_for

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'img', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # decode the array into an image
            img = cv2.imdecode(file.read(), cv2.IMREAD_UNCHANGED)
            bbox, label, conf = cv.detect_common_objects(img)
            output_image = draw_bbox(img, bbox, label, conf)
            plt.imshow(output_image)
            plt.show()
            print()
            return f'''Number of cars in the image is {label.count('car')}'''
    return '''
    <!doctype html>
    <title>test app rishat</title>
    <h1>Upload new image with car</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
