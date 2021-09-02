FROM tensorflow/tensorflow:latest-gpu
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update

# For opencv
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libglib2.0

# For Mask_RCNN
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y git
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libgl1-mesa-glx

# turbojpeg
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libturbojpeg wget

RUN pip3 install "torch>=1.8"
RUN pip3 install "torchvision>=0.9"
RUN pip3 install "PyYAML>=5.3"
RUN pip3 install scikit_image
RUN pip3 install Cython
RUN pip3 install pycocotools
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install opencv_python
RUN pip3 install "numpy>=1.16.*"
RUN pip3 install imgaug
RUN pip3 install asyncio
RUN pip3 install GitPython
RUN pip3 install pycocotools
RUN pip3 install tqdm
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git"
RUN pip3 install flask
RUN pip3 install python-telegram-bot
RUN pip3 install -U 'git+https://github.com/matterport/Mask_RCNN.git'
RUN tf_upgrade_v2 --intree Mask_RCNN --inplace --reportfile report.txt
RUN sed -i 's/import keras.engine as KE/import keras.engine as KE\nfrom keras.layers import Layer/g' /usr/local/lib/python3.6/site-packages/mrcnn/model.py
RUN sed -i 's/KE.Layer/Layer/g' /usr/local/lib/python3.6/site-packages/mrcnn/model.py
RUN sed -i 's/tf.log(/tf.math.log(/g' /usr/local/lib/python3.6/site-packages/mrcnn/model.py
RUN sed -i 's/tf.sets.set_intersection(/tf.sets.intersection(/g' /usr/local/lib/python3.6/site-packages/mrcnn/model.py
RUN sed -i 's/tf.sparse_tensor_to_dense(/tf.sparse.to_dense(/g' /usr/local/lib/python3.6/site-packages/mrcnn/model.py
RUN sed -i 's/tf.to_float()/tf.cast([value], tf.float32)/g' /usr/local/lib/python3.6/site-packages/mrcnn/model.py
WORKDIR /var/www/
