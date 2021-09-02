FROM  python:3.9
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libglib2.0 git libgl1-mesa-glx wget libglib2.0

RUN pip3 install ruamel.yaml opencv-python cvlib matplotlib tensorflow keras flask numpy ruamel.yaml tqdm seaborn Cython torch torchvision GitPython scikit-image python-telegram-bot tensorflow
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git" "git+https://github.com/ria-com/modelhub-client.git" "git+https://github.com/matterport/Mask_RCNN.git"
#fix masc_RCNN
RUN tf_upgrade_v2 --intree Mask_RCNN --inplace --reportfile report.txt
RUN sed -i 's/import keras.engine as KE/import keras.engine as KE\nfrom keras.layers import Layer/g' /usr/local/lib/python3.9/site-packages/mrcnn/model.py
RUN sed -i 's/KE.Layer/Layer/g' /usr/local/lib/python3.9/site-packages/mrcnn/model.py
RUN sed -i 's/tf.log(/tf.math.log(/g' /usr/local/lib/python3.9/site-packages/mrcnn/model.py
RUN sed -i 's/tf.sets.set_intersection(/tf.sets.intersection(/g' /usr/local/lib/python3.9/site-packages/mrcnn/model.py
RUN sed -i 's/tf.sparse_tensor_to_dense(/tf.sparse.to_dense(/g' /usr/local/lib/python3.9/site-packages/mrcnn/model.py
RUN sed -i 's/tf.to_float()/tf.cast([value], tf.float32)/g' /usr/local/lib/python3.9/site-packages/mrcnn/model.py

CMD [ "python", "./find_place.py" ]
