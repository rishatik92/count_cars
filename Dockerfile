FROM tensorflow/tensorflow:latest-gpu
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /var/www
WORKDIR /var/www

RUN apt-get update
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libglib2.0 git libgl1-mesa-glx libturbojpeg wget libglib2.0

RUN pip3 install ruamel.yaml opencv-python cvlib matplotlib tensorflow keras==2.1.6 flask numpy ruamel.yaml tqdm seaborn Cython torch torchvision GitPython scikit-image python-telegram-bot
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git" "git+https://github.com/ria-com/modelhub-client.git"
