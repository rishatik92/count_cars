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

RUN pip3 install ruamel.yaml
RUN pip3 install opencv-python cvlib matplotlib tensorflow keras flask numpy ruamel.yaml
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git"
