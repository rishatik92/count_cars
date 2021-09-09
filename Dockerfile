FROM  python:3.9
ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libglib2.0 git wget libglib2.0

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
RUN mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
RUN dpkg -i cuda-repo-wsl-ubuntu-11-4-local_11.4.0-1_amd64.deb
RUN apt-key add /var/cuda-repo-wsl-ubuntu-11-4-local/7fa2af80.pub
RUN apt-get update
RUN apt-get -y install cuda


RUN pip3 install ruamel.yaml cvlib matplotlib tensorflow keras flask numpy ruamel.yaml tqdm seaborn Cython torch torchvision GitPython scikit-image python-telegram-bot cvlib
RUN pip3 install -U "git+git://github.com/lilohuang/PyTurboJPEG.git" "git+https://github.com/ria-com/modelhub-client.git" "git+https://github.com/akTwelve/Mask_RCNN.git"

RUN pip3 install pyyaml

RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.3.tar.gz
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.3.tar.gz
RUN tar -zxvf opencv.zip
RUN tar -zxvf opencv_contrib.zip
RUN mv opencv-* opencv
RUN mv opencv_contrib-* opencv_contrib
RUN apt-get -o Acquire::Max-FutureTime=86400 install build-essential cmake pkg-config gcc -y
RUN apt-get -o Acquire::Max-FutureTime=86400 install -y libgtk-3-dev python3-dev python3-tk libhdf5-serial-dev libopenblas-dev libatlas-base-dev liblapack-dev gfortran libxvidcore-dev libx264-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libjpeg-dev libpng-dev libtiff-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev screen
# download and install cudNN
COPY "libcudnn8_8.2.4.15-1+cuda11.4_amd64.deb" "/libcudnn8.deb"
COPY "libcudnn8-dev_8.2.4.15-1+cuda11.4_amd64.deb" "/libcudnn8_dev.deb"
RUN dpkg -i ./libcudnn8.deb
RUN dpkg -i ./libcudnn8_dev.deb
RUN mkdir -p opencv/build;
WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D INSTALL_C_EXAMPLES=OFF -D OPENCV_ENABLE_NONFREE=ON -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=7.5 -D WITH_CUBLAS=1 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -D HAVE_opencv_python3=ON -D PYTHON_EXECUTABLE=$(which python3) -D BUILD_EXAMPLES=ON ..
RUN bash -c "make -j12"
RUN make install
RUN ldconfig
RUN mkdir -p /var/www
WORKDIR /var/www
COPY find_cars_v1.py /var/www/
COPY test_yolov5.py /var/www/
COPY find_place.py /var/www/
COPY main.py /var/www/
COPY main1.py /var/www/
COPY config.yaml /var/www/
#CMD [ "python", "./test_yolov5.py" ]
