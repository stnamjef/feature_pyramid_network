FROM nvidia/cuda:10.2-base
RUN apt-get update
RUN apt-get -y install \ 
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev \
    wget libbz2-dev liblzma-dev libgl1-mesa-glx libglib2.0-0
RUN mkdir /workspace
WORKDIR /workspace
RUN wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
RUN tar -xf Python-3.8.0.tgz
WORKDIR Python-3.8.0
RUN ./configure --enable-optimizations
RUN make -j 8
RUN make install
WORKDIR /workspace
RUN rm -rf Python-3.8.0
RUN rm -rf Python-3.8.0.tgz
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
RUN pip3 install torchnet pycocotools tqdm scikit-image opencv-python