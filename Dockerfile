FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt update && \
    apt install -y \
    wget \
    bzip2 \
    build-essential \
    git \
    git-lfs \
    imagemagick \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3 \
    python3-pip

RUN pip3 install --no-cache-dir -U pip

RUN pip3 install \
    requests \
    scikit-learn \
    torch \
    torchvision \
    pytest \
    numpy \
    pandas \
    matplotlib \
    torchtext \
    wandb \ 
    python-dotenv

RUN export PYTHONPATH="$(pwd)"