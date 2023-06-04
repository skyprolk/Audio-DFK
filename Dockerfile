FROM nvidia/cuda:11.8.0-base-ubuntu22.04 as cuda

ENV PYTHON_VERSION=3.10

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    libsndfile1-dev \
    git \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

FROM cuda as app
# 2. Copy files
COPY . /src

WORKDIR /src
# 3. Install dependencies
RUN pip install -r requirements-pip.txt

# 4. Install notebook
RUN pip install encodec rich-argparse

EXPOSE 8082
CMD ["python", "bark_webui.py"]
