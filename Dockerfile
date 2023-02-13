FROM python:3.10.10-slim-bullseye

# Create dev user
ARG GID=1000
ARG UID=1000
RUN groupadd --gid ${GID} dev
RUN useradd \
  --uid ${UID} \
  --gid ${GID} \
  --shell "/usr/bin/bash" \
  --create-home \
  dev

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  tini \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

USER dev
WORKDIR /home/dev

# Setup virtualenv
RUN python -m venv ./venv
ENV PATH=/home/dev/venv/bin:$PATH

# Install pytorch, default to CPU-only
ARG PYTORCH_INSTALL_FLAGS="--extra-index-url https://download.pytorch.org/whl/cpu"
RUN pip install torch torchvision torchaudio $PYTORCH_INSTALL_FLAGS

# Install jupyter notebook
RUN pip install \
  jupyterlab \
  notebook \
  ipywidgets \
  # fix broken nbextensions
  nbclassic==0.4.8

# Install jupyter extensions
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --sys-prefix

# Install base packages
RUN pip install \
  ipython \
  numpy \
  pandas \
  sympy \
  matplotlib \
  scikit-learn

RUN mkdir -p /home/dev/workspace
WORKDIR /home/dev/workspace
CMD bash
