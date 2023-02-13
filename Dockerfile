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
RUN mkdir -p /home/dev/workspace
WORKDIR /home/dev/workspace

# Setup virtualenv
USER root
RUN mkdir -p /opt/python && chown dev:dev /opt/python
USER dev
RUN python -m venv /opt/python
ENV PATH=/opt/python/bin:$PATH

# Install pip-tools
RUN pip install --upgrade pip==23.0 pip-tools==6.12.2

# Install packages
ARG COMPUTE_DEVICE=cpu
COPY ./requirements.$COMPUTE_DEVICE.txt ./
RUN pip install --no-deps -r requirements.$COMPUTE_DEVICE.txt

# Setup jupyter
RUN jupyter contrib nbextension install --sys-prefix

CMD bash
