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
  nodejs \
  npm \
  graphviz \
  tree \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

USER dev
RUN mkdir -p /home/dev/workspace
WORKDIR /home/dev/workspace
ENV PATH=/home/dev/workspace/.venv/bin:$PATH
CMD bash
