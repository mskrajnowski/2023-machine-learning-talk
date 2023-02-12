FROM debian:11-slim

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
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
ENV PATH=/opt/conda/bin:${PATH}

# Install mamba forge
# https://github.com/conda-forge/miniforge/releases/tag/22.11.1-4
# Based on https://github.com/conda-forge/miniforge-images/blob/050f7f2a954549fcfef2da3a30ef9fc5c328cc5b/ubuntu/Dockerfile
RUN mkdir -p /opt/conda && chown dev:dev /opt/conda
USER dev
RUN \
  ARCH=$(uname -m) \
  && curl -LO https://github.com/conda-forge/miniforge/releases/download/22.11.1-4/Mambaforge-22.11.1-4-Linux-$ARCH.sh \
  && echo \
  '16c7d256de783ceeb39970e675efa4a8eb830dcbb83187f1197abfea0bf07d30  Mambaforge-22.11.1-4-Linux-x86_64.sh\n' \
  '96191001f27e0cc76612d4498d34f9f656d8a7dddee44617159e42558651479c  Mambaforge-22.11.1-4-Linux-aarch64.sh' \
  | sha256sum --check --ignore-missing \
  && bash Mambaforge-22.11.1-4-Linux-$ARCH.sh -b -u -p /opt/conda \
  && rm Mambaforge-22.11.1-4-Linux-$ARCH.sh \
  && conda clean --tarballs --index-cache --packages --yes \
  && find /opt/conda -follow -type f -name '*.a' -delete \
  && find /opt/conda -follow -type f -name '*.pyc' -delete \
  && conda clean --force-pkgs-dirs --all --yes

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/dev/.bashrc
RUN echo ". /opt/conda/etc/profile.d/mamba.sh" >> /home/dev/.bashrc

# Install pytorch
ARG PYTORCH_INSTALL_FLAGS="cpuonly -c pytorch"
RUN mamba install -y pytorch torchvision torchaudio $PYTORCH_INSTALL_FLAGS

# Install jupyter notebook
RUN mamba install -y \
  jupyterlab \
  notebook \
  ipywidgets

# Workaround for error during jupyter_contrib_nbextensions installation:
# pkg_resources.DistributionNotFound: The 'webcolors>=1.11; extra == "format-nongpl"'
# distribution was not found and is required by jsonschema
RUN conda config --set pip_interop_enabled True
RUN pip install jsonschema[format-nongpl]

# Install jupyter extensions
RUN mamba install -y jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --sys-prefix --debug

# Install base packages
RUN mamba install -y \
  ipython \
  numpy \
  pandas \
  sympy \
  matplotlib \
  scikit-learn

RUN mkdir -p /home/dev/workspace
WORKDIR /home/dev/workspace
CMD bash
