# 2023-machine-learning-talk

## CPU-only setup

1. Install [docker](https://docs.docker.com/engine/install/)
2. Build container images

   ```sh
   docker compose build shell
   ```

3. Setup the virtualenv

   ```sh
   docker compose run --rm shell python -m venv .venv
   docker compose run --rm shell pip install --no-deps -r requirements.cpu.txt
   ```

4. Enable nbextensions

   ```sh
   jupyter contrib nbextension install --sys-prefix
   ```

5. Run jupyter notebook

   ```sh
   docker compose up notebook
   ```
6. Jupyter server should print out a link you can use to open the notebook, it should look like http://127.0.0.1:8888/?token=abcd1234...

## GPU acceleration

### Nvidia CUDA

1. Install and enable `nvidia-container-toolkit`

   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

2. Enable CUDA-specific compose overrides

    ```sh
    ln -s ./docker-compose.cuda.yml ./docker-compose.override.yml
    ```

3. Rebuild container images

    ```sh
    docker compose build shell
    ```

4. Recreate the virtualenv

    ```sh
    rm -rf .venv
    docker compose run --rm shell python -m venv .venv
    docker compose run --rm shell pip install --no-deps -r requirements.cuda.txt
    ```

5. Check if everything worked

    ```sh
    docker compose run --rm shell python -c \
        'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
    ```

    should print out

    ```
    CUDA available: True
    ```
