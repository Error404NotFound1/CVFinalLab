FROM nvcr.io/nvidia/pytorch:22.11-py3

ARG HTTP_PROXY=http://172.19.135.130:5000
ARG HTTPS_PROXY=http://172.19.135.130:5000
ARG USERNAME=app
ARG USER_UID=1000
ARG USER_GID=1000

ENV http_proxy=${HTTP_PROXY} \
	https_proxy=${HTTPS_PROXY} \
	HTTP_PROXY=${HTTP_PROXY} \
	HTTPS_PROXY=${HTTPS_PROXY} \
	NO_PROXY=localhost,127.0.0.1 \
	no_proxy=localhost,127.0.0.1 \
	HOME=/home/${USERNAME} \
	PYTHONUSERBASE=/home/${USERNAME}/.local \
	PYTHONPATH=/home/${USERNAME}/.local/lib/python3.8/site-packages \
	PATH=/home/${USERNAME}/.local/bin:$PATH \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_DEFAULT_TIMEOUT=120

RUN groupadd --gid "$USER_GID" "$USERNAME" \
	&& useradd --uid "$USER_UID" --gid "$USER_GID" --create-home --shell /bin/bash "$USERNAME"

RUN rm -rf /workspace/*
WORKDIR /workspace/unet

COPY requirements.txt .

USER ${USERNAME}

RUN pip install --user --no-cache-dir --upgrade --pre pip \
	&& pip install --user --no-cache-dir -r requirements.txt

COPY . .
