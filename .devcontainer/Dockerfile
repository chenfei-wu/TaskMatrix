FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt update && apt install -y --no-install-recommends git ssh python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

ARG USER_NAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER_NAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER_NAME \
    && chmod 0440 /etc/sudoers.d/$USER_NAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USER_NAME