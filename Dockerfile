FROM huggingface/transformers-pytorch-gpu:4.3.3

RUN groupadd -r docker-user && useradd -r -m -s /bin/false -g docker-user docker-user

RUN apt update && apt install -y less nano jq git

COPY bash.bashrc /etc/bash.bashrc

ARG DOCKER_WORKSPACE_PATH
RUN mkdir -p $DOCKER_WORKSPACE_PATH/src $DOCKER_WORKSPACE_PATH/.home
WORKDIR $DOCKER_WORKSPACE_PATH/src
ENV HOME=$DOCKER_WORKSPACE_PATH/.home

RUN pip install notebook \ 
                pytest
