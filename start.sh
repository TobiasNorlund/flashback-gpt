#!/bin/bash
set -e

# Docker image name for this project
export DOCKER_IMAGE_NAME="tobias/flashback-gpt"

# Path to where in the docker container the project root will be mounted
export DOCKER_WORKSPACE_PATH="/workspace"

# Path to repo/project root dir (independent of pwd)
export PROJECT_ROOT=$( cd $(dirname $( dirname "${BASH_SOURCE[0]}") ); pwd )

# Path to data dir
export DATA_DIR="$PROJECT_ROOT/data"

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --gpu)
    RUNTIME_ARGS="--gpus all"
    shift # past argument
    ;;
    -v|--mount)
    MOUNT="-v $2"
    shift # past argument
    shift # past value
    ;;
    -d|--detach)
    DETACH="--detach"
    shift # past argument
    ;;
    *)    # unknown option
    echo "Unrecognized argument '$1'"
    exit 1
    ;;
esac
done

USER_MAP="-u $(id -u):$(id -g)"
CONTAINER_NAME=${DOCKER_IMAGE_NAME##*/}

# Stop any potentially running container with the same name
docker stop $CONTAINER_NAME 2> /dev/null || true

set -x
docker build --rm --build-arg DOCKER_WORKSPACE_PATH -t $DOCKER_IMAGE_NAME $PROJECT_ROOT
docker run --rm -it \
  --name $CONTAINER_NAME \
  -v $PROJECT_ROOT:$DOCKER_WORKSPACE_PATH \
  -p 8888:8888 \
  $MOUNT \
  $USER_MAP \
  $RUNTIME_ARGS \
  $DETACH \
  $DOCKER_IMAGE_NAME bash
