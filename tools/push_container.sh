#!/bin/sh

NEW_TAG_ID=$(docker images | ag 1.2-train-ppyoloe | wc -l)
LOCAL_IMAGE_TAG=aiserver-train
REMOTE_IMAGE_TAG=registry.cn-hangzhou.aliyuncs.com/kaixjl/aiserver:1.2-train-ppyoloe
REMOTE_IMAGE_TAG_W_ID=${REMOTE_IMAGE_TAG}-${NEW_TAG_ID}

docker image tag ${LOCAL_IMAGE_TAG} ${REMOTE_IMAGE_TAG}
docker image tag ${LOCAL_IMAGE_TAG} ${REMOTE_IMAGE_TAG_W_ID}
docker push ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG_W_ID}