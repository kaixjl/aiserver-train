#!/bin/bash

IMAGE_TAG=aiserver-train
docker rmi ${IMAGE_TAG}
docker build -f docker/dockerfile -t ${IMAGE_TAG} .