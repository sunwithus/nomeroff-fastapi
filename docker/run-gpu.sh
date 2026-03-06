#!/bin/bash

docker run --rm -it \
	-p 8904:8904 \
	--privileged --gpus all \
	-v `pwd`/..:/project/nomeroff-net \
	nomeroff-net-gpu bash
