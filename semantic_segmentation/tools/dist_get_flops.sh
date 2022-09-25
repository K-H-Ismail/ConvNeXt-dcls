#!/usr/bin/env bash

CONFIG=$1
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python   \
    $(dirname "$0")/get_flops.py $CONFIG
