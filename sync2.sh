#!/bin/bash

set -e

# SPATH=$(readlink -f $(dirname $0))
SPATH='/Users/user/PycharmProjects/'

SES_NAME="anemerov@192.168.1.7"

ssh $SES_NAME 'mkdir -p /home/anemerov/PycharmProjects/MILTestTasks'
rsync -vazL \
 --exclude '.git' \
 --exclude 'env' \
 --exclude '.idea' \
 --exclude '*egg-info' \
 --exclude 'result' \
 --exclude 'spark-warehouse' \
 --exclude 'build' \
 --exclude '*.pyc' \
 --exclude '__pycache__' \
 --exclude 'py' \
 --exclude '*.npy' \
 --exclude '*.log' \
 $SES_NAME:/home/anemerov/PycharmProjects/MILTestTasks $SPATH

