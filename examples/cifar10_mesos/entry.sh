#!/bin/bash
cd /workspace
wget $1
tar zxf *.tar.gz
cd /usr/src/incubator-singa/examples/cifar10_mesos/
if [ $2 -eq '2' ]; then
  python main.py
  wait
else
  cd /usr/src/incubator-singa/
  python tool/python/examples/train_cifar10.py
  wait
fi
