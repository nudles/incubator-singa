#!/bin/bash
cd /workspace
wget $1
tar zxf *.tar.gz
cp /workspace/model.py /usr/src/incubator-singa/tool/python/examples/user1-cifar10/
if [ $2 -eq '2' ]; then
  cd /usr/src/incubator-singa/examples/cifar10_mesos/
  python main.py
else
  cd /usr/src/incubator-singa/
  python tool/python/examples/user1-cifar10/main.py $3 $4 $5 $6 $7
fi
