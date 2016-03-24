#!/bin/bash
cd /workspace
wget $1
tar zxf *.tar.gz
cd /usr/src/incubator-singa/examples/cifar10_mesos/
python main.py
