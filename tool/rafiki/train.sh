#!/bin/bash
cd /workspace
wget $1
tar zxf *.tar.gz
cp /workspace/model.py /usr/src/incubator-singa/tool/rafiki/
cd /usr/src/incubator-singa/
python tool/rafiki/main.py train $2 $3 $4 $5 $6
