#!/usr/bin/env python
#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/


from PIL import Image
import sys, glob, os, random, shutil, time
from flask import Flask, request, redirect, url_for
#all the settings
current_path_ = os.path.dirname(__file__)

singa_root_=os.path.abspath(os.path.join(current_path_,'../..'))

#data prepare settings
input_folder_=os.path.abspath(os.path.join(current_path_,"data/raw"))
output_folder_=os.path.abspath(os.path.join(current_path_,"data/out"))
temp_folder_=os.path.abspath(os.path.join(current_path_,"data/temp"))

meta_file_name_="meta.txt"
train_bin_file_name_="train.bin"
test_bin_file_name_="test.bin"
validate_bin_file_name_="validate.bin"
mean_bin_file_name_="mean.bin"
label_list_=[(0,"airplane"),
    (1,"truck"),
    (2,"ship"),
    (3,"dog"),
    (4,"cat"),
    (5,"deer"),
    (6,"bird"),
    (7,"automobile"),
    (8,"horse"),
    (9,"frog")]

#image size
size_=(32,32)  

#final label numbers
total_record_num_=60000
label_num_=10  

#data partial
train_rate_=5.0/6
test_rate_=1.0/6
validate_rate_=0.0

#training settings
model_name_="cifar10-cnn"
workspace_="examples/cifar10_py"
batch_size_=64
check_point_path_=workspace_+"/checkpoint/step1000-worker0"


allowd_extensions_ = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

#singa python libs
sys.path.append(os.path.join(singa_root_,'tool','python'))
from singa.driver import NeuralNet,Driver,strVector,layerVector,intVector,floatVector,DummyInputLayer,Layer,floatArray_frompointer
from singa.model import *
from singa.utils import kvstore, imgtool
from pb2.common_pb2 import RecordProto
app = Flask(__name__)

mean_record_=""
dummy_=""
net_=""
pixel_length_=0


def buildModel(argv):
    model = Sequential(model_name_,argv)

    model.add(Convolution2D(32, 5, 1, 2, w_std=0.0001, b_lr=2))
    model.add(MaxPooling2D(pool_size=(3,3), stride=2))
    model.add(Activation('relu'))
    model.add(LRN2D(3, alpha=0.00005, beta=0.75))

    model.add(Convolution2D(32, 5, 1, 2, b_lr=2))
    model.add(Activation('relu'))
    model.add(AvgPooling2D(pool_size=(3,3), stride=2))
    model.add(LRN2D(3, alpha=0.00005, beta=0.75))

    model.add(Convolution2D(64, 5, 1, 2))
    model.add(Activation('relu'))
    model.add(AvgPooling2D(pool_size=(3,3), stride=2))

    #label_num_ should be the same with input data label num 
    model.add(Dense(label_num_, w_wd=250, b_lr=2, b_wd=0, activation='softmax'))

    sgd = SGD(decay=0.004, momentum=0.9, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))

    topo = Cluster(workspace_)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)

    return model 



def generate_data_conf(
         backend = 'kvfile',
         batchsize = 1,
         random = 5000,
         shape = (3, 32, 32),
         std = 127.5,
         mean = 127.5
      ):

  # using cifar10 dataset
    path_train =os.path.join(output_folder_ ,train_bin_file_name_)
    path_test  =os.path.join(output_folder_ ,test_bin_file_name_)
    path_mean  =os.path.join(output_folder_ ,mean_bin_file_name_)


    store = Store(path=path_train, mean_file=path_mean, backend=backend,
              random_skip=random, batchsize=batchsize,
              shape=shape)

    data_train = Data(load='recordinput', phase='train', conf=store)

    store = Store(path=path_test, mean_file=path_mean, backend=backend,
              batchsize=batchsize,
              shape=shape)

    data_test = Data(load='recordinput', phase='test', conf=store)

    return data_train, data_test    


def train(model):

    X_train, X_test= generate_data_conf(batchsize=batch_size_)
    model.fit(X_train, nb_epoch=1000, with_test=True)
    result = model.evaluate(X_test, test_steps=100, test_freq=300)

def test(model):
    pass


@app.route("/")
def index():
    return "Hello World! This is SINGA DLAAS! Please send post request with image=file to '/predict' "

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in allowd_extensions_

@app.route('/predict', methods=['POST'])
def predict():
    global pixel_length_,mean_record_,net_,dummy_
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            im = Image.open(file).convert("RGB")
            im = imgtool.resize_to_center(im,size_)
            pixel = floatVector(pixel_length_) 
            byteArray = imgtool.toBin(im,size_)
            for i in range(pixel_length_):
                pixel[i]= byteArray[i]-mean_record_.data[i]

            #dummy data Layer
            
            shape = intVector(3)
            shape[0]=3
            shape[1]=size_[0]
            shape[2]=size_[1]
            dummy_.Feed(1,shape,pixel)

            #checkpoint_paths =getattr(m.jobconf, 'checkpoint_path')
            checkpoint_paths = strVector(1)
            checkpoint_paths[0]=check_point_path_
            net_.Load(checkpoint_paths)

            print "1" 
            dummyVector=layerVector(1)
            dummyVector[0]=dummy_.ToLayer()
            print len(net_.layers())
            for i,layer in enumerate(net_.layers()):
                #skip data layer
                if i==0:
                    continue 
                elif i==1:
                    layer.ComputeFeature(4,dummyVector)
                else:
                    layer.ComputeFeature(4,net_.srclayers(layer))
        
            #get result
            lastLayer=net_.layers()[-1]
            data = lastLayer.data(dummy_.ToLayer())
            prop =floatArray_frompointer(data.mutable_cpu_data())
            result=[]
            for i in range(label_num_):
                result.append((i,prop[i])) 
        
            result.sort(key=lambda tup: tup[1], reverse=True)
            
            label_map=dict()
            for item in label_list_:
               label_map[item[0]]=item[1] 
            response="" 
            for r in result:
                response+=str(label_map[r[0]])+str(r[1]) 
        
            return response 
    return "error"

def product(model):
    global pixel_length_,mean_record_,net_,dummy_
    #fake data layer
    X_train, X_test= generate_data_conf()

    model.layers.insert(0,X_test)
    model.build()
    #register layers
    d = Driver()
    d.Init(sys.argv)
    net_ = NeuralNet.CreateForTest(model.jobconf.neuralnet.SerializeToString())

   
    pixel_length_ = 3*size_[0]*size_[1]

    #minus mean and feed data
    key,mean_str = kvstore.FileStore().open(os.path.join(output_folder_,mean_bin_file_name_),"read").read()
    mean_record_ = RecordProto()  
    mean_record_.ParseFromString(mean_str)
    assert len(mean_record_.data)==pixel_length_

    dummy_ = DummyInputLayer()
    
    
    app.debug = True
    app.run()


if __name__=='__main__':
    
    print "please use -transform -data -test -product to specify different task"

    if "-transform" in sys.argv:
        total_record_num_=imgtool.transform_img(input_folder_,temp_folder_,size_)
    if "-data" in sys.argv:
        label_list_=imgtool.generate_kvrecord_data(temp_folder_,
                output_folder_,
                size_,     
                train_num=int(total_record_num_*train_rate_),
                test_num=int(total_record_num_*test_rate_),
                validate_num=int(total_record_num_*validate_rate_),
                meta_file_name=meta_file_name_,
                train_bin_file_name=train_bin_file_name_,
                test_bin_file_name=test_bin_file_name_,
                validate_bin_file_name=validate_bin_file_name_,
                mean_bin_file_name=mean_bin_file_name_
            )
        label_num_=len(label_list_)
    model=buildModel(sys.argv)
    if "-train" in sys.argv:
        train(model)
    elif "-test" in sys.argv:
        test(model)
    elif "-product" in sys.argv:
        product(model)