/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/neuralnet/input_layer.h"
namespace singa {

using std::vector;

DummyInputLayer::DummyInputLayer(){

}
void DummyInputLayer::Feed(int batchsize, vector<int> shape, vector<float>* data){

	batchsize_ = batchsize;
	shape.insert(shape.begin(),batchsize);
	data_.Reshape(shape);

	int size = data_.count() / batchsize_;
    CHECK_EQ(size, data->size());
    float* ptr = data_.mutable_cpu_data();
    for (int i = 0; i< size; i++)
	      ptr[i] = data->at(i);

    return;
}

}  // namespace singa
