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

/*interface file for swig */

%module driver
%include "std_vector.i"
%include "std_string.i"
%include "argcargv.i"
%include "carrays.i"
%array_class(float, floatArray);

%apply (int ARGC, char **ARGV) { (int argc, char **argv)  }
%{
#include "singa/driver.h"
#include "singa/neuralnet/neuralnet.h"
#include "singa/neuralnet/layer.h"
#include "singa/neuralnet/input_layer.h"
#include "singa/utils/blob.h"
%}

namespace std {
  %template(strVector) vector<string>;
  %template(intVector) vector<int>;
  %template(floatVector) vector<float>;
  %template(layerVector) vector<singa::Layer*>;
}

namespace singa{
  class Driver{
    public:
    void Train(bool resume, const std::string job_conf);
    void Init(int argc, char **argv);
    void InitLog(char* arg);
    void Test(const std::string job_conf);
  };

  class NeuralNet{
    public:
     static NeuralNet* CreateForTest(const std::string str);
     void Load(const std::vector<std::string>& paths);
     inline const std::vector<singa::Layer*>& layers();
     inline const std::vector<singa::Layer*>& srclayers(const singa::Layer* layer);
  };
    
  class DummyInputLayer{
    public:
      void Feed(int batchsize, std::vector<int> shape, std::vector<float>* data);
      singa::Layer* ToLayer();
  };

  %nodefault Layer;
  class Layer{
    public:
      virtual void ComputeFeature(int flag, const std::vector<singa::Layer*>& srclayers); 
      virtual const singa::Blob<float>& data(const singa::Layer* from); 
  };
  

  template <typename Dtype>
  class Blob{
    public:
      inline Dtype* mutable_cpu_data(); 
  };

  %template(floatBlob) Blob<float>;
}

