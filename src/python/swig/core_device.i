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

%module core_device
%include "std_vector.i"
%include "std_string.i"
%include "std_pair.i"
%include "std_shared_ptr.i"

%{
#include "singa/core/device.h"
%}

/* smart pointer to avoid memory leak */
%shared_ptr(singa::Device);

namespace std{
%template(Shape) std::vector<size_t>;
%template(SizePair) std::pair<size_t, size_t>;
%template(vectorPair) std::vector<std::pair<size_t, size_t>>;
%template(vectorSharedPtr) std::vector<std::shared_ptr<singa::Device>>;
}

namespace singa{

class Device {
  public:
  virtual void SetRandSeed(unsigned seed) = 0;
  std::shared_ptr<Device> host();
  int id() const;
};

class Platform {
 public:
  static int GetNumGPUs();
  static const vector<int> GetGPUIDs();
  static const std::pair<size_t, size_t> GetGPUMemSize(const int device);
  static const vector<std::pair<size_t, size_t>> GetGPUMemSize();
  static const string DeviceQuery(int id, bool verbose = false);
  static const vector<shared_ptr<Device> >
  CreateCudaGPUs(const size_t num_devices, size_t init_size = 0);
};
}

