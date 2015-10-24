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

#include "singa/io/store.h"
#include "singa/io/kvfile_store.h"
#include "singa/io/textfile_store.h"
#include "singa/io/lmdb_store.h"

namespace singa { namespace io {
Store* CreateStore(const std::string& backend) {
  Store *store = nullptr;
  if (backend.compare("textfile") == 0) {
    store = new TextFileStore();
  } else if (backend.compare("kvfile") == 0) {
    store = new KVFileStore();
  }

  if (backend == "lmdb") {
    return new LMDBStore();
  }

#ifdef USE_OPENCV
  if (backend == "imagefolder") {
    return new ImageFolderStore();
  }
#endif

#ifdef USE_HDFS
  if (backend == "hdfs") {
    return new HDFSStore();
  }
#endif
  return store;
}

Store* OpenStore(const string& backend, const string& path, Mode mode) {
  auto store = CreateStore(backend);
  store->Open(path, mode);
  return store;
}
} /* io */

} /* singa */
