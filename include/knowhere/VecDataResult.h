

//  Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
//  with the License. You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software distributed under the License
//  is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
//  or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <cstring>
namespace knowhere {

class VecDataResult {
 public:
    VecDataResult(int64_t* ids, float* distance, size_t* lims) : ids(ids), distance(distance), lims(lims) {
    }

    VecDataResult(int64_t* ids, float* distance) : ids(ids), distance(distance) {
    }

    VecDataResult(void* tensor) : tensor(tensor) {
    }

    void*
    getTensor() const {
        return tensor;
    }

    int64_t*
    getIds() const {
        return ids;
    }
    float*
    getDistance() const {
        return distance;
    }
    size_t*
    getLims() const {
        return lims;
    }

 private:
    void *tensor;

    int64_t *ids;

    float *distance;

    size_t *lims;
};

using VecDataResultPtr = std::shared_ptr<VecDataResult>;

}
