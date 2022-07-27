// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License
#pragma once

#include <string>

#include "IndexType.h"
#include "MetricType.h"
#include "include/common/Bitset.h"

namespace knowhere {

struct IndexMeta {
     /* index meta */
    std::string name;
    
    int64_t dim;

    int64_t count;
    
    int64_t size;

    IndexType type;

    IndexMode mode;

    bool is_trained;

    MetricType metricType;

    void *p_data;

    /**
     * search meta
     */
     int64_t nprobe;

     int64_t topk;

     BitSet bitset;

     float radius;

    virtual ~IndexMeta() {

    }
};

}