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

#include "Dataset.h"
#include "include/common/Object.h"

namespace knowhere {

class VecIndex : public Object {
    friend class VecIndexProxy;
public:
    virtual int
    Build(const DataSet& dataset) = 0;
    virtual int
    Train(const DataSet& dataset) = 0;
    virtual int
    Add(const DataSet& dataset) = 0;
    virtual DataSet
    Search(const DataSet& dataset, const IndexMeta& param) const = 0;
    virtual DataSet
    SearchByRange(const DataSet& dataset, const IndexMeta& param) const = 0;
    virtual DataSet
    GetVectorById(const DataSet& dataset, const IndexMeta& param) const = 0;
    virtual int
    Serialization(BinarySet& binset) = 0;
    virtual int
    Deserialization(const BinarySet& binset) = 0;
    virtual int64_t
    Size() const = 0;
protected:
    virtual IndexMeta&
    GetMeta() const = 0;
    virtual IndexMeta&
    GenParam() const = 0;
};

}