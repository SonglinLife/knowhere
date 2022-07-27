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

#include <memory>
#include <utility>
#include <vector>

#include "faiss/faiss/Index.h"

#include "include/common/BinarySet.h"
#include "include/knowhere/Dataset.h"
#include "include/knowhere/IndexMeta/IVFIndexMeta.h"
#include "include/knowhere/VecIndex.h"

namespace knowhere {

class IndexIVFFlat : public VecIndex
{
public:
    IndexIVFFlat(/* args */);
    
    ~IndexIVFFlat();
    int
    Build(const DataSet& dataset) override;
    int
    Train(const DataSet& dataset) override;
    int
    Add(const DataSet& dataset) = 0;
    DataSet
    Search(const DataSet& dataset, const IndexMeta& param) const override;
    DataSet
    SearchByRange(const DataSet& dataset, const IndexMeta& param) const override;
    DataSet
    GetVectorById(const DataSet& dataset, const IndexMeta& param) const override;
    int
    Serialization(BinarySet& binset) override;
    int
    Deserialization(const BinarySet& binset) override;
    int64_t
    Size() const = 0;
private:
    IVFIndexMeta meta;

    std::shared_ptr<faiss::Index> index_ = nullptr;
};

}