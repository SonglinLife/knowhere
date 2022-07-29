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

#include "VecIndex.h"
#include "include/common/BinarySet.h"
#include "include/common/Config.h"
#include "include/knowhere/IndexMeta/IndexMeta.h"

namespace knowhere {

class VecIndexProxy final {
 public:
    VecIndexProxy(const VecIndexProxy& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        idx.node->IncRef();
        node = idx.node;
    }

    VecIndexProxy(VecIndexProxy&& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    VecIndexProxy&
    operator=(const VecIndexProxy& idx) {
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        if (idx.node == nullptr) {
            node = nullptr;
            return *this;
        }
        node = idx.node;
        node->IncRef();
        return *this;
    }

    VecIndexProxy&
    operator=(VecIndexProxy&& idx) {
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        node = idx.node;
        idx.node = nullptr;
        return *this;
    }
    
    ~VecIndexProxy() {
        if (node == nullptr)
            return;
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

    void
    Build(const DataSet& dataset, const Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        node->Build(dataset, config);
    }
    void
    Train(const DataSet& dataset, const Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        node->Train(dataset, metaPtr);
    }

    void
    Add(const DataSet& dataset, const Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        node->Train(dataset, metaPtr);
    }

    DataSet
    Search(const DataSet& dataset, const Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        return node->Search(dataset, metaPtr);
    }

    DataSet
    SearchByRange(const DataSet& dataset, const Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        return node->SearchByRange(dataset, metaPtr);
    }

    DataSet
    GetVectorById(const DataSet& dataset, const Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        return node->SearchByRange(dataset, metaPtr);
    }

    int
    Serialization(Config& config) {
        IndexMetaPtr metaPtr = node->GetMetaPtr();
        metaPtr->load(config);
        return node->Serialization(metaPtr);
    }

    int
    Deserialization(const BinarySet& binset) {
        return node->Deserialization(binset);
    }

    int64_t
    Size() const {
        return node->Size();
    }

 private:
    VecIndexProxy(VecIndex * node_) : node(node_) {
    }
    VecIndex* node;
};

}