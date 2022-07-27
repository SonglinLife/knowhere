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

#include <any>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "IndexMeta/IndexParameter.h"

namespace knowhere {

using Value = std::any;
using ValuePtr = std::shared_ptr<Value>;


class DataSet {
 public:
    DataSet() = default;
    ~DataSet();

    void
    SetDim(int64_t&& v) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[meta::DIM] = std::make_shared<Value>(std::forward<int64_t>(v));
    }

    int64_t
    GetDim() {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<int64_t>(*(data_.at(meta::DIM)));
    }

    void
    SetRowCount(int64_t&& v) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[meta::ROWS] = std::make_shared<Value>(std::forward<int64_t>(v));
    }

    int64_t
    GetRowCount() {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<int64_t>(*(data_.at(meta::ROWS)));
    }

    void
    SetTensor(void* && data) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[meta::TENSOR] = std::make_shared<Value>(std::forward<void*>(data));
    }

    void*
    GetTensor() {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<void*>(*(data_.at(meta::TENSOR)));
    }

    void
    SetPid(int64_t* &&ids) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[meta::IDS] = std::make_shared<Value>(std::forward<void*>(ids));
    }

    int64_t*
    GetPid() {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<int64_t *>(*(data_.at(meta::IDS)));
    }

    void
    SetDis(float* &&dis) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[meta::DISTANCE] = std::make_shared<Value>(std::forward<float*>(dis));
    }

    float*
    GetDis() {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<float *>(*(data_.at(meta::DISTANCE)));
    }

    void
    SetLimits(size_t* &&lims) {
        std::lock_guard<std::mutex> lk(mutex_);
        data_[meta::LIMS] = std::make_shared<Value>(std::forward<size_t*>(lims));
    }

    size_t*
    GetLimits() {
        std::lock_guard<std::mutex> lk(mutex_);
        return std::any_cast<size_t *>(*(data_.at(meta::LIMS)));
    }

 private:
    std::mutex mutex_;
    std::map<std::string, ValuePtr> data_;
};

using DatasetPtr = std::shared_ptr<DataSet>;

DatasetPtr
GenResultDataset(const int64_t* ids, const float* distance = nullptr, const size_t* lims = nullptr) {
    auto ret_ds = std::make_shared<DataSet>();
    ret_ds->SetPid(ids);
    ret_ds->SetDis(distance);
    ret_ds->SetLimits(lims);
    return ret_ds;
}

}  // namespace knowhere
