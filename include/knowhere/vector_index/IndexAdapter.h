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

#include <unordered_map>

#include "faiss/MetricType.h"
#include "include/common/Exception.h"
#include "include/knowhere/IndexMeta/MetricType.h"

namespace knowhere {

static const std::unordered_map<knowhere::MetricType, faiss::MetricType> metric_map = {
    {MetricTypeEnum::L2, faiss::MetricType::METRIC_L2},
    {MetricTypeEnum::IP, faiss::MetricType::METRIC_INNER_PRODUCT},
    {MetricTypeEnum::JACCARD, faiss::MetricType::METRIC_Jaccard},
    {MetricTypeEnum::TANIMOTO, faiss::MetricType::METRIC_Tanimoto},
    {MetricTypeEnum::HAMMING, faiss::MetricType::METRIC_Hamming},
    {MetricTypeEnum::SUBSTRUCTURE, faiss::MetricType::METRIC_Substructure},
    {MetricTypeEnum::SUPERSTRUCTURE, faiss::MetricType::METRIC_Superstructure},
};

inline faiss::MetricType
GetFaissMetricType(const MetricType& type) {
    try {
        std::string type_str = type;
        std::transform(type_str.begin(), type_str.end(), type_str.begin(), toupper);
        return metric_map.at(type_str);
    } catch (...) {
        KNOWHERE_THROW_FORMAT("Metric type '%s' invalid", type.data());
    }
}

}