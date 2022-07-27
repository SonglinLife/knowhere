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