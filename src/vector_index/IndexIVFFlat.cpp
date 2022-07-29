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

#include "include/knowhere/vector_index/IndexIVFFlat.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>

#include <memory>
#include <utility>
#include <vector>

#include "include/knowhere/vector_index/faiss/FaissIVF.h"
#include "include/knowhere/vector_index/utils/MetricTypeAdapter.h"

namespace knowhere {

IndexIVFFlat::IndexIVFFlat() {

}

IndexIVFFlat::~IndexIVFFlat() {

}

int
IndexIVFFlat::Build(const DataSet& dataset, const IndexMetaPtr& meta) {
    Train(dataset, meta);
    Add(dataset, meta);
}

int
IndexIVFFlat::Train(const DataSet& dataset, const IndexMetaPtr& meta) {
    std::shared_ptr<IVFIndexMeta> metaPtr = std::dynamic_pointer_cast<IVFIndexMeta>(meta);
    faiss::MetricType metric_type = GetFaissMetricType(meta->metricType);
    faiss::Index* coarse_quantizer = new faiss::IndexFlat(meta->dim, metric_type);
    auto index = std::make_shared<faiss::IndexIVFFlat>(coarse_quantizer, metaPtr->dim, metaPtr->nlist, metric_type);
    index->own_fields = true;
    index->train(dataset.GetRowCount(), reinterpret_cast<const float*>(dataset.GetTensor()));
    index_ = index;
}

int
IndexIVFFlat::Add(const DataSet& dataset, const IndexMetaPtr& meta) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    index_->add(dataset.GetRowCount(), reinterpret_cast<const float*>(dataset.GetTensor()));
}

VecDataResultPtr
IndexIVFFlat::Search(const DataSet& dataset, const IndexMetaPtr& param) const {
    std::shared_ptr<IVFIndexMeta> searchConfig = std::dynamic_pointer_cast<IVFIndexMeta>(param);
    int64_t rows = dataset.GetRowCount();
    int64_t k = searchConfig -> topk;
    
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->nprobe = std::min(searchConfig->nprobe, searchConfig->nlist);

    auto elems = rows * k;

    size_t p_id_size = sizeof(int64_t) * elems;
    size_t p_dist_size = sizeof(float) * elems;
    auto p_id = static_cast<int64_t*>(malloc(p_id_size));
    auto p_dist = static_cast<float*>(malloc(p_dist_size));

    if (ivf_index->nprobe > 1 && rows <= 4) {
        ivf_index->parallel_mode = 1;
    } else {
        ivf_index->parallel_mode = 0;
    }

    ivf_index->search(rows, static_cast<const float *>(dataset.GetTensor()), k, p_dist, p_id, searchConfig->bitset);
    return std::make_shared<VecDataResult>(p_id, p_dist);
}

VecDataResultPtr
IndexIVFFlat::SearchByRange(const DataSet& dataset, const IndexMetaPtr& param) const {
    std::shared_ptr<IVFIndexMeta> searchConfig = std::dynamic_pointer_cast<IVFIndexMeta>(param);

    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->nprobe = std::min(searchConfig->nprobe, ivf_index->nlist);
    if (searchConfig->nprobe > 1 && dataset.GetRowCount() <= 4) {
        ivf_index->parallel_mode = 1;
    } else {
        ivf_index->parallel_mode = 0;
    }
    
    float radius = searchConfig->radius;
    if (index_->metric_type == faiss::MetricType::METRIC_L2) {
        radius *= radius;
    }

    int64_t row = dataset.GetRowCount();
    faiss::RangeSearchResult res(row);
    ivf_index->range_search(row, dataset.GetTensor(), radius, &res, searchConfig->bitset);

    return std::make_shared<VecDataResult>(res.labels, res.distances, res.lims);
}

VecDataResultPtr
IndexIVFFlat::GetVectorById(const DataSet& dataset, const IndexMetaPtr& param) const {
    std::shared_ptr<IVFIndexMeta> searchConfig = std::dynamic_pointer_cast<IVFIndexMeta>(param);
    float* p_x = nullptr;
    p_x = (float*)malloc(sizeof(float) * searchConfig->dim * dataset.GetRowCount());
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->get_vector_by_id(dataset.GetRowCount(), dataset.GetPid(), p_x);
    return  std::make_shared<VecDataResult>(p_x);
}

int
IndexIVFFlat::Serialization(IndexMetaPtr& param) {
    std::shared_ptr<IVFIndexMeta> config = std::dynamic_pointer_cast<IVFIndexMeta>(param);
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    SerializeImpl(ivf_index, config->metricType);
}

int
IndexIVFFlat::Deserialization(const BinarySet& binset) {
    faiss::Index* index;
    LoadImpl(binset, index);
    index_.reset(index);
}

IndexMetaPtr
IndexIVFFlat::GetMetaPtr() {
    return std::make_shared<IVFIndexMeta>();
}

}