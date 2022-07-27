

#include "include/knowhere/vector_index/IndexIVFFlat.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>

#include <memory>
#include <utility>
#include <vector>

#include "include/knowhere/vector_index/IndexAdapter.h"
#include "include/knowhere/vector_index/faiss/FaissIVF.h"

namespace knowhere {

IndexIVFFlat::IndexIVFFlat() {

}

IndexIVFFlat::~IndexIVFFlat() {

}

int
IndexIVFFlat::Build(const DataSet& dataset) {
    Train(dataset);
    Add(dataset);
}

int
IndexIVFFlat::Train(const DataSet& dataset) {
    faiss::MetricType metric_type = GetFaissMetricType(meta.metricType);
    faiss::Index* coarse_quantizer = new faiss::IndexFlat(meta.dim, metric_type);

    auto index = std::make_shared<faiss::IndexIVFFlat>(coarse_quantizer, meta.dim, meta.nlist, metric_type);
    index->own_fields = true;
    index->train(dataset.GetRowCount(), reinterpret_cast<const float*>(dataset.GetTensor()));
    index_ = index;
}

int
IndexIVFFlat::Add(const DataSet& dataset) {
    if (!index_ || !index_->is_trained) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }

    index_->add(dataset.GetRowCount(), reinterpret_cast<const float*>(dataset.GetTensor()));
}

DataSet
IndexIVFFlat::Search(const DataSet& dataset, const IndexMeta& param) const {
    const IVFIndexMeta *searchParam = dynamic_cast<const IVFIndexMeta*>(&param);
    int64_t rows = dataset.GetRowCount();
    int64_t k = searchParam -> topk;
    
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->nprobe = std::min(searchParam->nprobe, meta.nlist);

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
    ivf_index->search(rows, static_cast<const float *>(dataset.GetTensor()), k, p_dist, p_id, searchParam->bitset);
}

DatasetPtr
IndexIVFFlat::SearchByRange(const DataSet& dataset, const IndexMeta& param) const {
    const IVFIndexMeta *searchParam = dynamic_cast<const IVFIndexMeta*>(&param);

    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->nprobe = std::min(searchParam->nprobe, meta.nlist);
    if (searchParam->nprobe > 1 && dataset.GetRowCount() <= 4) {
        ivf_index->parallel_mode = 1;
    } else {
        ivf_index->parallel_mode = 0;
    }
    
    float radius = searchParam->radius;
    if (index_->metric_type == faiss::MetricType::METRIC_L2) {
        radius *= radius;
    }

    int64_t row = dataset.GetRowCount();
    faiss::RangeSearchResult res(row);
    ivf_index->range_search(row, dataset.GetTensor(), radius, &res, searchParam->bitset);
}

DatasetPtr
IndexIVFFlat::GetVectorById(const DataSet& dataset, const IndexMeta& param) const {
    float* p_x = nullptr;
    p_x = (float*)malloc(sizeof(float) * meta.dim * dataset.GetRowCount());
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    ivf_index->get_vector_by_id(dataset.GetRowCount(), dataset.GetPid(), p_x);
    return GenResultDataset(p_x);
}

int
IndexIVFFlat::Serialization(BinarySet& binset) {
    auto ivf_index = dynamic_cast<faiss::IndexIVF*>(index_.get());
    SerializeImpl(ivf_index, meta.type);
}

int
IndexIVFFlat::Deserialization(const BinarySet& binset) {
    faiss::Index* index;
    LoadImpl(binset, index);
    index_.reset(index);
}

}