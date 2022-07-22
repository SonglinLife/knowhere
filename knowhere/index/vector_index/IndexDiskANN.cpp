// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "index/vector_index/IndexDiskANN.h"

#include <omp.h>

#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "DiskANN/include/aux_utils.h"
#ifndef _WINDOWS
#include "DiskANN/include/linux_aligned_file_reader.h"
#else
#include "DiskANN/include/windows_aligned_file_reader.h"
#endif
#include "knowhere/common/Exception.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"

namespace knowhere {

template <typename T>
IndexDiskANN<T>::IndexDiskANN(std::string index_prefix, MetricType metric_type,
                              std::unique_ptr<FileManager> file_manager)
    : index_prefix_(index_prefix), file_manager_(std::move(file_manager)) {
    index_type_ = IndexEnum::INDEX_DISKANN;

    if (metric_type == metric::L2) {
        metric_ = diskann::L2;
    } else if (metric_type == metric::IP) {
        if (!std::is_same_v<T, float>) {
            KNOWHERE_THROW_MSG(
                "DiskANN currently only supports floating point data for Max "
                "Inner Product Search. ");
        }
        metric_ = diskann::INNER_PRODUCT;
    } else {
        KNOWHERE_THROW_MSG("DiskANN only support L2 and IP distance.");
    }
}

namespace {
void
CheckPreparation(bool is_prepared) {
    if (!is_prepared) {
        KNOWHERE_THROW_MSG("DiskANN is not prepared yet, plz call Prepare() to make it ready for queries.");
    }
}

/**
 * @brief Convert id from uint64 to int64. We can avoid this by either supporting int64 id in DiskANN or accepting
 * uint64 id in Milvus.
 *
 * @return false if any error.
 */
bool
ConvertId(const uint64_t* src, int64_t* des, size_t num) {
    int64_t max_musk = 1;
    max_musk = max_musk << 63;
    bool successful = true;
#pragma omp parallel for schedule(static, 65536)
    for (size_t n = 0; n < num; ++n) {
        if (src[n] & max_musk) {
            LOG_KNOWHERE_ERROR_ << "Id " << src[n] << " exceeds the limit of int64_t.";
            successful = false;
        } else {
            des[n] = static_cast<int64_t>(src[n]);
        }
    }
    return successful;
}
}  // namespace

template <typename T>
void
IndexDiskANN<T>::AddWithoutIds(const DatasetPtr& data_set, const Config& config) {
    auto build_conf = DiskANNBuildConfig::Get(config);

    std::stringstream stream;
    stream << build_conf.max_degree << " " << build_conf.search_list_size << " " << build_conf.search_dram_budget_gb
           << " " << build_conf.build_dram_budget_gb << " " << build_conf.num_threads << " "
           << build_conf.pq_disk_bytes;
    diskann::build_disk_index<T>(build_conf.data_path, index_prefix_, stream.str(), metric_);
}

template <typename T>
bool
IndexDiskANN<T>::Prepare(const Config& config) {
    std::lock_guard<std::mutex> lock(preparation_lock_);

    auto prep_conf = DiskANNPrepareConfig::Get(config);
    if (is_prepared_) {
        return true;
    }

    // load PQ file
    LOG_KNOWHERE_INFO_ << "Loading PQ from disk.";
    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    pq_flash_index_ = (new diskann::PQFlashIndex<T>(reader, metric_));

    if (pq_flash_index_->load(prep_conf.num_threads, index_prefix_.c_str()) != 0) {
        return false;
    }

    std::string warmup_query_file = index_prefix_ + "_sample_data.bin";
    // load cache
    if (prep_conf.num_nodes_to_cache > 0) {
        std::vector<uint32_t> node_list;
        LOG_KNOWHERE_INFO_ << "Caching " << prep_conf.num_nodes_to_cache << " sample nodes around medoid(s).";

        if (prep_conf.use_bfs_cache) {
            pq_flash_index_->cache_bfs_levels(prep_conf.num_nodes_to_cache, node_list);
        } else {
            pq_flash_index_->generate_cache_list_from_sample_queries(
                warmup_query_file, 15, 6, prep_conf.num_nodes_to_cache, prep_conf.num_threads, node_list);
        }
        pq_flash_index_->load_cache_list(node_list);
    }

    // set thread number
    omp_set_num_threads(prep_conf.num_threads);

    // warmup
    if (prep_conf.warm_up) {
        LOG_KNOWHERE_INFO_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        T* warmup = nullptr;
        if (file_exists(warmup_query_file)) {
            diskann::load_aligned_bin<T>(warmup_query_file, warmup, warmup_num, warmup_dim, warmup_aligned_dim);
        } else {
            warmup_num = (std::min)((_u32)150000, (_u32)15000 * prep_conf.num_threads);
            warmup_dim = pq_flash_index_->get_data_dim();
            warmup_aligned_dim = ROUND_UP(warmup_dim, 8);
            diskann::alloc_aligned(((void**)&warmup), warmup_num * warmup_aligned_dim * sizeof(T), 8 * sizeof(T));
            std::memset(warmup, 0, warmup_num * warmup_aligned_dim * sizeof(T));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(-128, 127);
            for (uint32_t i = 0; i < warmup_num; ++i) {
                for (uint32_t d = 0; d < warmup_dim; ++d) {
                    warmup[i * warmup_aligned_dim + d] = (T)dis(gen);
                }
            }
        }
        std::vector<uint64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<float> warmup_result_dists(warmup_num, 0);

#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            pq_flash_index_->cached_beam_search(warmup + (i * warmup_aligned_dim), 1, warmup_L,
                                                warmup_result_ids_64.data() + (i * 1),
                                                warmup_result_dists.data() + (i * 1), 4);
        }
        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }
    }

    is_prepared_ = true;
    return true;
}

template <typename T>
DatasetPtr
IndexDiskANN<T>::Query(const DatasetPtr& dataset_ptr, const Config& config,
                       const faiss::BitsetView /* not implemented yet */) {
    CheckPreparation(is_prepared_);

    auto query_conf = DiskANNQueryConfig::Get(config);
    auto& k = query_conf.k;

    GET_TENSOR_DATA_DIM(dataset_ptr);
    auto query = static_cast<T*>(p_data);
    auto p_id_u64 = std::vector<uint64_t>(k * rows).data();
    auto p_id = static_cast<int64_t*>(malloc(sizeof(int64_t) * k * rows));
    auto p_dist = static_cast<float*>(malloc(sizeof(float) * k * rows));

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t row = 0; row < rows; ++row) {
        pq_flash_index_->cached_beam_search(query + (row * dim), k, query_conf.search_list_size, p_id_u64 + (row * k),
                                            p_dist + (row * k), query_conf.beamwidth, false, nullptr);
    }

    if (!ConvertId(p_id_u64, p_id, k * rows)) {
        KNOWHERE_THROW_MSG("Failed to convert id from uint64 to int64.");
    }
    return GenResultDataset(p_id, p_dist);
}

template <typename T>
DatasetPtr
IndexDiskANN<T>::QueryByRange(const DatasetPtr& dataset_ptr, const Config& config,
                              const faiss::BitsetView /* not implemented yet */) {
    CheckPreparation(is_prepared_);

    auto query_conf = DiskANNQueryByRangeConfig::Get(config);
    auto& radius = query_conf.radius;

    GET_TENSOR_DATA_DIM(dataset_ptr);
    auto query = static_cast<T*>(p_data);

    std::vector<std::vector<uint64_t>> result_id_array(rows);
    std::vector<std::vector<float>> result_dist_array(rows);
    auto p_lims = static_cast<size_t*>(malloc((rows + 1) * sizeof(size_t)));
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t row = 0; row < rows; ++row) {
        std::vector<_u64> indices;
        std::vector<float> distances;
        auto res_count = pq_flash_index_->range_search(query + (row * dim), radius, query_conf.min_k, query_conf.max_k,
                                                       indices, distances, query_conf.beamwidth);
        result_id_array[row].resize(res_count);
        result_dist_array[row].resize(res_count);
        for (int32_t res_num = 0; res_num < res_count; ++res_num) {
            result_id_array[row][res_num] = indices[res_num];
            result_dist_array[row][res_num] = distances[res_num];
        }
        *(p_lims + row + 1) = *(p_lims + row) + res_count;
    }
    auto ans_size = *(p_lims + rows);
    auto p_id = static_cast<int64_t*>(malloc(ans_size * sizeof(int64_t)));
    auto p_dist = static_cast<float*>(malloc(ans_size * sizeof(float)));

    for (int64_t row = 0; row < rows; ++row) {
        auto start = *(p_lims + row);
        if (!ConvertId(result_id_array[row].data(), p_id + start, result_id_array[row].size())) {
            KNOWHERE_THROW_MSG("Failed to convert id from uint64 to int64.");
        }
        memcpy(p_dist + start, result_dist_array[row].data(), result_dist_array[row].size() * sizeof(float));
    }

    return GenResultDataset(p_id, p_dist, p_lims);
}

template <typename T>
int64_t
IndexDiskANN<T>::Count() {
    CheckPreparation(is_prepared_);
    return pq_flash_index_->get_num_points();
}

template <typename T>
int64_t
IndexDiskANN<T>::Dim() {
    CheckPreparation(is_prepared_);
    return pq_flash_index_->get_data_dim();
}
}  // namespace knowhere
