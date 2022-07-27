#pragma once


#include <algorithm>
#include <string>
#include <unordered_map>

namespace knowhere {

namespace meta {
constexpr const char* SLICE_SIZE = "SLICE_SIZE";
constexpr const char* METRIC_TYPE = "metric_type";
constexpr const char* COUNT = "count";
constexpr const char* TYPE = "type";
constexpr const char* DIM = "dim";
constexpr const char* TENSOR = "tensor";
constexpr const char* ROWS = "rows";
constexpr const char* IDS = "ids";
constexpr const char* DISTANCE = "distance";
constexpr const char* LIMS = "lims";
constexpr const char* TOPK = "k";
constexpr const char* RADIUS = "radius";
constexpr const char* INPUT_IDS = "input_ids";
constexpr const char* DEVICE_ID = "gpu_id";
}  // namespace meta


namespace indexparam {
// IVF Params
constexpr const char* NPROBE = "nprobe";
constexpr const char* NLIST = "nlist";
constexpr const char* NBITS = "nbits";   // PQ/SQ
constexpr const char* M = "m";           // PQ param for IVFPQ
constexpr const char* PQ_M = "PQM";      // PQ param for RHNSWPQ
// HNSW Params
constexpr const char* EFCONSTRUCTION = "efConstruction";
constexpr const char* HNSW_M = "M";
constexpr const char* EF = "ef";
constexpr const char* HNSW_K = "range_k";
// Annoy Params
constexpr const char* N_TREES = "n_trees";
constexpr const char* SEARCH_K = "search_k";
}
}