
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "include/utils.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>
#include <queue>
#include <random>
#include <utility>
#include <vector>
#include <omp.h>
#include "knowhere/index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include <boost/program_options.hpp>
#include "index/vector_index/IndexDiskANN.h"
#include "knowhere/index/vector_index/IndexDiskANNConfig.h"
#include "DiskANN/include/aux_utils.h"
#include "DiskANN/include/utils.h"
#ifndef _WINDOWS
#include "DiskANN/include/linux_aligned_file_reader.h"
#else
#include "DiskANN/include/windows_aligned_file_reader.h"
#endif
#include "knowhere/common/Exception.h"
#include <limits>
#include <sstream>
#include "unittest/LocalFileManager.h"

namespace po = boost::program_options;
using PC = knowhere::DiskANNPrepareConfig;
using QC = knowhere::DiskANNQueryConfig;
using RC = knowhere::DiskANNQueryByRangeConfig;


template< typename T>
void KNNSearchOne(const PC& prep_conf, const QC& query_conf,std::string prefix,
  std::string metric_type, T* query_data_, int64_t KQ, int64_t KD, 
  std::vector<unsigned>& query_ids, std::vector<float>& query_dis, std::chrono::duration<double>& diff){
  auto diskann = std::make_unique<knowhere::IndexDiskANN<float>>(prefix, metric_type,
                                                                  std::make_unique<knowhere::LocalFileManager>());
  knowhere::Config cfg;
  cfg.clear();
  knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(KQ, KD, (void*)query_data_);

  knowhere::DiskANNPrepareConfig::Set(cfg, prep_conf);
  diskann->Prepare(cfg);

  cfg.clear();
  knowhere::DiskANNQueryConfig::Set(cfg, query_conf);
  auto                  s = std::chrono::high_resolution_clock::now();
  auto result = diskann->Query(data_set_ptr, cfg, nullptr);
  auto e = std::chrono::high_resolution_clock::now();
  auto ids = knowhere::GetDatasetIDs(result);
  auto dis = knowhere::GetDatasetDistance(result);

  for(_u64 i=0; i< static_cast<_u64>(query_conf.k) * KQ; i++){
    query_ids.push_back(ids[i]);
    query_dis.push_back(dis[i]);
  }
  diff += e - s;
}

template<typename T>
void RangeSearchOne(const PC& prep_conf, RC& range_search_conf, std::string prefix,
  std::string metric_type, T* query_data_, int64_t KQ, int64_t KD, 
  unsigned long& ans_nums,std::chrono::duration<double>& diff){
  
  auto diskann = std::make_unique<knowhere::IndexDiskANN<float>>(prefix, metric_type,
                                                                  std::make_unique<knowhere::LocalFileManager>());
  knowhere::Config cfg;
  cfg.clear();
  PC::Set(cfg, prep_conf);
  diskann->Prepare(cfg);
  
  cfg.clear();
  knowhere::DatasetPtr data_set_ptr = knowhere::GenDataset(KQ, KD, (void*)query_data_);
  RC::Set(cfg, range_search_conf);
  auto                  s = std::chrono::high_resolution_clock::now();
  auto result = diskann->QueryByRange(data_set_ptr, cfg, nullptr);
  auto e = std::chrono::high_resolution_clock::now();
  auto lims = knowhere::GetDatasetLims(result);
  ans_nums += *(lims + KQ);
  diff += e - s;
}

void merge_res(std::vector<unsigned>& ids_res, std::vector<float>& dis_res,
               std::vector<unsigned>& ids_query, std::vector<float>& dis_query,
               unsigned recall_at, unsigned query_num, unsigned offset) {
  std::vector<unsigned> ids_tmp(ids_res.size());
  std::vector<float>    dis_tmp(ids_res.size());
  for (unsigned i = 0; i < query_num; i++) {
    unsigned s = 0 + i * recall_at, t = 0 + i * recall_at;
    unsigned k = i * recall_at;
    for (unsigned j = 0; j < recall_at; j++) {
      if (dis_res[s] < dis_query[t]) {
        ids_tmp[j + k] = ids_res[s];
        dis_tmp[j + k] = dis_res[s];
        s++;
      } else {
        ids_tmp[j + k] = ids_query[t] + offset;
        dis_tmp[j + k] = dis_query[t];
        t++;
      }
    }
  }
  ids_res.swap(ids_tmp);
  dis_res.swap(dis_tmp);
}

template<typename T>
int KNNSearch(
    std::string& metric, std::string& segments_path_prefix, unsigned N,
    const std::string& result_output_prefix, const std::string& query_file,
    std::string& gt_file, const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const std::vector<unsigned>& Lvec, const bool use_reorder_data = false) {
    
  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0){
    std::cout << "beamwidth should greater 1" << std::endl;
    return -1;
  }

  // load query bin
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      std::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  std::vector<unsigned> query_ids;
  std::vector<float>    query_dis;

  std::vector<unsigned> res_ids;
  std::vector<float>   res_dis;

  std::chrono::duration<double> diff;
  uint64_t offset = 0;
  unsigned L;
  L = Lvec[0];
  if(Lvec.size() != 1){
    std::cout<<"only using first L "<<L<<std::endl;
  }
  for (unsigned i = 0; i < N; i++) {
    std::cout << "searching on segment " << i << std::endl;
    auto disk_segment = segments_path_prefix + "_part" +std::to_string(i);
    PC pc;
    pc.num_nodes_to_cache = 0;
    pc.use_bfs_cache = false;
    pc.num_threads = num_threads;
    pc.warm_up = false;

    QC qc;
    qc.beamwidth = beamwidth;
    qc.k = recall_at;
    qc.search_list_size = L;
    KNNSearchOne<T>(pc, qc, disk_segment, metric, query, query_num, query_aligned_dim
    , query_ids, query_dis, diff);

    std::cout << "search done on segment " << i << std::endl;
    if (i == 0) {
      res_ids.resize(recall_at * query_num);
      res_dis.resize(recall_at * query_num);
      for (unsigned k = 0; k < recall_at * query_num; k++) {
        res_ids[k] = offset + query_ids[k];
        res_dis[k] = query_dis[k];
      }
    } else {
      merge_res(res_ids, res_dis, query_ids, query_dis, recall_at,
                  query_num, offset);
    }
    auto disk_segment_index = disk_segment + std::string("_disk.index");
    /********************************************************
    please make sure the offset is correct
    *********************************************************/
    int fd = open(disk_segment_index.c_str(), O_RDONLY);
    int err = pread(fd, (void*) &offset, sizeof(uint64_t),  sizeof(uint64_t)*11);
    if(err == -1){
      std::cout<< disk_segment_index<<" fd: "<<fd<<std::endl;
      std::cout << errno << std::endl;
      exit(-1);
    }
    close(fd);
  }

  auto peak = getPeakRSS();

  float qps = (1.0 * query_num) / (1.0 * diff.count());

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS";
  if (calc_recall_flag) {
    std::cout << std::setw(16) << recall_string << std::setw(16) <<"Peak Mem"<< std::endl;
  } else
    std::cout << std::endl;
  std::cout
      << "==============================================================="
        "======================================================="
      << std::endl;


  float recall = 0;
  if (calc_recall_flag) {
  recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                        res_ids.data(), recall_at,
                                        recall_at);
  }

  std::cout << std::setw(6) << L << std::setw(12) << beamwidth
                << std::setw(16) << qps;
  if (calc_recall_flag) {
    std::cout << std::setw(16) << recall <<std::setw(16)<<peak<< std::endl;
  } else {
    std::cout << std::endl;
  }

  diskann::aligned_free(query);
  return 0;
}


template<typename T>
int RangeSearch(
    std::string& metric, std::string& segments_path_prefix, unsigned N, float radius,
    const std::string& result_output_prefix, const std::string& query_file,
    std::string& gt_file, const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const std::vector<unsigned>& Lvec, const bool use_reorder_data = false) {
    
  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0){
    std::cout << "beamwidth should greater 1" << std::endl;
    return -1;
  }

  // load query bin
  T*        query = nullptr;
  std::vector<std::vector<_u32>> gt_ids;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_range_truthset(gt_file, gt_ids, gt_num);
    if (gt_num != query_num) {
      std::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  std::chrono::duration<double> diff;
  uint64_t offset = 0;
  uint64_t res = 0;
  unsigned L;
  L = Lvec[0];
  if(Lvec.size() != 1){
    std::cout<<"only using first L "<<L<<std::endl;
  }
  for (unsigned i = 0; i < N; i++) {
    std::cout << "searching on segment " << i << std::endl;
    auto disk_segment = segments_path_prefix + "_part" +std::to_string(i);
    PC pc;
    pc.num_nodes_to_cache = 0;
    pc.use_bfs_cache = false;
    pc.num_threads = num_threads;
    pc.warm_up = false;

    RC rc;
    rc.beamwidth = beamwidth;
    rc.max_k = 10000;
    rc.min_k = L;
    rc.radius = radius;
    RangeSearchOne(pc, rc, disk_segment, metric, query, query_num, query_aligned_dim, res, diff);
    std::cout << "search done on segment " << i << std::endl;
  }

  auto peak = getPeakRSS();

  float qps = (1.0 * query_num) / (1.0 * diff.count());

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth"
                << std::setw(16) << "QPS";
  if (calc_recall_flag) {
    std::cout << std::setw(16) << "AP" << std::setw(16) <<"Peak Mem"<< std::endl;
  } else
    std::cout << std::endl;
  std::cout
      << "==============================================================="
        "======================================================="
      << std::endl;

  float ap =0;
  if (calc_recall_flag) {
    _u32 total_pos = 0;
    for(unsigned i=0; i< query_num; i++){
      total_pos += gt_ids[i].size();
    }
    ap = (1.0 * res) / (1.0 * total_pos); 
  }

  std::cout << std::setw(6) << L << std::setw(12) << beamwidth
                << std::setw(16) << qps;
  if (calc_recall_flag) {
    std::cout << std::setw(16) << ap <<std::setw(16)<<peak<< std::endl;
  } else {
    std::cout << std::endl;
  }

  diskann::aligned_free(query);
  return 0;
}
int main(int argc, char** argv) {
  std::string              data_type, dist_fn, query_file, gt_file;
  std::string              result_path_prefix;
  std::string              segments_path_prefix;
  unsigned                 num_threads, K, W,N, num_nodes_to_cache;
  std::vector<unsigned>    Lvec;
  std::vector<std::string> segment_files;
  bool                     use_reorder_data = false;
  float                    radius;
  std::string              search_type;
  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()("segments_path_prefix",
                       po::value<std::string>(&segments_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("segment_nums,N", po::value<uint32_t>(&N)->required(),
                       "Number of segments");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path_prefix)->default_value("tmp"),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                       "Beamwidth for search. Set 0 to optimize internally.");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Beamwidth for search");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()(
        "search_type,S",
        po::value<std::string>(&search_type) ->default_value("knn"),
        "knn or range.");
    desc.add_options()(
        "radius,R",
        po::value<float>(&radius) ->default_value(0),
        "radius of range."
    );
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);

  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }
  std::string metric;
  if(dist_fn == std::string("l2")){
    metric = "L2";
  }else if (dist_fn == std::string("mips")) {
    metric = "IP";
  }else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }
  if ((data_type != std::string("float")) &&
      (metric == std::string("IP"))) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }
  /**************************************************************
                      default using knn
  **************************************************************/
  if (search_type == "knn"){
    if (data_type == std::string("float")) {
      return KNNSearch<float>(
          metric, segments_path_prefix, N,result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    } else if (data_type == std::string("int8")) {
      return KNNSearch<int8_t>(
          metric, segments_path_prefix,N, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    } else if (data_type == std::string("uint8")) {
      return KNNSearch<uint8_t>(
          metric, segments_path_prefix, N,result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    } else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  }else if(search_type == "range"){
    if (data_type == std::string("float")) {
      return RangeSearch<float>(
          metric, segments_path_prefix, N, radius, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    } else if (data_type == std::string("int8")) {
      return RangeSearch<int8_t>(
          metric, segments_path_prefix, N, radius,  result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    } else if (data_type == std::string("uint8")) {
      return RangeSearch<uint8_t>(
          metric, segments_path_prefix, N, radius, result_path_prefix, query_file, gt_file,
          num_threads, K, W, num_nodes_to_cache, Lvec, use_reorder_data);
    } else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  }else{
    std::cout << "No such way!"<<std::endl;
    return -1;
  }
}
