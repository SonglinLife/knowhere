#pragma once

#include <string>

#include "include/common/Bitset.h"
#include "include/knowhere/IndexType.h"
#include "include/knowhere/MetricType.h"

namespace knowhere {

struct IndexMeta {
     /* index meta */
    std::string name;
    
    int64_t dim;

    int64_t count;
    
    int64_t size;

    IndexType type;

    IndexMode mode;

    bool is_trained;

    MetricType metricType;

    void *p_data;

    /**
     * search meta
     */
     int64_t nprobe;

     int64_t topk;

     BitSet bitset;

     float radius;

    virtual ~IndexMeta() {

    }
};

}