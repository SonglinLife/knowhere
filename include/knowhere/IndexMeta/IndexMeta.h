#pragma once

#include <string>

#include "IndexType.h"
#include "MetricType.h"
#include "include/common/Bitset.h"

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