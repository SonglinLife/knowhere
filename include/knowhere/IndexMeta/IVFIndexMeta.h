#pragma once

#include <string>

#include "IndexMeta.h"

namespace knowhere {

struct IVFIndexMeta : public IndexMeta  {
    int64_t nlist;

    int64_t nprobe;
};

}