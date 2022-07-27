#pragma once

#include <string>

#include "Dataset.h"
#include "include/common/Object.h"

namespace knowhere {

class VecIndex : public Object {
    friend class VecIndexProxy;
public:
    virtual int
    Build(const DataSet& dataset) = 0;
    virtual int
    Train(const DataSet& dataset) = 0;
    virtual int
    Add(const DataSet& dataset) = 0;
    virtual DataSet
    Search(const DataSet& dataset, const IndexMeta& param) const = 0;
    virtual DataSet
    SearchByRange(const DataSet& dataset, const IndexMeta& param) const = 0;
    virtual DataSet
    GetVectorById(const DataSet& dataset, const IndexMeta& param) const = 0;
    virtual int
    Serialization(BinarySet& binset) = 0;
    virtual int
    Deserialization(const BinarySet& binset) = 0;
    virtual int64_t
    Size() const = 0;
protected:
    virtual IndexMeta&
    GetMeta() const = 0;
    virtual IndexMeta&
    GenParam() const = 0;
};

}