#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "include/common/BinarySet.h"
#include "include/knowhere/Dataset.h"
#include "include/knowhere/IndexMeta/IVFIndexMeta.h"
#include "include/knowhere/VecIndex.h"

namespace knowhere {

class IndexIVFFlat : public VecIndex
{
public:
    IndexIVFFlat(/* args */);
    
    ~IndexIVFFlat();
    int
    Build(const DataSet& dataset) override;
    int
    Train(const DataSet& dataset) override;
    int
    Add(const DataSet& dataset) = 0;
    DataSet
    Search(const DataSet& dataset, const IndexMeta& param) const override;
    DataSet
    SearchByRange(const DataSet& dataset, const IndexMeta& param) const override;
    DataSet
    GetVectorById(const DataSet& dataset, const IndexMeta& param) const override;
    int
    Serialization(BinarySet& binset) override;
    int
    Deserialization(const BinarySet& binset) override;
    int64_t
    Size() const = 0;
private:
    IVFIndexMeta meta;

    std::shared_ptr<faiss::Index> index_ = nullptr;
};

}