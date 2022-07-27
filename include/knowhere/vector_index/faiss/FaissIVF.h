#pragma once

#include "faiss/index_io.h"
#include "include/common/BinarySet.h"
#include "include/common/Exception.h"
#include "include/knowhere/IndexMeta/IndexType.h"
#include "include/knowhere/vector_index/utils/FaissIO.h"

namespace knowhere {

BinarySet SerializeImpl(faiss::Index* index , const IndexType& type) {
    try {
        MemoryIOWriter writer;
        faiss::write_index(index, &writer);
        std::shared_ptr<uint8_t[]> data(writer.data_);

        BinarySet res_set;
        // TODO(linxj): use virtual func Name() instead of raw string.
        res_set.Append("IVF", data, writer.rp);
        return res_set;
    } catch (std::exception& e) {
        KNOWHERE_THROW_MSG(e.what());
    }
}

faiss::Index* LoadImpl(const BinarySet& binary_set, faiss::Index* &index) {
    auto binary = binary_set.GetByName("IVF");

    MemoryIOReader reader;
    reader.total = binary->size;
    reader.data_ = binary->data.get();

    index = faiss::read_index(&reader);
}

void
SealImpl() {
}

}