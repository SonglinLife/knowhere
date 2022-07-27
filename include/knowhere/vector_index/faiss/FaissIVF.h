// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

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