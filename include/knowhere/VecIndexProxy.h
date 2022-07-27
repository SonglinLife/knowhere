#pragma once

#include "VecIndex.h"

namespace knowhere {

class VecIndexProxy {
 public:
    VecIndexProxy(const VecIndexProxy& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        idx.node->IncRef();
        node = idx.node;
    }

    VecIndexProxy(VecIndexProxy&& idx) {
        if (idx.node == nullptr) {
            node = nullptr;
            return;
        }
        node = idx.node;
        idx.node = nullptr;
    }

    VecIndexProxy&
    operator=(const VecIndexProxy& idx) {
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        if (idx.node == nullptr) {
            node = nullptr;
            return *this;
        }
        node = idx.node;
        node->IncRef();
        return *this;
    }

    VecIndexProxy&
    operator=(VecIndexProxy&& idx) {
        if (node != nullptr) {
            node->DecRef();
            if (!node->Ref())
                delete node;
        }
        node = idx.node;
        idx.node = nullptr;
        return *this;
    }
    
    ~VecIndexProxy() {
        if (node == nullptr)
            return;
        node->DecRef();
        if (!node->Ref())
            delete node;
    }

 private:
    VecIndexProxy(VecIndex * node_) : node(node_) {
    }
    VecIndex* node;
};

}