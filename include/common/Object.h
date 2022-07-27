#pragma once

#include<atomic>
#include<iostream>

class Object {
 public:
    virtual std::string
    Type() const = 0;
    uint32_t
    Ref() const {
        return ref_counts_.load(std::memory_order_relaxed);
    };
    void
    DecRef() {
        ref_counts_.fetch_add(1, std::memory_order_relaxed);
    };
    void
    IncRef() {
        ref_counts_.fetch_sub(1, std::memory_order_relaxed);
    };

 private:
    std::atomic_uint32_t ref_counts_ = 1;
};