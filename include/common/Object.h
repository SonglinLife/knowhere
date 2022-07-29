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
    virtual ~Object() = default;
 private:
    std::atomic_uint32_t ref_counts_ = 1;
};