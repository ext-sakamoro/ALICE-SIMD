// ALICE-SIMD — UE5 C++ Bindings
// License: see LICENSE
// Author: Moroya Sakamoto
//
// 15 extern "C" declarations + 2 RAII wrapper classes + 1 static utility

#pragma once

#include <cstdint>
#include <utility>

// ── extern "C" declarations ─────────────────────────────────────────

extern "C" {

// AlignedVec<f32> (5)
void*    alice_aligned_vec_new();
void     alice_aligned_vec_push(void* vec, float val);
float    alice_aligned_vec_get(const void* vec, uint32_t idx);
uint32_t alice_aligned_vec_len(const void* vec);
void     alice_aligned_vec_destroy(void* vec);

// BitMask64 (5)
uint64_t alice_bitmask_new(uint64_t bits);
uint64_t alice_bitmask_set(uint64_t mask, uint32_t i);
uint8_t  alice_bitmask_test(uint64_t mask, uint32_t i);
uint32_t alice_bitmask_count_ones(uint64_t mask);
uint64_t alice_bitmask_and(uint64_t a, uint64_t b);

// branchless (3)
float alice_select_f32(uint8_t condition, float a, float b);
float alice_branchless_min(float a, float b);
float alice_branchless_max(float a, float b);

// fast_math (2)
float alice_lerp(float a, float b, float t);
float alice_fma(float a, float b, float c);

} // extern "C"

// ── RAII wrappers ───────────────────────────────────────────────────

class AliceAlignedVec {
    void* ptr_;
public:
    AliceAlignedVec() : ptr_(alice_aligned_vec_new()) {}
    ~AliceAlignedVec() { if (ptr_) alice_aligned_vec_destroy(ptr_); }

    AliceAlignedVec(const AliceAlignedVec&) = delete;
    AliceAlignedVec& operator=(const AliceAlignedVec&) = delete;
    AliceAlignedVec(AliceAlignedVec&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)) {}
    AliceAlignedVec& operator=(AliceAlignedVec&& o) noexcept {
        if (this != &o) { if (ptr_) alice_aligned_vec_destroy(ptr_); ptr_ = std::exchange(o.ptr_, nullptr); }
        return *this;
    }

    void Push(float val) { alice_aligned_vec_push(ptr_, val); }
    float Get(uint32_t idx) const { return alice_aligned_vec_get(ptr_, idx); }
    uint32_t Len() const { return alice_aligned_vec_len(ptr_); }
};

struct AliceBitMask64 {
    uint64_t bits;

    explicit AliceBitMask64(uint64_t b = 0) : bits(alice_bitmask_new(b)) {}

    AliceBitMask64 Set(uint32_t i) const { return AliceBitMask64{alice_bitmask_set(bits, i)}; }
    bool Test(uint32_t i) const { return alice_bitmask_test(bits, i) != 0; }
    uint32_t CountOnes() const { return alice_bitmask_count_ones(bits); }
    AliceBitMask64 And(AliceBitMask64 other) const {
        return AliceBitMask64{alice_bitmask_and(bits, other.bits)};
    }
};

// ── Static math utilities ───────────────────────────────────────────

struct AliceMath {
    static float Select(bool cond, float a, float b) {
        return alice_select_f32(cond ? 1 : 0, a, b);
    }
    static float Min(float a, float b) { return alice_branchless_min(a, b); }
    static float Max(float a, float b) { return alice_branchless_max(a, b); }
    static float Lerp(float a, float b, float t) { return alice_lerp(a, b, t); }
    static float Fma(float a, float b, float c) { return alice_fma(a, b, c); }
};
