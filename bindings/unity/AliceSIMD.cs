// ALICE-SIMD — Unity C# Bindings
// License: see LICENSE
// Author: Moroya Sakamoto
//
// 15 DllImport + 3 wrapper classes

using System;
using System.Runtime.InteropServices;

namespace Alice.SIMD
{
    internal static class Native
    {
#if UNITY_IOS && !UNITY_EDITOR
        private const string Lib = "__Internal";
#else
        private const string Lib = "alice_simd";
#endif

        // AlignedVec<f32> (5)
        [DllImport(Lib)] public static extern IntPtr alice_aligned_vec_new();
        [DllImport(Lib)] public static extern void alice_aligned_vec_push(IntPtr vec, float val);
        [DllImport(Lib)] public static extern float alice_aligned_vec_get(IntPtr vec, uint idx);
        [DllImport(Lib)] public static extern uint alice_aligned_vec_len(IntPtr vec);
        [DllImport(Lib)] public static extern void alice_aligned_vec_destroy(IntPtr vec);

        // BitMask64 (5)
        [DllImport(Lib)] public static extern ulong alice_bitmask_new(ulong bits);
        [DllImport(Lib)] public static extern ulong alice_bitmask_set(ulong mask, uint i);
        [DllImport(Lib)] public static extern byte alice_bitmask_test(ulong mask, uint i);
        [DllImport(Lib)] public static extern uint alice_bitmask_count_ones(ulong mask);
        [DllImport(Lib)] public static extern ulong alice_bitmask_and(ulong a, ulong b);

        // branchless (3)
        [DllImport(Lib)] public static extern float alice_select_f32(byte condition, float a, float b);
        [DllImport(Lib)] public static extern float alice_branchless_min(float a, float b);
        [DllImport(Lib)] public static extern float alice_branchless_max(float a, float b);

        // fast_math (2)
        [DllImport(Lib)] public static extern float alice_lerp(float a, float b, float t);
        [DllImport(Lib)] public static extern float alice_fma(float a, float b, float c);
    }

    public sealed class AliceAlignedVec : IDisposable
    {
        private IntPtr _ptr;

        public AliceAlignedVec() { _ptr = Native.alice_aligned_vec_new(); }

        public void Push(float val) => Native.alice_aligned_vec_push(_ptr, val);
        public float Get(uint idx) => Native.alice_aligned_vec_get(_ptr, idx);
        public uint Len => Native.alice_aligned_vec_len(_ptr);

        public void Dispose()
        {
            if (_ptr != IntPtr.Zero) { Native.alice_aligned_vec_destroy(_ptr); _ptr = IntPtr.Zero; }
        }
        ~AliceAlignedVec() => Dispose();
    }

    public struct AliceBitMask64
    {
        public ulong Bits;

        public AliceBitMask64(ulong bits) { Bits = Native.alice_bitmask_new(bits); }

        public AliceBitMask64 Set(uint i) => new AliceBitMask64(Native.alice_bitmask_set(Bits, i));
        public bool Test(uint i) => Native.alice_bitmask_test(Bits, i) != 0;
        public uint CountOnes => Native.alice_bitmask_count_ones(Bits);
        public AliceBitMask64 And(AliceBitMask64 other)
            => new AliceBitMask64(Native.alice_bitmask_and(Bits, other.Bits));
    }

    public static class AliceMath
    {
        public static float Select(bool condition, float a, float b)
            => Native.alice_select_f32(condition ? (byte)1 : (byte)0, a, b);
        public static float Min(float a, float b) => Native.alice_branchless_min(a, b);
        public static float Max(float a, float b) => Native.alice_branchless_max(a, b);
        public static float Lerp(float a, float b, float t) => Native.alice_lerp(a, b, t);
        public static float Fma(float a, float b, float c) => Native.alice_fma(a, b, c);
    }
}
