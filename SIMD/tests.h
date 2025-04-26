#include "kernels.h"

template <typename T> void testVectorAdd(SIMD<T> *impl);
template <typename T> void testVectorReduce(SIMD<T> *impl);
template <typename T> void testPrefixSum(SIMD<T> *impl);
template <typename T> void testVectorMin(SIMD<T> *impl);
template <typename T> void testVectorMax(SIMD<T> *impl);
template <typename T> void testConv1D(SIMD<T> *impl, SIMD<int8_t> *impl2);
template <typename T> void testConv2D(SIMD<T> *impl, SIMD<int8_t> *impl2);
template <typename T> void testSoftmax(SIMD<T> *impl, SIMD<T> *impl2);
template <typename T> void testGEMM(SIMD<T> *impl, SIMD<int8_t> *impl2);
