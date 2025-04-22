#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <arm_neon.h>
#include <cstdint>
#include <vector>

// Abstract base class for SIMD-based numerical operations.
// Intended to be subclassed for AVX, NEON, or other SIMD instruction sets.
template <typename T>
class SIMD {
public:
  SIMD() = default;

  // Compute prefix sum (inclusive) over the input vector.
  virtual T* prefixSum(T* input, int size) = 0;

  // Perform element-wise addition of two vectors.
  virtual T* vectorAdd(T* v1, T* v2, int size) = 0;

  // Reduce the input vector to a single sum (accumulation).
  virtual T vectorReduce(T* input, int size) = 0;

  // Find the maximum value in the vector.
  virtual T vectorMax(T* input, int size) = 0;

  // Find the minimum value in the vector.
  virtual T vectorMin(T* input, int size) = 0;

  // Apply 1D convolution over the input with the given kernel.
  // Returns the output vector and sets its size via `oSize`.
  virtual T* convolution_1d(T* input, int inputSize,
                            T* kernel, int kernelSize,
                            int& oSize, int padding, int stride) = 0;

  // Apply 2D convolution over a square input matrix with a square kernel.
  // The output size is set via `oSide` pointer.
  virtual T* convolution_2d(T* input, int inputSide,
                            T* kernel, int kernelSide,
                            int** oSide, int padding, int stride) = 0;

  // Matrix multiplication: computes A (MxK) * B (KxN) = output (MxN).
  virtual T* matMul(T* A, int M, T* B, int N, int K) = 0;

  // Compute the softmax function across a 1D slice (e.g., row of 2D tensor).
  virtual void softMax(float32_t* input, float32_t* output, int length) = 0;

  virtual ~SIMD() = default;
};

#endif  // SIMD_KERNELS_H
