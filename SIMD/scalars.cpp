#ifndef SCALAR_H
#define SCALAR_H

#include "kernels.h"
#include <arm_neon.h>  // Included for consistency across SIMD backends
#include <cassert>
#include <cmath>
#include <cstdint>

// Scalar implementation of the SIMD interface using plain C++ (no intrinsics).
class Scalar : public SIMD<int8_t> {
public:
  // Compute running total (inclusive prefix sum)
  int8_t* prefixSum(int8_t* v, int size) override {
    int8_t* result = new int8_t[size];
    result[0] = v[0];
    int8_t cumulative = v[0];
    for (int i = 1; i < size; i++) {
      cumulative += v[i];
      result[i] = cumulative;
    }
    return result;
  }

  // Add two vectors element-wise
  int8_t* vectorAdd(int8_t* v1, int8_t* v2, int size) override {
    int8_t* result = new int8_t[size];
    for (int i = 0; i < size; i++) {
      result[i] = v1[i] + v2[i];
    }
    return result;
  }

  // Sum all elements in the vector
  int8_t vectorReduce(int8_t* v, int size) override {
    int8_t total = 0;
    for (int i = 0; i < size; i++) {
      total += v[i];
    }
    return total;
  }

  // Find the largest element in the vector
  int8_t vectorMax(int8_t* v, int size) override {
    int8_t maxVal = v[0];
    for (int i = 1; i < size; i++) {
      if (v[i] > maxVal) maxVal = v[i];
    }
    return maxVal;
  }

  // Find the smallest element in the vector
  int8_t vectorMin(int8_t* v, int size) override {
    int8_t minVal = v[0];
    for (int i = 1; i < size; i++) {
      if (v[i] < minVal) minVal = v[i];
    }
    return minVal;
  }

  // 1D convolution with configurable padding and stride
  int8_t* convolution_1d(int8_t* input, int iSize, int8_t* kernel,
                         int kSize, int& oSize, int padding, int stride) override {
    oSize = (iSize + 2 * padding - kSize) / stride + 1;
    int8_t* output = new int8_t[oSize];

    for (int i = 0; i < oSize; i++) {
      int32_t sum = 0;
      for (int j = 0; j < kSize; j++) {
        int idx = i * stride + j - padding;
        if (idx >= 0 && idx < iSize) {
          sum += input[idx] * kernel[j];
        }
      }
      // Clamp result to fit int8_t range
      output[i] = std::max(INT8_MIN, std::min(INT8_MAX, sum));
    }
    return output;
  }

  // 2D convolution over square matrix input and kernel
  int8_t* convolution_2d(int8_t* input, int iSide, int8_t* kernel,
                         int kSide, int** oSide, int padding, int stride) override {
    int outDim = (iSide + 2 * padding - kSide) / stride + 1;
    *oSide = new int(outDim);
    int8_t* output = new int8_t[outDim * outDim];

    for (int i = 0; i < outDim; i++) {
      for (int j = 0; j < outDim; j++) {
        int32_t sum = 0;
        for (int ki = 0; ki < kSide; ki++) {
          for (int kj = 0; kj < kSide; kj++) {
            int row = i * stride + ki - padding;
            int col = j * stride + kj - padding;
            if (row >= 0 && row < iSide && col >= 0 && col < iSide) {
              sum += input[row * iSide + col] * kernel[ki * kSide + kj];
            }
          }
        }
        output[i * outDim + j] = static_cast<int8_t>(
            std::max(INT8_MIN, std::min(INT8_MAX, sum)));
      }
    }

    return output;
  }

  // Multiply two matrices: A (MxK) * B (KxN) = C (MxN)
  int8_t* matMul(int8_t* A, int M, int8_t* B, int N, int K) override {
    int8_t* result = new int8_t[M * N];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
          sum += A[i * K + k] * B[k * N + j];
        }
        result[i * N + j] = static_cast<int8_t>(sum);
      }
    }
    return result;
  }

  // Compute softmax across a 1D input vector
  void softMax(float32_t* input, float32_t* output, int length) override {
    float32_t maxVal = input[0];
    for (int i = 1; i < length; i++) {
      if (input[i] > maxVal) maxVal = input[i];
    }

    float32_t sum = 0.0f;
    for (int i = 0; i < length; i++) {
      output[i] = std::exp(input[i] - maxVal);
      sum += output[i];
    }

    for (int i = 0; i < length; i++) {
      output[i] /= sum;
    }
  }
};

#endif  // SCALAR_H
