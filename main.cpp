#include "SIMD/kernels.h"
#include "SIMD/scalars.cpp"
#include "SIMD/tests.h"
#include "SIMD/vectorNeon.cpp"
#include <cstdint>
#include <iostream>

using namespace std;
int main() {
  auto neon = new Neon();
  auto scalar = new Scalar();

  // testVectorAdd(scalar);
  // testVectorAdd(neon);
  // testVectorReduce(scalar);
  // testVectorReduce(neon);
  // testPrefixSum(scalar);
  // testPrefixSum(neon);
  // testVectorMin(scalar);
  // testVectorMin(neon);
  // testVectorMax(scalar);
  // testVectorMax(neon);
  testSoftmax(scalar, neon);
  testConv1D(scalar, neon);
  testConv2D(scalar, neon);
  // testGEMM(scalar, neon);
  return 0;
}