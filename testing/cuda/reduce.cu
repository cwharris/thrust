#include <unittest/unittest.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__
void reduce_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T init, Iterator2 result)
{
  *result = thrust::reduce(exec, first, last, init);
}


template<typename T, typename ExecutionPolicy>
void TestReduceDevice(ExecutionPolicy exec, const size_t n)
{
  thrust::host_vector<T>   h_data = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;
  
  thrust::device_vector<T> d_result(1);
  
  T init = 13;
  
  T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
  
  reduce_kernel<<<1,1>>>(exec, d_data.begin(), d_data.end(), init, d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);
  
  ASSERT_EQUAL(h_result, d_result[0]);
}


template<typename T>
struct TestReduceDeviceSeq
{
  void operator()(const size_t n)
  {
    TestReduceDevice<T>(thrust::seq, n);
  }
};
VariableUnitTest<TestReduceDeviceSeq, IntegralTypes> TestReduceDeviceSeqInstance;


template<typename T>
struct TestReduceDeviceDevice
{
  void operator()(const size_t n)
  {
    TestReduceDevice<T>(thrust::device, n);
  }
};
VariableUnitTest<TestReduceDeviceDevice, IntegralTypes> TestReduceDeviceDeviceInstance;

struct noncommutative_reducer
{
  int __device__ operator()(int a, int b) {
    return a;
  }
};

void TestReduceCudaStreams()
{
  typedef thrust::device_vector<int> Vector;

  auto count = 1 << 30;
  auto zeros = thrust::make_constant_iterator<int>(0);

  // Vector v(3);
  // v[0] = 1; v[1] = -2; v[2] = 3;

  auto v = Vector(zeros, zeros + count);

  v[0] = 3;

  cudaStream_t s;
  cudaStreamCreate(&s);

  // // no initializer
  // ASSERT_EQUAL(thrust::reduce(thrust::cuda::par.on(s), v.begin(), v.end()), 2);

  // commutative
  ASSERT_EQUAL(thrust::reduce(thrust::cuda::par.on(s), v.begin(), v.end(), 7, noncommutative_reducer{}), 7);

  // // with initializer
  // ASSERT_EQUAL(thrust::reduce(thrust::cuda::par.on(s), v.begin(), v.end(), 10), 12);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReduceCudaStreams);

