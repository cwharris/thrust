#include <unittest/unittest.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          BinaryPredicate pred,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred);
}


template<typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename Iterator4, typename BinaryPredicate, typename BinaryFunction, typename Iterator5>
__global__
void reduce_by_key_kernel(ExecutionPolicy exec,
                          Iterator1 keys_first, Iterator1 keys_last,
                          Iterator2 values_first,
                          Iterator3 keys_result,
                          Iterator4 values_result,
                          BinaryPredicate pred,
                          BinaryFunction binary_op,
                          Iterator5 result)
{
  *result = thrust::reduce_by_key(exec, keys_first, keys_last, values_first, keys_result, values_result, pred, binary_op);
}


template<typename T>
struct is_equal_div_10_reduce
{
  __host__ __device__
  bool operator()(const T x, const T& y) const { return ((int) x / 10) == ((int) y / 10); }
};


template<typename Vector>
void initialize_keys(Vector& keys)
{
  keys.resize(9);
  keys[0] = 11;
  keys[1] = 11;
  keys[2] = 21;
  keys[3] = 20;
  keys[4] = 21;
  keys[5] = 21;
  keys[6] = 21;
  keys[7] = 37;
  keys[8] = 37;
}


template<typename Vector>
void initialize_values(Vector& values)
{
  values.resize(9);
  values[0] = 0; 
  values[1] = 1;
  values[2] = 2;
  values[3] = 3;
  values[4] = 4;
  values[5] = 5;
  values[6] = 6;
  values[7] = 7;
  values[8] = 8;
}


template<typename ExecutionPolicy>
void TestReduceByKeyDevice(ExecutionPolicy exec)
{
  typedef int T;
  
  thrust::device_vector<T> keys;
  thrust::device_vector<T> values;

  typedef typename thrust::pair<
    typename thrust::device_vector<T>::iterator,
    typename thrust::device_vector<T>::iterator
  > iterator_pair;

  thrust::device_vector<iterator_pair> new_last_vec(1);
  iterator_pair new_last;
  
  // basic test
  initialize_keys(keys);  initialize_values(values);
  
  thrust::device_vector<T> output_keys(keys.size());
  thrust::device_vector<T> output_values(values.size());
  
  reduce_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
  
  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);
  
  // test BinaryFunction
  initialize_keys(keys);  initialize_values(values);
  
  reduce_by_key_kernel<<<1,1>>>(exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];
  
  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);
}


void TestReduceByKeyDeviceSeq()
{
  TestReduceByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceSeq);


void TestReduceByKeyDeviceDevice()
{
  TestReduceByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestReduceByKeyDeviceDevice);


void TestReduceByKeyCudaStreams()
{
  typedef thrust::device_vector<int> Vector;
  typedef Vector::value_type T;

  Vector keys;
  Vector values;

  thrust::pair<Vector::iterator, Vector::iterator> new_last;

  // basic test
  initialize_keys(keys);  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  cudaStream_t s;
  cudaStreamCreate(&s);

  new_last = thrust::reduce_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  // test BinaryPredicate
  initialize_keys(keys);  initialize_values(values);
  
  new_last = thrust::reduce_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), is_equal_div_10_reduce<T>());

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   3);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1], 20);
  ASSERT_EQUAL(output_values[2], 15);

  // test BinaryFunction
  initialize_keys(keys);  initialize_values(values);

  new_last = thrust::reduce_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), thrust::equal_to<T>(), thrust::plus<T>());

  ASSERT_EQUAL(new_last.first  - output_keys.begin(),   5);
  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  ASSERT_EQUAL(output_keys[0], 11);
  ASSERT_EQUAL(output_keys[1], 21);
  ASSERT_EQUAL(output_keys[2], 20);
  ASSERT_EQUAL(output_keys[3], 21);
  ASSERT_EQUAL(output_keys[4], 37);
  
  ASSERT_EQUAL(output_values[0],  1);
  ASSERT_EQUAL(output_values[1],  2);
  ASSERT_EQUAL(output_values[2],  3);
  ASSERT_EQUAL(output_values[3], 15);
  ASSERT_EQUAL(output_values[4], 15);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestReduceByKeyCudaStreams);

struct func_props {
  int32_t lhs_idx;
  int32_t rhs_idx;
  bool expects_associativity;
  bool expects_commutativity;
};

__device__ func_props operator +(func_props const& lhs, func_props const& rhs)
{
    return func_props{
        lhs.lhs_idx,
        rhs.rhs_idx,
        lhs.expects_associativity || rhs.expects_associativity || rhs.lhs_idx != rhs.rhs_idx,
        lhs.expects_commutativity || rhs.expects_commutativity || lhs.rhs_idx != rhs.lhs_idx - 1
    };
}

__device__ func_props make_func_props(int32_t idx)
{
  return func_props{
    idx,
    idx,
    false,
    false
  };
}

struct func_prop_reduce
{
  __device__ func_props operator ()(func_props const& lhs, func_props const& rhs) const
  {
    return lhs + rhs;
  }
};

struct int_to_func_props
{
  __device__ func_props operator()(int32_t const x) const
  {
    return make_func_props(x);
  }
};

template<typename T>
struct always_equal
{
  __device__ bool operator ()(T const& lhs, T const& rhs) const
  {
    return true;
  }
};

template<typename T>
struct div_map
{
  T divisor;
  __device__ T operator ()(T value) const
  {
    return value / divisor;
  }
};

void TestNonCommutivitySupport()
{
  int32_t count = 1 << 30;
  int32_t segments = 43;
  int32_t elements_per_segment = count / segments;

  auto count_iter = thrust::make_counting_iterator<int32_t>(0);
  auto keys_iter = thrust::make_transform_iterator(count_iter, div_map<int32_t>{elements_per_segment});
  auto keys = thrust::device_vector<int32_t>(keys_iter, keys_iter + count);

  auto values_iter = thrust::make_transform_iterator(thrust::make_counting_iterator<int32_t>(0), int_to_func_props{});
  auto values = thrust::device_vector<func_props>(values_iter, values_iter + count);

  auto output_keys = thrust::device_vector<int32_t>(segments);
  auto output_values = thrust::device_vector<func_props>(segments);

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::reduce_by_key(thrust::cuda::par.on(s),
                        keys.begin(),
                        keys.end(),
                        values.begin(),
                        output_keys.begin(),
                        output_values.begin(),
                        thrust::equal_to<int>(),
                        func_prop_reduce{});

  thrust::host_vector<func_props> h_output_values = output_values;

  for (auto i = 0; i < segments; i ++)
  {
    ASSERT_EQUAL(h_output_values[i].expects_commutativity, true);
    ASSERT_EQUAL(h_output_values[i].expects_associativity, false);
  }

  auto result = thrust::reduce(thrust::cuda::par.on(s),
                               values.begin() + 1,
                               values.end(),
                               func_props{0, 0, false, false},
                               func_prop_reduce{});

  h_output_values = output_values;

  ASSERT_EQUAL(result.expects_commutativity, true);
  ASSERT_EQUAL(result.expects_associativity, false);

  cudaStreamDestroy(s);

}
DECLARE_UNITTEST(TestNonCommutivitySupport);

