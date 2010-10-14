#include <unittest/unittest.h>
#include <thrust/merge.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

//template<typename Vector>
//void TestMergeSimple(void)
//{
//  typedef typename Vector::iterator Iterator;
//
//  Vector a(3), b(4);
//
//  a[0] = 0; a[1] = 2; a[2] = 4;
//  b[0] = 0; b[1] = 3; b[2] = 3; b[3] = 4;
//
//  Vector ref(7);
//  ref[0] = 0;
//  ref[1] = 0;
//  ref[2] = 2;
//  ref[3] = 3;
//  ref[4] = 3;
//  ref[5] = 4;
//  ref[6] = 4;
//
//  Vector result(7);
//
//  Iterator end = thrust::merge(a.begin(), a.end(),
//                               b.begin(), b.end(),
//                               result.begin());
//
//  ASSERT_EQUAL_QUIET(result.end(), end);
//  ASSERT_EQUAL(ref, result);
//}
//DECLARE_VECTOR_UNITTEST(TestMergeSimple);


template<typename T>
//void TestMerge(const size_t n)
void TestMerge(size_t n)
{
  n = 129;

  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);
  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());

  thrust::sort(h_a.begin(), h_a.end());
  thrust::sort(h_b.begin(), h_b.end());

  thrust::device_vector<T> d_a = h_a;
  thrust::device_vector<T> d_b = h_b;

  thrust::host_vector<T> h_result(h_a.size() + h_b.size());
  thrust::device_vector<T> d_result(d_a.size() + d_b.size());

  typename thrust::host_vector<T>::iterator h_end;
  typename thrust::device_vector<T>::iterator d_end;
  
  h_end = thrust::merge(h_a.begin(), h_a.end(),
                        h_b.begin(), h_b.end(),
                        h_result.begin());

  std::cerr << "n: " << n << std::endl;

//  d_a.erase(thrust::unique(d_a.begin(), d_a.end()), d_a.end());
//  d_b.erase(thrust::unique(d_b.begin(), d_b.end()), d_b.end());

  d_end = thrust::merge(d_a.begin(), d_a.end(),
                        d_b.begin(), d_b.end(),
                        d_result.begin());

  for(unsigned int i = 0; i < h_result.size(); ++i)
  {
    std::cerr << "h_result[" << i << "]: " << (unsigned int)h_result[i] << std::endl;
  }

  ASSERT_EQUAL(h_result, d_result);
}
DECLARE_VARIABLE_UNITTEST(TestMerge);


//template<typename T>
//void TestSetIntersectionEquivalentRanges(const size_t n)
//{
//  thrust::host_vector<T> temp = unittest::random_integers<T>(n);
//  thrust::host_vector<T> h_a = temp; thrust::sort(h_a.begin(), h_a.end());
//  thrust::host_vector<T> h_b = h_a;
//
//  thrust::device_vector<T> d_a = h_a;
//  thrust::device_vector<T> d_b = h_b;
//
//  thrust::host_vector<T>   h_result(n);
//  thrust::device_vector<T> d_result(n);
//
//  typename thrust::host_vector<T>::iterator   h_end;
//  typename thrust::device_vector<T>::iterator d_end;
//  
//  h_end = thrust::set_intersection(h_a.begin(), h_a.end(),
//                                   h_b.begin(), h_b.end(),
//                                   h_result.begin());
//  h_result.resize(h_end - h_result.begin());
//
//  d_end = thrust::set_intersection(d_a.begin(), d_a.end(),
//                                   d_b.begin(), d_b.end(),
//                                   d_result.begin());
//
//  d_result.resize(d_end - d_result.begin());
//
//  ASSERT_EQUAL(h_result, d_result);
//}
//DECLARE_VARIABLE_UNITTEST(TestSetIntersectionEquivalentRanges);
//
//
//template<typename T>
//void TestSetIntersectionMultiset(const size_t n)
//{
//  thrust::host_vector<T> temp = unittest::random_integers<T>(2 * n);
//
//  // restrict elements to [min,13)
//  for(typename thrust::host_vector<T>::iterator i = temp.begin();
//      i != temp.end();
//      ++i)
//  {
//    int temp = static_cast<int>(*i);
//    temp %= 13;
//    *i = temp;
//  }
//
//  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
//  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());
//
//  thrust::sort(h_a.begin(), h_a.end());
//  thrust::sort(h_b.begin(), h_b.end());
//
//  thrust::device_vector<T> d_a = h_a;
//  thrust::device_vector<T> d_b = h_b;
//
//  thrust::host_vector<T> h_result(n);
//  thrust::device_vector<T> d_result(n);
//
//  typename thrust::host_vector<T>::iterator h_end;
//  typename thrust::device_vector<T>::iterator d_end;
//  
//  h_end = thrust::set_intersection(h_a.begin(), h_a.end(),
//                                   h_b.begin(), h_b.end(),
//                                   h_result.begin());
//  h_result.resize(h_end - h_result.begin());
//
//  d_end = thrust::set_intersection(d_a.begin(), d_a.end(),
//                                   d_b.begin(), d_b.end(),
//                                   d_result.begin());
//
//  d_result.resize(d_end - d_result.begin());
//
//  ASSERT_EQUAL(h_result, d_result);
//}
//DECLARE_VARIABLE_UNITTEST(TestSetIntersectionMultiset);
//
//
//struct non_arithmetic
//{
//  __host__ __device__
//  non_arithmetic(void)
//    {}
//
//  __host__ __device__
//  non_arithmetic(const non_arithmetic &x)
//    : key(x.key) {}
//
//  __host__ __device__
//  non_arithmetic(const int k)
//    : key(k) {}
//
//  __host__ __device__
//  bool operator<(const non_arithmetic &rhs) const
//  {
//    return key < rhs.key;
//  }
//
//  __host__ __device__
//  bool operator==(const non_arithmetic &rhs) const
//  {
//    return key == rhs.key;
//  }
//
//  int key;
//};
//
//
//void TestSetIntersectionNonArithmetic(void)
//{
//  const unsigned int n = 12345;
//
//  typedef non_arithmetic T;
//
//  thrust::host_vector<T> temp = unittest::random_integers<int>(2 * n);
//  thrust::host_vector<T> h_a(temp.begin(), temp.begin() + n);
//  thrust::host_vector<T> h_b(temp.begin() + n, temp.end());
//
//  thrust::sort(h_a.begin(), h_a.end());
//  thrust::sort(h_b.begin(), h_b.end());
//
//  thrust::device_vector<T> d_a = h_a;
//  thrust::device_vector<T> d_b = h_b;
//
//  thrust::host_vector<T> h_result(n);
//  thrust::device_vector<T> d_result(n);
//
//  typename thrust::host_vector<T>::iterator h_end;
//  typename thrust::device_vector<T>::iterator d_end;
//  
//  h_end = thrust::set_intersection(h_a.begin(), h_a.end(),
//                                   h_b.begin(), h_b.end(),
//                                   h_result.begin());
//  h_result.resize(h_end - h_result.begin());
//
//  d_end = thrust::set_intersection(d_a.begin(), d_a.end(),
//                                   d_b.begin(), d_b.end(),
//                                   d_result.begin());
//
//  d_result.resize(d_end - d_result.begin());
//
//  ASSERT_EQUAL_QUIET(h_result, d_result);
//}
//DECLARE_UNITTEST(TestSetIntersectionNonArithmetic);
//
//