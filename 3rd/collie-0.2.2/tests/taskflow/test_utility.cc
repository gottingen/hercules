#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <collie/testing/doctest.h>

#include <collie/taskflow/utility/traits.h>
#include <collie/taskflow/utility/object_pool.h>
#include <collie/taskflow/utility/math.h>



// --------------------------------------------------------
// Testcase: ObjectPool.Sequential
// --------------------------------------------------------
struct Poolable {
  std::string str;
  std::vector<int> vec;
  int a;
  char b;

  TF_ENABLE_POOLABLE_ON_THIS;
};

TEST_CASE("ObjectPool.Sequential" * doctest::timeout(300)) {

  for(unsigned w=1; w<=4; w++) {

    collie::tf::ObjectPool<Poolable> pool(w);

    REQUIRE(pool.num_heaps() > 0);
    REQUIRE(pool.num_local_heaps() > 0);
    REQUIRE(pool.num_global_heaps() > 0);
    REQUIRE(pool.num_bins_per_local_heap() > 0);
    REQUIRE(pool.num_objects_per_bin() > 0);
    REQUIRE(pool.num_objects_per_block() > 0);
    REQUIRE(pool.emptiness_threshold() > 0);

    // fill out all objects
    size_t N = 100*pool.num_objects_per_block();

    std::set<Poolable*> set;

    for(size_t i=0; i<N; ++i) {
      auto item = pool.animate();
      REQUIRE(set.find(item) == set.end());
      set.insert(item);
    }

    REQUIRE(set.size() == N);

    for(auto s : set) {
      pool.recycle(s);
    }

    REQUIRE(N == pool.capacity());
    REQUIRE(N == pool.num_available_objects());
    REQUIRE(0 == pool.num_allocated_objects());

    for(size_t i=0; i<N; ++i) {
      auto item = pool.animate();
      REQUIRE(set.find(item) != set.end());
    }

    REQUIRE(pool.num_available_objects() == 0);
    REQUIRE(pool.num_allocated_objects() == N);
  }
}

// --------------------------------------------------------
// Testcase: ObjectPool.Threaded
// --------------------------------------------------------

template <typename T>
void threaded_objectpool(unsigned W) {

  collie::tf::ObjectPool<T> pool;

  std::vector<std::thread> threads;

  for(unsigned w=0; w<W; ++w) {
    threads.emplace_back([&pool](){
      std::vector<T*> items;
      for(int i=0; i<65536; ++i) {
        auto item = pool.animate();
        items.push_back(item);
      }
      for(auto item : items) {
        pool.recycle(item);
      }
    });
  }

  for(auto& thread : threads) {
    thread.join();
  }

  REQUIRE(pool.num_allocated_objects() == 0);
  REQUIRE(pool.num_available_objects() == pool.capacity());
}

TEST_CASE("ObjectPool.1thread" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(1);
}

TEST_CASE("ObjectPool.2threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(2);
}

TEST_CASE("ObjectPool.3threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(3);
}

TEST_CASE("ObjectPool.4threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(4);
}

TEST_CASE("ObjectPool.5threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(5);
}

TEST_CASE("ObjectPool.6threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(6);
}

TEST_CASE("ObjectPool.7threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(7);
}

TEST_CASE("ObjectPool.8threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(8);
}

TEST_CASE("ObjectPool.9threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(9);
}

TEST_CASE("ObjectPool.10threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(10);
}

TEST_CASE("ObjectPool.11threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(11);
}

TEST_CASE("ObjectPool.12threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(12);
}

TEST_CASE("ObjectPool.13threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(13);
}

TEST_CASE("ObjectPool.14threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(14);
}

TEST_CASE("ObjectPool.15threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(15);
}

TEST_CASE("ObjectPool.16threads" * doctest::timeout(300)) {
  threaded_objectpool<Poolable>(16);
}

// --------------------------------------------------------
// Testcase: Reference Wrapper
// --------------------------------------------------------

TEST_CASE("RefWrapper" * doctest::timeout(300)) {

  static_assert(std::is_same<
    collie::tf::unwrap_reference_t<int>, int
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_reference_t<int&>, int&
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_reference_t<int&&>, int&&
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_reference_t<std::reference_wrapper<int>>, int&
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_reference_t<std::reference_wrapper<std::reference_wrapper<int>>>,
    std::reference_wrapper<int>&
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_ref_decay_t<int>, int
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_ref_decay_t<int&>, int
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_ref_decay_t<int&&>, int
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_ref_decay_t<std::reference_wrapper<int>>, int&
  >::value, "");

  static_assert(std::is_same<
    collie::tf::unwrap_ref_decay_t<std::reference_wrapper<std::reference_wrapper<int>>>,
    std::reference_wrapper<int>&
  >::value, "");

}

//// --------------------------------------------------------
//// Testcase: FunctionTraits
//// --------------------------------------------------------
//void func1() {
//}
//
//int func2(int, double, float, char) {
//  return 0;
//}
//
//TEST_CASE("FunctionTraits" * doctest::timeout(300)) {
//
//  SUBCASE("func1") {
//    using func1_traits = collie::tf::function_traits<decltype(func1)>;
//    static_assert(std::is_same<func1_traits::return_type, void>::value, "");
//    static_assert(func1_traits::arity == 0, "");
//  }
//
//  SUBCASE("func2") {
//    using func2_traits = collie::tf::function_traits<decltype(func2)>;
//    static_assert(std::is_same<func2_traits::return_type, int>::value, "");
//    static_assert(func2_traits::arity == 4, "");
//    static_assert(std::is_same<func2_traits::argument_t<0>, int>::value,   "");
//    static_assert(std::is_same<func2_traits::argument_t<1>, double>::value,"");
//    static_assert(std::is_same<func2_traits::argument_t<2>, float>::value, "");
//    static_assert(std::is_same<func2_traits::argument_t<3>, char>::value,  "");
//  }
//
//  SUBCASE("lambda1") {
//    auto lambda1 = [] () mutable {
//      return 1;
//    };
//    using lambda1_traits = collie::tf::function_traits<decltype(lambda1)>;
//    static_assert(std::is_same<lambda1_traits::return_type, int>::value, "");
//    static_assert(lambda1_traits::arity == 0, "");
//  }
//
//  SUBCASE("lambda2") {
//    auto lambda2 = [] (int, double, char&) {
//    };
//    using lambda2_traits = collie::tf::function_traits<decltype(lambda2)>;
//    static_assert(std::is_same<lambda2_traits::return_type, void>::value, "");
//    static_assert(lambda2_traits::arity == 3, "");
//    static_assert(std::is_same<lambda2_traits::argument_t<0>, int>::value, "");
//    static_assert(std::is_same<lambda2_traits::argument_t<1>, double>::value, "");
//    static_assert(std::is_same<lambda2_traits::argument_t<2>, char&>::value, "");
//  }
//
//  SUBCASE("class") {
//    struct foo {
//      int operator ()(int, float) const;
//    };
//    using foo_traits = collie::tf::function_traits<foo>;
//    static_assert(std::is_same<foo_traits::return_type, int>::value, "");
//    static_assert(foo_traits::arity == 2, "");
//    static_assert(std::is_same<foo_traits::argument_t<0>, int>::value, "");
//    static_assert(std::is_same<foo_traits::argument_t<1>, float>::value, "");
//  }
//
//  SUBCASE("std-function") {
//    using ft1 = collie::tf::function_traits<std::function<void()>>;
//    static_assert(std::is_same<ft1::return_type, void>::value, "");
//    static_assert(ft1::arity == 0, "");
//
//    using ft2 = collie::tf::function_traits<std::function<int(int&, double&&)>&>;
//    static_assert(std::is_same<ft2::return_type, int>::value, "");
//    static_assert(ft2::arity == 2, "");
//    static_assert(std::is_same<ft2::argument_t<0>, int&>::value, "");
//    static_assert(std::is_same<ft2::argument_t<1>, double&&>::value, "");
//
//    using ft3 = collie::tf::function_traits<std::function<int(int&, double&&)>&&>;
//    static_assert(std::is_same<ft3::return_type, int>::value, "");
//    static_assert(ft3::arity == 2, "");
//    static_assert(std::is_same<ft3::argument_t<0>, int&>::value, "");
//    static_assert(std::is_same<ft3::argument_t<1>, double&&>::value, "");
//
//    using ft4 = collie::tf::function_traits<const std::function<void(int)>&>;
//    static_assert(std::is_same<ft4::return_type, void>::value, "");
//    static_assert(ft4::arity == 1, "");
//    static_assert(std::is_same<ft4::argument_t<0>, int>::value, "");
//  }
//}

// --------------------------------------------------------
// Math utilities
// --------------------------------------------------------
TEST_CASE("NextPow2") {

  static_assert(collie::tf::next_pow2(0u) == 1);
  static_assert(collie::tf::next_pow2(1u) == 1);
  static_assert(collie::tf::next_pow2(100u) == 128u);
  static_assert(collie::tf::next_pow2(245u) == 256u);
  static_assert(collie::tf::next_pow2(512u) == 512u);
  static_assert(collie::tf::next_pow2(513u) == 1024u);

  REQUIRE(collie::tf::next_pow2(0u) == 1u);
  REQUIRE(collie::tf::next_pow2(2u) == 2u);
  REQUIRE(collie::tf::next_pow2(1u) == 1u);
  REQUIRE(collie::tf::next_pow2(33u) == 64u);
  REQUIRE(collie::tf::next_pow2(100u) == 128u);
  REQUIRE(collie::tf::next_pow2(211u) == 256u);
  REQUIRE(collie::tf::next_pow2(23u) == 32u);
  REQUIRE(collie::tf::next_pow2(54u) == 64u);

  uint64_t z = 0;
  uint64_t a = 1;
  REQUIRE(collie::tf::next_pow2(z) == 1);
  REQUIRE(collie::tf::next_pow2(a) == a);
  REQUIRE(collie::tf::next_pow2((a<<5)  + 0) == (a<<5));
  REQUIRE(collie::tf::next_pow2((a<<5)  + 1) == (a<<6));
  REQUIRE(collie::tf::next_pow2((a<<32) + 0) == (a<<32));
  REQUIRE(collie::tf::next_pow2((a<<32) + 1) == (a<<33));
  REQUIRE(collie::tf::next_pow2((a<<41) + 0) == (a<<41));
  REQUIRE(collie::tf::next_pow2((a<<41) + 1) == (a<<42));

  REQUIRE(collie::tf::is_pow2(0) == false);
  REQUIRE(collie::tf::is_pow2(1) == true);
  REQUIRE(collie::tf::is_pow2(2) == true);
  REQUIRE(collie::tf::is_pow2(3) == false);
  REQUIRE(collie::tf::is_pow2(0u) == false);
  REQUIRE(collie::tf::is_pow2(1u) == true);
  REQUIRE(collie::tf::is_pow2(54u) == false);
  REQUIRE(collie::tf::is_pow2(64u) == true);
}





