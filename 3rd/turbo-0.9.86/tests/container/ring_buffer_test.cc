// Copyright 2023 The Turbo Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "turbo/container/ring_buffer.h"
#include "turbo/random/random.h"
#include "gtest/gtest.h"
#include <deque>
#include <list>

using namespace turbo;

// Template instantations.
// These tell the compiler to compile all the functions for the given class.
template class turbo::ring_buffer<int, std::vector<int>>;

template class turbo::ring_buffer<int, std::deque<int>>;

template class turbo::ring_buffer<int, std::list<int>>;

TEST(ring_buffer, all) {

  { // regression for bug in the capacity() function for the case of capacity ==
    // 0.

    std::vector<int> emptyIntArray;
    ring_buffer<int, std::vector<int>> intRingBuffer(emptyIntArray);

    EXPECT_TRUE(intRingBuffer.validate());
    EXPECT_TRUE(intRingBuffer.capacity() == 0);

    intRingBuffer.resize(0);
    EXPECT_TRUE(intRingBuffer.validate());
    EXPECT_TRUE(intRingBuffer.size() == 0);

    intRingBuffer.resize(1);
    EXPECT_TRUE(intRingBuffer.validate());
    EXPECT_TRUE(intRingBuffer.size() == 1);
  }

  {
    turbo::BitGen rng;
    typedef ring_buffer<std::string, std::vector<std::string>> RBVectorString;

    int counter = 0;
    char counterBuffer[32];

    // explicit ring_buffer(size_type size = 0);
    const int kOriginalCapacity = 50;
    RBVectorString rbVectorString(50);

    // bool empty() const;
    // size_type size() const;
    // bool validate() const;
    EXPECT_TRUE(rbVectorString.validate());
    EXPECT_TRUE(rbVectorString.empty());
    EXPECT_TRUE(rbVectorString.size() == 0);
    EXPECT_TRUE(rbVectorString.capacity() == 50);

    // void clear();
    rbVectorString.clear();
    EXPECT_TRUE(rbVectorString.validate());
    EXPECT_TRUE(rbVectorString.empty());
    EXPECT_TRUE(rbVectorString.size() == 0);
    EXPECT_TRUE(rbVectorString.capacity() == 50);

    // container_type& get_container();
    RBVectorString::container_type &c = rbVectorString.get_container();
    EXPECT_TRUE(c.size() == (kOriginalCapacity +
                             1)); // We need to add one because the ring_buffer
                                  // mEnd is necessarily an unused element.

    // iterator begin();
    // iterator end();
    // int validate_iterator(const_iterator i) const;
    RBVectorString::iterator it = rbVectorString.begin();
    while (it != rbVectorString.end()) {
      ++it;
    }

    // void push_back(const value_type& value);
    ::sprintf(counterBuffer, "%d", counter++);
    rbVectorString.push_back(std::string(counterBuffer));
    EXPECT_TRUE(rbVectorString.validate());
    EXPECT_TRUE(!rbVectorString.empty());
    EXPECT_TRUE(rbVectorString.size() == 1);
    EXPECT_TRUE(rbVectorString.capacity() == 50);

    it = rbVectorString.begin();
    EXPECT_TRUE(*it == "0");

    // reference front();
    // reference back();
    std::string &sFront = rbVectorString.front();
    std::string &sBack = rbVectorString.back();
    EXPECT_TRUE(&sFront == &sBack);

    // void push_back();
    std::string &ref = rbVectorString.push_back();
    EXPECT_TRUE(rbVectorString.validate());
    EXPECT_TRUE(rbVectorString.size() == 2);
    EXPECT_TRUE(rbVectorString.capacity() == 50);
    EXPECT_TRUE(&ref == &rbVectorString.back());

    it = rbVectorString.begin();
    ++it;
    EXPECT_TRUE(it->empty());

    ::sprintf(counterBuffer, "%d", counter++);
    *it = counterBuffer;
    EXPECT_TRUE(*it == "1");

    ++it;
    EXPECT_TRUE(it == rbVectorString.end());

    it = rbVectorString.begin();
    while (it != rbVectorString.end()) {
      ++it;
    }

    // reference operator[](size_type n);
    std::string &s0 = rbVectorString[0];
    EXPECT_TRUE(s0 == "0");

    std::string &s1 = rbVectorString[1];
    EXPECT_TRUE(s1 == "1");

    // Now we start hammering the ring buffer with push_back.
    for (size_t i = 0, iEnd = rbVectorString.capacity() * 5; i != iEnd; i++) {
      ::sprintf(counterBuffer, "%d", counter++);
      rbVectorString.push_back(std::string(counterBuffer));
      EXPECT_TRUE(rbVectorString.validate());
    }

    int counterCheck = counter - 1;
    char counterCheckBuffer[32];
    ::sprintf(counterCheckBuffer, "%d", counterCheck);
    EXPECT_TRUE(rbVectorString.back() == counterCheckBuffer);

    // reverse_iterator rbegin();
    // reverse_iterator rend();
    for (RBVectorString::reverse_iterator ri = rbVectorString.rbegin();
         ri != rbVectorString.rend(); ++ri) {
      ::sprintf(counterCheckBuffer, "%d", counterCheck--);
      EXPECT_TRUE(*ri == counterCheckBuffer);
    }

    ++counterCheck;

    // iterator begin();
    // iterator end();
    for (RBVectorString::iterator i = rbVectorString.begin();
         i != rbVectorString.end(); ++i) {
      EXPECT_TRUE(*i == counterCheckBuffer);
      ::sprintf(counterCheckBuffer, "%d", ++counterCheck);
    }

    // void clear();
    rbVectorString.clear();
    EXPECT_TRUE(rbVectorString.validate());
    EXPECT_TRUE(rbVectorString.empty());
    EXPECT_TRUE(rbVectorString.size() == 0);
    EXPECT_TRUE(rbVectorString.capacity() == 50);

    // We make sure that after the above we still have some contents.
    if (rbVectorString.size() < 8)
      rbVectorString.resize(8);

    EXPECT_TRUE(rbVectorString.validate());

    // Test const functions
    // const_iterator begin() const;
    // const_iterator end() const;
    // const_reverse_iterator rbegin() const;
    // const_reverse_iterator rend() const;
    // const_reference front() const;
    // const_reference back() const;
    // const_reference operator[](size_type n) const;
    // const container_type& get_container() const;
    const RBVectorString &rbVSConst = rbVectorString;
    EXPECT_TRUE(rbVSConst.front() == rbVectorString.front());
    EXPECT_TRUE(rbVSConst.back() == rbVectorString.back());
    EXPECT_TRUE(rbVSConst[0] == rbVectorString[0]);
    EXPECT_TRUE(&rbVSConst.get_container() == &rbVectorString.get_container());

    // Test additional constructors.
    // ring_buffer(const this_type& x);
    // explicit ring_buffer(const Container& x);
    // this_type& operator=(const this_type& x);
    // void swap(this_type& x);
    RBVectorString rbVectorString2(rbVectorString);
    RBVectorString rbVectorString3(rbVectorString.get_container());
    RBVectorString rbVectorString4(rbVectorString.capacity() / 2);
    RBVectorString rbVectorString5(rbVectorString.capacity() * 2);

    EXPECT_TRUE(rbVectorString.validate());
    EXPECT_TRUE(rbVectorString2.validate());
    EXPECT_TRUE(rbVectorString3.validate());
    EXPECT_TRUE(rbVectorString4.validate());
    EXPECT_TRUE(rbVectorString5.validate());

    EXPECT_TRUE(rbVectorString == rbVectorString2);
    EXPECT_TRUE(rbVectorString3.get_container() ==
                rbVectorString2.get_container());

    rbVectorString3 = rbVectorString4;
    EXPECT_TRUE(rbVectorString3.validate());

    turbo::swap(rbVectorString2, rbVectorString4);
    EXPECT_TRUE(rbVectorString2.validate());
    EXPECT_TRUE(rbVectorString3.validate());
    EXPECT_TRUE(rbVectorString4.validate());
    EXPECT_TRUE(rbVectorString == rbVectorString4);
    EXPECT_TRUE(rbVectorString2 == rbVectorString3);

    // void ring_buffer<T, Container>::reserve(size_type n)
    size_t cap = rbVectorString2.capacity();
    rbVectorString2.reserve(cap += 2);
    EXPECT_TRUE(rbVectorString2.validate());
    EXPECT_TRUE(rbVectorString2.capacity() == cap);
    rbVectorString2.reserve(
        cap -= 4); // This should act as a no-op if we are following convention.
    EXPECT_TRUE(rbVectorString2.validate());

    // void ring_buffer<T, Container>::set_capacity(size_type n)
    cap = rbVectorString2.capacity();
    rbVectorString2.resize(cap);
    EXPECT_TRUE(rbVectorString2.size() == cap);
    rbVectorString2.set_capacity(cap += 2);
    EXPECT_TRUE(rbVectorString2.validate());
    EXPECT_TRUE(rbVectorString2.capacity() == cap);
    rbVectorString2.set_capacity(cap -= 4);
    EXPECT_TRUE(rbVectorString2.capacity() == cap);
    EXPECT_TRUE(rbVectorString2.validate());

    // template <typename InputIterator>
    // void assign(InputIterator first, InputIterator last);
    std::string stringArray[10];
    for (int q = 0; q < 10; q++)
      stringArray[q] = (char)('0' + (char)q);

    rbVectorString5.assign(stringArray, stringArray + 10);
    EXPECT_TRUE(rbVectorString5.validate());
    EXPECT_TRUE(rbVectorString5.size() == 10);
    EXPECT_TRUE(rbVectorString5.front() == "0");
    EXPECT_TRUE(rbVectorString5.back() == "9");
  }

  {
    // Additional testing
    typedef ring_buffer<int, std::vector<int>> RBVectorInt;

    RBVectorInt rbVectorInt(6);

    rbVectorInt.push_back(0);
    rbVectorInt.push_back(1);
    rbVectorInt.push_back(2);
    rbVectorInt.push_back(3);
    rbVectorInt.push_back(4);
    rbVectorInt.push_back(5);
    EXPECT_TRUE(rbVectorInt[0] == 0);
    EXPECT_TRUE(rbVectorInt[5] == 5);

    // iterator insert(iterator position, const value_type& value);
    rbVectorInt.insert(rbVectorInt.begin(), 999);
    EXPECT_TRUE(rbVectorInt[0] == 999);
    EXPECT_TRUE(rbVectorInt[1] == 0);
    EXPECT_TRUE(rbVectorInt[5] == 4);

    rbVectorInt.clear();
    rbVectorInt.push_back(0);
    rbVectorInt.push_back(1);
    rbVectorInt.push_back(2);
    rbVectorInt.push_back(3);
    rbVectorInt.push_back(4);

    // iterator insert(iterator position, const value_type& value);
    rbVectorInt.insert(rbVectorInt.begin(), 999);
    EXPECT_TRUE(rbVectorInt[0] == 999);
    EXPECT_TRUE(rbVectorInt[1] == 0);
    EXPECT_TRUE(rbVectorInt[5] == 4);

    rbVectorInt.clear();
    rbVectorInt.push_back(0);
    rbVectorInt.push_back(1);
    rbVectorInt.push_back(2);
    rbVectorInt.push_back(3);
    rbVectorInt.push_back(4);
    rbVectorInt.push_back(5);
    rbVectorInt.push_back(6);
    EXPECT_TRUE(rbVectorInt[0] == 1);
    EXPECT_TRUE(rbVectorInt[5] == 6);

    // iterator insert(iterator position, const value_type& value);
    rbVectorInt.insert(rbVectorInt.begin(), 999);
    EXPECT_TRUE(rbVectorInt[0] == 999);
    EXPECT_TRUE(rbVectorInt[1] == 1);
    EXPECT_TRUE(rbVectorInt[5] == 5);

    // iterator insert(iterator position, const value_type& value);
    RBVectorInt::iterator it = rbVectorInt.begin();
    std::advance(it, 3);
    rbVectorInt.insert(it, 888);
    EXPECT_TRUE(rbVectorInt[0] == 999);
    EXPECT_TRUE(rbVectorInt[1] == 1);
    EXPECT_TRUE(rbVectorInt[2] == 2);
    EXPECT_TRUE(rbVectorInt[3] == 888);
    EXPECT_TRUE(rbVectorInt[4] == 3);
    EXPECT_TRUE(rbVectorInt[5] == 4);
  }

  {
    turbo::BitGen rng;

    typedef ring_buffer<std::string, std::list<std::string>> RBListString;

    int counter = 0;
    char counterBuffer[32];

    // explicit ring_buffer(size_type size = 0);
    const int kOriginalCapacity = 50;
    RBListString rbListString(50);

    // bool empty() const;
    // size_type size() const;
    // bool validate() const;
    EXPECT_TRUE(rbListString.validate());
    EXPECT_TRUE(rbListString.empty());
    EXPECT_TRUE(rbListString.size() == 0);
    EXPECT_TRUE(rbListString.capacity() == 50);

    // void clear();
    rbListString.clear();
    EXPECT_TRUE(rbListString.validate());
    EXPECT_TRUE(rbListString.empty());
    EXPECT_TRUE(rbListString.size() == 0);
    EXPECT_TRUE(rbListString.capacity() == 50);

    // container_type& get_container();
    RBListString::container_type &c = rbListString.get_container();
    EXPECT_TRUE(c.size() == (kOriginalCapacity +
                             1)); // We need to add one because the ring_buffer
                                  // mEnd is necessarily an unused element.

    // iterator begin();
    // iterator end();
    // int validate_iterator(const_iterator i) const;
    RBListString::iterator it = rbListString.begin();

    while (it != rbListString.end()) // This loop should do nothing.
    {
      ++it;
    }

    // void push_back(const value_type& value);
    ::sprintf(counterBuffer, "%d", counter++);
    rbListString.push_back(std::string(counterBuffer));
    EXPECT_TRUE(rbListString.validate());
    EXPECT_TRUE(!rbListString.empty());
    EXPECT_TRUE(rbListString.size() == 1);
    EXPECT_TRUE(rbListString.capacity() == 50);

    it = rbListString.begin();
    EXPECT_TRUE(*it == "0");

    // reference front();
    // reference back();
    std::string &sFront = rbListString.front();
    std::string &sBack = rbListString.back();
    EXPECT_TRUE(&sFront == &sBack);

    // void push_back();
    std::string &ref = rbListString.push_back();
    EXPECT_TRUE(rbListString.validate());
    EXPECT_TRUE(rbListString.size() == 2);
    EXPECT_TRUE(rbListString.capacity() == 50);
    EXPECT_TRUE(&ref == &rbListString.back());

    it = rbListString.begin();
    ++it;
    EXPECT_TRUE(it->empty());

    ::sprintf(counterBuffer, "%d", counter++);
    *it = counterBuffer;
    EXPECT_TRUE(*it == "1");

    ++it;
    EXPECT_TRUE(it == rbListString.end());

    it = rbListString.begin();
    while (it != rbListString.end()) {
      ++it;
    }

    // reference operator[](size_type n);
    std::string &s0 = rbListString[0];
    EXPECT_TRUE(s0 == "0");

    std::string &s1 = rbListString[1];
    EXPECT_TRUE(s1 == "1");

    // Now we start hammering the ring buffer with push_back.
    for (size_t i = 0, iEnd = rbListString.capacity() * 5; i != iEnd; i++) {
      ::sprintf(counterBuffer, "%d", counter++);
      rbListString.push_back(std::string(counterBuffer));
      EXPECT_TRUE(rbListString.validate());
    }

    int counterCheck = counter - 1;
    char counterCheckBuffer[32];
    ::sprintf(counterCheckBuffer, "%d", counterCheck);
    EXPECT_TRUE(rbListString.back() == counterCheckBuffer);

    // reverse_iterator rbegin();
    // reverse_iterator rend();
    for (RBListString::reverse_iterator ri = rbListString.rbegin();
         ri != rbListString.rend(); ++ri) {
      ::sprintf(counterCheckBuffer, "%d", counterCheck--);
      EXPECT_EQ(*ri, counterCheckBuffer);
    }

    ++counterCheck;

    // iterator begin();
    // iterator end();
    for (RBListString::iterator i = rbListString.begin();
         i != rbListString.end(); ++i) {
      EXPECT_EQ(*i,  counterCheckBuffer);
      ::sprintf(counterCheckBuffer, "%d", ++counterCheck);
    }

    // void clear();
    rbListString.clear();
    EXPECT_TRUE(rbListString.validate());
    EXPECT_TRUE(rbListString.empty());
    EXPECT_TRUE(rbListString.size() == 0);
    EXPECT_TRUE(rbListString.capacity() == 50);

    // We make sure that after the above we still have some contents.
    if (rbListString.size() < 8)
      rbListString.resize(8);

    EXPECT_TRUE(rbListString.validate());

    // Test const functions
    // const_iterator begin() const;
    // const_iterator end() const;
    // const_reverse_iterator rbegin() const;
    // const_reverse_iterator rend() const;
    // const_reference front() const;
    // const_reference back() const;
    // const_reference operator[](size_type n) const;
    // const container_type& get_container() const;
    const RBListString &rbVSConst = rbListString;

    EXPECT_TRUE(rbVSConst.front() == rbListString.front());
    EXPECT_TRUE(rbVSConst.back() == rbListString.back());
    EXPECT_TRUE(rbVSConst[0] == rbListString[0]);
    EXPECT_TRUE(&rbVSConst.get_container() == &rbListString.get_container());

    // Test additional constructors.
    // ring_buffer(const this_type& x);
    // explicit ring_buffer(const Container& x);
    // this_type& operator=(const this_type& x);
    // void swap(this_type& x);
    RBListString rbListString2(rbListString);
    RBListString rbListString3(rbListString.get_container());
    RBListString rbListString4(rbListString.capacity() / 2);
    RBListString rbListString5(rbListString.capacity() * 2);

    EXPECT_TRUE(rbListString.validate());
    EXPECT_TRUE(rbListString2.validate());
    EXPECT_TRUE(rbListString3.validate());
    EXPECT_TRUE(rbListString4.validate());
    EXPECT_TRUE(rbListString5.validate());

    EXPECT_TRUE(rbListString == rbListString2);
    EXPECT_TRUE(rbListString3.get_container() == rbListString2.get_container());

    rbListString3 = rbListString4;
    EXPECT_TRUE(rbListString3.validate());

    turbo::swap(rbListString2, rbListString4);
    EXPECT_TRUE(rbListString2.validate());
    EXPECT_TRUE(rbListString3.validate());
    EXPECT_TRUE(rbListString4.validate());
    EXPECT_TRUE(rbListString == rbListString4);
    EXPECT_TRUE(rbListString2 == rbListString3);

    // void ring_buffer<T, Container>::reserve(size_type n)
    size_t cap = rbListString2.capacity();
    rbListString2.reserve(cap += 2);
    EXPECT_TRUE(rbListString2.validate());
    EXPECT_TRUE(rbListString2.capacity() == cap);
    rbListString2.reserve(
        cap -= 4); // This should act as a no-op if we are following convention.
    EXPECT_TRUE(rbListString2.validate());

    // template <typename InputIterator>
    // void assign(InputIterator first, InputIterator last);
    std::string stringArray[10];
    for (int q = 0; q < 10; q++)
      stringArray[q] = '0' + (char)q;

    rbListString5.assign(stringArray, stringArray + 10);
    EXPECT_TRUE(rbListString5.validate());
    EXPECT_TRUE(rbListString5.size() == 10);
    EXPECT_TRUE(rbListString5.front() == "0");
    EXPECT_TRUE(rbListString5.back() == "9");

    // ring_buffer(this_type&& x);
    // ring_buffer(this_type&& x, const allocator_type& allocator);
    // this_type& operator=(this_type&& x);

    RBListString rbListStringM1(std::move(rbListString5));
    EXPECT_TRUE(rbListStringM1.validate() && rbListString5.validate());
    EXPECT_TRUE((rbListStringM1.size() == 10) && (rbListString5.size() == 0));

    RBListString rbListStringM2(std::move(rbListStringM1),
                                RBListString::allocator_type());
    EXPECT_TRUE(rbListStringM2.validate() && rbListStringM1.validate());
    EXPECT_TRUE((rbListStringM2.size() == 10) && (rbListStringM1.size() == 0));

    rbListStringM1 = std::move(rbListStringM2);
    EXPECT_TRUE(rbListStringM1.validate() && rbListStringM2.validate());
    EXPECT_TRUE((rbListStringM1.size() == 10) && (rbListStringM2.size() == 0));
  }

  {
    // Regression for bug with iterator subtraction
    typedef turbo::ring_buffer<int> IntBuffer_t;
    IntBuffer_t intBuffer = {0, 1, 2, 3, 4, 5, 6, 7};
    IntBuffer_t::iterator it = intBuffer.begin();

    EXPECT_TRUE(*it == 0);
    it += 4;
    EXPECT_TRUE(*it == 4);
    it--;
    EXPECT_TRUE(*it == 3);
    it -= 2;
    EXPECT_TRUE(*it == 1);

    intBuffer.push_back(8);
    intBuffer.push_back(9);
    intBuffer.push_back(10);
    intBuffer.push_back(11);

    EXPECT_TRUE(*it == 10);
    it -= 3;
    EXPECT_TRUE(*it ==
                7); // Test looping around the end of the underlying container
    it -= 5;
    EXPECT_TRUE(*it ==
                11); // Test wrapping around begin to end of the ring_buffer
    it -= 2;
    EXPECT_TRUE(*it == 9); // It is important to test going back to the
                           // beginning of the underlying container.
  }
}
