// Copyright 2024 The Elastic AI Search Authors.
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
//


#include <iostream>
#include <string>
#include <unordered_set>

#include <collie/type_safe/strong_typedef.h>

namespace ts = collie::ts;

// we want some kind of handle to an int
struct handle : ts::strong_typedef<handle, int*>,                   // required
                ts::strong_typedef_op::equality_comparison<handle>, // for operator==/operator!=
                ts::strong_typedef_op::dereference<handle, int>     // for operator*/operator->
{
    using strong_typedef::strong_typedef;

    // we also want the operator bool()
    explicit operator bool() const noexcept
    {
        return static_cast<int*>(*this) != nullptr;
    }
};

void use_handle(handle h)
{
    // we can dereference it
    std::cout << *h << '\n';

    // and compare it
    (void)(h == handle(nullptr));

    // and reassign
    int a;
    h = handle(&a);
    // or get back
    int* ptr = static_cast<int*>(h);
    std::cout << &a << ' ' << ptr << '\n';
}

// integer representing a distance
struct distance
: ts::strong_typedef<distance, unsigned>,                 // required
  ts::strong_typedef_op::equality_comparison<distance>,   // for operator==/operator!=
  ts::strong_typedef_op::relational_comparison<distance>, // for operator< etc.
  ts::strong_typedef_op::integer_arithmetic<distance> // all arithmetic operators that make sense
                                                      // for integers
{
    using strong_typedef::strong_typedef;
};

// we want to output it
std::ostream& operator<<(std::ostream& out, const distance& d)
{
    return out << static_cast<unsigned>(d) << " distance";
}

namespace std
{
// we want to use it with the std::unordered_* containers
template <>
struct hash<::distance> : collie::ts::hashable<::distance>
{};

} // namespace std

int main()
{
    distance d(4);
    //    int      val = d; // error
    //    d += 3;           // error
    d += distance(3); // works

    std::unordered_set<distance> set{d};

    std::cout << *set.find(d) << '\n';
}
