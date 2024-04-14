// Copyright 2023 The Elastic-AI Authors.
// part of Elastic AI Search
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
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "turbo/testing/test.h"
#include <cstddef>
#include "turbo/container/intrusive_list.h"

namespace {
    using namespace turbo;
    using namespace turbo::testing;
    /// IntNode
    ///
    /// Test intrusive_list node.
    ///
    struct IntNode : public turbo::intrusive_list_node {
        int mX;

        IntNode(int x = 0)
                : mX(x) {}

        operator int() const { return mX; }
    };


    /// ListInit
    ///
    /// Utility class for setting up a list.
    ///
    class ListInit {
    public:
        ListInit(intrusive_list <IntNode> &container, IntNode *pNodeArray)
                : mpContainer(&container), mpNodeArray(pNodeArray) {
            mpContainer->clear();
        }

        ListInit &operator+=(int x) {
            mpNodeArray->mX = x;
            mpContainer->push_back(*mpNodeArray++);
            return *this;
        }

        ListInit &operator,(int x) {
            mpNodeArray->mX = x;
            mpContainer->push_back(*mpNodeArray++);
            return *this;
        }

    protected:
        intrusive_list <IntNode> *mpContainer;
        IntNode *mpNodeArray;
    };

} // namespace




// Template instantations.
// These tell the compiler to compile all the functions for the given class.
template
class turbo::intrusive_list<IntNode>;



TEST_CASE("list ") {

    int i;
    {
        IntNode nodes[20];

        intrusive_list <IntNode> ilist;

#ifndef __GNUC__ // GCC warns on this, though strictly specaking it is allowed to.
        // Enforce that offsetof() can be used with an intrusive_list in a struct;
            // it requires a POD type. Some compilers will flag warnings or even errors
            // when this is violated.
            struct Test {
                intrusive_list<IntNode> m;
            };
            (void)offsetof(Test, m);
#endif

        // begin / end
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "ctor()", -1));


        // push_back
        ListInit(ilist, nodes) += 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "push_back()", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1));


        // iterator / begin
        intrusive_list<IntNode>::iterator it = ilist.begin();
        REQUIRE(it->mX == 0);
        ++it;
        REQUIRE(it->mX == 1);
        ++it;
        REQUIRE(it->mX == 2);
        ++it;
        REQUIRE(it->mX == 3);


        // const_iterator / begin
        const intrusive_list <IntNode> cilist;
        intrusive_list<IntNode>::const_iterator cit;
        for (cit = cilist.begin(); cit != cilist.end(); ++cit)
            REQUIRE(cit == cilist.end()); // This is guaranteed to be false.


        // reverse_iterator / rbegin
        intrusive_list<IntNode>::reverse_iterator itr = ilist.rbegin();
        REQUIRE(itr->mX == 9);
        ++itr;
        REQUIRE(itr->mX == 8);
        ++itr;
        REQUIRE(itr->mX == 7);
        ++itr;
        REQUIRE(itr->mX == 6);


        // iterator++/--
        {
            intrusive_list<IntNode>::iterator it1(ilist.begin());
            intrusive_list<IntNode>::iterator it2(ilist.begin());

            ++it1;
            ++it2;
            if ((it1 != it2++) || (++it1 != it2))
                REQUIRE(!"[iterator::increment] fail\n");

            if ((it1 != it2--) || (--it1 != it2))
                REQUIRE(!"[iterator::decrement] fail\n");
        }


        // clear / empty
        REQUIRE(!ilist.empty());

        ilist.clear();
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "clear()", -1));
        REQUIRE(ilist.empty());


        // splice
        ListInit(ilist, nodes) += 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;

        ilist.splice(++ilist.begin(), ilist, --ilist.end());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(single)", 0, 9, 1, 2, 3, 4, 5, 6, 7, 8, -1));

        intrusive_list <IntNode> ilist2;
        ListInit(ilist2, nodes + 10) += 10, 11, 12, 13, 14, 15, 16, 17, 18, 19;

        ilist.splice(++ ++ilist.begin(), ilist2);
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "splice(whole)", -1));
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(whole)", 0, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                              18, 19, 1, 2, 3, 4, 5, 6, 7, 8, -1));

        ilist.splice(ilist.begin(), ilist, ++ ++ilist.begin(), -- --ilist.end());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(range)", 10, 11, 12, 13, 14, 15, 16, 17, 18,
                              19, 1, 2, 3, 4, 5, 6, 0, 9, 7, 8, -1));

        ilist.clear();
        ilist.swap(ilist2);
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "swap(empty)", -1));
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "swap(empty)", -1));

        ilist2.push_back(nodes[0]);
        ilist.splice(ilist.begin(), ilist2);
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(single)", 0, -1));
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "splice(single)", -1));


        // splice(single) -- evil case (splice at or right after current position)
        ListInit(ilist, nodes) += 0, 1, 2, 3, 4;
        ilist.splice(++ ++ilist.begin(), *++ ++ilist.begin());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(single)", 0, 1, 2, 3, 4, -1));
        ilist.splice(++ ++ ++ilist.begin(), *++ ++ilist.begin());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(single)", 0, 1, 2, 3, 4, -1));


        // splice(range) -- evil case (splice right after current position)
        ListInit(ilist, nodes) += 0, 1, 2, 3, 4;
        ilist.splice(++ ++ilist.begin(), ilist, ++ilist.begin(), ++ ++ilist.begin());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "splice(range)", 0, 1, 2, 3, 4, -1));


        // push_front / push_back
        ilist.clear();
        ilist2.clear();
        for (i = 4; i >= 0; --i)
            ilist.push_front(nodes[i]);
        for (i = 5; i < 10; ++i)
            ilist2.push_back(nodes[i]);

        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "push_front()", 0, 1, 2, 3, 4, -1));
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "push_back()", 5, 6, 7, 8, 9, -1));

        for (i = 4; i >= 0; --i) {
            ilist.pop_front();
            ilist2.pop_back();
        }
        auto em = ilist2.empty() && ilist.empty();
        REQUIRE(ilist.empty());
        REQUIRE(ilist2.empty());
        REQUIRE(em);
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "pop_front()", -1));
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "pop_back()", -1));


        // contains / locate
        for (i = 0; i < 5; ++i)
            ilist.push_back(nodes[i]);

        REQUIRE(ilist.contains(nodes[2]));
        REQUIRE(!ilist.contains(nodes[7]));

        it = ilist.locate(nodes[3]);
        REQUIRE(it->mX == 3);

        it = ilist.locate(nodes[8]);
        REQUIRE(it == ilist.end());


        // reverse
        ilist.reverse();
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "push_front()", 4, 3, 2, 1, 0, -1));


        // swap()
        ilist.swap(ilist2);
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "swap()", -1));
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "swap()", 4, 3, 2, 1, 0, -1));


        // erase()
        ListInit(ilist2, nodes) += 0, 1, 2, 3, 4;
        ListInit(ilist, nodes + 5) += 5, 6, 7, 8, 9;
        ilist.erase(++ ++ilist.begin());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "erase(single)", 5, 6, 8, 9, -1));

        ilist.erase(ilist.begin(), ilist.end());
        REQUIRE(check_sequence_eq(ilist.begin(), ilist.end(), int(), "erase(all)", -1));

        ilist2.erase(++ilist2.begin(), -- --ilist2.end());
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "erase(range)", 0, 3, 4, -1));


        // size
        REQUIRE(ilist2.size() == 3);


        // pop_front / pop_back
        ilist2.pop_front();
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "pop_front()", 3, 4, -1));

        ilist2.pop_back();
        REQUIRE(check_sequence_eq(ilist2.begin(), ilist2.end(), int(), "pop_back()", 3, -1));
    }


    {
        // void sort()
        // void sort(Compare compare)

        const int kSize = 10;
        IntNode nodes[kSize];

        intrusive_list <IntNode> listEmpty;
        listEmpty.sort();
        REQUIRE(check_sequence_eq(listEmpty.begin(), listEmpty.end(), int(), "list::sort", -1));

        intrusive_list <IntNode> list1;
        ListInit(list1, nodes) += 1;
        list1.sort();
        REQUIRE(check_sequence_eq(list1.begin(), list1.end(), int(), "list::sort", 1, -1));
        list1.clear();

        intrusive_list <IntNode> list4;
        ListInit(list4, nodes) += 1, 9, 2, 3;
        list4.sort();
        REQUIRE(check_sequence_eq(list4.begin(), list4.end(), int(), "list::sort", 1, 2, 3, 9, -1));
        list4.clear();

        intrusive_list <IntNode> listA;
        ListInit(listA, nodes) += 1, 9, 2, 3, 5, 7, 4, 6, 8, 0;
        listA.sort();
        REQUIRE(check_sequence_eq(listA.begin(), listA.end(), int(), "list::sort", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1));
        listA.clear();

        intrusive_list <IntNode> listB;
        ListInit(listB, nodes) += 1, 9, 2, 3, 5, 7, 4, 6, 8, 0;
        listB.sort(std::less<int>());
        REQUIRE(check_sequence_eq(listB.begin(), listB.end(), int(), "list::sort", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1));
        listB.clear();
    }


    {
        // void merge(this_type& x);
        // void merge(this_type& x, Compare compare);

        const int kSize = 8;
        IntNode nodesA[kSize];
        IntNode nodesB[kSize];

        intrusive_list <IntNode> listA;
        ListInit(listA, nodesA) += 1, 2, 3, 4, 4, 5, 9, 9;

        intrusive_list <IntNode> listB;
        ListInit(listB, nodesB) += 1, 2, 3, 4, 4, 5, 9, 9;

        listA.merge(listB);
        REQUIRE(check_sequence_eq(listA.begin(), listA.end(), int(), "list::merge", 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 9,
                              9, 9, 9, -1));
        REQUIRE(check_sequence_eq(listB.begin(), listB.end(), int(), "list::merge", -1));
    }


    {
        // void unique();
        // void unique(BinaryPredicate);

        const int kSize = 8;
        IntNode nodesA[kSize];
        IntNode nodesB[kSize];

        intrusive_list <IntNode> listA;
        ListInit(listA, nodesA) += 1, 2, 3, 4, 4, 5, 9, 9;
        listA.unique();
        REQUIRE(check_sequence_eq(listA.begin(), listA.end(), int(), "list::unique", 1, 2, 3, 4, 5, 9, -1));

        intrusive_list <IntNode> listB;
        ListInit(listB, nodesB) += 1, 2, 3, 4, 4, 5, 9, 9;
        listB.unique(std::equal_to<int>());
        REQUIRE(check_sequence_eq(listA.begin(), listA.end(), int(), "list::unique", 1, 2, 3, 4, 5, 9, -1));
    }


}












