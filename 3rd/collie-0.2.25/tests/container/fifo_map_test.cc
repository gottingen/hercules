/*!
@file
@copyright The code is licensed under the MIT License
           <http://opensource.org/licenses/MIT>,
           Copyright (c) 2015 Niels Lohmann.
@author Niels Lohmann <http://nlohmann.me>
@see https://github.com/nlohmann/fifo_map
*/

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <collie/testing/doctest.h>

// allow accessing private members
#define private public

#include <collie/container/fifo_map.h>
using collie::fifo_map;

#include <string>
#include <type_traits>

/// helper function to check order of keys
const auto collect_keys = [](const fifo_map<std::string, int>& m)
{
    std::string result;
    for (auto x : m)
    {
        result += x.first;
    }
    return result;
};

TEST_CASE("types")
{
    using map = std::map<std::string, int>;
    using fmap = fifo_map<std::string, int>;

    CHECK((std::is_same<map::key_type, fmap::key_type>::value));
    CHECK((std::is_same<map::mapped_type, fmap::mapped_type>::value));
    CHECK((std::is_same<map::value_type, fmap::value_type>::value));
    CHECK((std::is_same<map::size_type, fmap::size_type>::value));
    CHECK((std::is_same<map::difference_type, fmap::difference_type>::value));
    CHECK((std::is_same<fmap::key_compare, collie::fifo_map_compare<fmap::key_type>>::value));
    CHECK((std::is_same<map::allocator_type, fmap::allocator_type>::value));
    CHECK((std::is_same<map::reference, fmap::reference>::value));
    CHECK((std::is_same<map::const_reference, fmap::const_reference>::value));
    CHECK((std::is_same<map::pointer, fmap::pointer>::value));
    CHECK((std::is_same<map::const_pointer, fmap::const_pointer>::value));
}

TEST_CASE("element access")
{
    fifo_map<std::string, int> m = {{"C", 1}, {"A", 2}, {"B", 3}};
    const fifo_map<std::string, int> mc = m;

    SUBCASE("at")
    {
        CHECK(m.at("C") == 1);
        CHECK(m.at("A") == 2);
        CHECK(m.at("B") == 3);

        CHECK_THROWS_AS(m.at("Z"), std::out_of_range&);

        CHECK(mc.at("C") == 1);
        CHECK(mc.at("A") == 2);
        CHECK(mc.at("B") == 3);

        CHECK_THROWS_AS(mc.at("Z"), std::out_of_range&);
    }

    SUBCASE("operator[] (rvalue)")
    {
        CHECK(m["C"] == 1);
        CHECK(m["A"] == 2);
        CHECK(m["B"] == 3);

        CHECK_NOTHROW(m["Z"]);
        CHECK(m["Z"] == 0);
    }

    SUBCASE("operator[] (lvalue)")
    {
        const std::string s_C = "C";
        const std::string s_A = "A";
        const std::string s_B = "B";
        const std::string s_Z = "Z";
        CHECK(m[s_C] == 1);
        CHECK(m[s_A] == 2);
        CHECK(m[s_B] == 3);

        CHECK_NOTHROW(m[s_Z]);
        CHECK(m[s_Z] == 0);
    }
}

TEST_CASE("iterators")
{
    fifo_map<std::string, int> m = {{"C", 1}, {"A", 2}, {"B", 3}};
    const fifo_map<std::string, int> mc = m;

    SUBCASE("begin/end with nonconst object")
    {
        fifo_map<std::string, int>::iterator it = m.begin();
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it == m.end());
    }

    SUBCASE("begin/end with const object")
    {
        fifo_map<std::string, int>::const_iterator it = mc.begin();
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it == mc.end());
    }

    SUBCASE("cbegin/cend with nonconst object")
    {
        fifo_map<std::string, int>::const_iterator it = m.cbegin();
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it == m.cend());
    }

    SUBCASE("cbegin/cend with const object")
    {
        fifo_map<std::string, int>::const_iterator it = mc.cbegin();
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it == mc.cend());
    }

    SUBCASE("begin/end with nonconst object")
    {
        fifo_map<std::string, int>::iterator it = m.begin();
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it == m.end());
    }

    SUBCASE("rbegin/rend with nonconst object")
    {
        fifo_map<std::string, int>::reverse_iterator it = m.rbegin();
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it == m.rend());
    }

    SUBCASE("rbegin/rend with const object")
    {
        fifo_map<std::string, int>::const_reverse_iterator it = mc.rbegin();
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it == mc.rend());
    }

    SUBCASE("crbegin/crend with nonconst object")
    {
        fifo_map<std::string, int>::const_reverse_iterator it = m.crbegin();
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it == m.crend());
    }

    SUBCASE("crbegin/crend with const object")
    {
        fifo_map<std::string, int>::const_reverse_iterator it = mc.crbegin();
        CHECK(it->first == "B");
        CHECK(it->second == 3);
        ++it;
        CHECK(it->first == "A");
        CHECK(it->second == 2);
        ++it;
        CHECK(it->first == "C");
        CHECK(it->second == 1);
        ++it;
        CHECK(it == mc.crend());
    }
}

TEST_CASE("capacity")
{
    collie::fifo_map<std::string, int> m_empty;
    collie::fifo_map<std::string, int> m_filled = {{"A", 1}, {"B", 2}};

    SUBCASE("empty")
    {
        CHECK(m_empty.empty());
        CHECK(! m_filled.empty());
    }

    SUBCASE("size")
    {
        CHECK(m_empty.size() == 0);
        CHECK(m_filled.size() == 2);
    }

    SUBCASE("max_size")
    {
        CHECK(m_empty.max_size() >= m_empty.size());
        CHECK(m_filled.max_size() >= m_filled.size());

        CHECK(m_empty.max_size() == m_empty.m_map.max_size());
        CHECK(m_filled.max_size() == m_filled.m_map.max_size());
    }
}

TEST_CASE("modifiers")
{
    collie::fifo_map<std::string, int> m_empty;
    collie::fifo_map<std::string, int> m_filled = {{"X", 1}, {"C", 2}};

    SUBCASE("clear")
    {
        CHECK(!m_filled.empty());
        CHECK(m_empty.empty());
        m_filled.clear();
        m_empty.clear();
        CHECK(m_filled.empty());
        CHECK(m_empty.empty());

        CHECK(m_empty.m_map.empty());
        CHECK(m_empty.m_keys.empty());
        CHECK(m_filled.m_map.empty());
        CHECK(m_filled.m_keys.empty());
    }

    SUBCASE("insert")
    {
        SUBCASE("insert value_type (lvalue)")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            // insert new value
            const collie::fifo_map<std::string, int>::value_type v1 = {"A", 3};
            auto res1 = m_filled.insert(v1);
            CHECK(m_filled["A"] == 3);
            CHECK(res1.second == true);
            CHECK(res1.first->first == "A");
            CHECK(res1.first->second == 3);

            // check that key and value were inserted
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");

            // insert already present value
            const collie::fifo_map<std::string, int>::value_type v2 = {"A", 4};
            auto res2 = m_filled.insert(v2);
            CHECK(m_filled["A"] == 3);
            CHECK(res2.second == false);
            CHECK(res2.first->first == "A");
            CHECK(res2.first->second == 3);

            // check that map remained unchanged
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");
        }

        SUBCASE("insert value_type (rvalue)")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            // insert new value
            auto res1 = m_filled.insert({"A", 3});
            CHECK(m_filled["A"] == 3);
            CHECK(res1.second == true);
            CHECK(res1.first->first == "A");
            CHECK(res1.first->second == 3);

            // check that key and value were inserted
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");

            // insert already present value
            auto res2 = m_filled.insert({"A", 4});
            CHECK(m_filled["A"] == 3);
            CHECK(res2.second == false);
            CHECK(res2.first->first == "A");
            CHECK(res2.first->second == 3);

            // check that map remained unchanged
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");
        }

        SUBCASE("insert value_type (lvalue) with hint")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            // insert new value
            const collie::fifo_map<std::string, int>::value_type v1 = {"A", 3};
            auto res1 = m_filled.insert(m_filled.end(), v1);
            CHECK(m_filled["A"] == 3);
            CHECK(res1->first == "A");
            CHECK(res1->second == 3);

            // check that key and value were inserted
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");

            // insert already present value
            const collie::fifo_map<std::string, int>::value_type v2 = {"A", 4};
            auto res2 = m_filled.insert(m_filled.end(), v2);
            CHECK(m_filled["A"] == 3);
            CHECK(res2->first == "A");
            CHECK(res2->second == 3);

            // check that map remained unchanged
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");
        }

        SUBCASE("insert value_type (rvalue) with hint")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            // insert new value
            auto res1 = m_filled.insert(m_filled.end(), {"A", 3});
            CHECK(m_filled["A"] == 3);
            CHECK(res1->first == "A");
            CHECK(res1->second == 3);

            // check that key and value were inserted
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");

            // insert already present value
            auto res2 = m_filled.insert(m_filled.end(), {"A", 4});
            CHECK(m_filled["A"] == 3);
            CHECK(res2->first == "A");
            CHECK(res2->second == 3);

            // check that map remained unchanged
            CHECK(m_filled.size() == 3);
            CHECK(m_filled.m_keys.size() == 3);
            CHECK(collect_keys(m_filled) == "XCA");
        }

        SUBCASE("insert initializer list")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            // insert initializer list
            m_filled.insert({{"A", 3}, {"Z", 4}, {"B", 5}});
            CHECK(m_filled["A"] == 3);
            CHECK(m_filled["Z"] == 4);
            CHECK(m_filled["B"] == 5);

            // check that keys and values were inserted
            CHECK(m_filled.size() == 5);
            CHECK(m_filled.m_keys.size() == 5);
            CHECK(collect_keys(m_filled) == "XCAZB");

            // insert empty initializer list
            m_filled.insert({});

            // check that map remained unchanged
            CHECK(m_filled.size() == 5);
            CHECK(m_filled.m_keys.size() == 5);
            CHECK(collect_keys(m_filled) == "XCAZB");
        }

        SUBCASE("insert range")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            // insert range
            std::map<std::string, int> values = {{"A", 3}, {"Z", 4}, {"B", 5}};
            m_filled.insert(values.begin(), values.end());
            CHECK(m_filled["A"] == 3);
            CHECK(m_filled["Z"] == 4);
            CHECK(m_filled["B"] == 5);

            // check that keys and values were inserted
            CHECK(m_filled.size() == 5);
            CHECK(m_filled.m_keys.size() == 5);
            CHECK(collect_keys(m_filled) == "XCABZ");
        }
    }

    SUBCASE("erase")
    {
        SUBCASE("remove element")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            auto it = m_filled.erase(++m_filled.begin());
            CHECK(it == m_filled.end());

            // check that key and value was removed
            CHECK(m_filled.size() == 1);
            CHECK(m_filled.m_keys.size() == 1);
            CHECK(collect_keys(m_filled) == "X");
        }

        SUBCASE("remove element range")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            auto it = m_filled.erase(m_filled.begin(), m_filled.end());
            CHECK(it == m_filled.end());

            // check that key and value was removed
            CHECK(m_filled.size() == 0);
            CHECK(m_filled.m_keys.size() == 0);
            CHECK(collect_keys(m_filled) == "");
        }

        SUBCASE("remove element by key")
        {
            // check initial state
            CHECK(m_filled.size() == 2);
            CHECK(m_filled.m_keys.size() == 2);
            CHECK(collect_keys(m_filled) == "XC");

            auto count = m_filled.erase("X");
            CHECK(count == 1);

            // check that key and value was removed
            CHECK(m_filled.size() == 1);
            CHECK(m_filled.m_keys.size() == 1);
            CHECK(collect_keys(m_filled) == "C");
        }
    }

    SUBCASE("swap")
    {
        // precondition
        CHECK(m_empty.empty());
        CHECK(!m_filled.empty());

        // swap
        m_filled.swap(m_empty);

        // postcondition
        CHECK(!m_empty.empty());
        CHECK(m_filled.empty());

        // swap
        m_filled.swap(m_empty);

        // back to precondition
        CHECK(m_empty.empty());
        CHECK(!m_filled.empty());
    }

    SUBCASE("emplace")
    {
        // check initial state
        CHECK(m_filled.size() == 2);
        CHECK(m_filled.m_keys.size() == 2);
        CHECK(collect_keys(m_filled) == "XC");

        // insert new value
        auto res1 = m_filled.emplace("A", 3);
        CHECK(m_filled["A"] == 3);
        CHECK(res1.second == true);
        CHECK(res1.first->first == "A");
        CHECK(res1.first->second == 3);

        // check that key and value were inserted
        CHECK(m_filled.size() == 3);
        CHECK(m_filled.m_keys.size() == 3);
        CHECK(collect_keys(m_filled) == "XCA");

        // insert already present value
        auto res2 = m_filled.emplace("A", 4);
        CHECK(m_filled["A"] == 3);
        CHECK(res2.second == false);
        CHECK(res2.first->first == "A");
        CHECK(res2.first->second == 3);

        // check that map remained unchanged
        CHECK(m_filled.size() == 3);
        CHECK(m_filled.m_keys.size() == 3);
        CHECK(collect_keys(m_filled) == "XCA");
    }

    SUBCASE("emplace_hint")
    {
        // check initial state
        CHECK(m_filled.size() == 2);
        CHECK(m_filled.m_keys.size() == 2);
        CHECK(collect_keys(m_filled) == "XC");

        // insert new value
        auto res1 = m_filled.emplace_hint(m_filled.end(), "A", 3);
        CHECK(m_filled["A"] == 3);
        CHECK(res1->first == "A");
        CHECK(res1->second == 3);

        // check that key and value were inserted
        CHECK(m_filled.size() == 3);
        CHECK(m_filled.m_keys.size() == 3);
        CHECK(collect_keys(m_filled) == "XCA");

        // insert already present value
        auto res2 = m_filled.emplace_hint(m_filled.end(), "A", 4);
        CHECK(m_filled["A"] == 3);
        CHECK(res2->first == "A");
        CHECK(res2->second == 3);

        // check that map remained unchanged
        CHECK(m_filled.size() == 3);
        CHECK(m_filled.m_keys.size() == 3);
        CHECK(collect_keys(m_filled) == "XCA");
    }
}

TEST_CASE("lookup")
{
    collie::fifo_map<std::string, int> m = {{"A", 1}, {"B", 2}};
    const collie::fifo_map<std::string, int> m_c = m;

    SUBCASE("count")
    {
        CHECK(m.count("A") == 1);
        CHECK(m.count("B") == 1);
        CHECK(m.count("C") == 0);
    }

    SUBCASE("find")
    {
        SUBCASE("iterators")
        {
            auto it_A = m.find("A");
            CHECK(it_A == m.begin());
            CHECK(it_A->first == "A");
            CHECK(it_A->second == 1);

            auto it_B = m.find("B");
            CHECK(it_B == ++m.begin());
            CHECK(it_B->first == "B");
            CHECK(it_B->second == 2);

            auto it_C = m.find("C");
            CHECK(it_C == m.end());
        }

        SUBCASE("const_iterators")
        {
            auto it_A = m_c.find("A");
            CHECK(it_A == m_c.cbegin());
            CHECK(it_A->first == "A");
            CHECK(it_A->second == 1);

            auto it_B = m_c.find("B");
            CHECK(it_B == ++m_c.cbegin());
            CHECK(it_B->first == "B");
            CHECK(it_B->second == 2);

            auto it_C = m_c.find("C");
            CHECK(it_C == m_c.cend());
        }
    }

    SUBCASE("equal_range")
    {
        SUBCASE("iterators")
        {
            {
                std::pair<fifo_map<std::string, int>::iterator, fifo_map<std::string, int>::iterator> range =
                    m.equal_range("A");
                CHECK(range.first == m.begin());
                CHECK(range.second == ++m.begin());
            }
            {
                std::pair<fifo_map<std::string, int>::iterator, fifo_map<std::string, int>::iterator> range =
                    m.equal_range("C");
                CHECK(range.first == m.end());
                CHECK(range.second == m.end());
            }
        }

        SUBCASE("const_iterators")
        {
            {
                std::pair<fifo_map<std::string, int>::const_iterator, fifo_map<std::string, int>::const_iterator>
                range = m_c.equal_range("A");
                CHECK(range.first == m_c.cbegin());
                CHECK(range.second == ++m_c.cbegin());
            }
            {
                std::pair<fifo_map<std::string, int>::const_iterator, fifo_map<std::string, int>::const_iterator>
                range = m_c.equal_range("C");
                CHECK(range.first == m_c.cend());
                CHECK(range.second == m_c.cend());
            }
        }
    }

    SUBCASE("lower_bound")
    {
        SUBCASE("iterators")
        {
            {
                fifo_map<std::string, int>::iterator it = m.lower_bound("A");
                CHECK(it == m.begin());
            }
            {
                fifo_map<std::string, int>::iterator it = m.lower_bound("C");
                CHECK(it == m.end());
            }
        }

        SUBCASE("const_iterators")
        {
            {
                fifo_map<std::string, int>::const_iterator it = m_c.lower_bound("A");
                CHECK(it == m_c.cbegin());
            }
            {
                fifo_map<std::string, int>::const_iterator it = m_c.lower_bound("C");
                CHECK(it == m_c.cend());
            }
        }
    }

    SUBCASE("upper_bound")
    {
        SUBCASE("iterators")
        {
            {
                fifo_map<std::string, int>::iterator it = m.upper_bound("A");
                CHECK(it == ++m.begin());
            }
            {
                fifo_map<std::string, int>::iterator it = m.upper_bound("C");
                CHECK(it == m.end());
            }
        }

        SUBCASE("const_iterators")
        {
            {
                fifo_map<std::string, int>::const_iterator it = m_c.upper_bound("A");
                CHECK(it == ++m_c.cbegin());
            }
            {
                fifo_map<std::string, int>::const_iterator it = m_c.upper_bound("C");
                CHECK(it == m_c.cend());
            }
        }
    }
}

TEST_CASE("observers")
{
    collie::fifo_map<std::string, int> m = {{"A", 1}, {"B", 2}};

    SUBCASE("key_comp")
    {
        auto comp = m.key_comp();
        CHECK(comp("A", "B"));
        CHECK(comp("A", "C"));
        CHECK(! comp("A", "A"));
        CHECK(! comp("B", "A"));
        CHECK(! comp("C", "C"));
    }
}

TEST_CASE("non-member functions")
{
    collie::fifo_map<std::string, int> m1 = {{"A", 1}, {"B", 2}};
    collie::fifo_map<std::string, int> m2 = {{"B", 2}, {"A", 1}};
    collie::fifo_map<std::string, int> m3 = {{"A", 3}, {"B", 4}};

    SUBCASE("comparison")
    {
        CHECK(m1 == m1);

        CHECK(m1 != m2);
        CHECK(m1 != m3);

        CHECK(m1 < m3);
        CHECK(m1 <= m3);

        CHECK(m3 > m1);
        CHECK(m3 >= m1);
    }

    SUBCASE("std::swap")
    {
        collie::fifo_map<std::string, int> m_empty;
        collie::fifo_map<std::string, int> m_filled = {{"X", 1}, {"C", 2}};

        // precondition
        CHECK(m_empty.empty());
        CHECK(!m_filled.empty());

        // swap
        std::swap(m_filled, m_empty);

        // postcondition
        CHECK(!m_empty.empty());
        CHECK(m_filled.empty());

        // swap
        std::swap(m_filled, m_empty);

        // back to precondition
        CHECK(m_empty.empty());
        CHECK(!m_filled.empty());
    }
}

TEST_CASE("regression tests")
{
    SUBCASE("issue 2: find() adding a key")
    {
        collie::fifo_map<std::string, uint8_t> pe;
        for (int i = 0; i < 1000; i++)
        {

            std::string key = "custom_";
            key += std::to_string(i);

            //if(0 == pe.count(key)){
            CHECK(pe.find(key) == pe.end());
            pe[key] = (i + 128) & 0xFF;
        }
    }
}

TEST_CASE("constructors")
{
    SUBCASE("range iterators constructor")
    {
        std::map<std::string, int> inputs = {{"A", 1}, {"B", 2}};
        collie::fifo_map<std::string, int> m(inputs.begin(), inputs.end());
        CHECK(m.size() == 2);
        CHECK(m["A"] == 1);
        CHECK(m["B"] == 2);

    }
}