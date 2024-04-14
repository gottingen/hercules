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

#ifndef COLLIE_STRINGS_INLINED_STRING_H_
#define COLLIE_STRINGS_INLINED_STRING_H_

#include <string_view>
#include <collie/container/inlined_vector.h>

namespace collie {
    /// InlinedString - A InlinedString is just a InlinedVector with methods and accessors
    /// that make it work better as a string (e.g. operator+ etc).
    template<unsigned InternalLen>
    class InlinedString : public InlinedVector<char, InternalLen> {
    public:
        /// Default ctor - Initialize to empty.
        InlinedString() = default;

        /// Initialize from a std::string_view.
        InlinedString(std::string_view S) : InlinedVector<char, InternalLen>(S.begin(), S.end()) {}

        /// Initialize by concatenating a list of std::string_views.
        InlinedString(std::initializer_list<std::string_view> Refs)
                : InlinedVector<char, InternalLen>() {
            this->append(Refs);
        }

        /// Initialize with a range.
        template<typename ItTy>
        InlinedString(ItTy S, ItTy E) : InlinedVector<char, InternalLen>(S, E) {}

        /// @}
        /// @name String Assignment
        /// @{

        using InlinedVector<char, InternalLen>::assign;

        /// Assign from a std::string_view.
        void assign(std::string_view RHS) {
            InlinedVectorImpl<char>::assign(RHS.begin(), RHS.end());
        }

        /// Assign from a list of std::string_views.
        void assign(std::initializer_list<std::string_view> Refs) {
            this->clear();
            append(Refs);
        }

        /// @}
        /// @name String Concatenation
        /// @{

        using InlinedVector<char, InternalLen>::append;

        /// Append from a std::string_view.
        void append(std::string_view RHS) {
            InlinedVectorImpl<char>::append(RHS.begin(), RHS.end());
        }

        /// Append from a list of std::string_views.
        void append(std::initializer_list<std::string_view> Refs) {
            size_t CurrentSize = this->size();
            size_t SizeNeeded = CurrentSize;
            for (const std::string_view &Ref: Refs)
                SizeNeeded += Ref.size();
            this->resize_for_overwrite(SizeNeeded);
            for (const std::string_view &Ref: Refs) {
                std::copy(Ref.begin(), Ref.end(), this->begin() + CurrentSize);
                CurrentSize += Ref.size();
            }
            assert(CurrentSize == this->size());
        }

        /// @}
        /// @name String Comparison
        /// @{

        /// Check for string equality.  This is more efficient than compare() when
        /// the relative ordering of inequal strings isn't needed.
        bool equals(std::string_view RHS) const {
            return str().equals(RHS);
        }

        /// Check for string equality, ignoring case.
        bool equals_insensitive(std::string_view RHS) const {
            return str().equals_insensitive(RHS);
        }

        /// compare - Compare two strings; the result is negative, zero, or positive
        /// if this string is lexicographically less than, equal to, or greater than
        /// the \p RHS.
        int compare(std::string_view RHS) const {
            return str().compare(RHS);
        }

        /// compare_numeric - Compare two strings, treating sequences of digits as
        /// numbers.
        int compare_numeric(std::string_view RHS) const {
            return str().compare_numeric(RHS);
        }

        /// @}
        /// @name String Predicates
        /// @{

        /// startswith - Check if this string starts with the given \p Prefix.
        bool startswith(std::string_view Prefix) const {
            return str().start_with(Prefix);
        }

        /// endswith - Check if this string ends with the given \p Suffix.
        bool endswith(std::string_view Suffix) const {
            return str().endswith(Suffix);
        }

        /// @}
        /// @name String Searching
        /// @{

        /// find - Search for the first character \p C in the string.
        ///
        /// \return - The index of the first occurrence of \p C, or npos if not
        /// found.
        size_t find(char C, size_t From = 0) const {
            return str().find(C, From);
        }

        /// Search for the first string \p Str in the string.
        ///
        /// \returns The index of the first occurrence of \p Str, or npos if not
        /// found.
        size_t find(std::string_view Str, size_t From = 0) const {
            return str().find(Str, From);
        }

        /// Search for the last character \p C in the string.
        ///
        /// \returns The index of the last occurrence of \p C, or npos if not
        /// found.
        size_t rfind(char C, size_t From = std::string_view::npos) const {
            return str().rfind(C, From);
        }

        /// Search for the last string \p Str in the string.
        ///
        /// \returns The index of the last occurrence of \p Str, or npos if not
        /// found.
        size_t rfind(std::string_view Str) const {
            return str().rfind(Str);
        }

        /// Find the first character in the string that is \p C, or npos if not
        /// found. Same as find.
        size_t find_first_of(char C, size_t From = 0) const {
            return str().find_first_of(C, From);
        }

        /// Find the first character in the string that is in \p Chars, or npos if
        /// not found.
        ///
        /// Complexity: O(size() + Chars.size())
        size_t find_first_of(std::string_view Chars, size_t From = 0) const {
            return str().find_first_of(Chars, From);
        }

        /// Find the first character in the string that is not \p C or npos if not
        /// found.
        size_t find_first_not_of(char C, size_t From = 0) const {
            return str().find_first_not_of(C, From);
        }

        /// Find the first character in the string that is not in the string
        /// \p Chars, or npos if not found.
        ///
        /// Complexity: O(size() + Chars.size())
        size_t find_first_not_of(std::string_view Chars, size_t From = 0) const {
            return str().find_first_not_of(Chars, From);
        }

        /// Find the last character in the string that is \p C, or npos if not
        /// found.
        size_t find_last_of(char C, size_t From = std::string_view::npos) const {
            return str().find_last_of(C, From);
        }

        /// Find the last character in the string that is in \p C, or npos if not
        /// found.
        ///
        /// Complexity: O(size() + Chars.size())
        size_t find_last_of(
                std::string_view Chars, size_t From = std::string_view::npos) const {
            return str().find_last_of(Chars, From);
        }

        /// @}
        /// @name Substring Operations
        /// @{

        /// Return a reference to the substring from [Start, Start + N).
        ///
        /// \param Start The index of the starting character in the substring; if
        /// the index is npos or greater than the length of the string then the
        /// empty substring will be returned.
        ///
        /// \param N The number of characters to included in the substring. If \p N
        /// exceeds the number of characters remaining in the string, the string
        /// suffix (starting with \p Start) will be returned.
        std::string_view substr(size_t Start, size_t N = std::string_view::npos) const {
            if(Start >= this->size())
                return std::string_view();
            return str().substr(Start, N);
        }

        /// Return a reference to the substring from [Start, End).
        ///
        /// \param Start The index of the starting character in the substring; if
        /// the index is npos or greater than the length of the string then the
        /// empty substring will be returned.
        ///
        /// \param End The index following the last character to include in the
        /// substring. If this is npos, or less than \p Start, or exceeds the
        /// number of characters remaining in the string, the string suffix
        /// (starting with \p Start) will be returned.
        std::string_view slice(size_t Start, size_t End) const {
            if(Start >= this->size() || Start >= End)
                return std::string_view();

            return str().substr(Start, End - Start);
        }

        // Extra methods.

        /// Explicit conversion to std::string_view.
        std::string_view str() const { return std::string_view(this->data(), this->size()); }

        // TODO: Make this const, if it's safe...
        const char *c_str() {
            this->push_back(0);
            this->pop_back();
            return this->data();
        }

        /// Implicit conversion to std::string_view.
        operator std::string_view() const { return str(); }

        explicit operator std::string() const {
            return std::string(this->data(), this->size());
        }

        // Extra operators.
        InlinedString &operator=(std::string_view RHS) {
            this->assign(RHS);
            return *this;
        }

        InlinedString &operator+=(std::string_view RHS) {
            this->append(RHS.begin(), RHS.end());
            return *this;
        }

        InlinedString &operator+=(char C) {
            this->push_back(C);
            return *this;
        }
    };
}  // namespace collie
#endif  // COLLIE_STRINGS_INLINED_STRING_H_
