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

#ifndef TURBO_CONTAINER_INTRUSIVE_LIST_H_
#define TURBO_CONTAINER_INTRUSIVE_LIST_H_

#include <iterator>
#include <memory>
#include "turbo/meta/type_traits.h"
#include "turbo/platform/port.h"


namespace turbo {

    template<class T>
    class intrusive_list;

    struct intrusive_list_node {
        intrusive_list_node *next{this};
        intrusive_list_node *prev{this};

        void insert_before(intrusive_list_node* e) {
            this->next = e;
            this->prev = e->prev;
            e->prev->next = this;
            e->prev = this;
        }

        void insert_after(intrusive_list_node* e) {
            this->next = e->next;
            this->prev = e;
            e->next->prev = this;
            e->next = this;
        }

        void remove_from_list() {
            this->prev->next = this->next;
            this->next->prev = this->prev;
            this->next = this;
            this->prev = this;
        }

        void insert_before_as_list(intrusive_list_node* e) {
            intrusive_list_node* pprev = this->prev;
            pprev->next = e;
            this->prev = e->prev;
            e->prev->next = this;
            e->prev = pprev;
        }

        void insert_after_as_list(intrusive_list_node* e) {
            intrusive_list_node* pprev = this->prev;
            pprev->next = e->next;
            this->prev = e;
            e->next->prev = pprev;
            e->next = this;
        }
    };

    /// intrusive_list_iterator
    ///
    template<typename T, typename Pointer, typename Reference>
    class intrusive_list_iterator {
    public:
        typedef intrusive_list_iterator<T, Pointer, Reference> this_type;
        typedef intrusive_list_iterator<T, T *, T &> iterator;
        typedef intrusive_list_iterator<T, const T *, const T &> const_iterator;
        typedef T value_type;
        typedef T node_type;
        typedef intrusive_list_node base_node_type;
        typedef ptrdiff_t difference_type;
        typedef Pointer pointer;
        typedef Reference reference;
        typedef std::bidirectional_iterator_tag iterator_category;
    public:
        pointer mpNode;

    public:
        intrusive_list_iterator();

        // Note: you can also construct an iterator from T* via this, since T should inherit from
        // intrusive_list_node.
        explicit intrusive_list_iterator(const base_node_type *pNode);

        // Note: this isn't always a copy constructor, iterator is not always equal to this_type
        intrusive_list_iterator(const iterator &x);

        // Note: this isn't always a copy assignment operator, iterator is not always equal to this_type
        intrusive_list_iterator &operator=(const iterator &x);

        // Calling these on the end() of a list invokes undefined behavior.
        reference operator*() const;

        pointer operator->() const;

        // Returns a pointer to the fully typed node (the same as operator->) this is useful when
        // iterating a list to destroy all the nodes, calling this on the end() of a list results in
        // undefined behavior.
        pointer node_ptr() const;

        intrusive_list_iterator &operator++();

        intrusive_list_iterator &operator--();

        intrusive_list_iterator operator++(int);

        intrusive_list_iterator operator--(int);

        // The C++ defect report #179 requires that we support comparisons between const and non-const iterators.
        // Thus we provide additional template paremeters here to support this. The defect report does not
        // require us to support comparisons between reverse_iterators and const_reverse_iterators.
        template<class PointerB, class ReferenceB>
        bool operator==(const intrusive_list_iterator<T, PointerB, ReferenceB> &other) const {
            return mpNode == other.mpNode;
        }

        template<typename PointerB, typename ReferenceB>
        inline bool operator!=(const intrusive_list_iterator<T, PointerB, ReferenceB> &other) const {
            return mpNode != other.mpNode;
        }

        // We provide a version of operator!= for the case where the iterators are of the
        // same type. This helps prevent ambiguity errors in the presence of rel_ops.
        inline bool operator!=(const intrusive_list_iterator other) const { return mpNode != other.mpNode; }

    private:

        pointer toInternalNodeType(base_node_type *node) { return static_cast<pointer>(node); }

        // for the "copy" constructor, which uses non-const iterator even in the
        // const_iterator case.  Also, some of the internal member functions in
        // intrusive_list<T> want to use mpNode.
        friend const_iterator;
        friend intrusive_list<T>;

        // for the comparison operators.
        template<class U, class Pointer1, class Reference1>
        friend
        class intrusive_list_iterator;
    }; // class intrusive_list_iterator



    /// intrusive_list_base
    ///
    class intrusive_list_base {
    public:
        typedef size_t size_type;     // See config.h for the definition of this, which defaults to size_t.
        typedef ptrdiff_t difference_type;

    protected:
        intrusive_list_node mAnchor;          ///< Sentinel node (end). All data nodes are linked in a ring from this node.

    public:
        intrusive_list_base();

        ~intrusive_list_base();

        [[nodiscard]] bool empty() const noexcept;

        ///< Returns the number of elements in the list; O(n).
        [[nodiscard]] size_t size() const noexcept;
        ///< Clears the list; O(1). No deallocation occurs.
        void clear() noexcept;
        ///< Removes an element from the front of the list; O(1). The element must exist, but is not deallocated.
        void pop_front();

        ///< Removes an element from the back of the list; O(1). The element must exist, but is not deallocated.
        void pop_back();
        ///< Reverses a list so that front and back are swapped; O(n).
        void reverse() noexcept;

    }; // class intrusive_list_base


    /**
     * @ingroup turbo_container_sequence
     * @brief intrusive_list is a doubly-linked list that is intrusive in that the elements themselves
     *        contain the next/prev pointers. This means that the elements themselves must inherit from
     *        intrusive_list_node. This is useful for when you want to store a list of objects that are
     *        already part of another list, or when you want to store a list of objects that are part of
     *        a larger object.
     *        Note that intrusive_list do not manage the lifetime of the elements, so you must ensure
     *        that the elements are not destroyed while they are still in the list.
     *        Example usage:
     *        @code
     *        struct IntNode : public turbo::intrusive_list_node {
     *        int mX;
     *        IntNode(int x) : mX(x) { }
     *        };
     *        IntNode nodeA(0);
     *        IntNode nodeB(1);
     *        intrusive_list<IntNode> intList;
     *        intList.push_back(nodeA);
     *        intList.push_back(nodeB);
     *        intList.remove(nodeA);
     *        @endcode
     * @tparam T
     */
    template<typename T = intrusive_list_node>
    class intrusive_list : public intrusive_list_base {
    public:
        typedef intrusive_list<T> this_type;
        typedef intrusive_list_base base_type;
        typedef T node_type;
        typedef T value_type;
        typedef typename base_type::size_type size_type;
        typedef typename base_type::difference_type difference_type;
        typedef T &reference;
        typedef const T &const_reference;
        typedef T *pointer;
        typedef const T *const_pointer;
        typedef intrusive_list_iterator<T, T *, T &> iterator;
        typedef intrusive_list_iterator<T, const T *, const T &> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    public:
        /**
         * @brief constructs an empty list
         */
        intrusive_list();
        /**
         * @brief constructs a empty list ignoring the argument
         *        for that the elements can not be copied and the linked
         *        can not be copied.
         * @param x
         */
        intrusive_list(const this_type &x);

        /**
         * @brief clears the list ignoring the argument
         * @param x
         * @return
         */
        this_type &operator=(const this_type &x);

        /**
         * @brief swaps the contents of two intrusive lists; O(1).
         * @param x
         */
        void swap(this_type &);

        /**
         * @brief Returns an iterator pointing to the first element in the list.
         * @return
         */
        iterator begin() noexcept;

        /**
         * @brief Returns a const_iterator pointing to the first element in the list.
         * @return
         */
        const_iterator begin() const noexcept;

        /**
         * @brief Returns a const_iterator pointing to the first element in the list.
         * @return
         */
        const_iterator cbegin() const noexcept;

        /**
         * @brief Returns an iterator pointing one-after the last element in the list.
         * @return
         */
        iterator end() noexcept;
        /**
         * @brief Returns a const_iterator pointing one-after the last element in the list.
         * @return
         */
        const_iterator end() const noexcept;

        /**
         * @brief Returns a const_iterator pointing one-after the last element in the list.
         * @return
         */
        const_iterator cend() const noexcept;

        /**
         * @brief Returns a reverse_iterator pointing at the end of the list (start of the reverse sequence).
         * @return
         */
        reverse_iterator rbegin() noexcept;

        /**
         * @brief Returns a const_reverse_iterator pointing at the end of the list (start of the reverse sequence).
         * @return
         */
        const_reverse_iterator rbegin() const noexcept;

        /**
         * @brief Returns a const_reverse_iterator pointing at the end of the list (start of the reverse sequence).
         * @return
         */
        const_reverse_iterator crbegin() const noexcept;

        /**
         * @brief Returns a reverse_iterator pointing at the start of the list (end of the reverse sequence).
         * @return
         */
        reverse_iterator rend() noexcept;

        /**
         * @brief Returns a const_reverse_iterator pointing at the start of the list (end of the reverse sequence).
         * @return
         */
        const_reverse_iterator rend() const noexcept;

        /**
         * @brief Returns a const_reverse_iterator pointing at the start of the list (end of the reverse sequence).
         * @return
         */
        const_reverse_iterator crend() const noexcept;

        /**
         * @brief Returns a reference to the first element. The list must be non-empty.
         * @return
         */
        reference front();

        /**
         * @brief Returns a const reference to the first element. The list must be non-empty.
         * @return
         */
        const_reference front() const;

        /**
         * @brief Returns a reference to the last element. The list must be non-empty.
         * @return
         */
        reference back();

        /**
         * @brief Returns a const reference to the last element. The list must be non-empty.
         * @return
         */
        const_reference back() const;

        /**
         * @brief Adds an element to the front of the list; O(1). The element is not copied. The element must not be in any other list.
         * @param x
         */
        void push_front(value_type &x);

        /**
         * @brief Adds an element to the back of the list; O(1). The element is not copied. The element must not be in any other list.
         * @param x
         */
        void push_back(value_type &x);

        /**
         * @brief Returns true if the given element is in the list; O(n). Equivalent to (locate(x) != end()).
         * @param x
         * @return
         */
        bool contains(const value_type &x) const;

        /**
         * @brief Converts a reference to an object in the list back to an iterator, or returns end() if it is not part of the list. O(n)
         * @param x
         * @return
         */
        iterator locate(value_type &x);

        /**
         * @brief Converts a const reference to an object in the list back to a const iterator, or returns end() if it is not part of the list. O(n)
         * @param x
         * @return
         */
        const_iterator locate(const value_type &x) const;

        /**
         * @brief Inserts an element before the element pointed to by the iterator. O(1)
         * @param pos
         * @param x
         * @return
         */
        iterator insert(const_iterator pos, value_type &x);

        /**
         * @brief Erases the element pointed to by the iterator. O(1)
         * @param pos
         * @return
         */
        iterator erase(const_iterator pos);

        /**
         * @brief Erases elements within the iterator range [pos, last). O(1)
         * @param pos
         * @param last
         * @return
         */
        iterator erase(const_iterator pos, const_iterator last);

        /**
         * @brief Erases the element pointed to by the iterator. O(1)
         * @param pos
         * @return
         */
        reverse_iterator erase(const_reverse_iterator pos);

        /**
         * @brief Erases elements within the iterator range [pos, last). O(1)
         * @param pos
         * @param last
         * @return
         */
        reverse_iterator erase(const_reverse_iterator pos, const_reverse_iterator last);

        /**
         * @brief Removes an element from the list; O(1). Note that this is static so you don't need to know which list the element, although it must be in some list.
         * @param value
         */
        static void remove(value_type &value);

        /**
         * @brief Removes an element from the list; O(1). Note that this is static so you don't need to know which list the element, although it must be in some list.
         * @param value
         */
        void splice(const_iterator pos, value_type &x);

        /**
         * @brief Moves the contents of a list into this list before the element pointed to by pos; O(1).
         *        Required: x must in some list or have first/next pointers that point it itself.
         * @param value
         */
        void splice(const_iterator pos, intrusive_list &x);

        /**
         * @brief Moves the contents of a list into this list before the element pointed to by pos; O(1).
         *     Required: x must in some list or have first/next pointers that point it itself.
         * @param value
         */
        void splice(const_iterator pos, intrusive_list &x, const_iterator i);

        /**
         * @brief Moves the contents of a list into this list before the element pointed to by pos; O(1).
         *     Required: x must in some list or have first/next pointers that point it itself.
         * @param value
         */
        void splice(const_iterator pos, intrusive_list &x, const_iterator first, const_iterator last);

    public:
        // Sorting functionality
        // This is independent of the global sort algorithms, as lists are
        // linked nodes and can be sorted more efficiently by moving nodes
        // around in ways that global sort algorithms aren't privy to.

        void merge(this_type &x);

        template<typename Compare>
        void merge(this_type &x, Compare compare);

        void unique();

        template<typename BinaryPredicate>
        void unique(BinaryPredicate);

        void sort();

        template<typename Compare>
        void sort(Compare compare);


    private:
        intrusive_list_node *to_list_node(const node_type *node) {
            return static_cast<intrusive_list_node *>(const_cast<node_type *>(node));
        }
    }; // intrusive_list





    ///////////////////////////////////////////////////////////////////////
    // intrusive_list_iterator
    ///////////////////////////////////////////////////////////////////////

    template<typename T, typename Pointer, typename Reference>
    inline intrusive_list_iterator<T, Pointer, Reference>::intrusive_list_iterator() {
    }


    template<typename T, typename Pointer, typename Reference>
    inline intrusive_list_iterator<T, Pointer, Reference>::intrusive_list_iterator(const base_node_type *pNode)
            : mpNode(toInternalNodeType(const_cast<base_node_type *>(pNode))) {
        // Empty
    }


    template<typename T, typename Pointer, typename Reference>
    inline intrusive_list_iterator<T, Pointer, Reference>::intrusive_list_iterator(const iterator &x)
            : mpNode(x.mpNode) {
        // Empty
    }

    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::this_type &
    intrusive_list_iterator<T, Pointer, Reference>::operator=(const iterator &x) {
        mpNode = x.mpNode;
        return *this;
    }

    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::reference
    intrusive_list_iterator<T, Pointer, Reference>::operator*() const {
        return *static_cast<pointer>(mpNode);
    }


    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::pointer
    intrusive_list_iterator<T, Pointer, Reference>::operator->() const {
        return static_cast<pointer>(mpNode);
    }

    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::pointer
    intrusive_list_iterator<T, Pointer, Reference>::node_ptr() const {
        return static_cast<pointer>(mpNode);
    }


    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::this_type &
    intrusive_list_iterator<T, Pointer, Reference>::operator++() {
        mpNode = toInternalNodeType(mpNode->next);
        return *this;
    }


    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::this_type
    intrusive_list_iterator<T, Pointer, Reference>::operator++(int) {
        intrusive_list_iterator it(*this);
        mpNode = toInternalNodeType(mpNode->next);
        return it;
    }


    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::this_type &
    intrusive_list_iterator<T, Pointer, Reference>::operator--() {
        mpNode = toInternalNodeType(mpNode->prev);
        return *this;
    }


    template<typename T, typename Pointer, typename Reference>
    inline typename intrusive_list_iterator<T, Pointer, Reference>::this_type
    intrusive_list_iterator<T, Pointer, Reference>::operator--(int) {
        intrusive_list_iterator it(*this);
        mpNode = toInternalNodeType(mpNode->prev);
        return it;
    }

    ///////////////////////////////////////////////////////////////////////
    // intrusive_list_base
    ///////////////////////////////////////////////////////////////////////

    inline intrusive_list_base::intrusive_list_base() {
        mAnchor.next = mAnchor.prev = &mAnchor;
    }

    inline intrusive_list_base::~intrusive_list_base() {
        // We don't do anything here because we don't own the elements.
    }


    inline bool intrusive_list_base::empty() const noexcept {
        return mAnchor.prev == &mAnchor;
    }


    inline intrusive_list_base::size_type intrusive_list_base::size() const noexcept {
        const intrusive_list_node *p = &mAnchor;
        size_type n = (size_type) -1;

        do {
            ++n;
            p = p->next;
        } while (p != &mAnchor);

        return n;
    }


    inline void intrusive_list_base::clear() noexcept {
        mAnchor.next = mAnchor.prev = &mAnchor;
    }


    inline void intrusive_list_base::pop_front() {

        mAnchor.next->next->prev = &mAnchor;
        mAnchor.next = mAnchor.next->next;

    }


    inline void intrusive_list_base::pop_back() {
        mAnchor.prev->prev->next = &mAnchor;
        mAnchor.prev = mAnchor.prev->prev;
    }




    ///////////////////////////////////////////////////////////////////////
    // intrusive_list
    ///////////////////////////////////////////////////////////////////////

    template<typename T>
    inline intrusive_list<T>::intrusive_list() {
    }


    template<typename T>
    inline intrusive_list<T>::intrusive_list(const this_type & /*x*/)
            : intrusive_list_base() {
        // We intentionally ignore argument x.
        // To consider: Shouldn't this function simply not exist? Is there a useful purpose for having this function?
        // There should be a comment here about it, though my first guess is that this exists to quell VC++ level 4/-Wall compiler warnings.
    }


    template<typename T>
    inline typename intrusive_list<T>::this_type &intrusive_list<T>::operator=(const this_type & /*x*/) {
        // We intentionally ignore argument x.
        // See notes above in the copy constructor about questioning the existence of this function.
        return *this;
    }


    template<typename T>
    inline typename intrusive_list<T>::iterator intrusive_list<T>::begin() noexcept {
        return iterator(mAnchor.next);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_iterator intrusive_list<T>::begin() const noexcept {
        return const_iterator(mAnchor.next);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_iterator intrusive_list<T>::cbegin() const noexcept {
        return const_iterator(mAnchor.next);
    }


    template<typename T>
    inline typename intrusive_list<T>::iterator intrusive_list<T>::end() noexcept {
        return iterator(&mAnchor);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_iterator intrusive_list<T>::end() const noexcept {
        return const_iterator(&mAnchor);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_iterator intrusive_list<T>::cend() const noexcept {
        return const_iterator(&mAnchor);
    }


    template<typename T>
    inline typename intrusive_list<T>::reverse_iterator intrusive_list<T>::rbegin() noexcept {
        return reverse_iterator(iterator(&mAnchor));
    }


    template<typename T>
    inline typename intrusive_list<T>::const_reverse_iterator intrusive_list<T>::rbegin() const noexcept {
        return const_reverse_iterator(const_iterator(&mAnchor));
    }


    template<typename T>
    inline typename intrusive_list<T>::const_reverse_iterator intrusive_list<T>::crbegin() const noexcept {
        return const_reverse_iterator(const_iterator(&mAnchor));
    }


    template<typename T>
    inline typename intrusive_list<T>::reverse_iterator intrusive_list<T>::rend() noexcept {
        return reverse_iterator(iterator(mAnchor.next));
    }


    template<typename T>
    inline typename intrusive_list<T>::const_reverse_iterator intrusive_list<T>::rend() const noexcept {
        return const_reverse_iterator(const_iterator(mAnchor.next));
    }


    template<typename T>
    inline typename intrusive_list<T>::const_reverse_iterator intrusive_list<T>::crend() const noexcept {
        return const_reverse_iterator(const_iterator(mAnchor.next));
    }


    template<typename T>
    inline typename intrusive_list<T>::reference intrusive_list<T>::front() {
        return *static_cast<T *>(mAnchor.next);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_reference intrusive_list<T>::front() const {
        return *static_cast<const T *>(mAnchor.next);
    }


    template<typename T>
    inline typename intrusive_list<T>::reference intrusive_list<T>::back() {
        return *static_cast<T *>(mAnchor.prev);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_reference intrusive_list<T>::back() const {
        return *static_cast<const T *>(mAnchor.prev);
    }


    template<typename T>
    inline void intrusive_list<T>::push_front(value_type &x) {
        x.next = mAnchor.next;
        x.prev = &mAnchor;
        mAnchor.next = &x;
        x.next->prev = &x;
    }


    template<typename T>
    inline void intrusive_list<T>::push_back(value_type &x) {
        x.prev = mAnchor.prev;
        x.next = &mAnchor;
        mAnchor.prev = &x;
        x.prev->next = &x;
    }


    template<typename T>
    inline bool intrusive_list<T>::contains(const value_type &x) const {
        for (const intrusive_list_node *p = mAnchor.next; p != &mAnchor; p = p->next) {
            if (p == &x)
                return true;
        }

        return false;
    }


    template<typename T>
    inline typename intrusive_list<T>::iterator intrusive_list<T>::locate(value_type &x) {
        for (intrusive_list_node *p = (T *) mAnchor.next; p != &mAnchor; p = p->next) {
            if (p == &x)
                return iterator(p);
        }

        return iterator(&mAnchor);
    }


    template<typename T>
    inline typename intrusive_list<T>::const_iterator intrusive_list<T>::locate(const value_type &x) const {
        for (const intrusive_list_node *p = mAnchor.next; p != &mAnchor; p = p->next) {
            if (p == &x)
                return const_iterator(p);
        }

        return const_iterator(&mAnchor);
    }


    template<typename T>
    inline typename intrusive_list<T>::iterator intrusive_list<T>::insert(const_iterator pos, value_type &x) {
        intrusive_list_node &next = *to_list_node(pos.mpNode);
        intrusive_list_node &prev = *next.prev;

        prev.next = next.prev = &x;
        x.prev = &prev;
        x.next = &next;

        return iterator(&x);
    }


    template<typename T>
    inline typename intrusive_list<T>::iterator
    intrusive_list<T>::erase(const_iterator pos) {
        intrusive_list_node &prev = *pos.mpNode->prev;
        intrusive_list_node &next = *pos.mpNode->next;
        prev.next = &next;
        next.prev = &prev;

        return iterator(&next);
    }


    template<typename T>
    inline typename intrusive_list<T>::iterator
    intrusive_list<T>::erase(const_iterator first, const_iterator last) {
        intrusive_list_node &prev = *(first.mpNode->prev);
        intrusive_list_node &next = *to_list_node(last.mpNode);
        prev.next = &next;
        next.prev = &prev;

        return iterator(last.mpNode);
    }


    template<typename T>
    inline typename intrusive_list<T>::reverse_iterator
    intrusive_list<T>::erase(const_reverse_iterator position) {
        return reverse_iterator(erase((++position).base()));
    }


    template<typename T>
    inline typename intrusive_list<T>::reverse_iterator
    intrusive_list<T>::erase(const_reverse_iterator first, const_reverse_iterator last) {
        // Version which erases in order from first to last.
        // difference_type i(first.base() - last.base());
        // while(i--)
        //     first = erase(first);
        // return first;

        // Version which erases in order from last to first, but is slightly more efficient:
        return reverse_iterator(erase((++last).base(), (++first).base()));
    }


    template<typename T>
    void intrusive_list<T>::swap(intrusive_list &x) {
        // swap anchors
        intrusive_list_node temp(mAnchor);
        mAnchor = x.mAnchor;
        x.mAnchor = temp;

        // Fixup node pointers into the anchor, since the addresses of
        // the anchors must stay the same with each list.
        if (mAnchor.next == &x.mAnchor)
            mAnchor.next = mAnchor.prev = &mAnchor;
        else
            mAnchor.next->prev = mAnchor.prev->next = &mAnchor;

        if (x.mAnchor.next == &mAnchor)
            x.mAnchor.next = x.mAnchor.prev = &x.mAnchor;
        else
            x.mAnchor.next->prev = x.mAnchor.prev->next = &x.mAnchor;
    }


    template<typename T>
    void intrusive_list<T>::splice(const_iterator pos, value_type &value) {
        // Note that splice(pos, x, pos) and splice(pos+1, x, pos)
        // are valid and need to be handled correctly.

        if (pos.mpNode != &value) {
            // Unlink item from old list.
            intrusive_list_node &oldNext = *value.next;
            intrusive_list_node &oldPrev = *value.prev;
            oldNext.prev = &oldPrev;
            oldPrev.next = &oldNext;

            // Relink item into new list.
            intrusive_list_node &newNext = *to_list_node(pos.mpNode);
            intrusive_list_node &newPrev = *newNext.prev;

            newPrev.next = &value;
            newNext.prev = &value;
            value.prev = &newPrev;
            value.next = &newNext;
        }
    }


    template<typename T>
    void intrusive_list<T>::splice(const_iterator pos, intrusive_list &x) {
        // Note: &x == this is prohibited, so self-insertion is not a problem.
        if (x.mAnchor.next != &x.mAnchor) // If the list 'x' isn't empty...
        {
            intrusive_list_node &next = *to_list_node(pos.mpNode);
            intrusive_list_node &prev = *next.prev;
            intrusive_list_node &insertPrev = *x.mAnchor.next;
            intrusive_list_node &insertNext = *x.mAnchor.prev;

            prev.next = &insertPrev;
            insertPrev.prev = &prev;
            insertNext.next = &next;
            next.prev = &insertNext;
            x.mAnchor.prev = x.mAnchor.next = &x.mAnchor;
        }
    }


    template<typename T>
    void intrusive_list<T>::splice(const_iterator pos, intrusive_list & /*x*/, const_iterator i) {
        // Note: &x == this is prohibited, so self-insertion is not a problem.

        // Note that splice(pos, x, pos) and splice(pos + 1, x, pos)
        // are valid and need to be handled correctly.

        // We don't need to check if the source list is empty, because
        // this function expects a valid iterator from the source list,
        // and thus the list cannot be empty in such a situation.

        iterator ii(i.mpNode); // Make a temporary non-const version.

        if (pos != ii) {
            // Unlink item from old list.
            intrusive_list_node &oldNext = *ii.mpNode->next;
            intrusive_list_node &oldPrev = *ii.mpNode->prev;
            oldNext.prev = &oldPrev;
            oldPrev.next = &oldNext;

            // Relink item into new list.
            intrusive_list_node &newNext = *to_list_node(pos.mpNode);
            intrusive_list_node &newPrev = *newNext.prev;

            newPrev.next = ii.mpNode;
            newNext.prev = ii.mpNode;
            ii.mpNode->prev = &newPrev;
            ii.mpNode->next = &newNext;
        }
    }


    template<typename T>
    void
    intrusive_list<T>::splice(const_iterator pos, intrusive_list & /*x*/, const_iterator first, const_iterator last) {
        // Note: &x == this is prohibited, so self-insertion is not a problem.
        if (first != last) {
            intrusive_list_node &insertPrev = *to_list_node(first.mpNode);
            intrusive_list_node &insertNext = *last.mpNode->prev;

            // remove from old list
            insertNext.next->prev = insertPrev.prev;
            insertPrev.prev->next = insertNext.next;

            // insert into this list
            intrusive_list_node &next = *to_list_node(pos.mpNode);
            intrusive_list_node &prev = *next.prev;

            prev.next = &insertPrev;
            insertPrev.prev = &prev;
            insertNext.next = &next;
            next.prev = &insertNext;
        }
    }


    template<typename T>
    inline void intrusive_list<T>::remove(value_type &value) {
        intrusive_list_node &prev = *value.prev;
        intrusive_list_node &next = *value.next;
        prev.next = &next;
        next.prev = &prev;

    }


    template<typename T>
    void intrusive_list<T>::merge(this_type &x) {
        if (this != &x) {
            iterator first(begin());
            iterator firstX(x.begin());
            const iterator last(end());
            const iterator lastX(x.end());

            while ((first != last) && (firstX != lastX)) {
                if (*firstX < *first) {
                    iterator next(firstX);

                    splice(first, x, firstX, ++next);
                    firstX = next;
                } else
                    ++first;
            }

            if (firstX != lastX)
                splice(last, x, firstX, lastX);
        }
    }


    template<typename T>
    template<typename Compare>
    void intrusive_list<T>::merge(this_type &x, Compare compare) {
        if (this != &x) {
            iterator first(begin());
            iterator firstX(x.begin());
            const iterator last(end());
            const iterator lastX(x.end());

            while ((first != last) && (firstX != lastX)) {
                if (compare(*firstX, *first)) {
                    iterator next(firstX);

                    splice(first, x, firstX, ++next);
                    firstX = next;
                } else
                    ++first;
            }

            if (firstX != lastX)
                splice(last, x, firstX, lastX);
        }
    }


    template<typename T>
    void intrusive_list<T>::unique() {
        iterator first(begin());
        const iterator last(end());

        if (first != last) {
            iterator next(first);

            while (++next != last) {
                if (*first == *next)
                    erase(next);
                else
                    first = next;
                next = first;
            }
        }
    }


    template<typename T>
    template<typename BinaryPredicate>
    void intrusive_list<T>::unique(BinaryPredicate predicate) {
        iterator first(begin());
        const iterator last(end());

        if (first != last) {
            iterator next(first);

            while (++next != last) {
                if (predicate(*first, *next))
                    erase(next);
                else
                    first = next;
                next = first;
            }
        }
    }


    template<typename T>
    void intrusive_list<T>::sort() {
        // We implement the algorithm employed by Chris Caulfield whereby we use recursive
        // function calls to sort the list. The sorting of a very large list may fail due to stack overflow
        // if the stack is exhausted. The limit depends on the platform and the avaialble stack space.

        // Easier-to-understand version of the 'if' statement:
        // iterator i(begin());
        // if((i != end()) && (++i != end())) // If the size is >= 2 (without calling the more expensive size() function)...

        // Faster, more inlinable version of the 'if' statement:
        if ((mAnchor.next != &mAnchor) && (mAnchor.next != mAnchor.prev)) {
            // Split the array into 2 roughly equal halves.
            this_type leftList;     // This should cause no memory allocation.
            this_type rightList;

            iterator mid(begin()), tail(end());

            while ((mid != tail) && (++mid != tail))
                --tail;

            // Move the left half of this into leftList and the right half into rightList.
            leftList.splice(leftList.begin(), *this, begin(), mid);
            rightList.splice(rightList.begin(), *this);

            // Sort the sub-lists.
            leftList.sort();
            rightList.sort();

            // Merge the two halves into this list.
            splice(begin(), leftList);
            merge(rightList);
        }
    }


    template<typename T>
    template<typename Compare>
    void intrusive_list<T>::sort(Compare compare) {
        // We implement the algorithm employed by Chris Caulfield whereby we use recursive
        // function calls to sort the list. The sorting of a very large list may fail due to stack overflow
        // if the stack is exhausted. The limit depends on the platform and the avaialble stack space.

        // Easier-to-understand version of the 'if' statement:
        // iterator i(begin());
        // if((i != end()) && (++i != end())) // If the size is >= 2 (without calling the more expensive size() function)...

        // Faster, more inlinable version of the 'if' statement:
        if ((mAnchor.next != &mAnchor) && (mAnchor.next != mAnchor.prev)) {
            // Split the array into 2 roughly equal halves.
            this_type leftList;     // This should cause no memory allocation.
            this_type rightList;

            iterator mid(begin()), tail(end());

            while ((mid != tail) && (++mid != tail))
                --tail;

            // Move the left half of this into leftList and the right half into rightList.
            leftList.splice(leftList.begin(), *this, begin(), mid);
            rightList.splice(rightList.begin(), *this);

            // Sort the sub-lists.
            leftList.sort(compare);
            rightList.sort(compare);

            // Merge the two halves into this list.
            splice(begin(), leftList);
            merge(rightList, compare);
        }
    }



    ///////////////////////////////////////////////////////////////////////
    // global operators
    ///////////////////////////////////////////////////////////////////////

    template<typename T>
    bool operator==(const intrusive_list<T> &a, const intrusive_list<T> &b) {
        // If we store an mSize member for intrusive_list, we want to take advantage of it here.
        typename intrusive_list<T>::const_iterator ia = a.begin();
        typename intrusive_list<T>::const_iterator ib = b.begin();
        typename intrusive_list<T>::const_iterator enda = a.end();
        typename intrusive_list<T>::const_iterator endb = b.end();

        while ((ia != enda) && (ib != endb) && (*ia == *ib)) {
            ++ia;
            ++ib;
        }
        return (ia == enda) && (ib == endb);
    }

    template<typename T>
    bool operator!=(const intrusive_list<T> &a, const intrusive_list<T> &b) {
        return !(a == b);
    }

    template<typename T>
    bool operator<(const intrusive_list<T> &a, const intrusive_list<T> &b) {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }

    template<typename T>
    bool operator>(const intrusive_list<T> &a, const intrusive_list<T> &b) {
        return b < a;
    }

    template<typename T>
    bool operator<=(const intrusive_list<T> &a, const intrusive_list<T> &b) {
        return !(b < a);
    }

    template<typename T>
    bool operator>=(const intrusive_list<T> &a, const intrusive_list<T> &b) {
        return !(a < b);
    }

    template<typename T>
    void swap(intrusive_list<T> &a, intrusive_list<T> &b) {
        a.swap(b);
    }

    inline void intrusive_list_base::reverse() noexcept {
        intrusive_list_node *pNode = &mAnchor;
        do {
            intrusive_list_node *const pTemp = pNode->next;
            pNode->next = pNode->prev;
            pNode->prev = pTemp;
            pNode = pNode->prev;
        } while (pNode != &mAnchor);
    }


} // namespace turbo


#endif // TURBO_CONTAINER_INTRUSIVE_LIST_H_















