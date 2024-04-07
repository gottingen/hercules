/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay, Martin Renou          *
* Copyright (c) 2016, QuantStack                                           *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XEUS_THREAD_HPP
#define XEUS_THREAD_HPP

#include <thread>
#include <type_traits>

namespace xeus
{

    /**
     * Joining std::thread
     */
    class xthread
    {
    public:

        using id = std::thread::id;
        using native_handle_type = std::thread::native_handle_type;

        xthread() noexcept = default;

        // Last arguments SFINAE out copy constructor
        template <class Function, class... Args,
                 typename = std::enable_if_t<!std::is_same<std::decay_t<Function>, xthread>::value>>
        explicit xthread(Function&& f, Args&&... args);

        ~xthread();

        xthread(const xthread&) = delete;
        xthread& operator=(const xthread&) = delete;

        xthread(xthread&&) = default;
        xthread& operator=(xthread&&);

        bool joinable() const noexcept;
        id get_id() const noexcept;
        native_handle_type native_handle();
        static unsigned int hardware_concurrency() noexcept;

        void join();
        void detach();
        void swap(xthread& other) noexcept;

    private:

        std::thread m_thread;

    };

    /**************************
     * xthread implementation *
     **************************/
    template <class Function, class... Args, typename>
    inline xthread::xthread(Function&& func, Args&&... args)
        : m_thread{
            std::forward<Function>(func),
            std::forward<Args>(args)...
        }

    {
    }

    inline xthread::~xthread()
    {
        if (joinable())
        {
            join();
        }
    }

    inline xthread& xthread::operator=(xthread&& rhs)
    {
        if (joinable())
        {
            join();
        }
        m_thread = std::move(rhs.m_thread);
        return *this;
    }

    inline bool xthread::joinable() const noexcept
    {
        return m_thread.joinable();
    }

    inline xthread::id xthread::get_id() const noexcept
    {
        return m_thread.get_id();
    }

    inline xthread::native_handle_type xthread::native_handle()
    {
        return m_thread.native_handle();
    }

    inline unsigned int xthread::hardware_concurrency() noexcept
    {
        return std::thread::hardware_concurrency();
    }

    inline void xthread::join()
    {
        m_thread.join();
    }

    inline void xthread::detach()
    {
        m_thread.detach();
    }

    inline void xthread::swap(xthread& other) noexcept
    {
        m_thread.swap(other.m_thread);
    }
}

#endif
