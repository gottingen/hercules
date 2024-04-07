/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay, Martin Renou          *
* Copyright (c) 2016, QuantStack                                           *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XEUS_ZMQ_EXPORT_HPP
#define XEUS_ZMQ_EXPORT_HPP

#ifdef _WIN32
    #ifdef XEUS_ZMQ_STATIC_LIB
        #define XEUS_ZMQ_API
    #else
        #ifdef XEUS_ZMQ_EXPORTS
            #define XEUS_ZMQ_API __declspec(dllexport)
        #else
            #define XEUS_ZMQ_API __declspec(dllimport)
        #endif
    #endif
#else
    #define XEUS_ZMQ_API
#endif

// Project version
#define XEUS_ZMQ_VERSION_MAJOR 1
#define XEUS_ZMQ_VERSION_MINOR 0
#define XEUS_ZMQ_VERSION_PATCH 3

// Binary version
#define XEUS_ZMQ_BINARY_CURRENT 1
#define XEUS_ZMQ_BINARY_REVISION 1
#define XEUS_ZMQ_BINARY_AGE 0

#endif

