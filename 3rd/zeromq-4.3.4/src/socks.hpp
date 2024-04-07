/*
    Copyright (c) 2007-2016 Contributors as noted in the AUTHORS file

    This file is part of libzmq, the ZeroMQ core engine in C++.

    libzmq is free software; you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    As a special exception, the Contributors give you permission to link
    this library with independent modules to produce an executable,
    regardless of the license terms of these independent modules, and to
    copy and distribute the resulting executable under terms of your choice,
    provided that you also meet, for each linked independent module, the
    terms and conditions of the license of that module. An independent
    module is a module which is not derived from or based on this library.
    If you modify this library, you must extend this exception to your
    version of the library.

    libzmq is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
    License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __ZMQ_SOCKS_HPP_INCLUDED__
#define __ZMQ_SOCKS_HPP_INCLUDED__

#include <string>
#include "fd.hpp"
#include "stdint.hpp"

namespace zmq
{
struct socks_greeting_t
{
    socks_greeting_t (uint8_t method_);
    socks_greeting_t (const uint8_t *methods_, uint8_t num_methods_);

    uint8_t methods[UINT8_MAX];
    const size_t num_methods;
};

class socks_greeting_encoder_t
{
  public:
    socks_greeting_encoder_t ();
    void encode (const socks_greeting_t &greeting_);
    int output (fd_t fd_);
    bool has_pending_data () const;
    void reset ();

  private:
    size_t _bytes_encoded;
    size_t _bytes_written;
    uint8_t _buf[2 + UINT8_MAX];
};

struct socks_choice_t
{
    socks_choice_t (uint8_t method_);

    uint8_t method;
};

class socks_choice_decoder_t
{
  public:
    socks_choice_decoder_t ();
    int input (fd_t fd_);
    bool message_ready () const;
    socks_choice_t decode ();
    void reset ();

  private:
    unsigned char _buf[2];
    size_t _bytes_read;
};


struct socks_basic_auth_request_t
{
    socks_basic_auth_request_t (const std::string &username_,
                                const std::string &password_);

    const std::string username;
    const std::string password;
};

class socks_basic_auth_request_encoder_t
{
  public:
    socks_basic_auth_request_encoder_t ();
    void encode (const socks_basic_auth_request_t &req_);
    int output (fd_t fd_);
    bool has_pending_data () const;
    void reset ();

  private:
    size_t _bytes_encoded;
    size_t _bytes_written;
    uint8_t _buf[1 + 1 + UINT8_MAX + 1 + UINT8_MAX];
};

struct socks_auth_response_t
{
    socks_auth_response_t (uint8_t response_code_);
    uint8_t response_code;
};

class socks_auth_response_decoder_t
{
  public:
    socks_auth_response_decoder_t ();
    int input (fd_t fd_);
    bool message_ready () const;
    socks_auth_response_t decode ();
    void reset ();

  private:
    int8_t _buf[2];
    size_t _bytes_read;
};

struct socks_request_t
{
    socks_request_t (uint8_t command_, std::string hostname_, uint16_t port_);

    const uint8_t command;
    const std::string hostname;
    const uint16_t port;
};

class socks_request_encoder_t
{
  public:
    socks_request_encoder_t ();
    void encode (const socks_request_t &req_);
    int output (fd_t fd_);
    bool has_pending_data () const;
    void reset ();

  private:
    size_t _bytes_encoded;
    size_t _bytes_written;
    uint8_t _buf[4 + UINT8_MAX + 1 + 2];
};

struct socks_response_t
{
    socks_response_t (uint8_t response_code_,
                      const std::string &address_,
                      uint16_t port_);
    uint8_t response_code;
    std::string address;
    uint16_t port;
};

class socks_response_decoder_t
{
  public:
    socks_response_decoder_t ();
    int input (fd_t fd_);
    bool message_ready () const;
    socks_response_t decode ();
    void reset ();

  private:
    int8_t _buf[4 + UINT8_MAX + 1 + 2];
    size_t _bytes_read;
};
}

#endif
