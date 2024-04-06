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

#ifndef __ZMQ_RAW_DECODER_HPP_INCLUDED__
#define __ZMQ_RAW_DECODER_HPP_INCLUDED__

#include "msg.hpp"
#include "i_decoder.hpp"
#include "stdint.hpp"
#include "decoder_allocators.hpp"

namespace zmq
{
//  Decoder for 0MQ v1 framing protocol. Converts data stream into messages.

class raw_decoder_t ZMQ_FINAL : public i_decoder
{
  public:
    raw_decoder_t (size_t bufsize_);
    ~raw_decoder_t ();

    //  i_decoder interface.

    void get_buffer (unsigned char **data_, size_t *size_);

    int decode (const unsigned char *data_, size_t size_, size_t &bytes_used_);

    msg_t *msg () { return &_in_progress; }

    void resize_buffer (size_t) {}

  private:
    msg_t _in_progress;

    shared_message_memory_allocator _allocator;

    ZMQ_NON_COPYABLE_NOR_MOVABLE (raw_decoder_t)
};
}

#endif
