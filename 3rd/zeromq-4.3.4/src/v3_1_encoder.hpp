/*
    Copyright (c) 2020 Contributors as noted in the AUTHORS file

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

#ifndef __ZMQ_V3_1_ENCODER_HPP_INCLUDED__
#define __ZMQ_V3_1_ENCODER_HPP_INCLUDED__

#include "encoder.hpp"
#include "msg.hpp"

namespace zmq
{
//  Encoder for 0MQ framing protocol. Converts messages into data stream.

class v3_1_encoder_t ZMQ_FINAL : public encoder_base_t<v3_1_encoder_t>
{
  public:
    v3_1_encoder_t (size_t bufsize_);
    ~v3_1_encoder_t () ZMQ_FINAL;

  private:
    void size_ready ();
    void message_ready ();

    unsigned char _tmp_buf[9 + zmq::msg_t::sub_cmd_name_size];

    ZMQ_NON_COPYABLE_NOR_MOVABLE (v3_1_encoder_t)
};
}

#endif
