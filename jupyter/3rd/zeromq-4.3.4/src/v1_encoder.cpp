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

#include "precompiled.hpp"
#include "encoder.hpp"
#include "v1_encoder.hpp"
#include "msg.hpp"
#include "wire.hpp"

#include <limits.h>

zmq::v1_encoder_t::v1_encoder_t (size_t bufsize_) :
    encoder_base_t<v1_encoder_t> (bufsize_)
{
    //  Write 0 bytes to the batch and go to message_ready state.
    next_step (NULL, 0, &v1_encoder_t::message_ready, true);
}

zmq::v1_encoder_t::~v1_encoder_t ()
{
}

void zmq::v1_encoder_t::size_ready ()
{
    //  Write message body into the buffer.
    next_step (in_progress ()->data (), in_progress ()->size (),
               &v1_encoder_t::message_ready, true);
}

void zmq::v1_encoder_t::message_ready ()
{
    size_t header_size = 2; // flags byte + size byte
    //  Get the message size.
    size_t size = in_progress ()->size ();

    //  Account for the 'flags' byte.
    size++;

    //  Account for the subscribe/cancel byte.
    if (in_progress ()->is_subscribe () || in_progress ()->is_cancel ())
        size++;

    //  For messages less than 255 bytes long, write one byte of message size.
    //  For longer messages write 0xff escape character followed by 8-byte
    //  message size. In both cases 'flags' field follows.
    if (size < UCHAR_MAX) {
        _tmpbuf[0] = static_cast<unsigned char> (size);
        _tmpbuf[1] = (in_progress ()->flags () & msg_t::more);
    } else {
        _tmpbuf[0] = UCHAR_MAX;
        put_uint64 (_tmpbuf + 1, size);
        _tmpbuf[9] = (in_progress ()->flags () & msg_t::more);
        header_size = 10;
    }

    //  Encode the subscribe/cancel byte. This is done in the encoder as
    //  opposed to when the subscribe message is created to allow different
    //  protocol behaviour on the wire in the v3.1 and legacy encoders.
    //  It results in the work being done multiple times in case the sub
    //  is sending the subscription/cancel to multiple pubs, but it cannot
    //  be avoided. This processing can be moved to xsub once support for
    //  ZMTP < 3.1 is dropped.
    if (in_progress ()->is_subscribe ())
        _tmpbuf[header_size++] = 1;
    else if (in_progress ()->is_cancel ())
        _tmpbuf[header_size++] = 0;

    next_step (_tmpbuf, header_size, &v1_encoder_t::size_ready, false);
}
