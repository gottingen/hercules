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
#include "macros.hpp"
#include "req.hpp"
#include "err.hpp"
#include "msg.hpp"
#include "wire.hpp"
#include "random.hpp"
#include "likely.hpp"

zmq::req_t::req_t (class ctx_t *parent_, uint32_t tid_, int sid_) :
    dealer_t (parent_, tid_, sid_),
    _receiving_reply (false),
    _message_begins (true),
    _reply_pipe (NULL),
    _request_id_frames_enabled (false),
    _request_id (generate_random ()),
    _strict (true)
{
    options.type = ZMQ_REQ;
}

zmq::req_t::~req_t ()
{
}

int zmq::req_t::xsend (msg_t *msg_)
{
    //  If we've sent a request and we still haven't got the reply,
    //  we can't send another request unless the strict option is disabled.
    if (_receiving_reply) {
        if (_strict) {
            errno = EFSM;
            return -1;
        }

        _receiving_reply = false;
        _message_begins = true;
    }

    //  First part of the request is the request routing id.
    if (_message_begins) {
        _reply_pipe = NULL;

        if (_request_id_frames_enabled) {
            _request_id++;

            msg_t id;
            int rc = id.init_size (sizeof (uint32_t));
            memcpy (id.data (), &_request_id, sizeof (uint32_t));
            errno_assert (rc == 0);
            id.set_flags (msg_t::more);

            rc = dealer_t::sendpipe (&id, &_reply_pipe);
            if (rc != 0) {
                return -1;
            }
        }

        msg_t bottom;
        int rc = bottom.init ();
        errno_assert (rc == 0);
        bottom.set_flags (msg_t::more);

        rc = dealer_t::sendpipe (&bottom, &_reply_pipe);
        if (rc != 0)
            return -1;
        zmq_assert (_reply_pipe);

        _message_begins = false;

        // Eat all currently available messages before the request is fully
        // sent. This is done to avoid:
        //   REQ sends request to A, A replies, B replies too.
        //   A's reply was first and matches, that is used.
        //   An hour later REQ sends a request to B. B's old reply is used.
        msg_t drop;
        while (true) {
            rc = drop.init ();
            errno_assert (rc == 0);
            rc = dealer_t::xrecv (&drop);
            if (rc != 0)
                break;
            drop.close ();
        }
    }

    bool more = (msg_->flags () & msg_t::more) != 0;

    int rc = dealer_t::xsend (msg_);
    if (rc != 0)
        return rc;

    //  If the request was fully sent, flip the FSM into reply-receiving state.
    if (!more) {
        _receiving_reply = true;
        _message_begins = true;
    }

    return 0;
}

int zmq::req_t::xrecv (msg_t *msg_)
{
    //  If request wasn't send, we can't wait for reply.
    if (!_receiving_reply) {
        errno = EFSM;
        return -1;
    }

    //  Skip messages until one with the right first frames is found.
    while (_message_begins) {
        //  If enabled, the first frame must have the correct request_id.
        if (_request_id_frames_enabled) {
            int rc = recv_reply_pipe (msg_);
            if (rc != 0)
                return rc;

            if (unlikely (!(msg_->flags () & msg_t::more)
                          || msg_->size () != sizeof (_request_id)
                          || *static_cast<uint32_t *> (msg_->data ())
                               != _request_id)) {
                //  Skip the remaining frames and try the next message
                while (msg_->flags () & msg_t::more) {
                    rc = recv_reply_pipe (msg_);
                    errno_assert (rc == 0);
                }
                continue;
            }
        }

        //  The next frame must be 0.
        // TODO: Failing this check should also close the connection with the peer!
        int rc = recv_reply_pipe (msg_);
        if (rc != 0)
            return rc;

        if (unlikely (!(msg_->flags () & msg_t::more) || msg_->size () != 0)) {
            //  Skip the remaining frames and try the next message
            while (msg_->flags () & msg_t::more) {
                rc = recv_reply_pipe (msg_);
                errno_assert (rc == 0);
            }
            continue;
        }

        _message_begins = false;
    }

    const int rc = recv_reply_pipe (msg_);
    if (rc != 0)
        return rc;

    //  If the reply is fully received, flip the FSM into request-sending state.
    if (!(msg_->flags () & msg_t::more)) {
        _receiving_reply = false;
        _message_begins = true;
    }

    return 0;
}

bool zmq::req_t::xhas_in ()
{
    //  TODO: Duplicates should be removed here.

    if (!_receiving_reply)
        return false;

    return dealer_t::xhas_in ();
}

bool zmq::req_t::xhas_out ()
{
    if (_receiving_reply && _strict)
        return false;

    return dealer_t::xhas_out ();
}

int zmq::req_t::xsetsockopt (int option_,
                             const void *optval_,
                             size_t optvallen_)
{
    const bool is_int = (optvallen_ == sizeof (int));
    int value = 0;
    if (is_int)
        memcpy (&value, optval_, sizeof (int));

    switch (option_) {
        case ZMQ_REQ_CORRELATE:
            if (is_int && value >= 0) {
                _request_id_frames_enabled = (value != 0);
                return 0;
            }
            break;

        case ZMQ_REQ_RELAXED:
            if (is_int && value >= 0) {
                _strict = (value == 0);
                return 0;
            }
            break;

        default:
            break;
    }

    return dealer_t::xsetsockopt (option_, optval_, optvallen_);
}

void zmq::req_t::xpipe_terminated (pipe_t *pipe_)
{
    if (_reply_pipe == pipe_)
        _reply_pipe = NULL;
    dealer_t::xpipe_terminated (pipe_);
}

int zmq::req_t::recv_reply_pipe (msg_t *msg_)
{
    while (true) {
        pipe_t *pipe = NULL;
        const int rc = dealer_t::recvpipe (msg_, &pipe);
        if (rc != 0)
            return rc;
        if (!_reply_pipe || pipe == _reply_pipe)
            return 0;
    }
}

zmq::req_session_t::req_session_t (io_thread_t *io_thread_,
                                   bool connect_,
                                   socket_base_t *socket_,
                                   const options_t &options_,
                                   address_t *addr_) :
    session_base_t (io_thread_, connect_, socket_, options_, addr_),
    _state (bottom)
{
}

zmq::req_session_t::~req_session_t ()
{
}

int zmq::req_session_t::push_msg (msg_t *msg_)
{
    //  Ignore commands, they are processed by the engine and should not
    //  affect the state machine.
    if (unlikely (msg_->flags () & msg_t::command))
        return 0;

    switch (_state) {
        case bottom:
            if (msg_->flags () == msg_t::more) {
                //  In case option ZMQ_CORRELATE is on, allow request_id to be
                //  transfered as first frame (would be too cumbersome to check
                //  whether the option is actually on or not).
                if (msg_->size () == sizeof (uint32_t)) {
                    _state = request_id;
                    return session_base_t::push_msg (msg_);
                }
                if (msg_->size () == 0) {
                    _state = body;
                    return session_base_t::push_msg (msg_);
                }
            }
            break;
        case request_id:
            if (msg_->flags () == msg_t::more && msg_->size () == 0) {
                _state = body;
                return session_base_t::push_msg (msg_);
            }
            break;
        case body:
            if (msg_->flags () == msg_t::more)
                return session_base_t::push_msg (msg_);
            if (msg_->flags () == 0) {
                _state = bottom;
                return session_base_t::push_msg (msg_);
            }
            break;
    }
    errno = EFAULT;
    return -1;
}

void zmq::req_session_t::reset ()
{
    session_base_t::reset ();
    _state = bottom;
}
