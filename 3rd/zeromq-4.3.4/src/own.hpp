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

#ifndef __ZMQ_OWN_HPP_INCLUDED__
#define __ZMQ_OWN_HPP_INCLUDED__

#include <set>

#include "object.hpp"
#include "options.hpp"
#include "atomic_counter.hpp"
#include "stdint.hpp"

namespace zmq
{
class ctx_t;
class io_thread_t;

//  Base class for objects forming a part of ownership hierarchy.
//  It handles initialisation and destruction of such objects.

class own_t : public object_t
{
  public:
    //  Note that the owner is unspecified in the constructor.
    //  It'll be supplied later on when the object is plugged in.

    //  The object is not living within an I/O thread. It has it's own
    //  thread outside of 0MQ infrastructure.
    own_t (zmq::ctx_t *parent_, uint32_t tid_);

    //  The object is living within I/O thread.
    own_t (zmq::io_thread_t *io_thread_, const options_t &options_);

    //  When another owned object wants to send command to this object
    //  it calls this function to let it know it should not shut down
    //  before the command is delivered.
    void inc_seqnum ();

    //  Use following two functions to wait for arbitrary events before
    //  terminating. Just add number of events to wait for using
    //  register_tem_acks functions. When event occurs, call
    //  remove_term_ack. When number of pending acks reaches zero
    //  object will be deallocated.
    void register_term_acks (int count_);
    void unregister_term_ack ();

  protected:
    //  Launch the supplied object and become its owner.
    void launch_child (own_t *object_);

    //  Terminate owned object
    void term_child (own_t *object_);

    //  Ask owner object to terminate this object. It may take a while
    //  while actual termination is started. This function should not be
    //  called more than once.
    void terminate ();

    //  Returns true if the object is in process of termination.
    bool is_terminating () const;

    //  Derived object destroys own_t. There's no point in allowing
    //  others to invoke the destructor. At the same time, it has to be
    //  virtual so that generic own_t deallocation mechanism destroys
    //  specific type of the owned object correctly.
    ~own_t () ZMQ_OVERRIDE;

    //  Term handler is protected rather than private so that it can
    //  be intercepted by the derived class. This is useful to add custom
    //  steps to the beginning of the termination process.
    void process_term (int linger_) ZMQ_OVERRIDE;

    //  A place to hook in when physical destruction of the object
    //  is to be delayed.
    virtual void process_destroy ();

    //  Socket options associated with this object.
    options_t options;

  private:
    //  Set owner of the object
    void set_owner (own_t *owner_);

    //  Handlers for incoming commands.
    void process_own (own_t *object_) ZMQ_OVERRIDE;
    void process_term_req (own_t *object_) ZMQ_OVERRIDE;
    void process_term_ack () ZMQ_OVERRIDE;
    void process_seqnum () ZMQ_OVERRIDE;

    //  Check whether all the pending term acks were delivered.
    //  If so, deallocate this object.
    void check_term_acks ();

    //  True if termination was already initiated. If so, we can destroy
    //  the object if there are no more child objects or pending term acks.
    bool _terminating;

    //  Sequence number of the last command sent to this object.
    atomic_counter_t _sent_seqnum;

    //  Sequence number of the last command processed by this object.
    uint64_t _processed_seqnum;

    //  Socket owning this object. It's responsible for shutting down
    //  this object.
    own_t *_owner;

    //  List of all objects owned by this socket. We are responsible
    //  for deallocating them before we quit.
    typedef std::set<own_t *> owned_t;
    owned_t _owned;

    //  Number of events we have to get before we can destroy the object.
    int _term_acks;

    ZMQ_NON_COPYABLE_NOR_MOVABLE (own_t)
};
}

#endif
