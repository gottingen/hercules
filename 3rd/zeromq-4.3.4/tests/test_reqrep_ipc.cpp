/*
    Copyright (c) 2007-2017 Contributors as noted in the AUTHORS file

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

#include "testutil.hpp"
#include "testutil_unity.hpp"

#include <stdlib.h>

SETUP_TEARDOWN_TESTCONTEXT

void test_leak ()
{
    char my_endpoint[256];

    void *sb = test_context_socket (ZMQ_REP);
    bind_loopback_ipc (sb, my_endpoint, sizeof my_endpoint);

    void *sc = test_context_socket (ZMQ_REQ);
    TEST_ASSERT_SUCCESS_ERRNO (zmq_connect (sc, my_endpoint));

    static const char leakymsg[] = "leakymsg";
    send_string_expect_success (sc, leakymsg, 0);

    char *buf = s_recv (sb);
    free (buf);

    test_context_socket_close (sc);

    msleep (SETTLE_TIME);

    send_string_expect_success (sb, leakymsg, 0);

    test_context_socket_close (sb);
}

void test_simple (void)
{
    char my_endpoint[256];

    void *sb = test_context_socket (ZMQ_REP);
    bind_loopback_ipc (sb, my_endpoint, sizeof my_endpoint);

    void *sc = test_context_socket (ZMQ_REQ);
    TEST_ASSERT_SUCCESS_ERRNO (zmq_connect (sc, my_endpoint));

    bounce (sb, sc);

    test_context_socket_close (sc);
    test_context_socket_close (sb);
}

int main (void)
{
    setup_test_environment ();

    UNITY_BEGIN ();
    RUN_TEST (test_simple);
    RUN_TEST (test_leak);
    return UNITY_END ();
}
