/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <iostream>
#include <memory>

#include "xeus/xeus_context.hpp"
#include "xeus/xkernel.hpp"
#include "xeus/xkernel_configuration.hpp"
#include "xeus-zmq/xserver_zmq.hpp"
#include "xmock_interpreter.hpp"

int main(int argc, char* argv[])
{
    std::string file_name = (argc == 1) ? "connection.json" : argv[2];
    xeus::xconfiguration config = xeus::load_configuration(file_name);

    auto context = xeus::make_context<zmq::context_t>();

    using interpreter_ptr = std::unique_ptr<xeus::xmock_interpreter>;
    interpreter_ptr interpreter = interpreter_ptr(new xeus::xmock_interpreter());
    xeus::xkernel kernel(config,
                         xeus::get_user_name(),
                         std::move(context),
                         std::move(interpreter),
                         xeus::make_xserver_zmq);
    std::cout << "starting kernel" << std::endl;
    kernel.start();

    return 0;
}
