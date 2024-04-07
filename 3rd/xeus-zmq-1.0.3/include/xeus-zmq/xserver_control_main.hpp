/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay, Martin Renou          *
* Copyright (c) 2016, QuantStack                                           *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XEUS_SERVER_CONTROL_MAIN_HPP
#define XEUS_SERVER_CONTROL_MAIN_HPP

#include "xeus/xcontrol_messenger.hpp"
#include "xeus/xeus_context.hpp"
#include "xeus/xkernel_configuration.hpp"

#include "xeus-zmq.hpp"
#include "xserver_zmq_split.hpp"

namespace xeus
{
    class XEUS_ZMQ_API xserver_control_main : public xserver_zmq_split
    {
    public:

        xserver_control_main(zmq::context_t& context,
                             const xconfiguration& config,
                             nl::json::error_handler_t eh);
        virtual ~xserver_control_main();

    private:

        void start_server(zmq::multipart_t& wire_msg) override;
    };

    XEUS_ZMQ_API
    std::unique_ptr<xserver> make_xserver_control_main(xcontext& context,
                                                       const xconfiguration& config,
                                                       nl::json::error_handler_t eh = nl::json::error_handler_t::strict);
}

#endif

