// Copyright (c) 2017-2024, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <collie/cli/cli.h>
#include <string>

int main(int argc, char **argv) {

    collie::App app("Vision Application");
    app.set_help_all_flag("--help-all", "Expand all help");
    app.add_flag("--version", "Get version");

    collie::App *cameraApp = app.add_subcommand("camera", "Configure the app camera");
    cameraApp->require_subcommand(0, 1);  // 0 (default) or 1 camera

    std::string mvcamera_config_file = "mvcamera_config.json";
    collie::App *mvcameraApp = cameraApp->add_subcommand("mvcamera", "MatrixVision Camera Configuration");
    mvcameraApp->add_option("-c,--config", mvcamera_config_file, "Config filename")
        ->capture_default_str()
        ->check(collie::ExistingFile);

    std::string mock_camera_path;
    collie::App *mockcameraApp = cameraApp->add_subcommand("mock", "Mock Camera Configuration");
    mockcameraApp->add_option("-p,--path", mock_camera_path, "Path")->required()->check(collie::ExistingPath);

    COLLIE_CLI_PARSE(app, argc, argv);
}
