//
// Created by jeff on 23-7-14.
//

#ifndef TURBO_FILES_FILE_OPTION_H_
#define TURBO_FILES_FILE_OPTION_H_

#include <cstdint>
#include <string>

namespace turbo {

    /**
     * @ingroup turbo_files_operation
     * @brief FileOption is used to configure the behavior of the File class.
     *        it is used in the File open.
     *        open_tries: the number of times to try to open the file.
     *        open_interval: the interval between each open attempt. in milliseconds.
     *        create_dir_if_miss: if the directory does not exist, create it.
     *        prevent_child: if the directory does not exist, create it.
     *        kDefault:
     *        open_tries = 1
     *        open_interval = 0(ms)
     *        create_dir_if_miss = false
     *        prevent_child = true
     *        The default behavior is to try to open the file once, and if it fails, it will not be opened again.
     *        If the directory does not exist, it will not be created.
     *        by default, user no need to set this option. the classes and functions in the turbo/files using
     *        FileOption::kDefault as default value.
     *        if you want to change the default behavior, you can set the FileOption::kDefault to your own
     *        Example:
     *        @note {.cpp}
     *        FileOption option;
     *        option.open_tries = 3;
     *        option.open_interval = 1000;
     *        option.create_dir_if_miss = true;
     *        option.prevent_child = false;
     *
     *        SequenceWriteFile file;
     *        file.set_option(option);
     *        auto rs = file.open("path/to/file");
     *        @endnote
     *        the above code will try to open the file 3 times, and the interval between each open attempt is 1 second.
     *        if the directory does not exist, it will be created.
     */
    struct FileOption {
        int32_t                  open_tries{1};
        uint32_t                 open_interval{0};
        int                      mode{0644};
        bool                     create_dir_if_miss{false};
        bool                     prevent_child{true};
        static const FileOption  kDefault;
    };
}  // namespace turbo

#endif  // TURBO_FILES_FILE_OPTION_H_
