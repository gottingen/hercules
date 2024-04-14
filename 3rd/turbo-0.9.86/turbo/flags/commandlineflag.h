// Copyright 2023 The titan-search Authors.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef TURBO_FLAGS_COMMANDLINEFLAG_H_
#define TURBO_FLAGS_COMMANDLINEFLAG_H_

#include <memory>
#include <string>
#include <optional>
#include "turbo/platform/port.h"
#include "turbo/base/internal/fast_type_id.h"
#include "turbo/flags/internal/commandlineflag.h"
#include "turbo/strings/string_view.h"

namespace turbo {

    namespace flags_internal {
        class PrivateHandleAccessor;
    }  // namespace flags_internal

    // CommandLineFlag
    //
    // This type acts as a type-erased handle for an instance of an turbo Flag and
    // holds reflection information pertaining to that flag. Use CommandLineFlag to
    // access a flag's name, location, help string etc.
    //
    // To obtain an turbo::CommandLineFlag, invoke `turbo::find_command_line_flag()`
    // passing it the flag name string.
    //
    // Example:
    //
    //   // Obtain reflection handle for a flag named "flagname".
    //   const turbo::CommandLineFlag* my_flag_data =
    //        turbo::find_command_line_flag("flagname");
    //
    //   // Now you can get flag info from that reflection handle.
    //   std::string flag_location = my_flag_data->Filename();
    //   ...
    class CommandLineFlag {
    public:
        constexpr CommandLineFlag() = default;

        // Not copyable/assignable.
        CommandLineFlag(const CommandLineFlag &) = delete;

        CommandLineFlag &operator=(const CommandLineFlag &) = delete;

        // turbo::CommandLineFlag::is_of_type()
        //
        // Return true iff flag has type T.
        template<typename T>
        inline bool is_of_type() const {
            return type_id() == base_internal::FastTypeId<T>();
        }

        // turbo::CommandLineFlag::TryGet()
        //
        // Attempts to retrieve the flag value. Returns value on success,
        // std::nullopt otherwise.
        template<typename T>
        std::optional<T> try_get() const {
            if (is_retired() || !is_of_type<T>()) {
                return std::nullopt;
            }

            // Implementation notes:
            //
            // We are wrapping a union around the value of `T` to serve three purposes:
            //
            //  1. `U.value` has correct size and alignment for a value of type `T`
            //  2. The `U.value` constructor is not invoked since U's constructor does
            //     not do it explicitly.
            //  3. The `U.value` destructor is invoked since U's destructor does it
            //     explicitly. This makes `U` a kind of RAII wrapper around non default
            //     constructible value of T, which is destructed when we leave the
            //     scope. We do need to destroy U.value, which is constructed by
            //     CommandLineFlag::Read even though we left it in a moved-from state
            //     after std::move.
            //
            // All of this serves to avoid requiring `T` being default constructible.
            union U {
                T value;

                U() {}

                ~U() { value.~T(); }
            };
            U u;

            Read(&u.value);
            // allow retired flags to be "read", so we can report invalid access.
            if (is_retired()) {
                return std::nullopt;
            }
            return std::move(u.value);
        }

        // turbo::CommandLineFlag::Name()
        //
        // Returns name of this flag.
        virtual std::string_view name() const = 0;

        // turbo::CommandLineFlag::Filename()
        //
        // Returns name of the file where this flag is defined.
        virtual std::string filename() const = 0;

        // turbo::CommandLineFlag::Help()
        //
        // Returns help message associated with this flag.
        virtual std::string help() const = 0;

        // turbo::CommandLineFlag::is_retired()
        //
        // Returns true iff this object corresponds to retired flag.
        virtual bool is_retired() const;

        // turbo::CommandLineFlag::default_value()
        //
        // Returns the default value for this flag.
        virtual std::string default_value() const = 0;

        // turbo::CommandLineFlag::current_value()
        //
        // Returns the current value for this flag.
        virtual std::string current_value() const = 0;

        // turbo::CommandLineFlag::parse_from()
        //
        // Sets the value of the flag based on specified string `value`. If the flag
        // was successfully set to new value, it returns true. Otherwise, sets `error`
        // to indicate the error, leaves the flag unchanged, and returns false.
        bool parse_from(std::string_view value, std::string *error);

    protected:
        ~CommandLineFlag() = default;

    private:
        friend class flags_internal::PrivateHandleAccessor;

        // Sets the value of the flag based on specified string `value`. If the flag
        // was successfully set to new value, it returns true. Otherwise, sets `error`
        // to indicate the error, leaves the flag unchanged, and returns false. There
        // are three ways to set the flag's value:
        //  * Update the current flag value
        //  * Update the flag's default value
        //  * Update the current flag value if it was never set before
        // The mode is selected based on `set_mode` parameter.
        virtual bool parse_from(std::string_view value,
                               flags_internal::FlagSettingMode set_mode,
                               flags_internal::ValueSource source,
                               std::string &error) = 0;

        // Returns id of the flag's value type.
        virtual flags_internal::FlagFastTypeId type_id() const = 0;

        // Interface to save flag to some persistent state. Returns current flag state
        // or nullptr if flag does not support saving and restoring a state.
        virtual std::unique_ptr<flags_internal::FlagStateInterface> save_state() = 0;

        // Copy-construct a new value of the flag's type in a memory referenced by
        // the dst based on the current flag's value.
        virtual void Read(void *dst) const = 0;

        // To be deleted. Used to return true if flag's current value originated from
        // command line.
        virtual bool is_specified_on_command_line() const = 0;

        // Validates supplied value using validator or parseflag routine
        virtual bool validate_input_value(std::string_view value) const = 0;

        // Checks that flags default value can be converted to string and back to the
        // flag's value type.
        virtual void check_default_value_parsing_roundtrip() const = 0;
    };


}  // namespace turbo

#endif  // TURBO_FLAGS_COMMANDLINEFLAG_H_
