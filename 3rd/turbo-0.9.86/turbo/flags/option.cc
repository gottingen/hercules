// Copyright 2023 The Turbo Authors.
//
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

// This include is only needed for IDEs to discover symbols
#include "turbo/flags/option.h"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace turbo {

    Option *Option::expected(int value) {
        if (value < 0) {
            expected_min_ = -value;
            if (expected_max_ < expected_min_) {
                expected_max_ = expected_min_;
            }
            allow_extra_args_ = true;
            flag_like_ = false;
        } else if (value == detail::expected_max_vector_size) {
            expected_min_ = 1;
            expected_max_ = detail::expected_max_vector_size;
            allow_extra_args_ = true;
            flag_like_ = false;
        } else {
            expected_min_ = value;
            expected_max_ = value;
            flag_like_ = (expected_min_ == 0);
        }
        return this;
    }

    Option *Option::expected(int value_min, int value_max) {
        if (value_min < 0) {
            value_min = -value_min;
        }

        if (value_max < 0) {
            value_max = detail::expected_max_vector_size;
        }
        if (value_max < value_min) {
            expected_min_ = value_max;
            expected_max_ = value_min;
        } else {
            expected_max_ = value_max;
            expected_min_ = value_min;
        }

        return this;
    }

    Option *Option::check(Validator validator, const std::string &validator_name) {
        validator.non_modifying();
        validators_.push_back(std::move(validator));
        if (!validator_name.empty())
            validators_.back().name(validator_name);
        return this;
    }

    Option *Option::check(std::function<std::string(const std::string &)> Validator,
                          std::string Validator_description,
                          std::string Validator_name) {
        validators_.emplace_back(Validator, std::move(Validator_description), std::move(Validator_name));
        validators_.back().non_modifying();
        return this;
    }

    Option *Option::transform(Validator Validator, const std::string &Validator_name) {
        validators_.insert(validators_.begin(), std::move(Validator));
        if (!Validator_name.empty())
            validators_.front().name(Validator_name);
        return this;
    }

    Option *Option::transform(const std::function<std::string(std::string)> &func,
                              std::string transform_description,
                              std::string transform_name) {
        validators_.insert(validators_.begin(),
                           Validator(
                                   [func](std::string &val) {
                                       val = func(val);
                                       return std::string{};
                                   },
                                   std::move(transform_description),
                                   std::move(transform_name)));

        return this;
    }

    Option *Option::each(const std::function<void(std::string)> &func) {
        validators_.emplace_back(
                [func](std::string &inout) {
                    func(inout);
                    return std::string{};
                },
                std::string{});
        return this;
    }

    Validator *Option::get_validator(const std::string &Validator_name) {
        for (auto &Validator: validators_) {
            if (Validator_name == Validator.get_name()) {
                return &Validator;
            }
        }
        if ((Validator_name.empty()) && (!validators_.empty())) {
            return &(validators_.front());
        }
        throw OptionNotFound(std::string{"Validator "} + Validator_name + " Not Found");
    }

    Validator *Option::get_validator(int index) {
        // This is an signed int so that it is not equivalent to a pointer.
        if (index >= 0 && index < static_cast<int>(validators_.size())) {
            return &(validators_[static_cast<decltype(validators_)::size_type>(index)]);
        }
        throw OptionNotFound("Validator index is not valid");
    }

    bool Option::remove_needs(Option *opt) {
        auto iterator = std::find(std::begin(needs_), std::end(needs_), opt);

        if (iterator == std::end(needs_)) {
            return false;
        }
        needs_.erase(iterator);
        return true;
    }

    Option *Option::excludes(Option *opt) {
        if (opt == this) {
            throw (IncorrectConstruction("and option cannot exclude itself"));
        }
        excludes_.insert(opt);

        // Help text should be symmetric - excluding a should exclude b
        opt->excludes_.insert(this);

        // Ignoring the insert return value, excluding twice is now allowed.
        // (Mostly to allow both directions to be excluded by user, even though the library does it for you.)

        return this;
    }

    bool Option::remove_excludes(Option *opt) {
        auto iterator = std::find(std::begin(excludes_), std::end(excludes_), opt);

        if (iterator == std::end(excludes_)) {
            return false;
        }
        excludes_.erase(iterator);
        return true;
    }


    Option *Option::multi_option_policy(MultiOptionPolicy value) {
        if (value != multi_option_policy_) {
            if (multi_option_policy_ == MultiOptionPolicy::Throw && expected_max_ == detail::expected_max_vector_size &&
                expected_min_ > 1) {  // this bizarre condition is to maintain backwards compatibility
                // with the previous behavior of expected_ with vectors
                expected_max_ = expected_min_;
            }
            multi_option_policy_ = value;
            current_option_state_ = option_state::parsing;
        }
        return this;
    }

    [[nodiscard]] std::string Option::get_name(bool positional, bool all_options) const {
        if (get_group().empty())
            return {};  // Hidden

        if (all_options) {

            std::vector<std::string> name_list;

            /// The all list will never include a positional unless asked or that's the only name.
            if ((positional && (!pname_.empty())) || (snames_.empty() && lnames_.empty())) {
                name_list.push_back(pname_);
            }
            if ((get_items_expected() == 0) && (!fnames_.empty())) {
                for (const std::string &sname: snames_) {
                    name_list.push_back("-" + sname);
                    if (check_fname(sname)) {
                        name_list.back() += "{" + get_flag_value(sname, "") + "}";
                    }
                }

                for (const std::string &lname: lnames_) {
                    name_list.push_back("--" + lname);
                    if (check_fname(lname)) {
                        name_list.back() += "{" + get_flag_value(lname, "") + "}";
                    }
                }
            } else {
                for (const std::string &sname: snames_)
                    name_list.push_back("-" + sname);

                for (const std::string &lname: lnames_)
                    name_list.push_back("--" + lname);
            }

            return detail::join(name_list);
        }

        // This returns the positional name no matter what
        if (positional)
            return pname_;

        // Prefer long name
        if (!lnames_.empty())
            return std::string(2, '-') + lnames_[0];

        // Or short name if no long name
        if (!snames_.empty())
            return std::string(1, '-') + snames_[0];

        // If positional is the only name, it's okay to use that
        return pname_;
    }

    void Option::run_callback() {
        if (force_callback_ && results_.empty()) {
            add_result(default_str_);
        }
        if (current_option_state_ == option_state::parsing) {
            _validate_results(results_);
            current_option_state_ = option_state::validated;
        }

        if (current_option_state_ < option_state::reduced) {
            _reduce_results(proc_results_, results_);
            current_option_state_ = option_state::reduced;
        }
        if (current_option_state_ >= option_state::reduced) {
            current_option_state_ = option_state::callback_run;
            if (!(callback_)) {
                return;
            }
            const results_t &send_results = proc_results_.empty() ? results_ : proc_results_;
            bool local_result = callback_(send_results);

            if (!local_result)
                throw ConversionError(get_name(), results_);
        }
    }

    [[nodiscard]] const std::string &Option::matching_name(const Option &other) const {
        static const std::string estring;
        for (const std::string &sname: snames_)
            if (other.check_sname(sname))
                return sname;
        for (const std::string &lname: lnames_)
            if (other.check_lname(lname))
                return lname;

        if (ignore_case_ ||
            ignore_underscore_) {  // We need to do the inverse, in case we are ignore_case or ignore underscore
            for (const std::string &sname: other.snames_)
                if (check_sname(sname))
                    return sname;
            for (const std::string &lname: other.lnames_)
                if (check_lname(lname))
                    return lname;
        }
        return estring;
    }

    [[nodiscard]] bool Option::check_name(const std::string &name) const {

        if (name.length() > 2 && name[0] == '-' && name[1] == '-')
            return check_lname(name.substr(2));
        if (name.length() > 1 && name.front() == '-')
            return check_sname(name.substr(1));
        if (!pname_.empty()) {
            std::string local_pname = pname_;
            std::string local_name = name;
            if (ignore_underscore_) {
                local_pname = detail::remove_underscore(local_pname);
                local_name = detail::remove_underscore(local_name);
            }
            if (ignore_case_) {
                local_pname = detail::to_lower(local_pname);
                local_name = detail::to_lower(local_name);
            }
            if (local_name == local_pname) {
                return true;
            }
        }

        if (!envname_.empty()) {
            // this needs to be the original since envname_ shouldn't match on case insensitivity
            return (name == envname_);
        }
        return false;
    }

    [[nodiscard]] std::string Option::get_flag_value(const std::string &name,
                                                     std::string input_value) const {
        static const std::string trueString{"true"};
        static const std::string falseString{"false"};
        static const std::string emptyString{"{}"};
        // check for disable flag override_
        if (disable_flag_override_) {
            if (!((input_value.empty()) || (input_value == emptyString))) {
                auto default_ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
                if (default_ind >= 0) {
                    // We can static cast this to std::size_t because it is more than 0 in this block
                    if (default_flag_values_[static_cast<std::size_t>(default_ind)].second != input_value) {
                        throw (ArgumentMismatch::FlagOverride(name));
                    }
                } else {
                    if (input_value != trueString) {
                        throw (ArgumentMismatch::FlagOverride(name));
                    }
                }
            }
        }
        auto ind = detail::find_member(name, fnames_, ignore_case_, ignore_underscore_);
        if ((input_value.empty()) || (input_value == emptyString)) {
            if (flag_like_) {
                return (ind < 0) ? trueString : default_flag_values_[static_cast<std::size_t>(ind)].second;
            }
            return (ind < 0) ? default_str_ : default_flag_values_[static_cast<std::size_t>(ind)].second;
        }
        if (ind < 0) {
            return input_value;
        }
        if (default_flag_values_[static_cast<std::size_t>(ind)].second == falseString) {
            try {
                auto val = detail::to_flag_value(input_value);
                return (val == 1) ? falseString : (val == (-1) ? trueString : std::to_string(-val));
            } catch (const std::invalid_argument &) {
                return input_value;
            }
        } else {
            return input_value;
        }
    }

    Option *Option::add_result(std::string s) {
        _add_result(std::move(s), results_);
        current_option_state_ = option_state::parsing;
        return this;
    }

    Option *Option::add_result(std::string s, int &results_added) {
        results_added = _add_result(std::move(s), results_);
        current_option_state_ = option_state::parsing;
        return this;
    }

    Option *Option::add_result(std::vector<std::string> s) {
        current_option_state_ = option_state::parsing;
        for (auto &str: s) {
            _add_result(std::move(str), results_);
        }
        return this;
    }

    [[nodiscard]] results_t Option::reduced_results() const {
        results_t res = proc_results_.empty() ? results_ : proc_results_;
        if (current_option_state_ < option_state::reduced) {
            if (current_option_state_ == option_state::parsing) {
                res = results_;
                _validate_results(res);
            }
            if (!res.empty()) {
                results_t extra;
                _reduce_results(extra, res);
                if (!extra.empty()) {
                    res = std::move(extra);
                }
            }
        }
        return res;
    }

    Option *Option::type_size(int option_type_size) {
        if (option_type_size < 0) {
            // this section is included for backwards compatibility
            type_size_max_ = -option_type_size;
            type_size_min_ = -option_type_size;
            expected_max_ = detail::expected_max_vector_size;
        } else {
            type_size_max_ = option_type_size;
            if (type_size_max_ < detail::expected_max_vector_size) {
                type_size_min_ = option_type_size;
            } else {
                inject_separator_ = true;
            }
            if (type_size_max_ == 0)
                required_ = false;
        }
        return this;
    }

    Option *Option::type_size(int option_type_size_min, int option_type_size_max) {
        if (option_type_size_min < 0 || option_type_size_max < 0) {
            // this section is included for backwards compatibility
            expected_max_ = detail::expected_max_vector_size;
            option_type_size_min = (std::abs)(option_type_size_min);
            option_type_size_max = (std::abs)(option_type_size_max);
        }

        if (option_type_size_min > option_type_size_max) {
            type_size_max_ = option_type_size_min;
            type_size_min_ = option_type_size_max;
        } else {
            type_size_min_ = option_type_size_min;
            type_size_max_ = option_type_size_max;
        }
        if (type_size_max_ == 0) {
            required_ = false;
        }
        if (type_size_max_ >= detail::expected_max_vector_size) {
            inject_separator_ = true;
        }
        return this;
    }

    [[nodiscard]] std::string Option::get_type_name() const {
        std::string full_type_name = type_name_();
        if (!validators_.empty()) {
            for (const auto &Validator: validators_) {
                std::string vtype = Validator.get_description();
                if (!vtype.empty()) {
                    full_type_name += ":" + vtype;
                }
            }
        }
        return full_type_name;
    }

    void Option::_validate_results(results_t &res) const {
        // Run the Validators (can change the string)
        if (!validators_.empty()) {
            if (type_size_max_ > 1) {  // in this context index refers to the index in the type
                int index = 0;
                if (get_items_expected_max() < static_cast<int>(res.size()) &&
                    multi_option_policy_ == turbo::MultiOptionPolicy::TakeLast) {
                    // create a negative index for the earliest ones
                    index = get_items_expected_max() - static_cast<int>(res.size());
                }

                for (std::string &result: res) {
                    if (detail::is_separator(result) && type_size_max_ != type_size_min_ && index >= 0) {
                        index = 0;  // reset index for variable size chunks
                        continue;
                    }
                    auto err_msg = _validate(result, (index >= 0) ? (index % type_size_max_) : index);
                    if (!err_msg.empty())
                        throw ValidationError(get_name(), err_msg);
                    ++index;
                }
            } else {
                int index = 0;
                if (expected_max_ < static_cast<int>(res.size()) &&
                    multi_option_policy_ == turbo::MultiOptionPolicy::TakeLast) {
                    // create a negative index for the earliest ones
                    index = expected_max_ - static_cast<int>(res.size());
                }
                for (std::string &result: res) {
                    auto err_msg = _validate(result, index);
                    ++index;
                    if (!err_msg.empty())
                        throw ValidationError(get_name(), err_msg);
                }
            }
        }
    }

    void Option::_reduce_results(results_t &out, const results_t &original) const {

        // max num items expected or length of vector, always at least 1
        // Only valid for a trimming policy

        out.clear();
        // Operation depends on the policy setting
        switch (multi_option_policy_) {
            case MultiOptionPolicy::TakeAll:
                break;
            case MultiOptionPolicy::TakeLast: {
                // Allow multi-option sizes (including 0)
                std::size_t trim_size = std::min<std::size_t>(
                        static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
                if (original.size() != trim_size) {
                    out.assign(original.end() - static_cast<results_t::difference_type>(trim_size), original.end());
                }
            }
                break;
            case MultiOptionPolicy::TakeFirst: {
                std::size_t trim_size = std::min<std::size_t>(
                        static_cast<std::size_t>(std::max<int>(get_items_expected_max(), 1)), original.size());
                if (original.size() != trim_size) {
                    out.assign(original.begin(), original.begin() + static_cast<results_t::difference_type>(trim_size));
                }
            }
                break;
            case MultiOptionPolicy::Join:
                if (results_.size() > 1) {
                    out.push_back(detail::join(original, std::string(1, (delimiter_ == '\0') ? '\n' : delimiter_)));
                }
                break;
            case MultiOptionPolicy::Sum:
                out.push_back(detail::sum_string_vector(original));
                break;
            case MultiOptionPolicy::Throw:
            default: {
                auto num_min = static_cast<std::size_t>(get_items_expected_min());
                auto num_max = static_cast<std::size_t>(get_items_expected_max());
                if (num_min == 0) {
                    num_min = 1;
                }
                if (num_max == 0) {
                    num_max = 1;
                }
                if (original.size() < num_min) {
                    throw ArgumentMismatch::AtLeast(get_name(), static_cast<int>(num_min), original.size());
                }
                if (original.size() > num_max) {
                    throw ArgumentMismatch::AtMost(get_name(), static_cast<int>(num_max), original.size());
                }
                break;
            }
        }
        // this check is to allow an empty vector in certain circumstances but not if expected is not zero.
        // {} is the indicator for an empty container
        if (out.empty()) {
            if (original.size() == 1 && original[0] == "{}" && get_items_expected_min() > 0) {
                out.push_back("{}");
                out.push_back("%%");
            }
        } else if (out.size() == 1 && out[0] == "{}" && get_items_expected_min() > 0) {
            out.push_back("%%");
        }
    }

    std::string Option::_validate(std::string &result, int index) const {
        std::string err_msg;
        if (result.empty() && expected_min_ == 0) {
            // an empty with nothing expected is allowed
            return err_msg;
        }
        for (const auto &vali: validators_) {
            auto v = vali.get_application_index();
            if (v == -1 || v == index) {
                try {
                    err_msg = vali(result);
                } catch (const ValidationError &err) {
                    err_msg = err.what();
                }
                if (!err_msg.empty())
                    break;
            }
        }

        return err_msg;
    }

    int Option::_add_result(std::string &&result, std::vector<std::string> &res) const {
        int result_count = 0;
        if (allow_extra_args_ && !result.empty() && result.front() == '[' &&
            result.back() == ']') {  // this is now a vector string likely from the default or user entry
            result.pop_back();

            for (auto &var: turbo::detail::split(result.substr(1), ',')) {
                if (!var.empty()) {
                    result_count += _add_result(std::move(var), res);
                }
            }
            return result_count;
        }
        if (delimiter_ == '\0') {
            res.push_back(std::move(result));
            ++result_count;
        } else {
            if ((result.find_first_of(delimiter_) != std::string::npos)) {
                for (const auto &var: turbo::detail::split(result, delimiter_)) {
                    if (!var.empty()) {
                        res.push_back(var);
                        ++result_count;
                    }
                }
            } else {
                res.push_back(std::move(result));
                ++result_count;
            }
        }
        return result_count;
    }
}  // namespace turbo
