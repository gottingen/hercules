// Copyright 2024 The Elastic-AI Authors.
// part of Elastic AI Search
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

#pragma once

#include <collie/toml/impl/preprocessor.h>
#if TOML_ENABLE_FORMATTERS

#include <collie/toml/impl/formatter.h>
#include <collie/toml/impl/header_start.h>

TOML_NAMESPACE_START
{
	/// \brief	A wrapper for printing TOML objects out to a stream as formatted YAML.
	///
	/// \availability This class is only available when #TOML_ENABLE_FORMATTERS is enabled.
	///
	/// \detail \cpp
	/// auto some_toml = toml::parse(R"(
	///		[fruit]
	///		apple.color = "red"
	///		apple.taste.sweet = true
	///
	///		[fruit.apple.texture]
	///		smooth = true
	/// )"sv);
	///	std::cout << toml::yaml_formatter{ some_toml } << "\n";
	/// \ecpp
	///
	/// \out
	/// fruit:
	///   apple:
	///     color: red
	///     taste:
	///       sweet: true
	///     texture:
	///       smooth: true
	/// \eout
	class TOML_EXPORTED_CLASS yaml_formatter : impl::formatter
	{
	  private:
		/// \cond

		using base = impl::formatter;

		TOML_EXPORTED_MEMBER_FUNCTION
		void print_yaml_string(const value<std::string>&);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print(const toml::table&, bool = false);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print(const toml::array&, bool = false);

		TOML_EXPORTED_MEMBER_FUNCTION
		void print();

		static constexpr impl::formatter_constants constants = {
			//
			format_flags::quote_dates_and_times | format_flags::indentation, // mandatory
			format_flags::allow_multi_line_strings,							 // ignored
			".inf"sv,
			"-.inf"sv,
			".NAN"sv,
			"true"sv,
			"false"sv
		};

		/// \endcond

	  public:
		/// \brief	The default flags for a yaml_formatter.
		static constexpr format_flags default_flags = constants.mandatory_flags			  //
													| format_flags::allow_literal_strings //
													| format_flags::allow_unicode_strings //
													| format_flags::allow_octal_integers  //
													| format_flags::allow_hexadecimal_integers;

		/// \brief	Constructs a YAML formatter and binds it to a TOML object.
		///
		/// \param 	source	The source TOML object.
		/// \param 	flags 	Format option flags.
		TOML_NODISCARD_CTOR
		explicit yaml_formatter(const toml::node& source, format_flags flags = default_flags) noexcept
			: base{ &source, nullptr, constants, { flags, "  "sv } }
		{}

#if TOML_DOXYGEN || (TOML_ENABLE_PARSER && !TOML_EXCEPTIONS)

		/// \brief	Constructs a YAML formatter and binds it to a toml::parse_result.
		///
		/// \availability This constructor is only available when exceptions are disabled.
		///
		/// \attention Formatting a failed parse result will simply dump the error message out as-is.
		///		This will not be valid YAML, but at least gives you something to log or show up in diagnostics:
		/// \cpp
		/// std::cout << toml::yaml_formatter{ toml::parse("a = 'b'"sv) } // ok
		///           << "\n\n"
		///           << toml::yaml_formatter{ toml::parse("a = "sv) } // malformed
		///           << "\n";
		/// \ecpp
		/// \out
		/// a: b
		///
		/// Error while parsing key-value pair: encountered end-of-file
		///         (error occurred at line 1, column 5)
		/// \eout
		/// Use the library with exceptions if you want to avoid this scenario.
		///
		/// \param 	result	The parse result.
		/// \param 	flags 	Format option flags.
		TOML_NODISCARD_CTOR
		explicit yaml_formatter(const toml::parse_result& result, format_flags flags = default_flags) noexcept
			: base{ nullptr, &result, constants, { flags, "  "sv } }
		{}

#endif

		/// \brief	Prints the bound TOML object out to the stream as YAML.
		friend std::ostream& TOML_CALLCONV operator<<(std::ostream& lhs, yaml_formatter& rhs)
		{
			rhs.attach(lhs);
			rhs.print();
			rhs.detach();
			return lhs;
		}

		/// \brief	Prints the bound TOML object out to the stream as YAML (rvalue overload).
		friend std::ostream& TOML_CALLCONV operator<<(std::ostream& lhs, yaml_formatter&& rhs)
		{
			return lhs << rhs; // as lvalue
		}
	};
}
TOML_NAMESPACE_END;

#include <collie/toml/impl/header_end.h>
#endif // TOML_ENABLE_FORMATTERS
