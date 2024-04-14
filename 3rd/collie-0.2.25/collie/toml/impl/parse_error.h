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
#if TOML_ENABLE_PARSER

#include <collie/toml/impl/std_except.h>
#include <collie/toml/impl/source_region.h>
#include <collie/toml/impl/print_to_stream.h>
#include <collie/toml/impl/header_start.h>

#if TOML_DOXYGEN || !TOML_EXCEPTIONS
#define TOML_PARSE_ERROR_BASE
#else
#define TOML_PARSE_ERROR_BASE	 : public std::runtime_error
#endif

TOML_NAMESPACE_START
{
	TOML_ABI_NAMESPACE_BOOL(TOML_EXCEPTIONS, ex, noex);

	/// \brief	An error generated when parsing fails.
	///
	/// \remarks This class inherits from std::runtime_error when exceptions are enabled.
	/// 		 The public interface is the same regardless of exception mode.
	class parse_error TOML_PARSE_ERROR_BASE
	{
	  private:
#if !TOML_EXCEPTIONS
		std::string description_;
#endif
		source_region source_;

	  public:
#if TOML_EXCEPTIONS

		TOML_NODISCARD_CTOR
		TOML_ATTR(nonnull)
		parse_error(const char* desc, source_region&& src) noexcept //
			: std::runtime_error{ desc },
			  source_{ std::move(src) }
		{}

		TOML_NODISCARD_CTOR
		TOML_ATTR(nonnull)
		parse_error(const char* desc, const source_region& src) noexcept //
			: parse_error{ desc, source_region{ src } }
		{}

		TOML_NODISCARD_CTOR
		TOML_ATTR(nonnull)
		parse_error(const char* desc, const source_position& position, const source_path_ptr& path = {}) noexcept
			: parse_error{ desc, source_region{ position, position, path } }
		{}

#else

		TOML_NODISCARD_CTOR
		parse_error(std::string&& desc, source_region&& src) noexcept //
			: description_{ std::move(desc) },
			  source_{ std::move(src) }
		{}

		TOML_NODISCARD_CTOR
		parse_error(std::string&& desc, const source_region& src) noexcept //
			: parse_error{ std::move(desc), source_region{ src } }
		{}

		TOML_NODISCARD_CTOR
		parse_error(std::string&& desc, const source_position& position, const source_path_ptr& path = {}) noexcept
			: parse_error{ std::move(desc), source_region{ position, position, path } }
		{}

#endif

		/// \brief	Returns a textual description of the error.
		/// \remark The backing string is guaranteed to be null-terminated.
		TOML_NODISCARD
		std::string_view description() const noexcept
		{
#if TOML_EXCEPTIONS
			return std::string_view{ what() };
#else
			return description_;
#endif
		}

		/// \brief	Returns the region of the source document responsible for the error.
		TOML_NODISCARD
		const source_region& source() const noexcept
		{
			return source_;
		}

		/// \brief	Prints a parse_error to a stream.
		///
		/// \detail \cpp
		/// try
		/// {
		/// 	auto tbl = toml::parse("enabled = trUe"sv);
		/// }
		/// catch (const toml::parse_error & err)
		/// {
		/// 	std::cerr << "Parsing failed:\n"sv << err << "\n";
		/// }
		/// \ecpp
		///
		/// \out
		/// Parsing failed:
		/// Encountered unexpected character while parsing boolean; expected 'true', saw 'trU'
		///		(error occurred at line 1, column 13)
		/// \eout
		///
		/// \tparam Char The output stream's underlying character type. Must be 1 byte in size.
		/// \param 	lhs	The stream.
		/// \param 	rhs	The parse_error.
		///
		/// \returns	The input stream.
		friend std::ostream& operator<<(std::ostream& lhs, const parse_error& rhs)
		{
			impl::print_to_stream(lhs, rhs.description());
			impl::print_to_stream(lhs, "\n\t(error occurred at "sv);
			impl::print_to_stream(lhs, rhs.source());
			impl::print_to_stream(lhs, ")"sv);
			return lhs;
		}
	};

	TOML_ABI_NAMESPACE_END; // TOML_EXCEPTIONS
}
TOML_NAMESPACE_END;

#undef TOML_PARSE_ERROR_BASE

#include <collie/toml/impl/header_end.h>
#endif // TOML_ENABLE_PARSER
