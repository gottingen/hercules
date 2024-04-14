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
//# {{
#if !TOML_IMPLEMENTATION
#error This is an implementation-only header.
#endif
//# }}
#if TOML_ENABLE_FORMATTERS

#include <collie/toml/impl/json_formatter.h>
#include <collie/toml/impl/print_to_stream.h>
#include <collie/toml/impl/table.h>
#include <collie/toml/impl/array.h>
#include <collie/toml/impl/header_start.h>

TOML_NAMESPACE_START
{
	TOML_EXTERNAL_LINKAGE
	void json_formatter::print(const toml::table& tbl)
	{
		if (tbl.empty())
		{
			print_unformatted("{}"sv);
			return;
		}

		print_unformatted('{');

		if (indent_sub_tables())
			increase_indent();
		bool first = false;
		for (auto&& [k, v] : tbl)
		{
			if (first)
				print_unformatted(',');
			first = true;
			print_newline(true);
			print_indent();

			print_string(k.str(), false);
			print_unformatted(" : "sv);

			const auto type = v.type();
			TOML_ASSUME(type != node_type::none);
			switch (type)
			{
				case node_type::table: print(*reinterpret_cast<const table*>(&v)); break;
				case node_type::array: print(*reinterpret_cast<const array*>(&v)); break;
				default: print_value(v, type);
			}
		}
		if (indent_sub_tables())
			decrease_indent();
		print_newline(true);
		print_indent();

		print_unformatted('}');
	}

	TOML_EXTERNAL_LINKAGE
	void json_formatter::print(const toml::array& arr)
	{
		if (arr.empty())
		{
			print_unformatted("[]"sv);
			return;
		}

		print_unformatted('[');
		if (indent_array_elements())
			increase_indent();
		for (size_t i = 0; i < arr.size(); i++)
		{
			if (i > 0u)
				print_unformatted(',');
			print_newline(true);
			print_indent();

			auto& v			= arr[i];
			const auto type = v.type();
			TOML_ASSUME(type != node_type::none);
			switch (type)
			{
				case node_type::table: print(*reinterpret_cast<const table*>(&v)); break;
				case node_type::array: print(*reinterpret_cast<const array*>(&v)); break;
				default: print_value(v, type);
			}
		}
		if (indent_array_elements())
			decrease_indent();
		print_newline(true);
		print_indent();
		print_unformatted(']');
	}

	TOML_EXTERNAL_LINKAGE
	void json_formatter::print()
	{
		if (dump_failed_parse_result())
			return;

		switch (auto source_type = source().type())
		{
			case node_type::table: print(*reinterpret_cast<const table*>(&source())); break;
			case node_type::array: print(*reinterpret_cast<const array*>(&source())); break;
			default: print_value(source(), source_type);
		}
	}
}
TOML_NAMESPACE_END;

#include <collie/toml/impl/header_end.h>
#endif // TOML_ENABLE_FORMATTERS
