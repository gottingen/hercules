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
TOML_DISABLE_WARNINGS;
#if !TOML_HAS_CUSTOM_OPTIONAL_TYPE
#include <optional>
#endif
TOML_ENABLE_WARNINGS;

TOML_NAMESPACE_START
{
#if TOML_HAS_CUSTOM_OPTIONAL_TYPE

	template <typename T>
	using optional = TOML_OPTIONAL_TYPE<T>;

#else

	/// \brief	The 'optional' type used throughout the library.
	///
	/// \remarks By default this will be an alias for std::optional, but you can change the optional type
	/// 		 used by the library by defining #TOML_OPTIONAL_TYPE.
	template <typename T>
	using optional = std::optional<T>;

#endif
}
TOML_NAMESPACE_END;
