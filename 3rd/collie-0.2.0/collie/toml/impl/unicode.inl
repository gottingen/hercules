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

//# {{
#include <collie/toml/impl/preprocessor.h>
#if !TOML_IMPLEMENTATION
#error This is an implementation-only header.
#endif
//# }}

#include <collie/toml/impl/unicode.h>
#include <collie/toml/impl/simd.h>
#include <collie/toml/impl/header_start.h>

TOML_IMPL_NAMESPACE_START
{
	TOML_PURE_GETTER
	TOML_EXTERNAL_LINKAGE
	bool is_ascii(const char* str, size_t len) noexcept
	{
		const char* const end = str + len;

#if TOML_HAS_SSE2 && (128 % CHAR_BIT) == 0
		{
			constexpr size_t chars_per_vector = 128u / CHAR_BIT;

			if (const size_t simdable = len - (len % chars_per_vector))
			{
				__m128i mask = _mm_setzero_si128();
				for (const char* const e = str + simdable; str < e; str += chars_per_vector)
				{
					const __m128i current_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str));
					mask						= _mm_or_si128(mask, current_bytes);
				}
				const __m128i has_error = _mm_cmpgt_epi8(_mm_setzero_si128(), mask);

#if TOML_HAS_SSE4_1
				if (!_mm_testz_si128(has_error, has_error))
					return false;
#else
				if (_mm_movemask_epi8(_mm_cmpeq_epi8(has_error, _mm_setzero_si128())) != 0xFFFF)
					return false;
#endif
			}
		}
#endif

		for (; str < end; str++)
			if (static_cast<unsigned char>(*str) > 127u)
				return false;

		return true;
	}
}
TOML_IMPL_NAMESPACE_END;

#include <collie/toml/impl/header_end.h>
