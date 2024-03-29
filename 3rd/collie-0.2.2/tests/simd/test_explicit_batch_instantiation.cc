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


#include <collie/simd/simd.h>
#ifndef COLLIE_SIMD_NO_SUPPORTED_ARCHITECTURE

namespace collie::simd
{

    template class batch<char>;
    template class batch<unsigned char>;
    template class batch<signed char>;
    template class batch<unsigned short>;
    template class batch<signed short>;
    template class batch<unsigned int>;
    template class batch<signed int>;
    template class batch<unsigned long>;
    template class batch<signed long>;
    template class batch<float>;
#if !COLLIE_SIMD_WITH_NEON || COLLIE_SIMD_WITH_NEON64
    template class batch<double>;
#endif
}
#endif
