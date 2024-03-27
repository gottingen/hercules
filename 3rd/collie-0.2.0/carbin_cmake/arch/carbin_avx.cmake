
INCLUDE(CheckCXXSourceRuns)

SET(AVX_CODE "
#if !defined __AVX__ // MSVC supports this flag since MSVS 2013
#error \"__AVX__ define is missing\"
#endif
#include <immintrin.h>
void test()
{
    __m256 a = _mm256_set1_ps(0.0f);
}
int main() { return 0; }")

SET(AVX2_CODE "
#if !defined __AVX2__ // MSVC supports this flag since MSVS 2013
#error \"__AVX2__ define is missing\"
#endif
#include <immintrin.h>
void test()
{
    int data[8] = {0,0,0,0, 0,0,0,0};
    __m256i a = _mm256_loadu_si256((const __m256i *)data);
    __m256i b = _mm256_bslli_epi128(a, 1);  // available in GCC 4.9.3+
}
int main() { return 0; }")

SET(AVX512_CODE "
#if defined __AVX512__ || defined __AVX512F__
#include <immintrin.h>
void test()
{
    __m512i zmm = _mm512_setzero_si512();
#if defined __GNUC__ && defined __x86_64__
    asm volatile (\"\" : : : \"zmm16\", \"zmm17\", \"zmm18\", \"zmm19\");
#endif
}
#else
#error \"AVX512 is not supported\"
#endif
int main() { return 0; }")





MACRO(CHECK_AVX type flags)
    SET(__FLAG_I 1)
    SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    FOREACH(__FLAG ${flags})
        IF(NOT CXX_${type}_FOUND)
            SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
            CHECK_CXX_SOURCE_RUNS("${${type}_CODE}" CXX_HAS_${type}_${__FLAG_I})
            IF(CXX_HAS_${type}_${__FLAG_I})
                SET(CXX_${type}_FOUND TRUE)
            ENDIF()
            MATH(EXPR __FLAG_I "${__FLAG_I}+1")
        ENDIF()
    ENDFOREACH()
    SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    IF(NOT CXX_${type}_FOUND)
        SET(CXX_${type}_FOUND FALSE)
    ENDIF()

    MARK_AS_ADVANCED(CXX_${type}_FOUND CXX_${type}_FLAGS)

ENDMACRO()

CHECK_AVX("AVX" ";-mavx;/arch:AVX")
CHECK_AVX("AVX2" ";-mavx2;/arch:AVX2")
CHECK_AVX("AVX512" ";-mavx512;/arch:AVX512")