
INCLUDE(CheckCXXSourceRuns)

SET(SSE1_CODE "
  #include <xmmintrin.h>

  int main()
  {
    __m128 a;
    float vals[4] = {0,0,0,0};
    a = _mm_loadu_ps(vals);  // SSE1
    return 0;
  }")

SET(SSE2_CODE "
  #include <emmintrin.h>

  int main()
  {
    __m128d a;
    double vals[2] = {0,0};
    a = _mm_loadu_pd(vals);  // SSE2
    return 0;
  }")

SET(SSE3_CODE "
#include <pmmintrin.h>
int main() {
    __m128 u, v;
    u = _mm_set1_ps(0.0f);
    v = _mm_moveldup_ps(u); // SSE3
    return 0;
}")

SET(SSSE3_CODE "
  #include <tmmintrin.h>
  const double v = 0;
  int main() {
    __m128i a = _mm_setzero_si128();
    __m128i b = _mm_abs_epi32(a); // SSSE3
    return 0;
  }")

SET(SSE4_1_CODE "
  #include <smmintrin.h>

  int main ()
  {
    __m128i a = {0,0,0,0}, b = {0,0,0,0};
    __m128i res = _mm_max_epi8(a, b); // SSE4_1

    return 0;
  }
")

SET(SSE4_2_CODE "
  #include <nmmintrin.h>

  int main()
  {
    __m128i a = {0,0,0,0}, b = {0,0,0,0}, c = {0,0,0,0};
    c = _mm_cmpgt_epi64(a, b);  // SSE4_2
    return 0;
  }
")



MACRO(CHECK_SSE lang type flags)
    SET(__FLAG_I 1)
    SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
    FOREACH(__FLAG ${flags})
        IF(NOT ${lang}_${type}_FOUND)
            SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
            CHECK_CXX_SOURCE_RUNS("${${type}_CODE}" ${lang}_HAS_${type}_${__FLAG_I})
            IF(${lang}_HAS_${type}_${__FLAG_I})
                SET(${lang}_${type}_FOUND TRUE)
            ENDIF()
            MATH(EXPR __FLAG_I "${__FLAG_I}+1")
        ENDIF()
    ENDFOREACH()
    SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    IF(NOT ${lang}_${type}_FOUND)
        SET(${lang}_${type}_FOUND FALSE)
    ENDIF()
    MARK_AS_ADVANCED(${lang}_${type}_FOUND ${lang}_${type}_FLAGS)

ENDMACRO()

CHECK_SSE(CXX "SSE1" ";-msse;/arch:SSE")
CHECK_SSE(CXX "SSE2" ";-msse2;/arch:SSE2")
CHECK_SSE(CXX "SSE3" ";-msse3;/arch:SSE3")
CHECK_SSE(CXX "SSSE3" ";-mssse3;/arch:SSSE3")
CHECK_SSE(CXX "SSE4_1" ";-msse4.1;-msse4;/arch:SSE4")
CHECK_SSE(CXX "SSE4_2" ";-msse4.2;-msse4;/arch:SSE4")