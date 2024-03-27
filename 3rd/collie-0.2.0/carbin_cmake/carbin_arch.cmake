#
# Copyright 2023 The Carbin Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 
# 
#####################################################################
# gcc x86
# https://gcc.gnu.org/onlinedocs/gcc-7.3.0/gcc/x86-Options.html#x86-Options
# gcc arm
# https://gcc.gnu.org/onlinedocs/gcc-7.3.0/gcc/ARM-Options.html#ARM-Options
# gcc -march=native -c -Q –help=target check option supports stats
# gcc -march=haswell -c -Q --help=target
# The following options are target specific:
# -m128bit-long-double        		[enabled]
# -m16                        		[disabled]
# -m32                        		[disabled]
# -m3dnow                     		[disabled]
# -m3dnowa                    		[disabled]
# -m64                        		[enabled]
# -m80387                     		[enabled]
# -m8bit-idiv                 		[disabled]
# -m96bit-long-double         		[disabled]
# -mabi=                      		sysv
# -mabm                       		[disabled]
# -maccumulate-outgoing-args  		[disabled]
# -maddress-mode=             		long
# -madx                       		[disabled]
# -maes                       		[disabled]
# -malign-data=               		compat
# -malign-double              		[disabled]
# -malign-functions=          		0
# -malign-jumps=              		0
# -malign-loops=              		0
# -malign-stringops           		[enabled]
# -mandroid                   		[disabled]
# -march=                     		haswell
# -masm=                      		att
# -mavx                       		[enabled]
# -mavx2                      		[enabled]
# -mavx256-split-unaligned-load 	[disabled]
# -mavx256-split-unaligned-store 	[disabled]
# -mavx5124fmaps              		[disabled]
# -mavx5124vnniw              		[disabled]
# -mavx512bitalg              		[disabled]
# -mavx512bw                  		[disabled]
# -mavx512cd                  		[disabled]
# -mavx512dq                  		[disabled]
# -mavx512er                  		[disabled]
# -mavx512f                   		[disabled]
# -mavx512ifma                		[disabled]
# -mavx512pf                  		[disabled]
# -mavx512vbmi                		[disabled]
# -mavx512vbmi2               		[disabled]
# -mavx512vl                  		[disabled]
# -mavx512vnni                		[disabled]
# -mavx512vpopcntdq           		[disabled]
# -mbionic                    		[disabled]
# -mbmi                       		[enabled]
# -mbmi2                      		[enabled]
# -mbranch-cost=<0,5>         		3
# -mcall-ms2sysv-xlogues      		[disabled]
# -mcet-switch                		[disabled]
# -mcld                       		[disabled]
# -mcldemote                  		[disabled]
# -mclflushopt                		[disabled]
# -mclwb                      		[disabled]
# -mclzero                    		[disabled]
# -mcmodel=                   		[default]
# -mcpu=
# -mcrc32                     		[disabled]
# -mcx16                      		[enabled]
# -mdispatch-scheduler        		[disabled]
# -mdump-tune-features        		[disabled]
# -mf16c                      		[enabled]
# -mfancy-math-387            		[enabled]
# -mfentry                    		[disabled]
# -mfentry-name=
# -mfentry-section=
# -mfma                       		[enabled]
# -mfma4                      		[disabled]
# -mforce-drap                		[disabled]
# -mforce-indirect-call       		[disabled]
# -mfp-ret-in-387             		[enabled]
# -mfpmath=                   		sse
# -mfsgsbase                  		[enabled]
# -mfunction-return=          		keep
# -mfused-madd
# -mfxsr                      		[enabled]
# -mgeneral-regs-only         		[disabled]
# -mgfni                      		[disabled]
# -mglibc                     		[enabled]
# -mhard-float                		[enabled]
# -mhle                       		[enabled]
# -miamcu                     		[disabled]
# -mieee-fp                   		[enabled]
# -mincoming-stack-boundary=  		0
# -mindirect-branch-register  		[disabled]
# -mindirect-branch=          		keep
# -minline-all-stringops      		[disabled]
# -minline-stringops-dynamically 	[disabled]
# -minstrument-return=        		none
# -mintel-syntax
# -mlarge-data-threshold=<number> 	65536
# -mlong-double-128           		[disabled]
# -mlong-double-64            		[disabled]
# -mlong-double-80            		[enabled]
# -mlwp                       		[disabled]
# -mlzcnt                     		[enabled]
# -mmanual-endbr              		[disabled]
# -mmemcpy-strategy=
# -mmemset-strategy=
# -mmitigate-rop              		[disabled]
# -mmmx                       		[enabled]
# -mmovbe                     		[enabled]
# -mmovdir64b                 		[disabled]
# -mmovdiri                   		[disabled]
# -mmpx                       		[disabled]
# -mms-bitfields              		[disabled]
# -mmusl                      		[disabled]
# -mmwaitx                    		[disabled]
# -mno-align-stringops        		[disabled]
# -mno-default                		[disabled]
# -mno-fancy-math-387         		[disabled]
# -mno-push-args              		[disabled]
# -mno-red-zone               		[disabled]
# -mno-sse4                   		[disabled]
# -mnop-mcount                		[disabled]
# -momit-leaf-frame-pointer   		[disabled]
# -mpc32                      		[disabled]
# -mpc64                      		[disabled]
# -mpc80                      		[disabled]
# -mpclmul                    		[enabled]
# -mpcommit                   		[disabled]
# -mpconfig                   		[disabled]
# -mpku                       		[disabled]
# -mpopcnt                    		[enabled]
# -mprefer-avx128
# -mprefer-vector-width=      		none
# -mpreferred-stack-boundary= 		0
# -mprefetchwt1               		[disabled]
# -mprfchw                    		[disabled]
# -mptwrite                   		[disabled]
# -mpush-args                 		[enabled]
# -mrdpid                     		[disabled]
# -mrdrnd                     		[enabled]
# -mrdseed                    		[disabled]
# -mrecip                     		[disabled]
# -mrecip=
# -mrecord-mcount             		[disabled]
# -mrecord-return             		[disabled]
# -mred-zone                  		[enabled]
# -mregparm=                  		6
# -mrtd                       		[disabled]
# -mrtm                       		[disabled]
# -msahf                      		[enabled]
# -msgx                       		[disabled]
# -msha                       		[disabled]
# -mshstk                     		[disabled]
# -mskip-rax-setup            		[disabled]
# -msoft-float                		[disabled]
# -msse                       		[enabled]
# -msse2                      		[enabled]
# -msse2avx                   		[disabled]
# -msse3                      		[enabled]
# -msse4                      		[enabled]
# -msse4.1                    		[enabled]
# -msse4.2                    		[enabled]
# -msse4a                     		[disabled]
# -msse5
# -msseregparm                		[disabled]
# -mssse3                     		[enabled]
# -mstack-arg-probe           		[disabled]
# -mstack-protector-guard-offset=
# -mstack-protector-guard-reg=
# -mstack-protector-guard-symbol=
# -mstack-protector-guard=    		tls
# -mstackrealign              		[disabled]
# -mstringop-strategy=        		[default]
# -mstv                       		[enabled]
# -mtbm                       		[disabled]
# -mtls-dialect=              		gnu
# -mtls-direct-seg-refs       		[enabled]
# -mtune-ctrl=
# -mtune=                     		haswell
# -muclibc                    		[disabled]
# -mvaes                      		[disabled]
# -mveclibabi=                		[default]
# -mvect8-ret-in-mem          		[disabled]
# -mvpclmulqdq                		[disabled]
# -mvzeroupper                		[enabled]
# -mwaitpkg                   		[disabled]
# -mwbnoinvd                  		[disabled]
# -mx32                       		[disabled]
# -mxop                       		[disabled]
# -mxsave                     		[enabled]
# -mxsavec                    		[disabled]
# -mxsaveopt                  		[enabled]
# -mxsaves                    		[disabled]
# 
# Known assembler dialects (for use with the -masm= option):
# att intel
# 
# Known ABIs (for use with the -mabi= option):
# ms sysv
# 
# Known code models (for use with the -mcmodel= option):
# 32 kernel large medium small
# 
# Valid arguments to -mfpmath=:
# 387 387+sse 387,sse both sse sse+387 sse,387
# 
# Known indirect branch choices (for use with the -mindirect-branch=/-mfunction-return= options):
# keep thunk thunk-extern thunk-inline
# 
# Known choices for return instrumentation with -minstrument-return=:
# call none nop5
# 
# Known data alignment choices (for use with the -malign-data= option):
# abi cacheline compat
# 
# Known vectorization library ABIs (for use with the -mveclibabi= option):
# acml svml
# 
# Known address mode (for use with the -maddress-mode= option):
# long short
# 
# Known preferred register vector length (to use with the -mprefer-vector-width= option):
# 128 256 512 none
# 
# Known stack protector guard (for use with the -mstack-protector-guard= option):
# global tls
# 
# Valid arguments to -mstringop-strategy=:
# byte_loop libcall loop rep_4byte rep_8byte rep_byte unrolled_loop vector_loop
# 
# Known TLS dialects (for use with the -mtls-dialect= option):
# gnu gnu2
# 
# Known valid arguments for -march= option:
# i386 i486 i586 pentium lakemont pentium-mmx winchip-c6 winchip2 c3 samuel-2 c3-2 nehemiah c7 esther i686 pentiumpro pentium2 pentium3 pentium3m pentium-m pentium4 pentium4m prescott nocona core2 nehalem corei7 westmere sandybridge corei7-avx ivybridge core-avx-i haswell core-avx2 broadwell skylake skylake-avx512 cannonlake icelake-client icelake-server cascadelake tigerlake bonnell atom silvermont slm goldmont goldmont-plus tremont knl knm intel geode k6 k6-2 k6-3 athlon athlon-tbird athlon-4 athlon-xp athlon-mp x86-64 eden-x2 nano nano-1000 nano-2000 nano-3000 nano-x2 eden-x4 nano-x4 k8 k8-sse3 opteron opteron-sse3 athlon64 athlon64-sse3 athlon-fx amdfam10 barcelona bdver1 bdver2 bdver3 bdver4 znver1 znver2 btver1 btver2 generic native
# 
# Known valid arguments for -mtune= option:
# generic i386 i486 pentium lakemont pentiumpro pentium4 nocona core2 nehalem sandybridge haswell bonnell silvermont goldmont goldmont-plus tremont knl knm skylake skylake-avx512 cannonlake icelake-client icelake-server cascadelake tigerlake intel geode k6 athlon k8 amdfam10 bdver1 bdver2 bdver3 bdver4 btver1 btver2 znver1 znver2
# 
######################################################################
#
######################################################################
# make simple, by default set to Intel haswell arch. as equal to enable
# AVX2 option
#####################################################

#set(CARBIN_ARCH_OPTION "-march=haswell")
# avoid not set to global

set(CARBIN_ARCH_OPTION "")
if (CARBIN_ENABLE_ARCH)
    include(carbin_sse)
    include(carbin_avx)

    if (CXX_AVX2_FOUND)
        message(STATUS "AVX2 SUPPORTED for CXX")
        set(AVX2_SUPPORTED true)
        set(HIGHEST_SIMD_SUPPORTED "AVX2")
        list(APPEND SSE_SUPPORTED_LIST ${AVX2_SUPPORTED})
    else ()
        set(AVX2_SUPPORTED false)
    endif ()

    if (CXX_AVX512_FOUND)
        message(STATUS "AVX512 SUPPORTED for C and CXX")
        set(AVX512_SUPPORTED true)
        set(HIGHEST_SIMD_SUPPORTED "AVX512")
        list(APPEND SSE_SUPPORTED_LIST ${AVX512_SUPPORTED})
    else ()
        set(AVX512_SUPPORTED false)
    endif ()

    set(CARBIN_ARCH_OPTION)

    if (TURBO_USE_SSE1)
        message(STATUS "CARBIN SSE1 SELECTED")
        list(APPEND TURBO_SSE1_SIMD_FLAGS "-msse")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_SSE1_SIMD_FLAGS})
    endif ()

    if (CXX_SSE2_FOUND)
        message(STATUS "CARBIN SSE2 SELECTED")
        list(APPEND TURBO_SSE2_SIMD_FLAGS "-msse2")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_SSE2_SIMD_FLAGS})
    endif ()

    if (CXX_SSE3_FOUND)
        message(STATUS "CARBIN SSE3 SELECTED")
        list(APPEND TURBO_SSE3_SIMD_FLAGS "-msse3")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_SSE3_SIMD_FLAGS})
    endif ()

    if (CXX_SSSE3_FOUND)
        message(STATUS "CARBIN SSSE3 SELECTED")
        list(APPEND TURBO_SSSE3_SIMD_FLAGS "-mssse3")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_SSSE3_SIMD_FLAGS})
    endif ()

    if (CXX_SSE4_1_FOUND)
        message(STATUS "CARBIN SSE4_1 SELECTED")
        list(APPEND TURBO_SSE4_1_SIMD_FLAGS "-msse4.1")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_SSE4_1_SIMD_FLAGS})
    endif ()

    if (CXX_SSE4_2_FOUND)
        message(STATUS "CARBIN SSE4_2 SELECTED")
        list(APPEND TURBO_SSE4_2_SIMD_FLAGS "-msse4.2")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_SSE4_2_SIMD_FLAGS})
    endif ()

    if (CXX_AVX_FOUND)
        message(STATUS "CARBIN AVX SELECTED")
        list(APPEND TURBO_AVX_SIMD_FLAGS "-mavx")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_AVX_SIMD_FLAGS})
    endif ()

    if (CXX_AVX2_FOUND)
        message(STATUS "CARBIN AVX2 SELECTED")
        list(APPEND TURBO_AVX2_SIMD_FLAGS "-mavx2" "-mfma")
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_AVX2_SIMD_FLAGS})
    endif ()

    if (CXX_AVX512_FOUND)
        message(STATUS "CARBIN AVX512 SELECTED")
        list(APPEND TURBO_AVX512_SIMD_FLAGS "-mavx512f" "-mfma") # note that this is a bit platform specific
        list(APPEND CARBIN_ARCH_OPTION ${TURBO_AVX512_SIMD_FLAGS}) # note that this is a bit platform specific
    endif ()
endif ()
list(APPEND CARBIN_ARCH_OPTION ${CARBIN_RANDOM_RANDEN_COPTS})
MESSAGE(STATUS "CARBIN ARCH FLAGS ${CARBIN_ARCH_OPTION}")
