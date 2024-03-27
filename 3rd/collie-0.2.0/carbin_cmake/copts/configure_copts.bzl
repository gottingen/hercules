"""carbin specific copts.

This file simply selects the correct options from the generated files.  To
change carbin copts, edit carbin/copts/copts.py
"""

load(
    "//:copts/carbin_generated_copts.bzl",
    "CARBIN_CLANG_CL_FLAGS",
    "CARBIN_CLANG_CL_TEST_FLAGS",
    "CARBIN_GCC_FLAGS",
    "CARBIN_GCC_TEST_FLAGS",
    "CARBIN_LLVM_FLAGS",
    "CARBIN_LLVM_TEST_FLAGS",
    "CARBIN_MSVC_FLAGS",
    "CARBIN_MSVC_LINKOPTS",
    "CARBIN_MSVC_TEST_FLAGS",
    "CARBIN_RANDOM_HWAES_ARM32_FLAGS",
    "CARBIN_RANDOM_HWAES_ARM64_FLAGS",
    "CARBIN_RANDOM_HWAES_MSVC_X64_FLAGS",
    "CARBIN_RANDOM_HWAES_X64_FLAGS",
)

CARBIN_DEFAULT_COPTS = select({
    "//:windows": CARBIN_MSVC_FLAGS,
    "//:clang_compiler": CARBIN_LLVM_FLAGS,
    "//conditions:default": CARBIN_GCC_FLAGS,
})

CARBIN_TEST_COPTS = CARBIN_DEFAULT_COPTS + select({
    "//:windows": CARBIN_MSVC_TEST_FLAGS,
    "//:clang_compiler": CARBIN_LLVM_TEST_FLAGS,
    "//conditions:default": CARBIN_GCC_TEST_FLAGS,
})

CARBIN_DEFAULT_LINKOPTS =  select({
    "//:windows": CARBIN_MSVC_LINKOPTS,
    "//conditions:default": [],
})

# CARBIN_RANDOM_RANDEN_COPTS blaze copts flags which are required by each
# environment to build an accelerated RandenHwAes library.
CARBIN_RANDOM_RANDEN_COPTS = select({
    # APPLE
    ":cpu_darwin_x86_64": CARBIN_RANDOM_HWAES_X64_FLAGS,
    ":cpu_darwin": CARBIN_RANDOM_HWAES_X64_FLAGS,
    ":cpu_x64_windows_msvc": CARBIN_RANDOM_HWAES_MSVC_X64_FLAGS,
    ":cpu_x64_windows": CARBIN_RANDOM_HWAES_MSVC_X64_FLAGS,
    ":cpu_k8": CARBIN_RANDOM_HWAES_X64_FLAGS,
    ":cpu_ppc": ["-mcrypto"],

    # Supported by default or unsupported.
    "//conditions:default": [],
})

# carbin_random_randen_copts_init:
#  Initialize the config targets based on cpu, os, etc. used to select
#  the required values for CARBIN_RANDOM_RANDEN_COPTS
def carbin_random_randen_copts_init():
    """Initialize the config_settings used by CARBIN_RANDOM_RANDEN_COPTS."""

    # CPU configs.
    # These configs have consistent flags to enable HWAES intsructions.
    cpu_configs = [
        "ppc",
        "k8",
        "darwin_x86_64",
        "darwin",
        "x64_windows_msvc",
        "x64_windows",
    ]
    for cpu in cpu_configs:
        native.config_setting(
            name = "cpu_%s" % cpu,
            values = {"cpu": cpu},
        )
