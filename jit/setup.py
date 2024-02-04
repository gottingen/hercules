#
# Copyright 2023 EA Authors.
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

import os
import sys
import shutil
import subprocess
from pathlib import Path
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension

exec(open("hercules/version.py").read())

ext = "dylib" if sys.platform == "darwin" else "so"

hercules_path = os.environ.get("HERCULES_DIR")
if not hercules_path:
    c = shutil.which("hercules")
    if c:
        hercules_path = Path(c).parent / ".."
else:
    hercules_path = Path(hercules_path)
for path in [
    os.path.expanduser("~") + "/.hercules",
    os.getcwd() + "/..",
]:
    path = Path(path)
    if not hercules_path and path.exists():
        hercules_path = path
        break

if (
    not hercules_path
    or not (hercules_path / "include" / "hercules").exists()
    or not (hercules_path / "lib" / "hercules").exists()
):
    print(
        "Cannot find Hercules.",
        'Please either install Hercules (/bin/bash -c "$(curl -fsSL https://github.com/gottingen/hercules/install.sh)"),',
        "or set HERCULES_DIR if Hercules is not in PATH or installed in ~/.hercules",
        file=sys.stderr,
    )
    sys.exit(1)
hercules_path = hercules_path.resolve()
print("Hercules: " + str(hercules_path))


if sys.platform == "darwin":
    libraries=["herculesrt", "herculesc"]
    linker_args = ["-Wl,-rpath," + str(hercules_path / "lib" / "hercules")]
else:
    libraries=["herculesrt"]
    linker_args = [
        "-Wl,-rpath=" + str(hercules_path / "lib" / "hercules"),
        "-Wl,--no-as-needed",
        "-lherculesc",
    ]

    # TODO: handle ABI changes better
    out = subprocess.check_output(["nm", "-g", str(hercules_path / "lib" / "hercules" / "libherculesc.so")])
    out = [i for i in out.decode(sys.stdout.encoding).split("\n") if "jitExecuteSafe" in i]
    if out and "cxx11" not in out[0]:
        print("CXX11 ABI not detected")
        os.environ["CFLAGS"] = os.environ.get("CFLAGS", "") + " -D_GLIBCXX_USE_CXX11_ABI=0"

jit_extension = Extension(
    "hercules.hercules_jit",
    sources=["hercules/jit.pyx"],
    libraries=libraries,
    language="c++",
    extra_compile_args=["-w"],
    extra_link_args=linker_args,
    include_dirs=[str(hercules_path / "include")],
    library_dirs=[str(hercules_path / "lib" / "hercules")],
)

setup(
    name="hercules-jit",
    version=__version__,
    install_requires=["cython", "astunparse"],
    python_requires=">=3.6",
    description="Hercules JIT decorator",
    url="https://github.com/gottingen.hercules",
    long_description="Please see https://github.com/gottingen.hercules for more details.",
    author="Exaloop Inc.",
    author_email="info@exaloop.io",
    license="Commercial",
    ext_modules=[jit_extension],
    packages=["hercules"],
    include_package_data=True,
    cmdclass={
        "build_ext": build_ext,
    },
)
