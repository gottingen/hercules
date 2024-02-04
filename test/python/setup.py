import os
import sys
import shutil

from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


hercules_path = os.environ.get("HERCULES_DIR")
if not hercules_path:
    c = shutil.which("hercules")
    if c:
        hercules_path = Path(c).parent / ".."
else:
    hercules_path = Path(hercules_path)
for path in [
    os.path.expanduser("~") + "/.hs",
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
        'Please either install Hercules (/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"),',
        "or set HERCULES_DIR if Hercules is not in PATH or installed in ~/.hercules",
        file=sys.stderr,
    )
    sys.exit(1)
hercules_path = hercules_path.resolve()
print("Hercules: " + str(hercules_path))


class HerculesExtension(Extension):
    def __init__(self, name, source):
        self.source = source
        super().__init__(name, sources=[], language='c')

class BuildHerculesExt(build_ext):
    def build_extensions(self):
        pass

    def run(self):
        inplace, self.inplace = self.inplace, False
        super().run()
        for ext in self.extensions:
            self.build_hercules(ext)
        if inplace:
            self.copy_extensions_to_source()

    def build_hercules(self, ext):
        extension_path = Path(self.get_ext_fullpath(ext.name))
        build_dir = Path(self.build_temp)
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(extension_path.parent.absolute(), exist_ok=True)

        optimization = '-debug' if self.debug else '-release'
        self.spawn([
            str(hercules_path / "bin" / "hercules"), 'build', optimization, "--relocation-model=pic",
            '-pyext', '-o', str(extension_path) + ".o", '-module', ext.name, ext.source])

        print('-->', extension_path)
        ext.runtime_library_dirs = [str(hercules_path / "lib" / "hercules")]
        self.compiler.link_shared_object(
            [str(extension_path) + ".o"],
            str(extension_path),
            libraries=["herculesrt"],
            library_dirs=ext.runtime_library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_preargs=['-Wl,-rpath,@loader_path'],
            # export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
        )
        self.distribution.hercules_lib = extension_path

setup(
    name='myext',
    version='0.1',
    packages=['myext'],
    ext_modules=[
        HerculesExtension('myext', 'myextension.hs'),
    ],
    cmdclass={'build_ext': BuildHerculesExt}
)

setup(
    name='myext2',
    version='0.1',
    packages=['myext2'],
    ext_modules=[
        HerculesExtension('myext2', 'myextension2.hs'),
    ],
    cmdclass={'build_ext': BuildHerculesExt}
)
