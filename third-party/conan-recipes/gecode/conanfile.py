from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.build import check_min_cppstd
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get, rmdir
from conan.tools.scm import Version
import os

required_conan_version = ">=2.0"


class GecodeConan(ConanFile):
    name = "gecode"
    description = (
        "Gecode: a toolkit for developing constraint-based systems, with "
        "finite-domain propagators, branchers and search engines"
    )
    license = "MIT"
    url = "https://github.com/Gecode/gecode"
    homepage = "https://www.gecode.org/"
    topics = ("constraint-programming", "csp", "solver", "propagation", "search")

    package_type = "library"
    settings = "os", "arch", "compiler", "build_type"

    # The module options mirror upstream's GECODE_ENABLE_* switches. The
    # defaults build the smallest configuration that is useful here: integer
    # variables, search, and MiniModel (the operator-overloading front end a
    # constraint tree is translated through). Set variables, float variables,
    # the command-line driver and FlatZinc are each a sizeable chunk of build
    # time nothing in this repo consumes.
    #
    # Gist is not offered at all: it is the graphical search inspector and
    # needs Qt Widgets, which would drag a GUI toolkit into a compiler's
    # dependency graph. Debugging a model interactively is a job for a
    # scratch build outside Conan.
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_minimodel": [True, False],
        "with_set_vars": [True, False],
        "with_float_vars": [True, False],
        "with_driver": [True, False],
        "with_flatzinc": [True, False],
        "with_cpprofiler": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_minimodel": True,
        "with_set_vars": False,
        "with_float_vars": False,
        "with_driver": False,
        "with_flatzinc": False,
        "with_cpprofiler": False,
    }

    @property
    def _min_cppstd(self):
        # Upstream declares cxx_std_17 on every component library.
        return 17

    @property
    def _compilers_minimum_version(self):
        return {
            "gcc": "8",
            "clang": "7",
            "apple-clang": "12",
            "msvc": "192",
        }

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self, src_folder="src")

    def validate(self):
        if self.settings.compiler.get_safe("cppstd"):
            check_min_cppstd(self, self._min_cppstd)
        minimum_version = self._compilers_minimum_version.get(str(self.settings.compiler), False)
        if minimum_version and Version(self.settings.compiler.version) < minimum_version:
            raise ConanInvalidConfiguration(
                f"{self.ref} requires C++{self._min_cppstd}, which "
                f"{self.settings.compiler} {self.settings.compiler.version} does not support."
            )

        # Upstream silently forces the modules a selected one depends on back
        # ON (gecode_force_option_on). Refuse instead: a package whose option
        # values do not describe what is in it produces a package_id that lies
        # about its contents.
        if self.options.with_flatzinc and not all(
            [self.options.with_minimodel, self.options.with_set_vars,
             self.options.with_float_vars, self.options.with_driver]
        ):
            raise ConanInvalidConfiguration(
                "-o gecode/*:with_flatzinc=True also needs with_minimodel, "
                "with_set_vars, with_float_vars and with_driver"
            )
        if self.options.with_driver and not self.options.with_minimodel:
            raise ConanInvalidConfiguration(
                "-o gecode/*:with_driver=True also needs with_minimodel"
            )

    def source(self):
        get(self, **self.conan_data["sources"][self.version], strip_root=True)

    def generate(self):
        tc = CMakeToolchain(self)
        # GECODE_BUILD_SHARED/STATIC select which library variants exist;
        # upstream derives them from BUILD_SHARED_LIBS, but only when the user
        # has not set them, so set both rather than rely on that.
        tc.cache_variables["GECODE_BUILD_SHARED"] = bool(self.options.shared)
        tc.cache_variables["GECODE_BUILD_STATIC"] = not bool(self.options.shared)

        tc.cache_variables["GECODE_ENABLE_INT_VARS"] = True
        tc.cache_variables["GECODE_ENABLE_SEARCH"] = True
        tc.cache_variables["GECODE_ENABLE_MINIMODEL"] = bool(self.options.with_minimodel)
        tc.cache_variables["GECODE_ENABLE_SET_VARS"] = bool(self.options.with_set_vars)
        tc.cache_variables["GECODE_ENABLE_FLOAT_VARS"] = bool(self.options.with_float_vars)
        tc.cache_variables["GECODE_ENABLE_DRIVER"] = bool(self.options.with_driver)
        tc.cache_variables["GECODE_ENABLE_FLATZINC"] = bool(self.options.with_flatzinc)
        tc.cache_variables["GECODE_ENABLE_CPPROFILER"] = bool(self.options.with_cpprofiler)

        # QT/GIST are tri-state STRING caches ("AUTO", "ON", "OFF"), so they
        # take the string and not a CMake boolean. AUTO would link Qt if it
        # happened to be installed on the build machine, which is exactly the
        # kind of host-dependent package this must not produce.
        tc.cache_variables["GECODE_ENABLE_GIST"] = "OFF"
        tc.cache_variables["GECODE_ENABLE_QT"] = "OFF"

        # MPFR only refines float-variable transcendental functions, and is
        # found through Gecode's own FindMPFR rather than a Conan dependency;
        # leaving it on would make the package's contents depend on what is
        # installed on the build machine.
        tc.cache_variables["GECODE_ENABLE_MPFR"] = False

        tc.cache_variables["GECODE_ENABLE_EXAMPLES"] = False
        tc.cache_variables["GECODE_INSTALL"] = True
        tc.cache_variables["BUILD_TESTING"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        copy(self, "LICENSE", self.source_folder, os.path.join(self.package_folder, "licenses"))
        cmake = CMake(self)
        cmake.install()
        # Upstream's own GecodeConfig.cmake would shadow the one CMakeDeps
        # generates, and it resolves components against the build machine's
        # Qt/MPFR rather than against this package.
        rmdir(self, os.path.join(self.package_folder, "lib", "cmake"))
        rmdir(self, os.path.join(self.package_folder, "lib", "pkgconfig"))
        rmdir(self, os.path.join(self.package_folder, "share"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "Gecode")
        # Consumers written against upstream link Gecode::gecode<component>;
        # keep those names so a translation unit does not have to know whether
        # Gecode came from Conan or from a system install.
        self.cpp_info.set_property("cmake_target_name", "Gecode::gecode")

        def add(component, libname, requires):
            c = self.cpp_info.components[component]
            c.set_property("cmake_target_name", f"Gecode::gecode{component}")
            c.libs = [libname]
            c.requires = requires
            # Only meaningful to MSVC (it disables the #pragma comment(lib)
            # autolink in the headers), but the headers check it everywhere.
            c.defines = ["GECODE_NO_AUTOLINK"]
            return c

        support = add("support", "gecodesupport", [])
        if self.settings.os in ["Linux", "FreeBSD"]:
            support.system_libs = ["m", "pthread", "rt"]

        add("kernel", "gecodekernel", ["support"])
        add("search", "gecodesearch", ["kernel"])
        add("int", "gecodeint", ["kernel"])

        minimodel_requires = ["int", "search"]
        if self.options.with_set_vars:
            add("set", "gecodeset", ["int"])
            minimodel_requires.append("set")
        if self.options.with_float_vars:
            add("float", "gecodefloat", ["int", "kernel"])
            minimodel_requires.append("float")
        if self.options.with_minimodel:
            add("minimodel", "gecodeminimodel", minimodel_requires)
        if self.options.with_driver:
            add("driver", "gecodedriver", ["int", "search", "minimodel"])
        if self.options.with_flatzinc:
            add("flatzinc", "gecodeflatzinc", ["minimodel", "driver"])
