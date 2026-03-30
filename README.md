# OpenBEM

OpenBEM is an open-source framework for electromagnetic simulation
with the boundary element method.

OpenBEM is not a solver. Rather, it is a _framework_ of components
that can be fit together in highly customizable ways, to allow users
to implement their own application-specific solvers. It is intended as
a research tool to facilitate the testing and prototyping of new
ideas, formulations, and algorithms, without the researcher having to
re-implement aspects of the boundary element method that are not
pertinent to their specific research.

As a starting point, it is highly recommended that users go through
the provided <a href="examples.html">Examples</a>.  To gain
familiarity with the code and its organization, the <a
href="topics.html">Topics</a> page is the recommended way to navigate
through the available functionality.


# Requirements <a name="req"></a>

OpenBEM requires at least C++17.

To compile your project with OpenBEM, the following should be noted:

- GCC version 10 or above, Clang version 8 or above, or an equivalent
  compiler is recommended.

- OpenBEM uses a relatively recent version of the [Eigen
  library](https://libeigen.gitlab.io/eigen/docs-5.0.1/GettingStarted.html).
  This is a header-only linear algebra toolkit which is included as
  part of the OpenBEM distribution, so you do not need to download or
  include it separately. However, if using an older compiler, you may
  need to disable certain compile options such as `march=native`,
  `mavx`, etc., for compatibility with this version of Eigen. I've
  experienced this issue with GCC 8.5.


# Usage <a name="use"></a>

There are two ways to incorporate OpenBEM in your project.


## Option 1: Header-only mode <a name="head"></a>

The simplest and quickest way to use OpenBEM is to import it into your
project it in header-only mode, which involves no installation,
building or linking; you simply `#include` the appropriate header into
your C++ files, and you're all set.

For example, if you want to include OpenBEM in a C++ file called
`main.cpp`, your compilation command may look like

```make
g++ -I/path/to/OpenBEM/source -std=c++17 main.cpp -o main
```

where the `-I` specifier adds the OpenBEM source directory to the
compiler's include search path.

If you're using [CMake](https://cmake.org) to build your project
(recommended), you can include OpenBEM in header-only mode by doing
the following:

```CMake
add_executable(main main.cpp)
target_include_directories(main PRIVATE /path/to/OpenBEM/source)
target_compile_features(main PRIVATE cxx_std_17)
```

In either case, OpenBEM's headers can then be included
directly into your project, for example,

```C++
#include "openbem.hpp"
```

and that's it.


## Option 2: Dynamically linked mode <a name="link"></a>

The compile time of your project can be somewhat improved by using
OpenBEM in linked mode, which involves building and dynamically
linking OpenBEM to your project.

Doing this requires you to have at least version 3.16.0 of
[CMake](https://cmake.org) installed on your machine. Then, just run

```bash
cmake -B /path/to/build/directory -S /path/to/OpenBEM
make -C /path/to/build/directory
```

This will build OpenBEM in any `/path/to/build/directory` of your
choice.

The following options are available to use in the `cmake` command to
customize your build, by augmenting the `cmake` command as

```bash
cmake -DOPTION=VAL -B /path/to/build/directory -S /path/to/OpenBEM
```

where `OPTION` is one of the following options, and `VAL` is one of
the possible corresponding values for that option:

- `PRECISION`, which can be `single`, `double` (default), or `extended`.
- `DEBUG_LEVEL`, which can be `release` (default, optimized build) or `debug`.
- `BUILD_EXAMPLES`, which can be `yes` or `no` (default).
- `BUILD_TESTS`, which can be `yes` or `no` (default).

Once built, you still need to follow the steps in Option 1 to add the
include path to your project's compilation. In addition, you'll need
to link your project to the OpenBEM dynamic library
`/path/to/build/directory/libopenbem.[so/dylib/dll]`.

Then you can include OpenBEM's headers in your project as usual, for
example,

```C++
#include "openbem.hpp"
```

# Citing

If OpenBEM benefits your research, please cite this GitHub repository via the following [![DOI](https://zenodo.org/badge/1060257425.svg)](https://doi.org/10.5281/zenodo.19337991) in your publications and presentations.