// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

#include "io.hpp"

#include "openbem.hpp"


int main(int argc, char** argv)
{

    std::vector<std::string> args;
    if (argc > 1)
        args.assign(argv + 1, argv + argc);

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_compile.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    return 0;

}
