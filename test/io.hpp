// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file Helpers for writing and reading json files.
*/

#ifndef IO_H
#define IO_H

#include <complex>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "types.hpp"
#include "external/json.hpp"


using json = nlohmann::json;


namespace std
{

template <typename T>
void to_json(json& j, const std::complex<T>& p)
{ j = json { p.real(), p.imag() }; return; }

template <typename T>
void from_json(const json& j, std::complex<T>& p)
{ p.real(j.at(0)); p.imag(j.at(1)); return; }


template <typename T>
void to_json(json& j, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
    j = json {
        { "rows", mat.rows() },
        { "cols", mat.cols() },
        { "values", std::vector<T> (mat.data(), mat.data() + mat.rows() * mat.cols()) }
    };
    return;
}

template <typename T>
void from_json(const json& j, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
    std::size_t num_rows = j["rows"];
    std::size_t num_cols = j["cols"];
    mat = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> (
        ((std::vector<T>) j["values"]).data(), num_rows, num_cols
        ).eval();
    return;
}

}

#endif

