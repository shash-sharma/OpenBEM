// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Iterative trapezoidal integration over a line segment.
*/

#include "quadrature/line/iterative_trapz.hpp"

#include <functional>
#include <stdexcept>

#include "types.hpp"
#include "constants.hpp"
#include "quadrature/utility.hpp"
#include "quadrature/line/trapz.hpp"


namespace bem
{

template <uint8_t dim>
void IterativeTrapzLineQuadrature<dim>::compute_points_weights(
    ConstEigRef<EigColVecN<Float, dim>> p1,
    ConstEigRef<EigColVecN<Float, dim>> p2,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{
    if (!eval)
        throw std::invalid_argument(
            "IterativeTrapzLineQuadrature::compute_points_weights(): invalid or missing eval."
            );

    uint16_t starting_iter = ceil(std::log2((Float)starting_num_segments_));
    uint16_t num_segments = std::pow(2, starting_iter);

    TrapzLineQuadrature<dim> trapz_quad (num_segments);
    trapz_quad.compute_points_weights(p1, p2);

    EigRowVec<Complex> vals = eval(trapz_quad.points());
    Complex val_ref = trapz_quad.weights().dot(vals);

    EigMatNX<Float, dim> points_temp = trapz_quad.points();
    EigRowVec<Complex> vals_temp = vals;
    EigRowVec<Float> weights_temp;

    if (max_iters_ > LINE_MAX_ORDER)
        throw std::domain_error(
            "IterativeTrapzLineQuadrature::compute_points_weights(): max iterations exceeded.");

    EigColVecN<Float, dim> p = p2 - p1;
    Float line_size = p.norm();

    if (line_size == 0.0)
    {
        base::points_ = p1;
        base::weights_ = EigRowVec<Float>::Zero(1, 1);
        base::points_weights_computed_ = true;
        converged_ = true;
        return;
    }

    uint16_t max_iters_mod = std::min(max_iters_, (uint16_t)15);
    converged_ = false;

    for (uint16_t iter = starting_iter; iter <= max_iters_mod; ++iter)
    {
        Float segment_size = line_size / (points_temp.cols() - 1);
        Float segment_ratio = segment_size / line_size / 2;

        EigColVecN<Float, dim> p1_new = p1 + p * segment_ratio;
        EigColVecN<Float, dim> p2_new = p2 - p * segment_ratio;
        uint16_t num_segments_new = std::pow(2, iter) - 1;

        trapz_quad.set_num_segments(num_segments_new);
        trapz_quad.compute_points_weights(p1_new, p2_new);

        vals = eval(trapz_quad.points());
        Complex val = trapz_quad.weights().dot(vals);

        points_temp.conservativeResize(dim, points_temp.cols() + trapz_quad.points().cols());
        points_temp.rightCols(trapz_quad.points().cols()) = trapz_quad.points();

        weights_temp.resize(points_temp.cols());
        weights_temp.setConstant(segment_size);
        weights_temp[0] = segment_size * 0.5;
        weights_temp[starting_num_segments_] = segment_size * 0.5;

        vals_temp = eval(points_temp);
        val = weights_temp.dot(vals_temp);

        converged_ = compare_with_tol(val, val_ref, tol_, 1);
        if (converged_)
        {
            converged_num_segments_ = points_temp.cols() - 1;
            break;
        }
        val_ref = val;
    }

    Float segment_size = line_size / (points_temp.cols() - 1);
    weights_temp.resize(points_temp.cols());
    weights_temp.setConstant(segment_size);
    weights_temp[0] = segment_size * 0.5;
    weights_temp[starting_num_segments_] = segment_size * 0.5;

    base::points_ = points_temp;
    base::weights_ = weights_temp;
    base::points_weights_computed_ = true;

    return;
};


template class IterativeTrapzLineQuadrature<1>;
template class IterativeTrapzLineQuadrature<2>;
template class IterativeTrapzLineQuadrature<3>;

}
