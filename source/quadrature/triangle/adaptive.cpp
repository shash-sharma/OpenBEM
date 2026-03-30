// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Adaptive quadrature over a triangle.
*/

#include "quadrature/triangle/adaptive.hpp"

#include <functional>

#include "types.hpp"
#include "constants.hpp"
#include "geometry/primitives/triangle.hpp"
#include "quadrature/utility.hpp"
#include "quadrature/triangle/iterative_gauss.hpp"


namespace bem
{

template <uint8_t dim>
void AdaptiveTriangleQuadrature<dim>::compute_points_weights(
    const Triangle<dim>& tri,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval
    )
{
    if (!eval)
        throw std::invalid_argument(
            "AdaptiveTriangleQuadrature::compute_points_weights(): invalid or missing eval."
            );

    base::points_.resize(dim, 0);
    base::weights_.resize(1, 0);
    vals_.resize(1, 0);
    reset_recursion_count();

    // The adaptive routine requires at least 11 function evals
    // As a first step, try iterative Gaussian quadrature with four
    // evals to see if that's already enough, to possibly save time.
    IterativeGaussTriangleQuadrature<dim> iq;
    iq.set_tol(tol_);
    iq.set_max_iters(2);
    iq.set_starting_order(1);
    iq.compute_points_weights(tri, eval);
    if (iq.converged())
    {
        base::points_ = iq.points();
        base::weights_ = iq.weights();
        base::points_weights_computed_ = true;
        vals_ = eval(base::points_);
        converged_ = true;
        return;
    }

    int level = 0;
    uint8_t longest_edge = tri.longest_edge_index();

    get_5_points(p_main_, tri, longest_edge);
    vals_main_ = eval(p_main_);

    run_recursion(level, tri, eval, longest_edge);

    if (level == max_levels_)
    {
        base::points_weights_computed_ = true;
        converged_ = false;
        return;
    }

    converged_iter_ = level;
    base::points_weights_computed_ = true;
    converged_ = true;

};


template <uint8_t dim>
Complex AdaptiveTriangleQuadrature<dim>::run_recursion(
    int& level,
    const Triangle<dim>& tri,
    std::function<EigRowVec<Complex> (ConstEigRef<EigMatNX<Float, dim>>)> eval,
    const uint8_t longest_edge
    )
{

    level++;

    EigMatMN<Float, dim, 3> v_subtri1, v_subtri2;
    get_subtris(v_subtri1, v_subtri2, tri, longest_edge);

    Triangle<dim> subtri1 (v_subtri1), subtri2 (v_subtri2);

    // if (level == 2) std::cout << "level: " << level << "\n" << subtri1(0) << "\n--\n" << subtri1(1) << "\n--\n" << subtri1(2) << "\n==" << std::endl;
    // if (level == 2) std::cout << "level: " << level << "\n" << subtri2(0) << "\n--\n" << subtri2(1) << "\n--\n" << subtri2(2) << "\n==" << std::endl;

    uint8_t longest_edge1 = subtri1.longest_edge_index();
    uint8_t longest_edge2 = subtri2.longest_edge_index();

    EigMatMN<Float, dim, 5> p_sub1, p_sub2;
    get_5_points(p_sub1, subtri1, longest_edge1);
    get_5_points(p_sub2, subtri2, longest_edge2);

    // if (level == 2) std::cout << "level: " << level << "\n" << p_sub1[0] << "\n--\n" << p_sub1[1] << "\n--\n" << p_sub1[2] << "\n--\n" << p_sub1[3] << "\n--\n" << p_sub1[4] << "\n==" << std::endl;
    // if (level == 2) std::cout << "level: " << level << "\n" << p_sub2[0] << "\n--\n" << p_sub2[1] << "\n--\n" << p_sub2[2] << "\n--\n" << p_sub2[3] << "\n--\n" << p_sub2[4] << "\n==" << std::endl;

    p_sub_.col(0) = p_sub1.col(0);
    p_sub_.col(1) = p_sub1.col(1);
    p_sub_.col(2) = p_sub1.col(4);
    p_sub_.col(3) = p_sub2.col(0);
    p_sub_.col(4) = p_sub2.col(1);
    p_sub_.col(5) = p_sub2.col(4);

    // if (level == 1) std::cout << "level: " << level << "\n" << p_sub_[0] << "\n--\n" << p_sub_[1] << "\n--\n" << p_sub_[2] << "\n--\n" << p_sub_[3] << "\n--\n" << p_sub_[4] << "\n--\n" << p_sub_[5] << "\n==" << std::endl;

    vals_sub_ = eval(p_sub_);

    Float area1 = subtri1.area();
    Float area2 = subtri2.area();
    Float area = area1 + area2;

    // 5-point integral result
    Complex I5 = (vals_main_.sum() + vals_main_[4]) * area / (Float)6.0;

    // 10-point integral result
    Complex I10 = vals_sub_.sum() + vals_sub_[2] + vals_sub_[5];

    if (longest_edge1 == 1)
    {
        idx_extra1_[0] = 4;
        idx_extra1_[1] = 3;
    }
    else if (longest_edge1 == 2)
    {
        idx_extra1_[0] = 3;
        idx_extra1_[1] = 0;
    }
    else if (longest_edge1 == 0)
    {
        idx_extra1_[0] = 0;
        idx_extra1_[1] = 4;
    }
    I10 += vals_main_[idx_extra1_[0]] + vals_main_[idx_extra1_[1]];

    if (longest_edge2 == 1)
    {
        idx_extra2_[0] = 2;
        idx_extra2_[1] = 4;
    }
    else if (longest_edge2 == 2)
    {
        idx_extra2_[0] = 4;
        idx_extra2_[1] = 1;
    }
    else if (longest_edge2 == 0)
    {
        idx_extra2_[0] = 1;
        idx_extra2_[1] = 2;
    }
    I10 += vals_main_[idx_extra2_[0]] + vals_main_[idx_extra2_[1]];
    I10 *= area / 12.0;

    // if (level == 2) std::cout << "level: " << level << ", " << (int)idx_extra1_[0] << ", " << (int)idx_extra1_[1] << "\n==" << std::endl;
    // if (level == 2) std::cout << "level: " << level << ", " << (int)idx_extra2_[0] << ", " << (int)idx_extra2_[1] << "\n==" << std::endl;

    bool converged = check_convergence(I5, I10);

    // {
    // 	std::cout << tri.v[0] << ", " << std::endl << tri.v[1] << ", " << std::endl << tri.v[2] << ", " << std::endl << "---" << std::endl;
    // 	for (uint8_t ii = 0; ii < 5; ii++)
    // 		std::cout << p[ii] << ", " << std::endl;
    // 	std::cout << "---" << std::endl;
    // 	for (uint8_t ii = 0; ii < 5; ii++)
    // 		std::cout << p_sub1[ii] << ", " << std::endl;
    // 	std::cout << "---" << std::endl;
    // 	for (uint8_t ii = 0; ii < 5; ii++)
    // 		std::cout << p_sub2[ii] << ", " << std::endl;
    // 	std::cout << "------" << std::endl;
    // 	std::cout << idx_extra1_[0] << ", " << idx_extra1_[1] << ", " << idx_extra2_[0] << ", " << idx_extra2_[1] << std::endl;
    // 	std::cout << "------------" << std::endl;
    // }

    if (converged || level == max_levels_)
    {

        const uint16_t num_add = 10;
        const uint16_t idx_add = base::points_.cols();

        base::points_.conservativeResize(dim, base::points_.cols() + num_add);
        base::weights_.conservativeResize(1, base::weights_.size() + num_add);
        vals_.conservativeResize(1, vals_.size() + num_add);

        for (uint8_t ii = 0; ii < 2; ii++)
        {
            base::points_.col(idx_add + 2 * ii) = p_main_.col(idx_extra1_[ii]);
            base::points_.col(idx_add + 2 * ii + 1) = p_main_.col(idx_extra2_[ii]);

            vals_[idx_add + 2 * ii] = vals_main_[idx_extra1_[ii]];
            vals_[idx_add + 2 * ii + 1] = vals_main_[idx_extra2_[ii]];

            // if (level == 2) std::cout << "level: " << level << "\n" << points_[points_.cols()-2] << "\n--\n" << points_[points_.cols()-1] << "\n==" << std::endl;
        }

        base::points_.rightCols(p_sub_.cols()) = p_sub_;
        base::weights_.rightCols(10) = weights10_ * area / (Float)12.0;
        vals_.rightCols(vals_sub_.size()) = vals_sub_;

        total_recursions_++;

        return I10;

    }

    // ------ Subtriangle 1 ------

    Complex I3_subtri1 = (vals_main_[0] + vals_main_[3] + vals_main_[4]) * area1 / (Float)3.0;

    vals_subtri1_ <<
        vals_sub_[0],
        vals_sub_[1],
        vals_main_[idx_extra1_[0]],
        vals_main_[idx_extra1_[1]],
        vals_sub_[2];

    Complex I5_subtri1 = (vals_subtri1_.sum() + vals_subtri1_[4]) * area1 / (Float)6.0;

    // ------ Subtriangle 2 ------

    Complex I3_subtri2 = (vals_main_[1] + vals_main_[2] + vals_main_[4]) * area2 / (Float)3.0;

    vals_subtri2_ <<
        vals_sub_[3],
        vals_sub_[4],
        vals_main_[idx_extra2_[0]],
        vals_main_[idx_extra2_[1]],
        vals_sub_[5];

    Complex I5_subtri2 = (vals_subtri2_.sum() + vals_subtri2_[4]) * area2 / (Float)6.0;

    // ------ Subtriangle recursion ------

    Complex I10_subtri1;
    bool converged_subtri1 = check_convergence(I3_subtri1, I5_subtri1);
    if (!converged_subtri1)
    {
        p_main_ = p_sub1;
        vals_main_ = vals_subtri1_;
        I10_subtri1 = run_recursion(level, subtri1, eval, longest_edge1);
        level--;
    }
    else
    {
        const uint16_t num_add = 5;

        base::points_.conservativeResize(dim, base::points_.cols() + num_add);
        base::points_.rightCols(num_add) = p_sub1;

        base::weights_.conservativeResize(1, base::weights_.size() + num_add);
        base::weights_.rightCols(num_add) = weights5_ * area1 / (Float)6.0;

        vals_.conservativeResize(1, vals_.size() + num_add);
        vals_.rightCols(num_add) = vals_subtri1_;

        I10_subtri1 = I5_subtri1;
    }

    Complex I10_subtri2;
    bool converged_subtri2 = check_convergence(I3_subtri2, I5_subtri2);
    if (!converged_subtri2)
    {
        p_main_ = p_sub2;
        vals_main_ = vals_subtri2_;
        I10_subtri2 = run_recursion(level, subtri2, eval, longest_edge2);
        level--;
    }
    else
    {
        const uint16_t num_add = 5;

        base::points_.conservativeResize(dim, base::points_.cols() + num_add);
        base::points_.rightCols(num_add) = p_sub2;

        base::weights_.conservativeResize(1, base::weights_.size() + num_add);
        base::weights_.rightCols(num_add) = weights5_ * area2 / (Float)6.0;

        vals_.conservativeResize(1, vals_.size() + num_add);
        vals_.rightCols(num_add) = vals_subtri2_;

        I10_subtri2 = I5_subtri2;
    }

    return I10_subtri1 + I10_subtri2;

}

template <uint8_t dim>
void AdaptiveTriangleQuadrature<dim>::get_5_points(
    EigMatMN<Float, dim, 5>& p,
    const Triangle<dim>& tri,
    const uint8_t longest_edge
    )
{

    const EigColVecN<Float, dim>& v0 = tri.v((longest_edge + 2) % 3);
    const EigColVecN<Float, dim>& v1 = tri.v(longest_edge);
    const EigColVecN<Float, dim>& v2 = tri.v((longest_edge + 1) % 3);

    const EigColVecN<Float, dim> vm = (v1 + v2) / two;

    p.col(0) = (v1 + vm) / two;
    p.col(1) = (v2 + vm) / two;
    p.col(2) = (v0 + v2) / two;
    p.col(3) = (v0 + v1) / two;
    p.col(4) = (v0 + vm) / two;

    return;

}


template <uint8_t dim>
void AdaptiveTriangleQuadrature<dim>::get_subtris(
    EigMatMN<Float, dim, 3>& v_subtri1,
    EigMatMN<Float, dim, 3>& v_subtri2,
    const Triangle<dim>& tri,
    const uint8_t longest_edge
    )
{
    const EigColVecN<Float, dim>& v0 = tri.v((longest_edge + 2) % 3);
    const EigColVecN<Float, dim>& v1 = tri.v(longest_edge);
    const EigColVecN<Float, dim>& v2 = tri.v((longest_edge + 1) % 3);

    const EigColVecN<Float, dim> vm = (v1 + v2) / two;

    v_subtri1.col(0) = v0;
    v_subtri1.col(1) = v1;
    v_subtri1.col(2) = vm;

    v_subtri2.col(0) = v0;
    v_subtri2.col(1) = vm;
    v_subtri2.col(2) = v2;

    return;
};


template class AdaptiveTriangleQuadrature<2>;
template class AdaptiveTriangleQuadrature<3>;

}
