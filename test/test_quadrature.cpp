// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>
#include <ctime>
#include <limits>
#include <cmath>
#include <vector>
#include <array>
#include <memory>

#include "types.hpp"
#include "geometry/operations.hpp"

#include "quadrature/triangle/gauss.hpp"
#include "quadrature/triangle/iterative_gauss.hpp"
#include "quadrature/triangle/adaptive.hpp"

#include "quadrature/line/gauss.hpp"
#include "quadrature/line/iterative_gauss.hpp"
#include "quadrature/line/trapz.hpp"
#include "quadrature/line/iterative_trapz.hpp"


using namespace bem;


void test_line_integration()
{

    // 3D
    EigColVecN<Float, 3> p1 (1.0, 2.5, -3.0);
    EigColVecN<Float, 3> p2 (2.0, -10.0, -2.0);
    // std::cout << std::setprecision(15) << (p2-p1).norm() << std::endl;

    struct Eval
    {
        EigColVecN<Float, 3> p1, p2;
        bool fast;
        Eval(EigColVecN<Float, 3> &_p1, EigColVecN<Float, 3> &_p2, bool _fast) { p1 = _p1; p2 = _p2; fast = _fast; }

        EigRowVec<Complex> operator() (const EigMatNX<Float, 3> &r)
        {
            EigRowVec<Complex> vals = EigRowVec<Complex>::Zero(1, r.cols());
            for (uint32_t ii = 0; ii < r.cols(); ++ii)
            {
                Float arg = (r.col(ii) - p1).norm();
                if (!fast)
                    vals[ii] = std::sin(2.0 * pi * arg / (p2 - p1).norm());
                else
                    vals[ii] = std::exp(-arg) * std::sin(3.0 * arg);
            }
            return vals;
        }
    };

    std::vector<Eval> evals = {
        // Eval(p1, p2, false),
        // Eval(p1, p2, true),
        // Eval(p1, p2, false),
        // Eval(p1, p2, true)
        Eval(p1, p2, true),
        Eval(p1, p2, true),
        Eval(p1, p2, true),
        Eval(p1, p2, true)
    };

    std::vector<std::shared_ptr<LineQuadratureBase<3>>> lq (4);
    lq[0] = std::make_shared<GaussLineQuadrature<3>> (GaussLineQuadrature<3> (10));
    lq[1] = std::make_shared<IterativeGaussLineQuadrature<3>> (IterativeGaussLineQuadrature<3> ());
    lq[2] = std::make_shared<TrapzLineQuadrature<3>> (TrapzLineQuadrature<3> (100));
    lq[3] = std::make_shared<IterativeTrapzLineQuadrature<3>> (IterativeTrapzLineQuadrature<3> ());

    std::vector<Float> tols = { 1.1e-2, 1.3e-4, 1.4e-2, 1.3e-4 };

    for (uint8_t ii = 0; ii < lq.size(); ++ii)
    {
        // std::cout << "------" << std::endl;

        // To check that all the methods compute the length of the line correctly
        QuadratureData<3> qd = lq[ii]->compute(p1, p2, evals[ii]);
        EigRowVec<Complex> vals = evals[ii](qd.points);
        Complex result = qd.weights.dot(vals);

        Float ref_val, rel_err;
        if (evals[ii].fast)
        {
            // source: https://www.wolframalpha.com/input?i=integrate+exp%28-x%29+sin+%283*x%29+dx+from+0+to+12.5797456254091
            ref_val = 0.299999;
            rel_err = std::abs(ref_val - std::abs(result)) / ref_val;
        }
        else
        {
            ref_val = 0.0;
            rel_err = std::abs(ref_val - std::abs(result));
        }

        if (rel_err > tols[ii])
        {
            std::cout << "ref. val: " << ref_val << ", result: " << result << ", rel. error: " << rel_err << std::endl;
            throw std::runtime_error(
                "test_line_integration(): tolerance violation for line integral calculation, quadrature method " + std::to_string(ii) + "."
            );
        }

    }

    return;
}


// Test that all methods of integration agree with one another within set tolerances
int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_quadrature.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    test_line_integration();

    return 0;

}

