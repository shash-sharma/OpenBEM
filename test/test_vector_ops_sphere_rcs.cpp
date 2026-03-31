// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


#include <iostream>

#include "io.hpp"

#include "types.hpp"
#include "constants.hpp"

#include "matrix/eigen_dense.hpp"

#include "geometry/point_cloud.hpp"
#include "geometry/structure.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/mesh/mesh_transfer.hpp"

#include "quadrature/triangle/gauss.hpp"
#include "quadrature/line/gauss.hpp"

#include "rwg/assemblers/operator_matrix.hpp"
#include "rwg/assemblers/excitation_matrix.hpp"
#include "rwg/assemblers/projector_matrix.hpp"

#include "rwg/integrators/src/strategic.hpp"
#include "rwg/integrators/obs/quadrature.hpp"

#include "rwg/operators/gram.hpp"
#include "rwg/operators/incidence.hpp"
#include "rwg/operators/vector_ops.hpp"

#include "rwg/excitations/plane_wave.hpp"

#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/function_space.hpp"


using namespace bem;
using namespace bem::rwg;
using json = nlohmann::json;


void test_cfie_pec()
{

    std::size_t path_pos = std::string(__FILE__).find_last_of("/");
    std::string path = std::string(__FILE__).substr(0, path_pos) + "/";

    std::string msh_filename = path + "/msh/sphere.msh";

    Structure<TriangleMesh<3>> structure;
    MeshTransfer::read_gmsh_v2(structure, msh_filename);
    TriangleMesh<3>& mesh = structure.mesh();

    Float f = 250e6;
    Float omega = two_pi * f;
    Float mu = mu0;
    Float eps = eps0;
    Complex k = omega * std::sqrt(eps * mu);
    Float lambda = std::real(two_pi / k);
    Float dist = 1e2 * lambda;


    VectorRwgOps ops;
    VectorOperatorsAssembler assembler (mesh, mesh);

    EigenDenseMatrix<Complex> mats;
    assembler.assemble(mats, ops, k);

    RwgRwgOp op_I;
    EdgeOperatorAssembler I_assembler (mesh, mesh);


    EigenDenseMatrix<Complex> T;
    mats.get_block(T, 0, 0, mesh.num_edges(), mesh.num_edges());
    T.scale(-J * omega * mu);

    EigenDenseMatrix<Complex> K;
    mats.get_block(K, mesh.num_edges() * 3, 0, mesh.num_edges(), mesh.num_edges());

    EigenDenseMatrix<Complex> I;
    I_assembler.assemble(I, op_I, k);
    I.scale(half);

    EigenDenseMatrix<Complex> A;
    A.set_axpby(T, K);
    A.add_ax(I);


    EigColVecN<Float, 3> dir, pol_e, pol_h, pos;
    EigRowVecN<Complex, 1> amp;
    dir << 0, 0, 1;
    pol_e << 1, 0, 0;
    pol_h << 0, 1, 0;
    pos << 0, 0, -dist;
    amp << 1;

    RwgPlaneWave pwe (dir, pol_e, pos, amp);
    NxRwgPlaneWave pwh (dir, pol_h, pos, amp / std::sqrt(mu / eps));
    EdgeExcitationAssembler pw_assembler (mesh);

    EigenDenseMatrix<Complex> Einc;
    pw_assembler.assemble(Einc, pwe, k);

    EigenDenseMatrix<Complex> Hinc;
    pw_assembler.assemble(Hinc, pwh, k);

    EigenDenseMatrix<Complex> b;
    b.set_axpby(Einc, Hinc);


    EigenDenseMatrix<Complex> x;
    A.mat_solve(x, b);


    EigColVecN<Float, 3> start, stop;
    EigColVecN<Index, 3> num_pts;
    start << dist, 0, 0;
    stop << dist, 0, pi;
    num_pts << 1, 1, 100;
    EigColVecN<Float, 3> center;
    center << 0, 0, 0;

    PointCloud<3> cloud;
    cloud.set_polar_data(start, stop, center, num_pts);


    VectorHypersingularProj op_T_proj;
    EdgeProjectorAssembler<3> T_proj_assembler (cloud, mesh);

    EigenDenseMatrix<Complex> T_proj;
    T_proj_assembler.assemble(T_proj, op_T_proj, k);
    T_proj.scale(-J * omega * mu);


    EigenDenseMatrix<Complex> Escat;
    Escat.set_mat_mul(T_proj, x);

    EigenDenseMatrix<Float> Escatmag;
    Escatmag.raw_matrix() = Escat.raw_matrix().reshaped(3, 100).colwise().norm();

    EigenDenseMatrix<Float> rcs;
    rcs.raw_matrix() = Eigen::pow(Escatmag.raw_matrix().array(), 2) * four_pi * std::pow(dist, 2);

    std::ifstream in_stream (path + "/ref/sphere_pec_ref.json");
    json sphere_ref = json::parse(in_stream);

    EigMat<Float> rcs_ref = sphere_ref["rcs"].template get<EigMat<Float>> ();
    EigMat<Float> Escatmag_ref = sphere_ref["escat_mag"].template get<EigMat<Float>> ();

    json out;
    out["escat_mag"] = Escatmag.raw_matrix();
    out["rcs"] = rcs.raw_matrix();

    std::ofstream out_stream (path + "/dump/sphere_efie_pec.json");
    out_stream << std::setw(4) << out << std::endl;

    Float rcs_err = (
        Eigen::abs((rcs_ref - rcs.raw_matrix()).array()) / Eigen::abs(rcs_ref.array())
        ).maxCoeff();
    Float escat_err = (
        Eigen::abs((Escatmag_ref - Escatmag.raw_matrix()).array()) / Eigen::abs(Escatmag_ref.array())
        ).maxCoeff();

    if (rcs_err > 2.2e-2 || escat_err > 1.2e-2)
    {
        std::cout << "====== test_cfie_pec ======" << std::endl;
        std::cout << "Fail:" << std::endl;
        std::cout << "--- RCS error " << rcs_err * 100 << " %" << std::endl;
        std::cout << "--- Escat_mag error " << escat_err * 100 << " %" << std::endl;
    }

    return;

}


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "test_vector_set_sphere_rcs.cpp" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    test_cfie_pec();

    return 0;

}


