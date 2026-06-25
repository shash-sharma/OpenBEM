// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file Example 2: Defining operators and assembling them into matrices for more customized integral
* equations. Repeats the computations in Example 1, but with user-defined setup of the integral
* operators and their assembly into matrices, for more customization. Assumes knowledge from Example
* 1.
*/

#include <iostream>
#include <string>

// The following are OpenBEM-specific headers that we need to include for this example to run. The
// specific functionality associated with each header included will be indicated in the main code.

#include "types.hpp"
#include "constants.hpp"

#include "matrix/eigen_matrix.hpp"

#include "geometry/structure.hpp"
#include "geometry/mesh/mesh_transfer.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/point_cloud.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "rwg/excitations/plane_wave.hpp"

#include "rwg/projectors/single_layer.hpp"
#include "rwg/projectors/double_layer.hpp"

#include "rwg/assemblers/operator_assembler.hpp"
#include "rwg/assemblers/excitation_assembler.hpp"
#include "rwg/assemblers/projector_assembler.hpp"

#include "rwg/integral_equations/tefie.hpp"


// Define the matrix data type, as in Example 1.

using MatrixType = bem::EigenMatrix<bem::Complex>;


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "OpenBEM example 2" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    // Read in the mesh file as in Example 1.

    std::size_t path_pos = std::string(__FILE__).find_last_of("/");
    std::string path = std::string(__FILE__).substr(0, path_pos) + "/";

    std::string msh_filename = path + "msh/sphere.msh";

    bem::Structure<bem::TriangleMesh<3>> structure;
    bem::MeshTransfer::read_gmsh_v2(structure, msh_filename);

    std::cout << "Number of vertices: " << structure.mesh().num_verts() << std::endl;
    std::cout << "Number of triangles: " << structure.mesh().num_elems() << std::endl;
    std::cout << "Number of edges: " << structure.mesh().num_edges() << std::endl;

    // Set the simulation frequency in Hz, and get the background medium (vacuum by default) wave
    // number and permeability for later use. See the `Material` class for more details.

    bem::Float f = 250e6;
    bem::Complex k = structure.background_material().k(f);
    bem::Complex mu = structure.background_material().mu();

    // This time, instead of using the the `Tefie` and `Tmfie` classes to implement the respective
    // integral equations, the associated operators will be defined and assembled into matrices
    // manually, to demonstrate the ability to create customized integral equations and operators.

    // The basic procedure is to first define an operator, and then pass it into an assembler, as
    // shown step-by-step below.

    // For the TEFIE, we require the tangentially tested vector hypersingular operator (usually
    // referred to as the L- or T-operator in the literature). For the NMFIE, we'll need the
    // rotationally tested vector double layer operator.

    // Both of these operators, as well as other commonly used operators, are predefined and
    // available in OpenBEM. To use them, we need the `rwg/operators/single_layer.hpp` and
    // `rwg/operators/double_layer.hpp` headers.

    bem::rwg::VectorHypersingularOp L_operator;
    bem::rwg::RotVectorDoubleLayerPvOp Kpv_operator;

    // First, note that the operators can take an optional constructor argument to define the source
    // and observation triangle integration routines for computing the operator.  Using custom
    // integration settings will be discussed in later examples; here, we just use the default by
    // providing no constructor arguments.

    // Second, for the double layer operator, its singularity must be handled separately in a
    // residue-and-principal-value sense, as is standard in the literature. The `Pv` in the class
    // name above indicates that self-terms (i.e. the interaction between two fully-overlapping
    // triangles) are ignored in the operator computations, and those contributions will be
    // considered separately below.

    // Third, the `Rot` in the double layer operator class name indicates that this is the
    // rotationally tested version of the operator. If we were using the TMFIE instead, we would
    // have used the `VectorDoubleLayerPvOp` class, without the `Rot`.

    // For the self-interactions involving the singularity, we compute the identity operator
    // separately, as follows. This requires the `rwg/operators/gram.hpp` header.

    bem::rwg::VectorIdentityOp I_operator;

    // Each operator has a `compute()` method which takes the wave number, a source triangle, and an
    // observation triangle as argument, to compute the operator values associated with that pair of
    // triangles, for each associated degree of freedom, for the given wave number.

    // However, we want to assemble the operator values for all the triangles in the mesh. Instead
    // of having to write loops over triangles manually, OpenBEM provides assembler classes to do
    // this automatically.

    // For each operator, we define an `OperatorAssembler` to assemble the matrix. This requires the
    // header `rwg/assemblers/operator_assembler.hpp`.

    bem::rwg::OperatorAssembler assembler (structure.mesh());

    // The assembler takes the mesh as its constructor argument. Once created, we can perform the
    // actual matrix assembly for both operators using the assember's `assemble()` method.

    // Note: this is the step when the BEM matrices are actually assembled by computing
    // triangle-triangle integrations and computing the associated Green's functions, so it is a
    // computationally heavy step that may take a couple minutes, depending on your machine and
    // access to OpenMP threads.

    MatrixType L, K, I;
    assembler.assemble(L, L_operator, k);
    assembler.assemble(K, Kpv_operator, k);
    assembler.assemble(I, I_operator, k);

    // For the TEFIE, the L operator must be scaled appropriately.

    L.scale(-bem::J * bem::two_pi * f * mu);

    // For the NMFIE, we need to combine the identity operator (halved) with the principal value K
    // operator, using the appropriate sign, keeping in mind that the unit normal vectors point out
    // of the sphere and into the region of interest (free space). See the documentation for the
    // `MatrixBase` class for more information about the matrix operations used below.

    K.add_ax(I, 0.5);

    // This time, instead of solving the TEFIE and NMFIE individually, we'll combine them in equal
    // parts to obtain the CFIE formulation for PECs. In the following, `A` is the combined-field
    // system matrix that we will eventually solve, and alpha is the CFIE combination factor.

    bem::Float alpha = 0.5;

    MatrixType A;
    A.set_axpby(L, K, alpha, 1.0 - alpha);

    // Now let's create the right-hand side vectors that represent a plane wave excitation,
    // similarly to Example 1. However, again, instead of using the `Tefie` and `Tmfie` classes to
    // help with this, we'll use an assembler to manually assemble the excitation vectors.

    // First, as in example 1, we'll set the direction, polarization, and phase reference.

    bem::EigColVecN<bem::Float, 3> dir = { 0, 0, 1 };
    bem::EigColVecN<bem::Float, 3> pol_e = { 1, 0, 0 };
    bem::EigColVecN<bem::Float, 3> pol_h = { 0, 1, 0 };

    bem::Float dist = 100 * (bem::c0 / f);
    bem::EigColVecN<bem::Float, 3> pos = -dir * dist;

    bem::EigColVecN<bem::Complex, 1> amp_e { 1 };
    bem::EigColVecN<bem::Complex, 1> amp_h { 1 / bem::eta0 };

    // We'll create the plane wave classes as before, for both the TEFIE and the NMFIE.

    bem::rwg::RwgPlaneWave pw_e (dir, pol_e, pos, amp_e);
    bem::rwg::NxRwgPlaneWave pw_h (dir, pol_h, pos, amp_h);

    // Now we'll use an `ExcitationAssembler` which works similarly to the `OperatorAssembler`
    // above. This requires the header `rwg/assemblers/excitation_assembler.hpp`.

    bem::rwg::ExcitationAssembler exc_assembler (structure.mesh());

    // Now we'll use the assembler and its `assemble()` method, providing each plane wave object, to
    // obtain the excitation vectors for the TEFIE and the NMFIE.

    MatrixType inc_e;
    exc_assembler.assemble(inc_e, pw_e, k);

    MatrixType inc_h;
    exc_assembler.assemble(inc_h, pw_h, k);

    // Since we're solving the CFIE, we need to also combine the excitation matrices.

    MatrixType b;
    b.set_axpby(inc_e, inc_h, alpha, 1.0 - alpha);

    // Finally, we solve the system of equations to obtain the electric surface current density.

    MatrixType x;
    A.factorize();
    A.mat_solve(x, b);

    // As in Example 1, the solution matrix will have as many columns as the number of excitation
    // vectors, which in this case is just one.

    // For comparison purposes, let's also compute the RCS using the `Tefie` class as in Example 1.

    bem::rwg::Tefie<MatrixType> tefie (structure.mesh());
    MatrixType A_tefie = tefie.j_matrix(f, structure.background_material());

    MatrixType x_tefie;
    A_tefie.factorize();
    A_tefie.mat_solve(x_tefie, inc_e);

    // Now we'll follow the same approach as in Example 1 to define a point cloud in the far field
    // on which to project the scattered field and compute the RCS, via our manually-built CFIE and
    // the built-in `Tefie` class.

    bem::PointCloud<3> projection_points;

    bem::EigColVecN<bem::Float, 3> arc_begin = { dist, 0, 0 };
    bem::EigColVecN<bem::Float, 3> arc_end = { dist, 0, bem::pi };

    bem::EigColVecN<bem::Index, 3> num_pts = { 1, 1, 100 };
    bem::EigColVecN<bem::Float, 3> center = { 0, 0, 0 };

    projection_points.set_polar_data(arc_begin, arc_end, center, num_pts);

    // Since we're learning the concept of assemblers in this example, we will also build the
    // E-field projector ourselves.

    // First, similarly to how we needed an operator object to assemble the integral equation
    // operators, we need a projector object that represents the same integral equation, but where
    // the observation points are a point cloud rather than a Galerkin testing scheme.

    // As in Example 1, all we're doing is evaluating the integral equation operators at a given set
    // of observation points. Since we just want to project the E-field here, to get the RCS, we
    // only need the projector associated with the EFIE.

    // The projector is defined analogously to the BEM operators above. This requires the header
    // `rwg/projectors/single_layer.hpp`.

    bem::rwg::VectorHypersingularProj<> L_projector;

    // Next, similar to the RWG operators, define an assembler to assemble the projector matrix,
    // using the `ProjectorAssembler` class, which works in much the same way as the assembler
    // classes we've already seen. This requires the header
    // `rwg/assemblers/projector_assembler.hpp`.

    bem::rwg::ProjectorAssembler<3> proj_assembler (projection_points, structure.mesh());

    // The template argument `3` indicates the dimension of the projected field, which is 3 in this
    // case since it's a vector field.

    // Analogously to the operators, let's construct the projector matrix by passing the projector
    // into its assembler, with the appropriate L-operator scaling.

    MatrixType Lproj;
    proj_assembler.assemble(Lproj, L_projector, k);
    Lproj.scale(-bem::J * bem::two_pi * f * mu);

    // Now let's put the projector into action by computing the electric field on our point cloud
    // for the manually-constructed CFIE case.

    MatrixType e_cfie;
    e_cfie.set_matmul(Lproj, x);

    // We'll also do the same using the built-in `Tefie` class, as in Example 1, just for
    // comparison.

    MatrixType tefie_proj = tefie.j_projector(f, structure.background_material(), projection_points);
    MatrixType e_tefie;
    e_tefie.set_matmul(tefie_proj, x_tefie);

    // As in Example 1, the components of the projected fields are stacked consecutively, so we
    // should reshape the projected fields into a more convenient shape.

    e_cfie.raw_matrix() = e_cfie.raw_matrix().reshaped(3, 100);
    e_tefie.raw_matrix() = e_tefie.raw_matrix().reshaped(3, 100);

    // Finally, let's compute the RCS using the manual CFIE and built-in TEFIE.

    MatrixType e_cfie_mag;
    e_cfie_mag.raw_matrix() = e_cfie.raw_matrix().colwise().norm();

    MatrixType e_tefie_mag;
    e_tefie_mag.raw_matrix() = e_tefie.raw_matrix().colwise().norm();

    MatrixType rcs_cfie;
    rcs_cfie.raw_matrix() = Eigen::pow(e_cfie_mag.raw_matrix().array(), 2) * bem::four_pi * std::pow(dist, 2);

    MatrixType rcs_tefie;
    rcs_tefie.raw_matrix() = Eigen::pow(e_tefie_mag.raw_matrix().array(), 2) * bem::four_pi * std::pow(dist, 2);

    // Next, let's compute and display the worst-case point-wise relative error in RCS between the
    // CFIE and TEFIE solutions.

    MatrixType rcs_error;
    rcs_error.raw_matrix() = (rcs_cfie.raw_matrix() - rcs_tefie.raw_matrix()).array().abs();

    MatrixType rcs_relative_error;
    rcs_relative_error.raw_matrix() = rcs_error.raw_matrix().array() / rcs_tefie.raw_matrix().array().abs();

    bem::Float rcs_max_relative_error = rcs_relative_error.raw_matrix().array().abs().maxCoeff();
    std::cout << "CFIE vs. TEFIE RCS maximum relative error: "
              << rcs_max_relative_error * 100
              << " %" << std::endl;

    // The worst-case relative error between the two methods is less than 0.5%.

    // Finally, we'll export the surface current density computed with the CFIE into the Gmsh
    // post-processing format, as in Example 1.

    bem::EigMatNX<bem::Float, 3> j_surf = bem::Rwg::reconstruct_field(
        structure.mesh(), // mesh with which the field is associated
        x, // RWG coefficients computed using the CFIE
        structure.mesh().elem_centroids() // points at which to compute field values
        ).array().real(); // Eigen syntax to keep only the real part of the computed currents

    bem::MeshTransfer::write_gmsh_v2_vector_field(structure, path + "msh/ex02_jsurf_cfie", j_surf);

    // Now you are equipped to set up your own integral equations in any way of combining different
    // operators that you'd like. Enjoy!

    return 0;

}

