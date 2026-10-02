// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file Example 3: Setting the source and observation triangle integrators explicitly. Demonstrates
* how to create custom integrators. Assumes knowledge from Examples 1 and 2.
*/

#include <iostream>
#include <string>

// The following are OpenBEM-specific headers that we need to include for this example to run. The
// specific functionality associated with each header will be indicated in the main code.

#include "types.hpp"
#include "constants.hpp"

#include "matrix/eigen_matrix.hpp"

#include "geometry/structure.hpp"
#include "geometry/mesh/mesh_transfer.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "geometry/primitives/triangle.hpp"
#include "geometry/point_cloud.hpp"

#include "quadrature/triangle/gauss.hpp"

#include "rwg/integrators/src/base.hpp"
#include "rwg/integrators/src/quadrature.hpp"
#include "rwg/integrators/src/singularity.hpp"
#include "rwg/integrators/obs/quadrature.hpp"

#include "rwg/operators/single_layer.hpp"
#include "rwg/operators/double_layer.hpp"
#include "rwg/operators/gram.hpp"

#include "rwg/excitations/plane_wave.hpp"

#include "rwg/projectors/single_layer.hpp"

#include "rwg/assemblers/operator_assembler.hpp"
#include "rwg/assemblers/excitation_assembler.hpp"
#include "rwg/assemblers/projector_assembler.hpp"


// Define the matrix data type, as in Examples 1 and 2.

using MatrixType = bem::EigenMatrix<bem::Complex>;


int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "OpenBEM example 3" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    // Read in the mesh file as in Example 1.

    std::size_t path_pos = std::string(__FILE__).find_last_of("/");
    std::string path = std::string(__FILE__).substr(0, path_pos) + "/";

    std::string msh_filename = path + "msh/sphere.msh";

    bem::Structure<3> structure;
    bem::MeshTransfer::read_gmsh_v2(structure, msh_filename);

    std::cout << "Number of vertices: " << structure.mesh().num_vertices() << std::endl;
    std::cout << "Number of triangles: " << structure.mesh().num_faces() << std::endl;
    std::cout << "Number of edges: " << structure.mesh().num_edges() << std::endl;

    // Set the simulation frequency, wave number, and permeability, as in Example 2.

    bem::Float f = 250e6;
    bem::Complex k = structure.background_material().k(f);
    bem::Complex mu = structure.background_material().mu();

    // Each entry of an RWG operator matrix involves a double integral: an outer integral over an
    // observation triangle, and an inner integral over a source triangle. OpenBEM splits these
    // into two separate objects, called integrators.

    // The observation integrator computes the outer integral. At each of its quadrature points, it
    // asks a source integrator to compute the inner integral. The source integrator is responsible
    // for handling any singularities in the kernel, such as those in standard Green's functions.

    // In Examples 1 and 2, we never had to think about this, because the operators default to
    // `ObsStrategic` and `SrcStrategic` integrators, which look at the distance between the
    // triangles, the frequency, and the material, and pick a suitable method automatically. In
    // this example, we'll create our own source integrator, and set the integrators explicitly.

    // OpenBEM ships with several source integrators. The two we'll use here are:

    // - `SrcQuadrature`, which applies plain Gaussian quadrature to the Green's function. This is
    //    accurate when the observation points are well separated from the source triangle. It 
    //    requires the `rwg/integrators/src/quadrature.hpp` header.

    // - `SrcSingularity`, which subtracts the singular part of the Green's function, integrates
    //   that part analytically, and applies quadrature only to the smooth remainder. This is more
    //   expensive, but stays accurate when the observation points are close to, or even on, the
    //   source triangle. It requires the `rwg/integrators/src/singularity.hpp` header.

    // To decide between the two, we'll write our own source integrator. Any class that inherits
    // from `SrcIntegratorBase` (defined in `rwg/integrators/src/base.hpp`) and implements its
    // `integrate()` method can be used as a source integrator anywhere in OpenBEM.

    // Our rule is simple: if any observation point lies within two longest-edge-lengths of the
    // source triangle's centroid, use `SrcSingularity`; otherwise, use `SrcQuadrature`.

    // Since we only need this class here, we can define it right inside `main()`.

    class CustomSrcIntegrator: public bem::rwg::SrcIntegratorBase
    {
    public:

        bem::rwg::SrcResult integrate(
            const bem::Complex k,
            const bem::Triangle<2>& src_tri,
            bem::ConstEigRef<bem::EigMatNX<bem::Float, 3>> r_obs,
            const bool g_terms = true,
            const bool grad_g_terms = true
            ) override
        {

            // Create the two integrators that we'll choose between. Each takes a triangle
            // quadrature object, for where we set the quadrature order explicitly. A higher order
            // is used for the near interactions, and a lower one for the far interactions. These
            // happen to be the same orders that `SrcStrategic` uses by default.

            bem::rwg::SrcSingularity singularity_ { bem::GaussTriangleQuadrature<2>(8) };
            bem::rwg::SrcQuadrature quadrature_ { bem::GaussTriangleQuadrature<2>(4) };

            // Normally, it might be advisable to create the integrator objects as members of the
            // class rather than here, because this method might be called millions of times in a 
            // loop to compute the integrals, which may lead to a lot of unnecessary initialization
            // which we're choosing to ignore for the purposes of this example.

            // Note that `GaussTriangleQuadrature<2>` is a Gaussian quadrature rule over a triangle
            // in 2D, which is what we want because the source triangle lives in its own local
            // xy-plane. It requires the `quadrature/triangle/gauss.hpp` header.

            // The source integrator works in the local coordinate system of the source triangle,
            // which lies in the xy-plane. So `src_tri` is a 2D triangle, and `r_obs` holds the
            // observation points (one per column) expressed in that same local coordinate system.

            // Get the centroid of the source triangle as a 3D point lying in the xy-plane, so that
            // we can compute its distance to the observation points.

            bem::EigColVecN<bem::Float, 3> centroid = bem::EigColVecN<bem::Float, 3>::Zero();
            centroid.topRows(2) = src_tri.centroid();

            // Find the distance from the centroid to the closest observation point.

            bem::Float min_dist = (r_obs.colwise() - centroid).colwise().norm().minCoeff();

            // Now hand the work to whichever integrator our rule picks.

            if (min_dist <= 2 * src_tri.longest_edge_length())
                return singularity_.integrate(k, src_tri, r_obs, g_terms, grad_g_terms);
            else
                return quadrature_.integrate(k, src_tri, r_obs, g_terms, grad_g_terms);

        };

    };

    // Now let's create our source integrator.

    CustomSrcIntegrator src_integrator;

    // Next, we need an observation integrator to compute the outer integral, and we need to tell it
    // to use our source integrator for the inner integral. We'll use `ObsQuadrature`, which applies
    // plain Gaussian quadrature over the observation triangle. It requires the
    // `rwg/integrators/obs/quadrature.hpp` header. Of course, we could create another custom
    // integrator for observation triangles too, but we'll skip that here.

    bem::rwg::ObsQuadrature obs_integrator (bem::GaussTriangleQuadrature<3>(4), src_integrator);

    // The first argument is the quadrature rule over the observation triangle. This time it is a
    // `GaussTriangleQuadrature<3>`, because the observation triangle lives in 3D space. Unlike
    // `ObsStrategic`, `ObsQuadrature` uses a single quadrature order for every pair of triangles.

    // Now we set up the operators manually as in Example 2, but this time, we pass in our
    // observation integrator as a constructor argument. That's all it takes; every integral
    // computed for these operators will now go through `obs_integrator`, and in turn, through
    // `src_integrator`.

    bem::rwg::VectorHypersingularOp L_operator (obs_integrator);
    bem::rwg::RotVectorDoubleLayerPvOp Kpv_operator (obs_integrator);

    // The identity operator doesn't involve a Green's function, and it doesn't need an integrator.

    bem::rwg::VectorIdentityOp I_operator;

    // Assemble the operator matrices as in Example 2.

    bem::rwg::OperatorAssembler assembler (structure.mesh());

    MatrixType L, K, I;
    assembler.assemble(L, L_operator, k);
    assembler.assemble(K, Kpv_operator, k);
    assembler.assemble(I, I_operator, k);

    // Scale the L operator to get the TEFIE operator, and combine the K and identity operators to
    // get the NMFIE operator, as in Example 2.

    L.scale(-bem::J * bem::two_pi * f * mu);
    K.add_ax(I, 0.5);

    // Set up the plane wave excitations as in Examples 1 and 2.

    bem::EigColVecN<bem::Float, 3> dir = { 0, 0, 1 };
    bem::EigColVecN<bem::Float, 3> pol_e = { 1, 0, 0 };
    bem::EigColVecN<bem::Float, 3> pol_h = { 0, 1, 0 };

    bem::Float dist = 100 * (bem::c0 / f);
    bem::EigColVecN<bem::Float, 3> pos = -dir * dist;

    bem::EigColVecN<bem::Complex, 1> amp_e { 1 };
    bem::EigColVecN<bem::Complex, 1> amp_h { 1 / bem::eta0 };

    bem::rwg::RwgPlaneWave pw_e (dir, pol_e, pos, amp_e);
    bem::rwg::NxRwgPlaneWave pw_h (dir, pol_h, pos, amp_h);

    bem::rwg::ExcitationAssembler exc_assembler (structure.mesh());

    MatrixType inc_e;
    exc_assembler.assemble(inc_e, pw_e, k);

    MatrixType inc_h;
    exc_assembler.assemble(inc_h, pw_h, k);

    // Now solve the TEFIE and the NMFIE separately, as in Example 1.

    MatrixType j_tefie;
    L.factorize();
    L.mat_solve(j_tefie, inc_e);

    MatrixType j_nmfie;
    K.factorize();
    K.mat_solve(j_nmfie, inc_h);

    // Define the far-field points on which to compute the RCS, as in Examples 1 and 2.

    bem::PointCloud<3> projection_points;

    bem::EigColVecN<bem::Float, 3> arc_begin = { dist, 0, 0 };
    bem::EigColVecN<bem::Float, 3> arc_end = { dist, 0, bem::pi };

    bem::EigColVecN<bem::Index, 3> num_pts = { 1, 1, 100 };
    bem::EigColVecN<bem::Float, 3> center = { 0, 0, 0 };

    projection_points.set_polar_data(arc_begin, arc_end, center, num_pts);

    // Projectors can be given an integrator too. The difference is that for a projector, the
    // "observation triangle" is replaced by a set of discrete points, so there's no outer integral
    // to compute. That's why a projector takes a source integrator directly, rather than an
    // observation integrator. Let's give it the same custom source integrator.

    bem::rwg::VectorHypersingularProj L_projector (src_integrator);

    // Our projection points are 100 wavelengths away from the sphere, so our custom rule will pick
    // `SrcQuadrature` for every triangle. But if we were projecting fields close to the mesh, the
    // same rule would automatically switch to `SrcSingularity` where needed.

    // Assemble the E-field projector matrix, with the appropriate scaling, as in Example 2.

    bem::rwg::ProjectorAssembler<3> proj_assembler (projection_points, structure.mesh());

    MatrixType Lproj;
    proj_assembler.assemble(Lproj, L_projector, k);
    Lproj.scale(-bem::J * bem::two_pi * f * mu);

    // Project the E-field from both solutions onto the point cloud, and reshape the results as in
    // Example 1.

    MatrixType e_tefie, e_nmfie;
    Lproj.matmul(e_tefie, j_tefie);
    Lproj.matmul(e_nmfie, j_nmfie);

    e_tefie.raw_matrix() = e_tefie.raw_matrix().reshaped(3, 100);
    e_nmfie.raw_matrix() = e_nmfie.raw_matrix().reshaped(3, 100);

    // Compute the RCS from both solutions.

    MatrixType e_tefie_mag;
    e_tefie_mag.raw_matrix() = e_tefie.raw_matrix().colwise().norm();

    MatrixType e_nmfie_mag;
    e_nmfie_mag.raw_matrix() = e_nmfie.raw_matrix().colwise().norm();

    MatrixType rcs_tefie;
    rcs_tefie.raw_matrix() = Eigen::pow(e_tefie_mag.raw_matrix().array(), 2) * bem::four_pi * std::pow(dist, 2);

    MatrixType rcs_nmfie;
    rcs_nmfie.raw_matrix() = Eigen::pow(e_nmfie_mag.raw_matrix().array(), 2) * bem::four_pi * std::pow(dist, 2);

    // Finally, compute and display the worst-case point-wise relative error in RCS between the
    // TEFIE and NMFIE solutions, as in Example 1.

    MatrixType rcs_error;
    rcs_error.raw_matrix() = (rcs_tefie.raw_matrix() - rcs_nmfie.raw_matrix()).array().abs();

    MatrixType rcs_relative_error;
    rcs_relative_error.raw_matrix() = rcs_error.raw_matrix().array() / rcs_tefie.raw_matrix().array().abs();

    bem::Float rcs_max_relative_error = rcs_relative_error.raw_matrix().array().abs().maxCoeff();
    std::cout << "TEFIE vs. NMFIE RCS maximum relative error: "
              << rcs_max_relative_error * 100
              << " %" << std::endl;

    // We see a larger error than in Example 1, around 6%, because we've used much looser
    // quadrature and singularity settings than the default `SrcStrategic` and `ObsStrategic`
    // classes.

    // The same approach works for any rule you'd like. For example, you could switch integrators
    // based on the frequency or the material, or add a third option such as `SrcLineIntegrator`
    // for electrically large triangles. As long as your class inherits from `SrcIntegratorBase`,
    // OpenBEM will use it wherever a source integrator is expected. Enjoy!

    return 0;

}

