// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file Example 1: TEFIE and NMFIE solvers for closed PEC objects. Demonstrates the computation of
* bistatic RCS for a PEC sphere with two different integral equation formulations.
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
#include "geometry/point_cloud.hpp"

#include "rwg/integral_equations/tefie.hpp"
#include "rwg/integral_equations/nmfie.hpp"

#include "rwg/excitations/plane_wave.hpp"

#include "rwg/function_space.hpp"

// Instead of including individual headers, one could just write #include "openbem.hpp" which will
// pull in all OpenBEM headers in a single line. However, this may increase code compile times a
// bit.

// All classes, variables, and types of OpenBEM are contained within the `bem` namespace and its
// sub-namespaces. Functionality of OpenBEM that is specific to RWG-based discretization is housed
// in the `bem::rwg` namespace. Discretizations other than those based on RWG functions may be added
// to OpenBEM in the future.

int main(int argc, char** argv)
{

    std::cout << "\n====================================================" << std::endl;
    std::cout << "OpenBEM example 1" << std::endl;
    std::cout << "====================================================\n" << std::endl;

    // A Gmsh-generated mesh file is used in this example. OpenBEM has built-in functionality to
    // read and write Gmsh files. First, we'll define where on disk the mesh file is located.

    // To make sure that the Gmsh mesh file can be found on disk no matter where this executable is
    // called from, we'll use C++'s `__FILE__` macro to get the path to this source file, and define
    // the path to the mesh file relative to that. This is not OpenBEM-specific, just a way to not
    // have to worry about which directory the executable is run from.

    std::size_t path_pos = std::string(__FILE__).find_last_of("/");
    std::string path = std::string(__FILE__).substr(0, path_pos) + "/";

    std::string msh_filename = path + "msh/sphere.msh";

    // Next, we'll create a `Structure` object, which will store the mesh and material properties
    // for each object in the structure being modeled. This requires the `geometry/structure.hpp`
    // header.

    bem::Structure<bem::TriangleMesh<3>> structure;

    // The template parameter of a `Structure` specifies the type of mesh that will be stored, and
    // its dimensionality. In this example, we're considering a 3D problem with a triangular mesh,
    // so we invoke the `TriangleMesh<3>` class. This requires the `geometry/mesh/triangle_mesh.hpp`
    // header.

    // Now we'll use OpenBEM's built-in Gmsh reader function to parse mesh data into the `Structure`
    // object we created. This requires the `geometry/mesh/io.hpp` header.

    bem::MeshTransfer::read_gmsh_v2(structure, msh_filename);

    // Now `structure` has been populated with all the mesh data. Mesh edges are automatically
    // identified from triangle connectivity during construction of the `TriangleMesh<3>` object
    // stored inside `structure`, accessed via `structure.mesh()`.

    // Let's print out some of the mesh info.

    std::cout << "Number of vertices: " << structure.mesh().num_verts() << std::endl;
    std::cout << "Number of triangles: " << structure.mesh().num_elems() << std::endl;
    std::cout << "Number of edges: " << structure.mesh().num_edges() << std::endl;

    // Next, we'll set the simulation frequency in Hz.

    bem::Float f = 250e6;

    // OpenBEM defines various types in the `types.hpp` header, which are just wrappers around the
    // usual C++ types. This allows easily switching types throughout the codebase, if ever
    // needed. It also allows easily switching between single, double, and extended double precision
    // at compile time.

    // For the default double precision case, `Float` used above is the same as `double`, and
    // `Complex` is `std::complex<double>`. In single precision, `Float` is the same as `float`, and
    // `Complex` would be `std::complex<float>`. See `types.hpp`.

    // In the boundary element method, integral operators are discretized and tested to obtain a
    // matrix representation of the electromagnetic problem. So, we need a way to assemble, store,
    // and manipulate complex-valued matrices, which is usually accomplished in C++ using
    // third-party open-source linear algebra libraries.

    // OpenBEM allows you to use any such library, with a little work. Basically, you can define a
    // matrix class that wraps around a third-party matrix algebra library. As long as the wrapper
    // class inherits from `MatrixBase` (defined in `matrix/base.hpp`) and implements basic
    // functions defining how values are written into and read from the matrix, your wrapper class
    // can then be used throughout OpenBEM.

    // This way, the internals of OpenBEM are agnostic to whatever linear algebra library you'd like
    // to use, as long as you're willing to write the associated wrapper class.

    // OpenBEM ships with default wrapper classes around dense and sparse Eigen matrices. If you're
    // happy with Eigen as your underlying matrix algebra library, then you do not need to do any
    // extra work; just use these wrappers.

    // In this example, we'll use dense complex-valued Eigen matrices. The following matrix wrapper
    // class requires the `matrix/eigen_matrix.hpp` header.

    using MatrixType = bem::EigenMatrix<bem::Complex>;

    // `EigenMatrix` is the aforementioned wrapper class that wraps dense Eigen
    // matrices. `MatrixType` is just an alias that allows slightly cleaner code.

    // The classical tangentially-tested EFIE, or TEFIE as it is often called in the literature, is
    // implemented in the (surprise) `Tefie` class. Let's make a `Tefie` object, which takes a
    // `TriangleMesh<3>` object as a constructor argument, so that the `Tefie` knows what mesh to
    // use. This requires the `rwg/integral_equations/tefie.hpp` header.

    // The `Tefie` class lives in the `bem::rwg` namespace, so that in the future, we can define
    // other discretizations of the TEFIE without naming conflicts with this one.

    bem::rwg::Tefie<MatrixType> tefie (structure.mesh());

    // Notice that `Tefie` takes an optional template parameter defining the matrix data type to
    // use. To use the default (which is a dense, complex-valued Eigen matrix) we still need to
    // remember to use the syntax `Tefie<> tefie;`.

    // Now, we compute the hypersingular TEFIE matrix operator. Internally, the associated integrals
    // are computed using a default strategy for singularity subtraction, which is automatically
    // invoked for source and observation triangles that are sufficiently close to each other. These
    // settings can all be completely customized - see other examples.

    MatrixType L = tefie.j_matrix(f, structure.background_material());

    // Three things to notice here:

    // First, notice that we need to pass in the frequency and the material to compute the matrix
    // operator. In this case, the material is just the background material in which the sphere is
    // immersed. The `Structure` has a `background_material()` object of type `Material`, which is
    // set to vacuum by default. If the sphere were not PEC, we could set its `Material` via the
    // `Component` objects stored in `structure`; see other examples.

    // Second, notice that we are calling the function `Tefie::j_matrix()`, where the `j` implies
    // that we are computing the matrix that operates upon the (unknown) electric surface current
    // density, since it is a PEC sphere.

    // If this were a PMC sphere, we would want to use `Tefie::m_matrix()` to compute the vector
    // double-layer operator instead (although in that case, we may want to use the `TMFIE` class
    // instead). If it were a dielectric, we'd probably need to compute both matrices, and also use
    // additional equations for the region interior to the sphere - see other examples for more
    // details.

    // Third, if the OpenMP library is available and the number of threads has been set to more than
    // 1, these computations will be parallelized.

    // Just for fun, let's also solve the same problem using the NMFIE, for which we can use the
    // `Nmfie` class. In this case, the relevant matrix operator associated with the electric
    // surface current density is the rotationally tested vector double-layer operator. The `Nmfie`
    // class requires the `rwg/integral_equations/nmfie.hpp` header.

    // Important note: the integral equation classes in OpenBEM are defined under the assumption
    // that the normal vector points outside of the relevant region. Here, the relevant region is
    // the free space outside the sphere. However, the normal vectors for the sphere's mesh point
    // outside the sphere (that's usually how Gmsh defines the triangles of its meshes), and
    // therefore into the air region. So, from the point of view of the relevant region, i.e., free
    // space, the normal vectors are pointing inwards.

    // To handle this, OpenBEM's integral equation classes provide an optional constructor argument
    // `flip_normals`, which, as the name suggests, computes operators for the case where the
    // normals point into the region of interest. This wasn't necessary for the TEFIE because the
    // tangentially tested L operator is unaffected by flipping normals. However, it is recommended
    // to always specify this explicitly, to be clear.

    bool flip_normals = true;

    bem::rwg::Nmfie<MatrixType> nmfie (structure.mesh(), flip_normals);
    MatrixType K = nmfie.j_matrix(f, structure.background_material());

    // Note that both the TEFIE's L operator and the NMFIE's K operator require computing Green's
    // function values for the same set of source and observation triangles. This means that a lot
    // of the same computations are repeated.

    // OpenBEM is designed for usability, readability, and maintainability, and the above way of
    // computing operators separately is more user-friendly despite the redundant computations.

    // However, if performance is an important consideration, OpenBEM also offers a way to
    // batch-compute several operators without redundant computations, by defining custom
    // operators. But this is saved for later and more advanced examples.

    // Now let's create the right-hand side vectors that represent a plane wave excitation. For
    // this, we need to set the direction, polarization, and phase reference.

    // For arrays, matrices, and vector algebra, the Eigen library's data structures are used, which
    // provide a simple Matlab-like interface and save us from having to write our own loops every
    // time we want to do element-wise operations on an array of numbers, etc.

    // In `types.hpp`, there are several simple aliases that have been created for commonly used
    // Eigen types, just to shorten the type names and make the code a little more readable. We've
    // used these aliases below.

    // For example (dropping the namespace specifier for brevity), `EigColVecN<Float, 3>` means
    // Eigen-based column vector of a fixed size 3, containing `Float` data.

    // First, set the direction of the plane wave to be a three-element vector pointing along +z.

    bem::EigColVecN<bem::Float, 3> dir = { 0, 0, 1 };

    // Next, let the incident plane wave have an E-field polarization along +x.

    bem::EigColVecN<bem::Float, 3> pol_e = { 1, 0, 0 };

    // Finally, set the "position" from which the plane wave originates. This defines the phase
    // reference, and can also be used as the sensor distance for RCS calculations, as will be shown
    // later in this example.

    // Let's assume the plane wave originates from a point 100 free space wavelengths along `-dir`,
    // i.e., the plane wave travels from `pos` along `dir` for a distance of 100 free space
    // wavelengths from the origin.

    bem::Float dist = 100 * (bem::c0 / f);
    bem::EigColVecN<bem::Float, 3> pos = -dir * dist;

    // Two things to note here: first, notice how easy Eigen makes it to multiply scalars with an
    // array. Second, OpenBEM defines several common constants such as `pi`, `eps0`, `mu0`, and
    // `c0` in `constants.hpp`.

    // Finally, define the E-field amplitude of the plane wave. It must be defined as an element of
    // a vector because in general, we can have multiple excitations, each with a different
    // amplitude; what follows is the special case of a single excitation.

    bem::EigColVecN<bem::Complex, 1> amp_e { 1 };

    // When there's more than one excitation, the `dir`, `pol_e` and `pos` parameters would be a
    // matrix rather than a column vector, where each column of the matrix corresponds to a single
    // excitation.

    // Now, from our knowledge of the RWG-discretized TEFIE, we know that the analytical expression
    // for the plane wave must be tangentially tested with RWG functions. So, we define an
    // `RwgPlaneWave` class. This requires the `rwg/excitations/plane_wave.hpp` header.

    bem::rwg::RwgPlaneWave pw_e (dir, pol_e, pos, amp_e);

    // We can use the `Tefie` class to now generate the excitation vector(s) by passing in
    // `pw_e`. In this case, we have only a single excitation, but in general, we can have as many
    // as we'd like. The excitation matrix would have as many columns as the number of excitations.

    MatrixType inc_e = tefie.exc_matrix(f, structure.background_material(), pw_e);

    // For solving the NMFIE, the excitation is not a tangentially tested incident E-field, but
    // rather a rotationally tested H-field. In order to be consistent with `pol_e` and the
    // direction of propagation, the H-field must be polarized along +y, and its amplitude must be
    // appropriately scaled by the wave impedance in free space, as below.

    bem::EigColVecN<bem::Float, 3> pol_h = { 0, 1, 0 };
    bem::EigColVecN<bem::Complex, 1> amp_h { 1 / bem::eta0 };

    bem::rwg::NxRwgPlaneWave pw_h (dir, pol_h, pos, amp_h);
    MatrixType inc_h = nmfie.exc_matrix(f, structure.background_material(), pw_h);

    // Notice that we use the `NxRwgPlaneWave` class because we are testing the incident field with
    // nxRWG functions. Notice also that `RwgPlaneWave` and `NxRwgPlaneWave` do not care whether we
    // are referring to an E or H field; it's up to us to supply the correct amplitude and
    // polarization depending on our use case.

    // Finally, we solve the system of equations to obtain the electric surface current density.
    // The wrapper classes for matrix algebra have `.factorize` and `.mat_solve()` methods which
    // can be used to solve a matrix system with a given right-hand side matrix, which we'll use here.

    MatrixType j_tefie;
    L.factorize();
    L.mat_solve(j_tefie, inc_e);

    // The solution matrix will have as many columns as the number of excitation vectors, which in
    // this case is just one.

    // Now let's do the same for the NMFIE case.

    MatrixType j_nmfie;
    K.factorize();
    K.mat_solve(j_nmfie, inc_h);

    // Having computed the electric surface current density using both approaches, we can now
    // compute far-field quantities like RCS by using OpenBEM's projectors.

    // First, let's define the points at which to compute far fields - these are the points to which
    // fields will be projected. We can use OpenBEM's `PointCloud` class (in three dimensions) to
    // create the set of projection points. Let's initialize an empty object of this class, and then
    // we'll populate it below. This requires the `rwg/geometry/point_cloud.hpp` header.

    bem::PointCloud<3> projection_points;

    // We can define these points directly in spherical coordinates. Let's project the fields to a
    // circular arc containing 100 points, that passes through the plane wave's `pos`, lies along
    // phi (azimuth) = 0, and spans theta (elevation) = 0 to pi. The following 3-element vectors
    // contain (r, phi, theta) coordinates.

    bem::EigColVecN<bem::Float, 3> arc_begin = { dist, 0, 0 };
    bem::EigColVecN<bem::Float, 3> arc_end = { dist, 0, bem::pi };

    // Based on the above discussion, our point cloud will contain one point along the r direction,
    // one point along the phi direction, and 100 points along theta.

    bem::EigColVecN<bem::Index, 3> num_pts = { 1, 1, 100 };

    // Note that the `Index` data type by default refers to unsigned integers, and is used
    // throughout OpenBEM to hold sizes and indices.

    // The `PointCloud` also needs to know what to use as the origin relative to which the points
    // are defined.

    bem::EigColVecN<bem::Float, 3> center = { 0, 0, 0 };

    // Now populate the point cloud in spherical coordinates.

    projection_points.set_polar_data(arc_begin, arc_end, center, num_pts);

    // Projecting surface currents to the far field involves integrating over all source triangles
    // in the mesh. Basically, we are evaluating the integral equation for a given set of
    // observation points.

    // Note: if the observation points lie very close to the mesh, for example, if we'd like to
    // compute fields near or on the mesh itself, the appropriate singularity extraction treatment
    // would automatically be applied. So the projectors are not just far-field projectors.

    // To do the projections, we compute a projector matrix. This is similar to the usual RWG matrix
    // operators, but instead of using testing functions defined on the mesh itself, we are
    // "testing" the operators at a set of discrete points in space.

    // The projector matrix, when applied to a given set of electric surface currents (which were
    // computed above by solving the system of equations), will give the E-field generated by those
    // currents.

    // Note that since we're just evaluating the EFIE (or MFIE) at a given set of discrete
    // observation points, there's no concept of "tangentially tested" or "rotationally tested"
    // projector matrices. So we would get exactly the same projector matrix if we had used the
    // NEFIE here instead.

    MatrixType e_proj = tefie.j_projector(f, structure.background_material(), projection_points);

    // If we would like to compute the projected H-field, we would need to use the projector from
    // the NMFIE. As in the TEFIE and NEFIE case, the NMFIE or TMFIE would give the same projector
    // matrix.

    MatrixType h_proj = nmfie.j_projector(f, structure.background_material(), projection_points);

    // Of course, if we had a penetrable object instead of a perfect electric conductor, we would
    // have both electric and magnetic surface current densities, so we would also need to compute
    // the associated `m_projector()`. The projected E and H fields would be a superposition of the
    // fields generated by the electric and magnetic surface current densities; see other examples.

    // Now let's put these projectors into action by computing the electric and magnetic fields on
    // our point cloud - this just requires applying the projector matrix to the computed currents.
    // First, let's get the far fields that result from the currents computed by solving the TEFIE.

    MatrixType e_tefie, h_tefie;
    e_tefie.set_matmul(e_proj, j_tefie);
    h_tefie.set_matmul(h_proj, j_tefie);

    // Note: do not confuse the projectors for the matrix operators. In the above, we are using the
    // TEFIE to solve for the currents. Then, we are using the EFIE to get the E-field generated by
    // those TEFIE-solved currents, and we are using the MFIE to get the H-field generated by the
    // same TEFIE-solved currents.

    // Next, for the sake of comparison, let's compute the far fields that result from our NMFIE
    // solution, using the same projectors, but this time applying them to the electric surface
    // currents computed using the NMFIE.

    MatrixType e_nmfie, h_nmfie;
    e_nmfie.set_matmul(e_proj, j_nmfie);
    h_nmfie.set_matmul(h_proj, j_nmfie);

    // Note that since the projected fields are vectorial, we have three field components at each
    // observation point in our point cloud. The components are stored contiguously along matrix
    // rows.

    // For example, the projected E-field in `e_tefie` is stored as Ex1, Ey1, Ez1, Ex2, Ey2, Ez2,
    // etc. Each column corresponds to a different column of the source currents, `j_tefie` in this
    // case. Here, there's only one column because we had only one excitation vector.

    // Next, we'll compute the RCS from the projected E-fields. For convenience, let's reshape the
    // projected field vectors into a 3 x N matrix where the first row contains the x-component, the
    // second contains the y-component, and the third the z-component. The N columns correspond to
    // the N observation points.

    // OpenBEM's matrix class wrappers don't directly provide this reshaping functionality, but
    // since we're using Eigen's matrix datastructures underneath, we can directly use Eigen's API
    // by accessing the underlying raw matrix as shown below.

    MatrixType e_tefie_reshaped;
    e_tefie_reshaped.raw_matrix() = e_tefie.raw_matrix().reshaped(3, 100);

    MatrixType e_nmfie_reshaped;
    e_nmfie_reshaped.raw_matrix() = e_nmfie.raw_matrix().reshaped(3, 100);

    // To compute the RCS, we'll continue to take advantage of Eigen's underlying matrix API, which
    // makes it a lot easier to do element-wise mathematical and geometric operations.

    // First, let's compute the E-field magnitudes.

    MatrixType e_tefie_mag;
    e_tefie_mag.raw_matrix() = e_tefie_reshaped.raw_matrix().colwise().norm();

    MatrixType e_nmfie_mag;
    e_nmfie_mag.raw_matrix() = e_nmfie_reshaped.raw_matrix().colwise().norm();

    // Next, let's use this to compute the RCS for both the TEFIE and the NMFIE solutions.

    MatrixType rcs_tefie;
    rcs_tefie.raw_matrix() = Eigen::pow(e_tefie_mag.raw_matrix().array(), 2) * bem::four_pi * std::pow(dist, 2);

    MatrixType rcs_nmfie;
    rcs_nmfie.raw_matrix() = Eigen::pow(e_nmfie_mag.raw_matrix().array(), 2) * bem::four_pi * std::pow(dist, 2);

    // Note that applying elementwise mathematical operations to Eigen matrices requires calling the
    // `.array()` method on the Eigen matrix, which has no overhead but just informs Eigen to treat
    // the operations as elementwise rather than in a matrix sense. This is similar to Matlab, where
    // we use, for example, `.*` for elementwise multiplication, while just `*` implies matrix
    // multiplication.

    // Note also that for some Eigen operations, you may need to explicitly include Eigen's headers,
    // which are available in `external/Eigen`, for example, `#include <external/Eigen/Dense>`. The
    // entire Eigen library is redistributed in the `external/Eigen` directory.

    // Next, let's compute and display the worst-case point-wise relative error in RCS between the
    // TEFIE and NMFIE solutions.

    MatrixType rcs_error;
    rcs_error.raw_matrix() = (rcs_tefie.raw_matrix() - rcs_nmfie.raw_matrix()).array().abs();

    MatrixType rcs_relative_error;
    rcs_relative_error.raw_matrix() = rcs_error.raw_matrix().array() / rcs_tefie.raw_matrix().array().abs();

    bem::Float rcs_max_relative_error = rcs_relative_error.raw_matrix().array().abs().maxCoeff();
    std::cout << "TEFIE vs. NMFIE RCS maximum relative error: "
              << rcs_max_relative_error * 100
              << " %" << std::endl;

    // The worst-case relative error between the two methods is less than 2.5%, which is reasonable.

    // Finally, we'll export the computed surface current densities in Gmsh's post-processing file
    // format, so that they can be directly visualized in the Gmsh GUI.

    // To do this, we need to recall that the quantity we solved for is not the actual surface
    // current density, but rather the RWG expansion coefficients for the surface current density.

    // To plot the actual surface current density, we will apply those RWG coefficients to the RWG
    // function definition on each edge, and compute the field values at the centroid of each mesh
    // triangle, which can be directly visualized in Gmsh.

    // All of this is accomplished easily using OpenBEM's built-in functionality to reconstruct an
    // RWG-based field, given the coefficients, at given points on the mesh. We'll do this for both
    // the TEFIE and NMFIE cases to compare. This requires the `rwg/function_space.hpp` header.

    bem::EigMatNX<bem::Float, 3> j_surf_tefie = bem::rwg::Rwg::reconstruct_field(
        structure.mesh(), // mesh with which the field is associated
        j_tefie, // RWG coefficients computed using the TEFIE
        structure.mesh().elem_centroids(), // points at which to compute field values
        false // whether the field was expanded using rotated RWGs; `false` means regular RWGs
        ).array().real(); // Eigen syntax to keep only the real part of the computed currents

    bem::EigMatNX<bem::Float, 3> j_surf_nmfie = bem::rwg::Rwg::reconstruct_field(
        structure.mesh(), // mesh with which the field is associated
        j_nmfie, // RWG coefficients computed using the NMFIE
        structure.mesh().elem_centroids(), // points at which to compute field values
        false // whether the field was expanded using rotated RWGs; `false` means regular RWGs
        ).array().real(); // Eigen syntax to keep only the real part of the computed currents

    // Use OpenBEM's built-in Gmsh field writer to write the fields to disk.

    bem::MeshTransfer::write_gmsh_v2_vector_field(structure, path + "msh/ex01_jsurf_tefie", j_surf_tefie);
    bem::MeshTransfer::write_gmsh_v2_vector_field(structure, path + "msh/ex01_jsurf_nmfie", j_surf_nmfie);

    // This will create two files in the above location with the `.pos` extension, which can be
    // opened with the Gmsh GUI to show the electric surface current density magnitude and direction
    // on the mesh surface. Enjoy!

    return 0;

}

