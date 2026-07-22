// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Class for assembling an ACA-compressed RWG-based BEM operator matrix block.
*/

#ifndef BEM_ACA_ASSEMBLER_H
#define BEM_ACA_ASSEMBLER_H

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "matrix/base.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/assemblers/base.hpp"
#include "rwg/assemblers/indexing.hpp"
#include "compression/aca/base.hpp"


namespace bem::rwg
{

/**
* \ingroup compression
* @{
*/

/**
* @brief Class for assembling an ACA-compressed operator matrix block.
*/
class AcaAssembler: public OperatorAssemblerBase
{
public:

    /**
    * @brief Constructs an `AcaAssembler` for a given mesh.
    * @param[in] mesh - Triangle mesh for which the operator matrix is to be assembled.
    * @param[in] aca - ACA algorithm used to compress the block.
    * @param[in] index_set - Row/column index definition of the block to compress (optional).
    * @param[in] recompress - Whether to recompress the block using a QR decomposition (optional).
    * @param[in] tol - Relative tolerance for ACA (optional).
    */
    AcaAssembler(
        const TriangleMesh<3>& mesh,
        const AcaBase& aca,
        const IndexSet& index_set = IndexSet(EigRowVec<Index>(), EigRowVec<Index>()),
        const bool recompress = true,
        const Float tol = 1e-4
        ):
        mesh_(mesh),
        aca_(aca),
        index_set_(index_set),
        recompress_(recompress),
        tol_(tol) {};


    /**
    * @brief Rescopes this assembler to a new index set.
    * @param[in] index_set - New block index definition.
    */
    void set_indices(const IndexSet& index_set) override
    {
        index_set_ = index_set;
        return;
    };


    /**
    * @brief Assembles the ACA-compressed operator matrix block for a given operator object.
    * @param[out] mat - Matrix to store the compressed block, must be a `LowRankMatrix`.
    * @param[in] op - Operator object that computes the coefficients to be compressed into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k
        ) override;


protected:

    const TriangleMesh<3>& mesh_;
    const AcaBase& aca_;
    IndexSet index_set_;
    const bool recompress_;
    const Float tol_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "compression/assemblers/aca_assembler.cpp"
#endif

#endif
