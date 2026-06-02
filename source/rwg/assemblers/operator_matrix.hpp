// OpenBEM - Copyright (C) 2026 Shashwat Sharma

// This file is part of OpenBEM.

// OpenBEM is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, either version 3
// of the License, or (at your option) any later version.

// You should have received a copy of the GNU General Public License along with OpenBEM.
// If not, see <https://www.gnu.org/licenses/>.


/**
* @file
* Classes for assembling RWG-based BEM operator matrices.
*/

#ifndef BEM_RWG_OP_ASSEMBLER_H
#define BEM_RWG_OP_ASSEMBLER_H

#include "types.hpp"
#include "rwg/assemblers/base.hpp"


namespace bem
{
// Forward declarations
template <typename T> class MatrixBase;
}


namespace bem::rwg
{

const Index EDGE_ELEM_RATIO = 2;

/**
* \addtogroup assm
* @{
*/

/**
* @brief Class for generating operator matrices for RWG observation and source functions.
*/
class EdgeOperatorAssembler: public OperatorAssemblerBase<3, 3>
{

    using base = OperatorAssemblerBase<3, 3>;
    using base::base;

public:

    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    */
    void prep_matrix(MatrixBase<Complex>& mat) override;


    /**
    * @brief Fills operator values in the matrix for edge-based RWG observation and source functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, 3, 3>> values
        ) override;

};


/**
* @brief Class for generating operator matrices for pulse observation and source functions.
*/
class FaceOperatorAssembler: public OperatorAssemblerBase<1, 1>
{

    using base = OperatorAssemblerBase<1, 1>;
    using base::base;

public:

    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source faces, and rows corresponding to observation faces.
    */
    void prep_matrix(MatrixBase<Complex>& mat) override;


    /**
    * @brief Fills operator values in the matrix for face-based pulse observation and source functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source faces, and rows corresponding to observation faces.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, 1, 1>> values
        ) override;

};


/**
* @brief Class for generating operator matrices for pulse observation and RWG source functions.
*/
class FaceEdgeOperatorAssembler: public OperatorAssemblerBase<1, 3>
{

    using base = OperatorAssemblerBase<1, 3>;
    using base::base;

public:

    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation faces.
    */
    void prep_matrix(MatrixBase<Complex>& mat) override;


    /**
    * @brief Fills operator values in the matrix for face-based pulse observation functions and edge-based
    * RWG source functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation faces.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, 1, 3>> values
        ) override;

};


/**
* @brief Class for generating operator matrices for RWG observation and pulse source functions.
*/
class EdgeFaceOperatorAssembler: public OperatorAssemblerBase<3, 1>
{

    using base = OperatorAssemblerBase<3, 1>;
    using base::base;

public:

    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source faces, and rows corresponding to observation edges.
    */
    void prep_matrix(MatrixBase<Complex>& mat) override;


    /**
    * @brief Fills operator values in the matrix for face-based pulse source functions and edge-based
    * RWG observation functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source faces, and rows corresponding to observation edges.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, 3, 1>> values
        ) override;

};


/**
* @brief Class for generating the full set of vector operator matrices for RWG observation and source functions.
*/
class VectorOperatorsAssembler: public OperatorAssemblerBase<3, 12>
{

    using base = OperatorAssemblerBase<3, 12>;
    using base::base;

public:

    /**
    * @brief Prepares the matrix for assembly (e.g., resizing and preallocation).
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges. The four vector operator matrices
    * are stacked along the horizontal direction.
    */
    void prep_matrix(MatrixBase<Complex>& mat) override;


    /**
    * @brief Fills operator values in the matrix for edge-based RWG observation and source functions.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source edges, and rows corresponding to observation edges, for all operators stacked along
    * the vertical direction.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom,
    * for all operators stacked along the horizontal direction.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMatMN<Complex, 3, 12>> values
        ) override;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/operator_matrix.cpp"
#endif

#endif
