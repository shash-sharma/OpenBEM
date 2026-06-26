/**
* @file
* Classes for assembling RWG-based BEM operator matrix sub-blocks.
*/

#ifndef BEM_BLOCK_ASSEMBLER_H
#define BEM_BLOCK_ASSEMBLER_H

#include <vector>
#include <unordered_map>

#include "types.hpp"
#include "geometry/mesh/triangle_mesh.hpp"
#include "matrix/base.hpp"
#include "rwg/integrators/obs/base.hpp"
#include "rwg/integrators/obs/strategic.hpp"
#include "rwg/operators/base.hpp"
#include "rwg/assemblers/base.hpp"
#include "rwg/assemblers/indexing.hpp"


namespace bem::rwg
{

/**
* \ingroup assm
* @{
*/

/**
* @brief Class for assembling matrix sub-blocks for RWG observation and source functions.
*/
class BlockAssembler: public OperatorAssemblerBase
{
public:

    /**
    * @brief Constructs a `BlockAssembler` for a given mesh.
    * @param[in] mesh - Triangle mesh for which the operator matrix is to be assembled.
    * @param[in] index_set - Block index definition.
    */
    BlockAssembler(
        const TriangleMesh<3>& mesh,
        const IndexSet& index_set
        ):
        mesh_(mesh),
        index_set_(index_set)
    {
        for (Index ii = 0; ii < index_set_.num_rows(); ++ii)
            row_map_.insert({ index_set_.rows()[ii], ii });

        for (Index ii = 0; ii < index_set_.num_cols(); ++ii)
            col_map_.insert({ index_set_.cols()[ii], ii });

        return;
    };


    /**
    * @brief Assembles the operator matrix blocks for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k
        ) override;


    /**
    * @brief Assembles operator matrices for given operator objects.
    * @param[out] mats - Matrices to store the assembled operator coefficients, with columns corresponding
    * to source degrees of freedom, and rows corresponding to observation degrees of freedom.
    * @param[in] ops - Operator objects that compute the coefficients to assemble into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    template <typename MatrixType, typename ObsIntegratorType = ObsStrategic, typename... Ops>
    void assemble(
        std::vector<MatrixType>& mats,
        const Complex k,
        const ObsIntegratorType obs_integrator = ObsStrategic()
        );


    /**
    * @brief Computes and retrieves a block of operator matrix values for a given operator object.
    * @param[out] mat - Block of values for requested row and column indices.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void get_block(
        EigMat<Complex>& mat,
        const OperatorBase& op,
        const Complex k
        );


    /**
    * @brief Fills operator values in the matrix.
    * @param[out] mat - Matrix to store the assembled operator coefficients.
    * @param[in] elem_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    */
    void fill_matrix(
        EigMat<Complex>& mat,
        const OperatorBase& op,
        ConstEigRef<EigColVecN<Index, 2>> elem_pair,
        ConstEigRef<EigMat<Complex>> values
        );


protected:

    const TriangleMesh<3>& mesh_;
    const IndexSet index_set_;
    std::unordered_map<Index, Index> row_map_;
    std::unordered_map<Index, Index> col_map_;

};


template <typename MatrixType, typename ObsIntegratorType, typename... Ops>
void BlockAssembler::assemble(
    std::vector<MatrixType>& mats,
    const Complex k,
    ObsIntegratorType obs_integrator
    )
{

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/block_assembler.cpp"
#endif

#endif
