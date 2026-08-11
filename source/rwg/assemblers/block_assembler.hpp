/**
* @file
* Classes for assembling RWG-based BEM operator matrix sub-blocks.
*/

#ifndef BEM_BLOCK_ASSEMBLER_H
#define BEM_BLOCK_ASSEMBLER_H

#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>
#include <mutex>

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
* @brief Hash for a pair of face indices, used to key `BlockAssembler`'s per-face-pair
* integration cache.
*/
struct FacePairHash
{
    std::size_t operator()(const std::pair<Index, Index>& p) const noexcept
    {
        std::size_t h1 = std::hash<Index>{}(p.first);
        std::size_t h2 = std::hash<Index>{}(p.second);

        if constexpr (sizeof(std::size_t) >= 8)
            h2 += 0x9e3779b97f4a7c15ULL;
        else
            h2 += 0x9e3779b9U;

        return h1 ^ (h2 + (h1 << 6) + (h1 >> 2));
    }
};


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
    * @param[in] use_integration_cache - Whether to cache and reuse triangle-pair integrals (optional).
    */
    BlockAssembler(
        const TriangleMesh<3>& mesh,
        const IndexSet& index_set,
        const bool use_integration_cache = true
        ):
        mesh_(mesh),
        index_set_(index_set),
        use_integration_cache_(use_integration_cache)
    {
        for (Index ii = 0; ii < index_set_.num_rows(); ++ii)
            row_map_.insert({ index_set_.rows()[ii], ii });

        for (Index ii = 0; ii < index_set_.num_cols(); ++ii)
            col_map_.insert({ index_set_.cols()[ii], ii });

        row_faces_from_edges_ = IndexGenerator::faces_from_edges(mesh_, index_set_.rows());
        col_faces_from_edges_ = IndexGenerator::faces_from_edges(mesh_, index_set_.cols());

        return;
    };


    /**
    * @brief Assembles the operator matrix block for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator block, sized to the block
    * defined by `index_set_`, with columns corresponding to source degrees of freedom, and rows
    * corresponding to observation degrees of freedom.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k
        ) override;


    /**
    * @brief Assembles a sub-block of the operator matrix block for a given operator object.
    * @param[out] mat - Matrix to store the assembled operator sub-block, sized to exactly
    * `local_rows.size() x local_cols.size()`.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    * @param[in] k - Complex wavenumber.
    * @param[in] local_rows - Local row indices (positions within `index_set_.rows()`) to assemble.
    * @param[in] local_cols - Local column indices (positions within `index_set_.cols()`) to assemble.
    * @details
    * Out-of-bounds entries in `local_rows`, `local_cols` are ignored (the corresponding row/column
    * of `mat` is left zero). A selection whose size equals `index_set_.num_rows()` (or `num_cols()`)
    * is assumed, not verified, to be `[0, 1, ..., n-1]`.
    */
    void assemble(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k,
        ConstEigRef<EigRowVec<Index>> local_rows,
        ConstEigRef<EigRowVec<Index>> local_cols
        );

        
protected:

    using OperatorAssemblerBase::assemble;


    /**
    * @brief Computes face pairs from `obs_faces` and `src_faces`, evaluates `op` for each,
    * and places results in `mat` (`out_rows x out_cols`) via `active_row_map`, `active_col_map`.
    */
    void assemble_from_faces(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        const Complex k,
        ConstEigRef<EigRowVec<Index>> obs_faces,
        ConstEigRef<EigRowVec<Index>> src_faces,
        const std::unordered_map<Index, Index>& active_row_map,
        const std::unordered_map<Index, Index>& active_col_map,
        const Index out_rows,
        const Index out_cols
        );


    /**
    * @brief Fills operator values in the matrix.
    * @param[out] mat - Matrix to store the assembled operator coefficients.
    * @param[in] op - Operator object that computes the coefficients to be assembled into `mat`.
    * @param[in] face_pair - Observation (first entry) and source (second entry) triangle index pair.
    * @param[in] values - Operator values for each pair of observation and source degrees of freedom.
    * @param[in] active_row_map - Map from global observation indices to local row indices in `mat`.
    * @param[in] active_col_map - Map from global source indices to local column indices in `mat`.
    */
    void fill_matrix(
        MatrixBase<Complex>& mat,
        const OperatorBase& op,
        ConstEigRef<EigColVecN<Index, 2>> face_pair,
        ConstEigRef<EigMat<Complex>> values,
        const std::unordered_map<Index, Index>& active_row_map,
        const std::unordered_map<Index, Index>& active_col_map
        );


    const TriangleMesh<3>& mesh_;
    const IndexSet index_set_;
    std::unordered_map<Index, Index> row_map_;
    std::unordered_map<Index, Index> col_map_;

    EigRowVec<Index> row_faces_from_edges_;
    EigRowVec<Index> col_faces_from_edges_;

    const bool use_integration_cache_ = true;
    std::unordered_map<std::pair<Index, Index>, EigMat<Complex>, FacePairHash> integration_cache_;
    const OperatorBase* integration_cache_op_ = nullptr;
    Complex integration_cache_k_ = 0;
    std::mutex integration_cache_mutex_;

};

/**
* @}
*/

}

#ifndef BEM_LINKED
#include "rwg/assemblers/block_assembler.cpp"
#endif

#endif
