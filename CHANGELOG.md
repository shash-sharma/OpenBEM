# Changes

## [Unreleased]

### Added

- Ability to compute and cache a component sub-mesh inside a `Component` object.

### Changed

- Refactored `EigenMatrix` to replace variants with a dispatch pattern that improves sparse / dense iteroperability.

### Fixed

- Bug in `EigenMatrix` where GMRES was used even if BiCGStab was requested.


## [1.1.0] - July 11, 2026

### Added

- Changelog.
- `IndexGenerator` helper class.
- `BoundingBox` geometry class.
- `IndexSet` to manage index blocks.
- `BlockAssembler` to allow assembling (not necessarily contiguous) index blocks.

### Changed

- Simplified the Eigen matrix wrapper class and unified its sparse and dense interface.
- Improved interoperability between sparse and dense Eigen matrices.
- `factorize()` must now be called explicitly before calling `mat_solve` in `EigenMatrix`.
- Line integration is now only vectorized in source points, not observation points, which is a little faster.
- Line integration speed improvements and fixed a bug in the directed angle computation.
- Simplified `TriangleMesh` interface and added a map from edges to associated elements.
- Refactored operator and assembler classes and added support for multiple operator generation.
- Changed assembler file names for consistency with class names.
- Refactored operator, integrator, and quadrature classes to remove templates, use cpp instead of tpp files, and make the computational routines all thread-safe.

### Fixed

- AEFIE's divergence matrix orientation fixed.


## [1.0.2] - April 05, 2025

### Added

- Virtual destructors added to all base classes.

### Changed

- Vectorized kernels for source points.
- Improved and simplified CMakeLists compile options.

### Fixed

- Fixed compiler warnings.
- Typos fixed in docs.


## [1.0.0] - March 30, 2026

Initial release.
