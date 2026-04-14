# Changes

## [Unreleased]

### Added

- Added this changelog.

### Changed

- Simplified the Eigen matrix wrapper class and unified its sparse and dense interface.
- Line integration is now only vectorized in source points, not observation points, which is a little faster.
- Line integration speed improvements and fixed a bug in the directed angle computation.

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

## [1.0.0] - March 30, 2025

Initial release.
