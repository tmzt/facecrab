# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3](https://github.com/tmzt/rusty-genius/compare/facecrab-v0.1.2...facecrab-v0.1.3) - 2026-02-05

### Other

- Commit ogenius changes
- Final quality fixes for clippy and redundant patterns
- Fix linting and serialization caught by pre-push
- Commit ogenius CLI

### Added
- **Configuration Architecture Refactor**: Split model discovery into `manifest.toml` (static configuration in `GENIUS_HOME`) and `registry.toml` (dynamic tracking in `GENIUS_CACHE`). 
- **Automatic Model Recording**: New models downloaded via repository paths are now automatically recorded in the dynamic `registry.toml`.
- *Note: This change aligns the implementation with the intended local-first architecture for separating user intent from system state.*

## [0.1.2](https://github.com/tmzt/rusty-genius/compare/facecrab-v0.1.1...facecrab-v0.1.2) - 2026-02-03

### Other

- (chore) update examples
- Update documentation and examples
- (chore) site: update badges
