# Changelog

All notable changes to this fork are documented in this file.
This project follows the "Keep a Changelog" format.

## [Unreleased]

### Added
- Coordinate-based centering via `--center "lat,lon"`.
- Output sizing controls with `--width`, `--height`, and `--dpi`.
- Text-free rendering with `--no-text`.
- Additional map layers: railways, subways, waterways, buildings, contours, bridges, tunnels, airports.
- New themes: `minimalist_dark`, `minimalist_light`.
- Theme defaults extended to support new layer colors.
- Theme-driven visibility: omit a key in a theme to skip rendering that layer or road class.

### Changed
- README updated with new options, layers, and theme keys.

### Removed
- N/A

### Fixed
- Preserve edges that cross the bbox so long motorways render fully (`truncate_by_edge=True`, `retain_all=True`).
