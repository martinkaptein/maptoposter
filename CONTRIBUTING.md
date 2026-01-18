# Contributing

Thanks for your interest in contributing to this fork of the City Map Poster Generator.

## Upstream
This project is a fork of:
https://github.com/originalankur/maptoposter.git

Fork repository:
https://github.com/martinkaptein/maptoposter.git

If a change is broadly useful, consider opening an issue or PR upstream as well.

## How to Contribute

1. Open an issue describing the change or bug.
2. Fork the repo and create a feature branch.
3. Keep changes focused and easy to review.
4. Update documentation when behavior changes.
5. Submit a PR with a clear description and screenshots (if visual output changed).

## Development Tips

- Use `python create_map_poster.py --list-themes` to validate theme discovery.
- Prefer small `--distance` and lower `--dpi` for faster iterations.
- If you add a new layer, update `README.md` and the theme keys list.
- If you add CLI options, update usage examples and the options table.

## Theme Guidelines

- New themes live in `themes/` as JSON.
- Include keys for all layers used by the renderer.
- Keep contrast sufficient for print.

## License

By contributing, you agree that your contributions will be licensed under the project license.
