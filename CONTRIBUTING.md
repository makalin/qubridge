# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install
```

## Development Workflow

1. Create a feature branch.
2. Run quality checks locally:

```bash
ruff check .
ruff format .
mypy qubridge
pytest
```

3. Update documentation for behavior changes.
4. Open a pull request with clear summary and test evidence.

## Commit Guidelines

- Use focused commits.
- Keep commit messages short and descriptive.
- Include tests for new functionality or bug fixes.
