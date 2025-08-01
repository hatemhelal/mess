# Contributing to MESS

We are interested in hearing any and all feedback so feel free to raise any questions,
bugs encountered, or enhancement requests as
[Issues](https://github.com/hatemhelal/mess/issues).

## Setting up a development environment

The following assumes that you have already set up an install of [uv](https://docs.astral.sh/uv/) and that the
`uv` command is available on your system path. Refer to the [uv install instructions](https://docs.astral.sh/uv/#installation) for your platform.

1. Create a virtual environment with the minimum python version required:

   ```bash
   uv venv --python=3.11
   ```

1. Install all required packages for developing MESS:

   ```bash
   uv pip install -e .[dev]
   ```

1. Install the pre-commit hooks

   ```bash
   uv run pre-commit install
   ```

1. Create a feature branch, make changes, and when you commit them the pre-commit hooks
   will run.

   ```bash
   git checkout -b feature
   ...
   git push --set-upstream origin feature
   ```

   The last command will print a link that you can follow to open a PR.

## Testing

Run all the tests using `pytest`

```bash
uv run pytest
```

## Building Documentation

From the project root, you can build the documentation with:

```bash
uv run jupyter-book build docs
```
