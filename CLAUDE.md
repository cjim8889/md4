# Python Package Management with uv

Use uv exclusively for Python package management in this project.

## Package Management Commands

- All Python dependencies **must be installed, synchronized, and locked** using uv
- Never use pip, pip-tools, poetry, or conda directly for dependency management

Use these commands:

- Install dependencies: `uv add <package>`
- Remove dependencies: `uv remove <package>`
- Sync dependencies: `uv sync`

## Running Python Code

- Run a Python script with `uv run <script-name>.py`
- Run Python tools like Pytest with `uv run pytest` or `uv run ruff`
- Launch a Python repl with `uv run python`

## Managing Scripts with PEP 723 Inline Metadata

- Run a Python script with inline metadata (dependencies defined at the top of the file) with: `uv run script.py`
- You can add or remove dependencies manually from the `dependencies =` section at the top of the script, or
- Or using uv CLI:
    - `uv add package-name --script script.py`
    - `uv remove package-name --script script.py`

After modifying individual python file always use the command `uvx ty check <filepath> --output-format concise` to consider any typing issues to follow the best practices. FIX ALL Type issues if they are in our control, i.e. not result of external libraries. ONLY use --output-format full when you don't know what to do next. Documentation of uvx ty type checker is the following:

uvx ty check --help
Check a project for type errors

Usage: ty check [OPTIONS] [PATH]...

Arguments:
  [PATH]...
          List of files or directories to check [default: the project root]

Options:
      --project <PROJECT>
          Run the command within the given project directory.

          All `pyproject.toml` files will be discovered by walking up the directory tree from the given
          project directory, as will the project's virtual environment (`.venv`) unless the `venv-path`
          option is set.

          Other command-line arguments (such as relative paths) will be resolved relative to the current
          working directory.

      --typeshed <PATH>
          Custom directory to use for stdlib typeshed stubs

      --extra-search-path <PATH>
          Additional path to use as a module-resolution source (can be passed multiple times)

  -v, --verbose...
          Use verbose output (or `-vv` and `-vvv` for more verbose output)

  -q, --quiet...
          Use quiet output (or `-qq` for silent output)

  -c, --config <CONFIG_OPTION>

          A TOML `<KEY> = <VALUE>` pair (such as you might find in a `ty.toml` configuration file)
          overriding a specific configuration option.

          Overrides of individual settings using this option always take precedence
          over all configuration files.

      --config-file <PATH>
          The path to a `ty.toml` file to use for configuration.

          While ty configuration can be included in a `pyproject.toml` file, it is not allowed in this
          context.

          [env: TY_CONFIG_FILE=]

      --output-format <OUTPUT_FORMAT>
          The format to use for printing diagnostic messages

          Possible values:
          - full:    Print diagnostics verbosely, with context and helpful hints \[default\]
          - concise: Print diagnostics concisely, one per line

      --color <WHEN>
          Control when colored output is used

          Possible values:
          - auto:   Display colors if the output goes to an interactive terminal
          - always: Always display colors
          - never:  Never display colors

      --error-on-warning
          Use exit code 1 if there are any warning-level diagnostics

      --exit-zero
          Always use exit code 0, even when there are error-level diagnostics

  -W, --watch
          Watch files for changes and recheck files related to the changed files

Enabling / disabling rules:
      --error <RULE>
          Treat the given rule as having severity 'error'. Can be specified multiple times.

      --warn <RULE>
          Treat the given rule as having severity 'warn'. Can be specified multiple times.

      --ignore <RULE>
          Disables the rule. Can be specified multiple times.

File selection:
      --respect-ignore-files
          Respect file exclusions via `.gitignore` and other standard ignore files. Use
          `--no-respect-gitignore` to disable

      --exclude <EXCLUDE>
          Glob patterns for files to exclude from type checking.

          Uses gitignore-style syntax to exclude files and directories from type checking. Supports
          patterns like `tests/`, `*.tmp`, `**/__pycache__/**`.