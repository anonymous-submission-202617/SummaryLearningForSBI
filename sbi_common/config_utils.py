"""Shared configuration-file resolution for all experiment scripts.

Every experiment accepts the same command-line convention::

    python <experiment>/run_experiment.py --config <path>

where ``<path>`` may be absolute, relative to the current working directory,
or relative to the experiment script's directory.  Resolution priority is:

1. ``--config <path>`` on the command line,
2. the environment variable ``<EXPERIMENT>_CONFIG_PATH`` (e.g.
   ``SIR_CONFIG_PATH``, ``GAUSSIAN_CONFIG_PATH``),
3. ``config.json`` next to the experiment script.

For backward compatibility, the Lotka-Volterra and DataSizeDependence scripts
still accept a positional config path (``python run_experiment.py config.json``).
"""

import os
import sys


def cli_config_arg(argv=None):
    """Return the value of ``--config <path>`` / ``--config=<path>`` from argv.

    Args:
        argv: argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The config path as a string, or ``None`` if no ``--config`` flag is
        present.  The first occurrence wins; later flags are ignored.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    for i, arg in enumerate(argv):
        if arg == "--config":
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def resolve_config_path(script_dir, env_var_name, cli_value=None):
    """Resolve the experiment configuration file path.

    Args:
        script_dir: directory containing the experiment script.
        env_var_name: environment variable name consulted when no CLI value
            is given (e.g. ``"SIR_CONFIG_PATH"``).
        cli_value: optional path from ``--config``.

    Returns:
        ``(path, requested)`` where ``path`` is the first existing candidate
        (or ``None``) and ``requested`` is the non-empty user-provided value
        (CLI or environment) used for diagnostics, or ``""`` when falling
        back to the default ``config.json``.
    """
    requested = (cli_value or "").strip() or os.environ.get(env_var_name, "").strip()
    candidates = []
    if requested:
        if os.path.isabs(requested):
            candidates.append(requested)
            # Fall back to the same filename beside the script when the
            # absolute path does not exist on this machine.
            candidates.append(os.path.join(script_dir, os.path.basename(requested)))
        else:
            candidates.append(os.path.join(os.getcwd(), requested))
            candidates.append(os.path.join(script_dir, requested))
    else:
        candidates.append(os.path.join(os.getcwd(), "config.json"))
        candidates.append(os.path.join(script_dir, "config.json"))

    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            if os.path.exists(path):
                return path, requested
    return None, requested


def is_evaluation(config):
    """Validate the experiment mode, defaulting to the original training workflow."""
    mode = config.get("run_mode", "new_experiment")
    if mode not in ("new_experiment", "evaluation"):
        raise ValueError(f"Unknown run_mode {mode!r}; use 'new_experiment' or 'evaluation'.")
    return mode == "evaluation"
