"""Command-line entry point: run a scenario from a Python config file.

Usage:
    ccr examples/sample_scenario.py [--json]

The config file must define a module-level ``scenario`` of type
:class:`ccr.config.ScenarioConfig`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .config import ScenarioConfig
from .engine import run


def _load_scenario(path: Path) -> ScenarioConfig:
    spec = importlib.util.spec_from_file_location("ccr_scenario", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load scenario from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "scenario"):
        raise AttributeError(f"{path} must define a top-level `scenario`.")
    return module.scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a counterparty-risk scenario.")
    parser.add_argument("config", type=Path, help="Path to a Python scenario config file.")
    parser.add_argument("--json", action="store_true", help="Emit metrics as JSON.")
    args = parser.parse_args(argv)

    scenario = _load_scenario(args.config)
    result = run(scenario)

    if args.json:
        payload = {
            "name": scenario.name,
            "epe": result.epe,
            "max_pfe": result.max_pfe,
            "cva": result.cva,
            "pfe_level": result.pfe_level,
            "n_paths": result.n_paths,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Scenario: {scenario.name}")
        print(f"  Trades      : {len(scenario.portfolio)}")
        print(f"  Paths       : {result.n_paths:,}")
        print(f"  EPE         : {result.epe:,.0f}")
        print(f"  Max PFE @{result.pfe_level:.1%}: {result.max_pfe:,.0f}")
        print(f"  CVA         : {result.cva:,.0f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
