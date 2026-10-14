"""YAML configuration loading with command-line overrides."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, then apply `key.sub=value` overrides."""
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    cfg = OmegaConf.load(p)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def add_config_args(parser: argparse.ArgumentParser, default: str) -> None:
    """Add `--config` plus trailing `key=value` overrides to a parser."""
    parser.add_argument("--config", default=default, help="path to a YAML config")
    parser.add_argument(
        "opts",
        nargs="*",
        metavar="key=value",
        help="config overrides, e.g. codec.tau=0.2 data.pairs=ALL",
    )


def resolve_ckpt(path: str) -> str:
    """A checkpoint path as given if it exists, else the same file name under CKPTS_DIR."""
    p = Path(path)
    if p.exists():
        return str(p)
    alt = Path(os.environ.get("CKPTS_DIR", _REPO_ROOT / "ckpts")) / p.name
    return str(alt) if alt.exists() else str(p)


def print_config(cfg: DictConfig) -> None:
    print("[config]\n" + OmegaConf.to_yaml(cfg).rstrip(), flush=True)
