"""
src/config.py
-------------
Centralized configuration container and logging factory.

Responsibility:
    - Load settings from config.yaml relative to project root.
    - Provide path helpers for raw data, processed data, models, reports, and figures.
    - Provide get_logger() for consistent logging across all modules.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Root directory: one level above src/
_SRC_DIR = Path(__file__).resolve().parent
_ROOT = _SRC_DIR.parent


class Config:
    """Central configuration container loaded from config.yaml."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = _ROOT / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as fh:
            self._cfg: Dict[str, Any] = yaml.safe_load(fh)

        # Base directories
        self.root: Path          = _ROOT
        self.raw_data_dir: Path  = _ROOT / self._cfg["paths"]["raw_data"]
        self.proc_data_dir: Path = _ROOT / self._cfg["paths"]["processed_data"]
        self.models_dir: Path    = _ROOT / self._cfg["paths"]["models"]
        self.reports_dir: Path   = _ROOT / self._cfg["paths"]["reports"]
        self.figures_dir: Path   = _ROOT / self._cfg["paths"]["figures"]
        self.metrics_dir: Path   = _ROOT / self._cfg["paths"]["metrics"]

        # File names
        self._raw_files: Dict[str, str]       = self._cfg["raw_files"]
        self._processed_files: Dict[str, str] = self._cfg["processed_files"]

        # Core parameters
        self.random_state: int       = self._cfg["project"]["random_state"]
        self.target_column: str      = self._cfg["model"]["target_column"]
        self.model_filename: str     = self._cfg["model"]["model_filename"]

        self.train_size: float       = self._cfg["splitting"]["train_size"]
        self.dev_size: float         = self._cfg["splitting"]["dev_size"]
        self.test_size: float        = self._cfg["splitting"]["test_size"]
        self.stratify: bool          = self._cfg["splitting"]["stratify"]

        self.numerical_features: List[str]   = self._cfg["numerical_features"]
        self.categorical_features: List[str] = self._cfg["categorical_features"]
        self.cv_folds: int                   = self._cfg["evaluation"]["cv_folds"]

        self.project_name: str       = self._cfg["project"]["name"]
        self.institution: str        = self._cfg["project"]["institution"]

    def raw_data_path(self, table_key: str) -> Path:
        """Return absolute path to raw CSV file."""
        return self.raw_data_dir / self._raw_files[table_key]

    def processed_data_path(self, file_key: str) -> Path:
        """Return absolute path to processed CSV file."""
        return self.proc_data_dir / self._processed_files[file_key]

    def model_path(self) -> Path:
        """Return absolute path to saved model artifact."""
        return self.models_dir / self.model_filename

    def figure_path(self, filename: str) -> Path:
        """Return absolute path for saving a figure."""
        return self.figures_dir / filename

    def metrics_path(self, filename: str) -> Path:
        """Return absolute path for saving metrics."""
        return self.metrics_dir / filename

    def reports_path(self, filename: str) -> Path:
        """Return absolute path for reports."""
        return self.reports_dir / filename

    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieve arbitrary nested keys from config."""
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


CFG = Config()

# --------------------------------------------------------------------------- #
# Unified Logger Setup                                                        #
# --------------------------------------------------------------------------- #
_configured_root = False


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger with console and rotating file output."""
    global _configured_root
    if not _configured_root:
        log_cfg = CFG.get("logging") or {}
        level_name = log_cfg.get("level", "INFO")
        fmt_str = log_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        log_file = log_cfg.get("log_file", "reports/training.log")

        lvl = getattr(logging, level_name.upper(), logging.INFO)
        formatter = logging.Formatter(fmt_str, datefmt="%Y-%m-%d %H:%M:%S")

        root = logging.getLogger()
        root.setLevel(lvl)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(lvl)
        ch.setFormatter(formatter)
        root.addHandler(ch)

        # File handler
        log_path = CFG.root / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(lvl)
        fh.setFormatter(formatter)
        root.addHandler(fh)

        _configured_root = True

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger

