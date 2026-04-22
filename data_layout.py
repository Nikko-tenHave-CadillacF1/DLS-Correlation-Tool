"""Project data layout paths (strict, no legacy fallback)."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "Data"

INPUTS_DIR = DATA_ROOT / "inputs"
TEMPLATES_DIR = DATA_ROOT / "templates"
OUTPUTS_DIR = DATA_ROOT / "outputs"
ARCHIVE_DIR = DATA_ROOT / "archive"

CORRELATION_INPUT_DIR = INPUTS_DIR / "correlation"
BOXPLOT_INPUT_DIR = INPUTS_DIR / "boxplots" / "fuel_investigation"

CORRELATION_OUTPUT_DIR = OUTPUTS_DIR / "correlation"
BOXPLOT_OUTPUT_DIR = OUTPUTS_DIR / "boxplots"
CORRELATION_PLOTS_DIR = CORRELATION_OUTPUT_DIR / "plots"
BOXPLOT_PLOTS_DIR = BOXPLOT_OUTPUT_DIR / "plots"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data_dirs() -> None:
    """Create the required project data directory structure."""
    for path in (
        DATA_ROOT,
        INPUTS_DIR,
        TEMPLATES_DIR,
        OUTPUTS_DIR,
        ARCHIVE_DIR,
        CORRELATION_INPUT_DIR,
        BOXPLOT_INPUT_DIR,
        CORRELATION_OUTPUT_DIR,
        BOXPLOT_OUTPUT_DIR,
        CORRELATION_PLOTS_DIR,
        BOXPLOT_PLOTS_DIR,
    ):
        _ensure_dir(path)


def resolve_correlation_input_dir() -> Path:
    """Return the fixed correlation input directory."""
    ensure_data_dirs()
    return CORRELATION_INPUT_DIR


def resolve_boxplot_input_dir() -> Path:
    """Return the fixed box-plot input directory."""
    ensure_data_dirs()
    return BOXPLOT_INPUT_DIR


def resolve_template_path(filename: str = "template.pptx") -> Path:
    """Return a template path inside the fixed templates directory."""
    ensure_data_dirs()
    return TEMPLATES_DIR / filename


ensure_data_dirs()
