from pathlib import Path
from torch import cuda

PROJECT_PATH = (Path(__file__).parent / "..").resolve()
DATA_PATH = PROJECT_PATH / "data"
DEVICE = "cuda" if cuda.is_available() else "cpu"
RNG_STATE = 13
