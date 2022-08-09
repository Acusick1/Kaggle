from pathlib import Path
from torch import cuda

DATA_PATH = Path(Path(__file__).parent.resolve(), "../data")
DEVICE = "cuda" if cuda.is_available() else "cpu"
RNG_STATE = 13
