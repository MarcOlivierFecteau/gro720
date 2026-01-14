import numpy as np
from typing import TypeAlias

# Type-hinting constructs
Float: TypeAlias = float | np.float32
Array: TypeAlias = np.typing.NDArray[np.float32 | np.float64]
