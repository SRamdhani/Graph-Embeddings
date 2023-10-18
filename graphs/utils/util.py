from dataclasses import dataclass, field
from graph_generator import Generation
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Utility(Generation):
    pass


