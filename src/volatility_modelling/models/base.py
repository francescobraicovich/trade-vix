from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any

class BaseModel(ABC):
    def __init__(self, cfg: Dict[str, Any]): self.cfg = cfg
    @abstractmethod
    def fit(self, **kwargs): ...
    @abstractmethod
    def predict(self, **kwargs) -> pd.Series: ...
