from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import joblib
from data.models import Fixture, Prediction


class BasePredictor(ABC):
    @abstractmethod
    def train(self, fixtures: List[Fixture]) -> None: ...

    @abstractmethod
    def predict(self, fixture: Fixture, history: List[Fixture]) -> Prediction: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BasePredictor":
        return joblib.load(path)
