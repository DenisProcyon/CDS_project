from abc import ABC, abstractmethod

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import RegressorMixin

class BaseLinearModel(ABC):
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, model: RegressorMixin) -> None:
        super().__init__()
        self.train, self.test = train, test
        self.model = model
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        pass

class LinearModel(BaseLinearModel):
    def __init__(self, data: pd.DataFrame) -> None:
        super().__init__(LinearRegression(), data)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

class RidgeModel(BaseLinearModel):
    def __init__(self, data: pd.DataFrame, alpha: float = 1.0) -> None:
        super().__init__(Ridge(alpha=alpha), data)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

class LassoModel(BaseLinearModel):
    def __init__(self, data: pd.DataFrame, alpha: float = 1.0) -> None:
        super().__init__(Lasso(alpha=alpha), data)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)
        