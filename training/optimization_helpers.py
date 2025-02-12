import numpy as np
import scipy
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TModel = TypeVar("TModel")


class IModelEvaluator(ABC, Generic[TModel]):
    """Interface for a class that can interface with the scipy.optimize.minimize() function.
    The minimize() function uses a flat array of parameters. This class implements the mapping
    params<->model and a function to evaluate the model on the data. Data should be captured in the
    __init__() method."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def model2params(self, model: TModel) -> np.ndarray: ...

    @abstractmethod
    def params2model(self, params: np.ndarray) -> TModel: ...

    @abstractmethod
    def getParamBounds(self) -> scipy.optimize.Bounds | None: ...

    @abstractmethod
    def evaluateModel(self, model: TModel) -> float: ...

    def evaluateParams(self, params: np.ndarray) -> float:
        return self.evaluateModel(self.params2model(params))

    def __call__(self, params: np.ndarray) -> float:
        return self.evaluateParams(params)


def optimizeModel(
    evaluator: IModelEvaluator[TModel], model0: TModel, debug=False
) -> TModel:
    params0 = evaluator.model2params(model0)
    res = scipy.optimize.minimize(
        evaluator, params0, bounds=evaluator.getParamBounds(), options={"disp": debug}
    )
    if debug:
        print(res)
    model = evaluator.params2model(res.x)
    return model
