from typing import *
import torch
from typing import Any

class Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x: Any): ...

class NormedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x: Any): ...

ObsTensor: Any
ActionTensor: Any
GoalTensor: Any
NormalObsTensor: Any
NormalActionTensor: Any
NormalGoalTensor: Any
Array: Any
NormedArray: Any
ObsArray: Any
ActionArray: Any
GoalArray: Any
NormalObsArray: Any
NormalActionArray: Any
NormalGoalArray: Any
