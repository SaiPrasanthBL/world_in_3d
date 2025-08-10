from abc import ABC, abstractmethod 

import torch

class BaseLoss(ABC):
    def __init__(self, config) -> None:
        super().__init__()

        @abstractmethod
        def get_loss_metric_names(self) -> list[str]:
            """
            Returns a list of metric names that this loss computes.
            These names will be used to log the loss values during training.
            """
            ...
        
        @abstractmethod
        def __call__(self, data) -> dict[str, torch.Tensor]:
            ...