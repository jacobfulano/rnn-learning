import numpy as np
import logging
import warnings
import dataclasses
from dataclasses import dataclass
from typing import Optional, List



from dataclasses import dataclass, field

@dataclass
class Task():
    """
    Task Class
    
    Should contain targets, specification of time to run task?

    N.B. what is SL signal is calculated on the fly?
    """
    
    x_in: np.array
    y_target: np.array
    y_teaching_signal: Optional[np.array] = None
    trial_duration: int = field(init=False)
        
    def __post_init__(self):
        self.trial_duration = len(self.x_in)