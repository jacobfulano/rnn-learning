import numpy as np
import logging
import warnings
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Task():
    """
    Task
    
    This class associates an input signal with a "reach" target in x,y coordinates. 
    Optionally, a teaching signal 'y_teaching_signal' can be specified for each 
    step
    """
    
    x_in: np.array
    y_target: np.array
    y_teaching_signal: Optional[np.array] = None
    trial_duration: int = field(init=False)
        
    def __post_init__(self):
        self.trial_duration = len(self.x_in)