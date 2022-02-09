from typing import Optional, List
from rnn import RNN

class LearningAlgorithm():
    
    """ Base class for learning algorithms """

    def __init__(self, rnn: RNN, apply_to: List[str]=['w_rec'], online: bool = False) -> None:
        
        self.rnn = rnn
        self.apply_to = apply_to
        self.online = online

    def update_learning_vars(self):
        """ Updates Learning Variables """
        pass
    
    def reset_learning_vars(self):
        """ Reset Learning Variables """
        pass
    
    def print_params(self) -> None:
        
        """ Print Hyperparameters """
        for k in ['apply_to', 'online']:
            print(k,': ',vars(self)[k])
