from rnn import RNN

class LearningAlgorithm():
    
    """ Base class for learning algorithms """

    def __init__(self, rnn: RNN) -> None:
        
        self.rnn = rnn

    def update_learning_vars(self):
        pass
    
    def reset_learning_vars(self):
        pass
