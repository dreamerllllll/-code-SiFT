'''
The base class for overlap scenario
Author: lxy 5-18
'''
from model.incrementNet import IncrementNet
from methods.base import BaseLearner

class OverlapBase(BaseLearner):
    '''
    The class allowing the Class Overlap Between Clients, and there is no memory. We can see the overlap
    class as memory or something else, which means we can use continue-learning algorithms
    '''
    def __init__(self, args, client, logger) -> None:
        super().__init__(args, client, logger)
        
        self.last_class_list = None #上一个client的class list
    
    def client_download(self, oldKnowledge:dict):
        '''
        Accept information from last client, including the `class list`
        '''
        if 'class_list' not in oldKnowledge:
            assert self._round_id == 1 and self._pos_current_round == 1
        else:
            self.last_class_list = oldKnowledge['class_list']
        super().client_download(oldKnowledge)
    
    def prepare_model(self, net: IncrementNet):
        assert getattr(self._args, 'full_classifier')==True
        self.all_class_list = net.known_class_list
        