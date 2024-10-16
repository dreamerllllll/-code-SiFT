'''
Extend the incrementalNet
'''
import torch
import json
from .incrementNet import IncrementNet
import torch.nn.functional as F
from utils.tools import get_class_name
import copy
import logging
class SemanticNet(IncrementNet):
    '''
    use the text embedding as fc
    '''
    def __init__(self, args, logger, backbone_type, dev) -> None:
        super().__init__(args, logger, backbone_type, dev)

        self.class_names = get_class_name(self.args, self.args.dataset)
        self.is_unitnormalize = getattr(self.args, 'is_unitnormalize', True)
        self._logger.info('The normalize is {}'.format(self.is_unitnormalize))

        if getattr(args, 'random_weight', False):
            self.fc = torch.randn(len(self.class_names), self.feature_dim) #random1
            # self.fc = torch.rand(len(self.class_names), self.feature_dim) #random2
            self._logger.info('Use the random weight for fc')
        else:
            self.fc = self.get_weight(self.args.embedding_path)
            if getattr(args, 'embed_addnoise', False):
                assert hasattr(self.args, 'noise_alpha')
                alpha = getattr(self.args, 'noise_alpha')

                ##normalize first
                self.fc = (self.fc-self.fc.mean())/self.fc.std() #the mean if 0.0247 little,so not change too much
                ## add the noise
                noise = torch.randn_like(self.fc)
                self.fc = alpha*noise + (1-alpha)*self.fc
                self._logger.info('Add the noise to fc weight, the alpha is {}'.format(alpha))
        
        if self.is_unitnormalize:
            self.fc = F.normalize(self.fc)
        self.fc = FC(self.fc, args, dev, feature_dim=self.feature_dim).to(dev)

        self.aux_projector = None
        
    def get_weight(self, embedding_path):
        with open(embedding_path, 'r') as f:
            self.data = json.load(f)
            self.file_data = copy.deepcopy(self.data)
        for i in self.data.keys():
            self.data[i] = torch.tensor(self.data[i]).float().mean(dim=0)
        weight = []
        for n in self.class_names:
            weight.append(self.data[n])
        return torch.stack(weight)

    def update_fc(self, classes_list):
        '''
        need not to update the fc, only log the class infor
        '''
        if set(classes_list) & set(self.known_class_list) == set(classes_list):
            return
        elif len(set(classes_list) & set(self.known_class_list)) == 0: # 完全不相交
            self.known_class_list.extend(classes_list)
        else:
            raise NotImplemented
    
    def forward(self, x):
        feature = self.feature_extractor(x)
        if self.aux_projector is not None:
            p_features = self.aux_projector(feature)
        else:
            p_features = None
            
        out, d = self.fc(feature)
        res_d = {'features':feature, 'p_features':p_features}
        res_d.update(d)
        return out, res_d

class FC(torch.nn.Module):
    def __init__(self, weight, args, dev, feature_dim=512) -> None:
        super().__init__()
        self.args = args
        self.is_unitnormalize = getattr(self.args, 'is_unitnormalize', True)
        self.is_eculi_logit = getattr(self.args, 'is_eculi_logit', False) ## use the eculi-distance to calculate the logit
        if self.is_eculi_logit:
            logging.info('Will calculate the eculi logit')

        if getattr(args, 'freeze_fc', True):
            self.weight = weight.to(dev)
        else:
            self.weight = torch.nn.Parameter(weight) #if using to(dev) here, will treat the weight as a normal tensor, not one part of model

        self.scalar = None ; self.projector = None
        if self.args.scalar:
            self.scalar = torch.nn.Parameter(torch.tensor(1.0))
        if self.args.projector and getattr(self.args, 'random_weight', False)==False: #if use random_weight, then no projector
            if getattr(self.args, 'projector_type', 'mlp')=='mlp':
                self.projector = torch.nn.Linear(feature_dim, self.weight.shape[1], bias=False)
            elif getattr(self.args, 'projector_type', 'mlp')=='bottle':
                self.projector = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, 50, bias=False),
                    torch.nn.Linear(50, self.weight.shape[1], bias=False)
                )
            elif getattr(self.args, 'projector_type', 'mlp') == 'random':
                self.projector = torch.nn.Linear(feature_dim, self.weight.shape[1], bias=False)
                #torch.nn.init.kaiming_normal_(self.projector)
                torch.nn.init.normal_(self.projector.weight, 1/feature_dim, 1)
                self.projector.requires_grad_(False)
            logging.info(f'Use the {getattr(self.args, "projector_type", "mlp")} projecter, dim from{feature_dim} to {self.weight.shape[1]}')
            
    def forward(self, feature):
        if self.is_unitnormalize: #normalize
            feature = F.normalize(feature)
        
        if self.projector is not None:
            feature = self.projector(feature)
        
        if self.is_eculi_logit:
            out = -1*torch.cdist(feature, self.weight)
        else:
            out = F.linear(feature, self.weight)
        
        if self.scalar is not None:
            out = self.scalar*out
        return out, {'features':feature}


class BottoleProject(torch.nn.Module):
    '''
    '''
    def __init__(self, inplanes, outplanes, hidden_dim=50, init_type='kaiming'):
        super().__init__()

        self.high2low = torch.nn.Parameter(torch.randn(inplanes, hidden_dim))
        self.low2high = torch.nn.Parameter(torch.randn(hidden_dim, outplanes))
        self.relu = torch.nn.ReLU()
        
        if init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(self.high2low)
            torch.nn.init.kaiming_normal_(self.low2high)
        else:
            raise NotImplemented
        
    def forward(self, features):
        res = features @ self.high2low.to(features.device)
        res = self.relu(res)
        res = res @ self.low2high.to(features.device)
        res = self.relu(res)
        return res 
    