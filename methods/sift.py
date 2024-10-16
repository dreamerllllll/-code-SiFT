'''
Author lxy
7-12
'''
from model.incrementNet import IncrementNet
from .overlap_base import OverlapBase
import torch.nn.functional as F
import torch
import numpy as np
from utils.tools import tensor2numpy

ADAPTIVE_KD=False

class SiFT(OverlapBase):
    def __init__(self, args, client, logger) -> None:
        super().__init__(args, client, logger)

        self._old_model = None
        self.global_model = None
        self.T = self._args.temperature

        self.optimizer, self.scheduler = None, None

        ### add the support for cwl(Cyclically weighted loss)
        if not self._args.use_focal_loss and getattr(self._args, 'loss_type', 'ce')=='cwl_ce':
            trainTarget = client.train_dataset.targets
            n_kj = [np.sum(trainTarget==i)/len(trainTarget) for i in client.class_list]
            self._logger.info(f'{client.get_clientName()}=>Class list:{client.class_list} ; n_kj:{n_kj}')
            self.n_kj = torch.tensor(n_kj).float().to(client.dev)
            self.class_list = np.array(client.class_list)

    def prepare_model(self, net: IncrementNet):
        super().prepare_model(net)
    
    def _epoch_train(self, model, train_dataloader, optimizer, scheduler):
        losses = 0
        correct = 0
        total = 0
        clf_losses = 0
        kd_losses = 0

        if self.optimizer is not None:
            work_optimizer = self.optimizer; work_scheduler = self.scheduler
        else:
            work_optimizer = optimizer ; work_scheduler=scheduler
        
        model.train()
        for x, y in train_dataloader:
            x, y = x.to(self._client.dev), y.to(self._client.dev)
            
            logits, outputs = model(x)

            ## 以下代码不考虑 施加约束
            if getattr(self._args, 'local_ce', True): 
                mask = list(self._client.class_list)
                ce_y = self.reflect_targets(y)
            else:
                mask = slice(None)
                ce_y = y

            if not self._args.use_focal_loss:
                loss_type = getattr(self._args, 'loss_type', 'ce')
                if loss_type=='bce':
                    clf_loss = F.binary_cross_entropy_with_logits(logits[:, mask]/self.T, F.one_hot(ce_y, logits[:, mask].shape[1]).float())
                elif loss_type == 'ce':
                    clf_loss = F.cross_entropy(logits[:, mask]/self.T, ce_y)
                elif loss_type == 'cwl_ce': #Cyclically weighted loss in 'Accounting for data variability in multi-institutional distributed deep learning for medical imaging'
                    clf_loss = F.cross_entropy(logits[:, mask]/self.T, ce_y, reduction='none')
                    L = logits.shape[1]
                    idx = [list(self.class_list).index(i) for i in y]
                    clf_loss = torch.mean(clf_loss / (self.n_kj[idx]*L))
                else:
                    raise NotImplemented
            else:
                ## Use focal loss
                focal_lambda = 2
                p = torch.softmax(logits[:, mask]/self.T, dim=-1)
                clf_loss = -1*torch.log(p+1e-10)[range(len(ce_y)), ce_y]
                focal_weight = (1-p[range(len(ce_y)), ce_y])**focal_lambda
                clf_loss = torch.mul(focal_weight, clf_loss).mean()
            clf_losses += clf_loss.item()
            loss = clf_loss

            preds = torch.max(logits, dim=1)[1]
            work_optimizer.zero_grad()
            loss.backward()
            work_optimizer.step()

            losses += loss.item()
            correct += preds.eq(y).cpu().data.sum()
            total += len(y)
        
        if work_scheduler != None:
            work_scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100/total, decimals=4)
        train_loss = [
            'loss', np.around(losses/len(train_dataloader), decimals=4), 
            'clf_loss', np.around(clf_losses/len(train_dataloader), decimals=4),
            'kd_loss', np.around(kd_losses/len(train_dataloader), decimals=4),
        ]
        return model, train_acc, train_loss
    
    def reflect_targets(self, targets):
        '''
        将target做映射, 映射为client的class list的下标
        '''
        dev = targets.device
        cls_list = list(self._client.class_list)
        return torch.tensor(list(map(lambda i:cls_list.index(i), targets)), dtype=torch.int64, device=dev)

    def after_train(self, net):
        self._old_model = None
        self.global_model = None
        
        return super().after_train(net)