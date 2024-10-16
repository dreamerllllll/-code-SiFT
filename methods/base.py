'''
用来实现一个device的本地更新策略过程 的基类
author: lxy 2-9
'''

# from clients import client
import os
import copy
import torch
import torch.optim as optim
from model.incrementNet import IncrementNet
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

from utils.tools import tensor2numpy, SimpleDataset

EPSILON = 1e-8

class BaseLearner(object):
    '''
    提供方法：
    数据的准备，已经准备好了，不用再次准备了
    prepare_model: 准备本次本地更新所需要的模型,需要判断是否要改变模型结构。
    client_train: 增量训练
    epoch_train: 做一个epoch的训练
    '''
    def __init__(self, args, client, logger) -> None:
        self._args = args
        self._logger = logger
        self._client = client # 为某一个client所服务

        self._lr = args.lr
        self._momentum = args.momentum
        self._fixed_memory = getattr(args, 'fixed_memory', None)
        self._memory_per_class = getattr(args, 'memory_per_class', None)
        self._memory_size = getattr(args, 'memory_size', None)

        self._multiple_gpus = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        # print("Let's use", len(self._multiple_gpus), "GPUs!")

        self._round_id = None
        self._pos_current_round = None
        self._local_optim = None
        self._data_memory = None
        self._final_train_acc = None #向服务器上传最后一个epoch的acc，但是不需要下载这个数据

        self.is_focal_loss = getattr(args, 'is_focal_loss', False) #是否采用focal loss，需要使用参数控制
        self.epoch = -1

    def client_upload(self):
        oldKnowledge = {}
        oldKnowledge['data_memory'] = self._data_memory
        oldKnowledge['targets_memory'] = self._targets_memory
        oldKnowledge['final_train_acc'] = self._final_train_acc
        oldKnowledge['class_list'] = self._client.class_list
        # oldKnowledge['class_mean'] = self._class_means
        oldKnowledge['class_mean'] = self._class_means
        if self._data_memory is not None and len(self._data_memory)>0:
            self._logger.info('{} Send the memory len {}'.format(self._client.get_clientName(), len(self._targets_memory)))
        return oldKnowledge

    def client_download(self, oldKnowledge):
        if oldKnowledge is None or oldKnowledge.get('data_memory', None) is None:
            self._data_memory = np.array([])
            self._targets_memory = np.array([])
            self._class_means = np.array([])
        else:
            self._data_memory = oldKnowledge['data_memory']
            self._targets_memory = oldKnowledge['targets_memory']
            self._class_means = oldKnowledge['class_mean']
            if self._data_memory is not None and len(self._data_memory)>0:
                self._logger.info('{} Receive the memory len {}'.format(self._client.get_clientName(), len(self._targets_memory)))

        if oldKnowledge is None or oldKnowledge.get('previous_class_lists', None) is None:
            self.previous_class_lists = None
        else:
            self.previous_class_lists = oldKnowledge['previous_class_lists']
        
        self.oldKnowledge = oldKnowledge

    def prepare_model(self, net:IncrementNet):
        self.known_class_list = net.known_class_list.copy()
        net.update_fc(self._client.class_list) # 更新逻辑基于class_list中的class有没有见过，在net中实现
        self.all_class_list = net.known_class_list
    
    def client_train(self, net):
        net = self._train(net)
        return net
    
        
    def _get_memory(self):
        if self._data_memory is None or len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)
        
    def _add_memory(self):
        if self._get_memory() is None:
            pass
        else:
            memory_data, memory_targets = self._get_memory()
            data, targets = np.concatenate((self._client.trainData, memory_data)), np.concatenate((self._client.trainTarget, memory_targets))
            dataset = SimpleDataset(data, targets, self._client.train_tfs, use_path=self._client.use_path)
            self._client.train_dataloader = DataLoader(dataset, batch_size=self._args.batchsize, shuffle=True, num_workers=self._args.num_workers)
            self._logger.info('add memory ({} + {} => {}) success'.format(len(self._client.trainData), len(memory_data), len(dataset)))
    
    def _train(self, net, own_test=False):
        '''
        Args:
            own_test: 只在当前client自己的数据集上进行测试,默认根据test_dataloader来决定
        '''
        if len(self._multiple_gpus) > 1:
            net = torch.nn.DataParallel(net, self._multiple_gpus)
        
        self._logger.info('learning on class: {}'.format(self._client.class_list))

        self._add_memory()

        optimizer = self._get_optimizer(self._args, filter(lambda p: p.requires_grad, net.parameters()))
        scheduler = self._get_scheduler(self._args, optimizer)

        if self._round_id == 1:
            epoch_num = self._args.client_init_epoch if self._pos_current_round == 1 else self._args.round_init_epoch
        else:
            epoch_num = self._args.epoch
        
        train_acc = -1
        for epoch in range(epoch_num):
            self.epoch = epoch
            net, train_acc, train_loss = self._epoch_train(net, self._client.train_dataloader, optimizer, scheduler)
            if (epoch+1) % self._args.local_val_freq == 0:
                if own_test:
                    test_dataloader = DataLoader(self._client.own_testDataSet, batch_size=self._args.batchsize, shuffle=False, num_workers=self._args.num_workers)
                    test_acc = self._epoch_test(net, test_dataloader)
                else:
                    test_acc = self._epoch_test(net, self._client.test_dataloader)
                if not isinstance(test_acc, list):
                    test_acc = ['all_tAcc', test_acc]
            record_dict = {'name':self._client.get_clientName(), 'round':self._round_id, 'epoch':epoch}
            # record_dict[self._client.get_clientName()+f'_r({self._round_id})'+'_train_loss'] = train_loss
            # for i in range(int(len(train_loss)/2)):
            #     record_dict[self._client.get_clientName()+'_train_'+train_loss[i*2]] = train_loss[i*2+1]
            # record_dict[self._client.get_clientName()+'_train_acc'] = train_acc
            
            if (epoch+1) % self._args.local_val_freq == 0:
                for i in range(int(len(test_acc)/2)):
                    record_dict[test_acc[i*2]] = test_acc[i*2+1]

                self._logger.info('{}, Epoch {}/{} ==> '.format(self._client.name, epoch+1, epoch_num)+\
                                ('{} {:.4f}, '*int(len(train_loss)/2)).format(*train_loss)+\
                                'train acc: {:.4f}'.format(train_acc)+\
                                ', test acc:{:.4f}'.format(test_acc[1]))
                
            else:
                self._logger.info('{}, Epoch {}/{} ==> '.format(self._client.name, epoch+1, epoch_num)+\
                                ('{} {:.4f}, '*int(len(train_loss)/2)).format(*train_loss)+\
                                'train acc: {:.4f}'.format(train_acc))
            self._logger.visual_logging('train', self._client.name, record_dict, epoch+1+self._client.history_epoch)
            
            self._logger.dumps_dict(record_dict)

        self._final_train_acc = train_acc
        self._client.history_epoch += epoch_num

        if self._fixed_memory is not None:
            self.build_rehearsal_memory(net, self._memory_per_class)

        if len(self._multiple_gpus) > 1:
            net = net.module
        
        if getattr(self._args, 'return_optim', False) and getattr(self, '_local_optimal_model', None) is not None:
            self._logger.info('E={}, return the local optimal model'.format(epoch_num))
            return getattr(self,'_local_optimal_model')
        else:
            return net

    def _epoch_train(self, model, train_dataloader, optimizer, scheduler):
        losses = 0
        correct = 0
        total = 0

        model.train()
        for x, y in train_dataloader:
            x, y = x.to(self._client.dev), y.to(self._client.dev)
            logits, _ = model(x)

            if self.is_focal_loss:
                focal_lambda = 2
                p = torch.softmax(logits, dim=-1)
                clf_loss = -1*torch.log(p+1e-10)[range(len(y)), y]
                focal_weight = (1-p[range(len(y)), y])**focal_lambda
                loss = torch.mul(focal_weight, clf_loss).mean()
            else:
                loss = cross_entropy(logits, y)
            
            preds = torch.max(logits, dim=1)[1]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()
            correct += preds.eq(y).cpu().data.sum()
            total += len(y)
        
        if scheduler != None:
            scheduler.step()
        train_acc = np.around(tensor2numpy(correct)*100/total, decimals=4)
        train_loss = ['focal_loss', np.around(losses/len(train_dataloader), decimals=4)] if self.is_focal_loss else ['loss', np.around(losses/len(train_dataloader), decimals=4)]
        return model, train_acc, train_loss

    def _epoch_test(self, model, test_dataloader):
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(self._client.dev), y.to(self._client.dev)
                logits, d = model(x)
                preds = torch.max(logits, dim=1)[1]

                correct += preds.eq(y).cpu().data.sum()
                total += len(y)

                all_preds.append(preds.data.cpu())
                all_targets.append(y.cpu())
            test_acc = np.around(tensor2numpy(correct)*100/total, decimals=4)
        
        if getattr(self._args, 'verbose_acc', 'False') == True:
            res = []
            res.extend(['all_tAcc', test_acc])
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            class_indx = torch.unique(all_targets)
            # for i in class_indx:
            #     indx = all_targets == i
            #     res.append(str(i.item())+'_tAcc')
            #     acc = all_preds[indx].eq(all_targets[indx]).sum() / indx.sum()
            #     res.append(np.around(tensor2numpy(acc*100), decimals=4))
            all_preds = all_preds.numpy()
            all_targets = all_targets.numpy()
            for i in range(0, len(class_indx), self._client.get_client_class_num()):
                indx = np.where(np.logical_and(all_targets >= i, all_targets < i + self._client.get_client_class_num()))[0]
                label = '{}-{}'.format(str(i).rjust(2, '0'), str(i+self._client.get_client_class_num()-1).rjust(2, '0'))
                res.append(label)
                res.append(np.around((all_targets[indx] == all_preds[indx]).sum()*100 / len(indx), decimals=4))
            return res
        return test_acc
    
    def client_eval(self, net):
        if getattr(self._args,'apply_nme', None):
            net.eval()
            vectors, targets = self._extract_vectors(net, self._client.test_dataloader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            dists = cdist(self._class_means, vectors, "sqeuclidean")  # [nb_classes, N]
            scores = dists.T  # [N, nb_classes]
            preds = np.argmin(scores, axis=1)
            nme_acc = np.around((preds == targets).sum()*100/len(targets), decimals=4)
            self._logger.info('Eval {}, NME Acc: {:.4f}\n'.format(self._client.name, nme_acc))

    def _get_optimizer(self, args, params):
        optimizer = None
        if self._round_id == 1:
            if self._pos_current_round == 1:
                if args.opt_type == 'sgd':
                    optimizer = optim.SGD(params, lr=args.client_init_lr, momentum=args.round_init_momentum, weight_decay=args.client_init_weight_decay)
                elif args.opt_type == 'adam':
                    optimizer = optim.Adam(params, lr=args.client_init_lr, weight_decay=self._args.client_init_weight_decay)
                    # optimizer = optim.Adam(params, lr=args.round_init_lr)
                elif args.opt_type == 'adamw':
                    optimizer = optim.AdamW(params, lr=args.client_init_lr, weight_decay=self._args.client_init_weight_decay)
                else: 
                    raise ValueError('No optimazer: {}'.format(args.opt_type))
            else:
                if args.opt_type == 'sgd':
                    optimizer = optim.SGD(params, lr=args.round_init_lr, momentum=args.round_init_momentum, weight_decay=args.round_init_weight_decay)
                elif args.opt_type == 'adam':
                    optimizer = optim.Adam(params, lr=args.round_init_lr, weight_decay=self._args.round_init_weight_decay)
                    # optimizer = optim.Adam(params, lr=args.round_init_lr)
                elif args.opt_type == 'adamw':
                    optimizer = optim.AdamW(params, lr=args.client_init_lr, weight_decay=self._args.client_init_weight_decay)
                else: 
                    raise ValueError('No optimazer: {}'.format(args.opt_type))
        else:
            # if args.round_milestones is not None:               # lr随着round变化
            #     if isinstance(args.round_milestones, list):
            #         self._lr = self._lr * 0.1 if self._round_id in args.round_milestones else self._lr
            #     elif isinstance(args.round_milestones, int):
            #         self._lr = self._lr * 0.998 if (self._round_id-1)%args.round_milestones==0 else self._lr
            if getattr(args,'round_lr_decay', None) is not None:
                ratio = args.round_lr_decay
                if ratio > 0:
                    self._lr = args.lr*(ratio**(self._round_id-1))
            self._logger.info('now lr is {}'.format(self._lr))
            if args.opt_type == 'sgd':
                optimizer = optim.SGD(params, lr=self._lr, momentum=self._momentum, weight_decay=args.weight_decay)
            elif args.opt_type == 'adam':
                optimizer = optim.Adam(params, lr=self._lr, weight_decay=self._args.weight_decay)
                # optimizer = optim.Adam(params, lr=args.lr)
            elif args.opt_type == 'adamw':
                optimizer = optim.AdamW(params, lr=args.client_init_lr, weight_decay=self._args.client_init_weight_decay)
            else: 
                raise ValueError('No optimazer: {}'.format(args.opt_type))
        return optimizer

    def _get_scheduler(self, args, optimizer):
        scheduler = None
        if self._round_id == 1:
            if self._pos_current_round == 1:
                if args.scheduler == 'multi_step':
                    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.client_init_milestones, gamma=args.lrate_decay)
                elif args.scheduler == 'cos':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.client_init_epoch)
                elif args.scheduler == None:
                    scheduler = None
                else: 
                    raise ValueError('No scheduler: {}'.format(args.scheduler))
            else:
                if args.scheduler == 'multi_step':
                    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.round_init_milestones, gamma=args.lrate_decay)
                elif args.scheduler == 'cos':
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.round_init_epoch)
                elif args.scheduler == None:
                    scheduler = None
                else: 
                    raise ValueError('No scheduler: {}'.format(args.scheduler))
        else:
            if args.scheduler == 'multi_step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.lrate_decay)
            elif args.scheduler == 'cos':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch)
            elif args.scheduler == None:
                scheduler = None
            else: 
                raise ValueError('No scheduler: {}'.format(args.scheduler))
        return scheduler
    
    def after_train(self, net):
        net.last_class_list = self._client.class_list
        return net
    
    def _extract_vectors(self, net, loader):
        net.eval()
        vectors, targets = [], []
        for _inputs, _targets in loader:
            _targets = _targets.numpy()
            _vectors = tensor2numpy(net.extract_features(_inputs.to(self._client.dev)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def receive(obj):
        '''
        Receive something from server, and then do something
        '''
        raise NotImplemented
    
    def build_rehearsal_memory(self, net, per_class):
        # if self._fixed_memory:
        if self._fixed_memory and self._round_id == 1:                                  # changed
            per_class = self._memory_size // len(self.all_class_list)
            self._reduce_exemplar(net, per_class)
            self._construct_exemplar(net, per_class)
        else:
            self._construct_exemplar_unified(net, per_class)

    def _construct_exemplar_unified(self, net, m):
        self._logger.info("Constructing exemplars for new classes...({} per classes)\n".format(m))
        _class_means = np.zeros((len(self.all_class_list), net.feature_dim))

        # Calculate the means of old classes with newly trained network
        # for class_idx in range(len(self.known_class_list)):
        # idx = list(set(self.all_class_list)-set(self._client.class_list)) if self._round_id == 1 else self.all_class_list  # changed
        if self._round_id == 1:
            idx = list(set(self.all_class_list)-set(self._client.class_list))
            for class_idx in idx:  # changed
                mask = np.where(self._targets_memory == class_idx)[0]
                class_data, class_targets = (
                    self._data_memory[mask],
                    self._targets_memory[mask],
                )

                class_dset = SimpleDataset(class_data, class_targets, self._client.test_tfs, use_path=self._client.use_path)
                class_loader = DataLoader(class_dset, batch_size=64, shuffle=False, num_workers=4)

                vectors, _ = self._extract_vectors(net, class_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        # for class_idx in range(len(self.known_class_list), len(self.all_class_list)):
        # if self._round_id == 1:
        for class_idx in self._client.class_list:                                       # changed
            mask = np.where(self._client.trainTarget == class_idx)[0]
            class_data, class_targets = (
                self._client.trainData[mask],
                self._client.trainTarget[mask],
            )
            class_dset = SimpleDataset(class_data, class_targets, self._client.test_tfs, use_path=self._client.use_path)
            class_loader = DataLoader(class_dset, batch_size=64, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(net, class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            if self._round_id != 1:
                self._class_means[class_idx, :] = class_mean
            else:
            # Select
                selected_exemplars = []
                exemplar_vectors = []
                for k in range(1, m+1):
                    S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                    selected_exemplars.append(np.array(class_data[i]))  # New object to avoid passing by inference
                    exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                    vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                    class_data = np.delete(class_data, i, axis=0)  # Remove it to avoid duplicative selection

                selected_exemplars = np.array(selected_exemplars)
                exemplar_targets = np.full(m, class_idx)
                if self._round_id == 1:                                                     # changed
                    self._data_memory = (
                        np.concatenate((self._data_memory, selected_exemplars))
                        if len(self._data_memory) != 0
                        else selected_exemplars
                    )
                    self._targets_memory = (
                        np.concatenate((self._targets_memory, exemplar_targets))
                        if len(self._targets_memory) != 0
                        else exemplar_targets
                    )
                # else:                                                                       # changed
                #     self._data_memory[class_idx*m : (class_idx+1)*m] = selected_exemplars
                #     self._targets_memory[class_idx*m : (class_idx+1)*m] = exemplar_targets

                # Exemplar mean
                exemplar_dset = SimpleDataset(selected_exemplars, exemplar_targets, self._client.test_tfs, use_path=self._client.use_path)
                exemplar_loader = DataLoader(exemplar_dset, batch_size=64, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(net, exemplar_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

                _class_means[class_idx, :] = mean

        if self._round_id == 1:
            self._class_means = _class_means

    def _reduce_exemplar(self, net, m):
        self._logger.info("Reducing exemplars...(to {} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)

        self._class_means = np.zeros((len(self.all_class_list), net.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(len(self.known_class_list)):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = SimpleDataset(dd, dt, self._client.test_tfs, use_path=self._client.use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(net, idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, net, m):
        self._logger.info("Constructing exemplars...({} per classes)\n".format(m))
        for class_idx in range(len(self.known_class_list), len(self.all_class_list)):
            mask = np.where(self._client.trainTarget == class_idx)[0]
            class_data, class_targets = (
                self._client.trainData[mask],
                self._client.trainTarget[mask],
            )
            idx_dataset = SimpleDataset(class_data, class_targets, self._client.test_tfs, use_path=self._client.use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(net, idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(np.array(class_data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                class_data = np.delete(class_data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = SimpleDataset(selected_exemplars, exemplar_targets, self._client.test_tfs, use_path=self._client.use_path)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(net, idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
    @property
    def nb_classes(self):
        if self._args.dataset == 'cifar10':
            return 10
        elif self._args.dataset == 'cifar100':
            return 100