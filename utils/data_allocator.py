import warnings
import numpy as np
import copy
import os
import copy
import logging

import pandas as pd
import matplotlib.pyplot as plt

class DataAllocator():
    '''
    To split the whole dataset to each client.
    '''
    def __init__(self, args, train_targets, test_targets, total_class_num) -> None:
        self.args = args
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.total_class_num = total_class_num
        self.client_num = args.num_of_clients

    def getAllocDict(self):
        if self.args.allocate_type == 'iid':
            train_slice_dct = self.iid(self.train_targets, self.client_num)
            test_slice_dct = self.iid(self.test_targets, self.client_num)
        elif self.args.allocate_type == 'noverlap':
            train_slice_dct = self.noverlap_noniid(self.train_targets, self.client_num, self.total_class_num, self.args.base_class)
            test_slice_dct = self.noverlap_noniid(self.test_targets, self.client_num, self.total_class_num, self.args.base_class)
        elif self.args.allocate_type == 'fedavg_noniid':
            train_slice_dct, test_slice_dct = self.fedavg_noniid(self.train_targets, self.test_targets, num_shards=getattr(self.args, 'num_shards', None))
        elif self.args.allocate_type == 'circle_noniid':
            train_slice_dct = self.circle_noniid(self.train_targets, self.client_num, self.args.overlap_class_num)
            test_slice_dct = self.circle_noniid(self.test_targets, self.client_num, self.args.overlap_class_num)
        elif self.args.allocate_type == 'label_quality_noniid':
            train_slice_dct, test_slice_dct = self.label_skew_quantity_based_partition(self.train_targets, self.test_targets, self.client_num, self.total_class_num, self.args.class_per_cli)
        elif self.args.allocate_type == 'label_dir_noniid':
            train_slice_dct, test_slice_dct = self.hetero_dir_partition(self.train_targets, self.test_targets, self.client_num, self.total_class_num, self.args.dir_alpha, min_require_size=getattr(self.args, 'min_require_size', None))
            if getattr(self.args, 'show_partition', False):
                os.makedirs('../imgs', exist_ok=True)
                self.show_partition(copy.deepcopy(train_slice_dct), len(train_slice_dct), '../imgs/label_dir_noniid2_train.jpg', self.train_targets)
                self.show_partition(copy.deepcopy(test_slice_dct), len(test_slice_dct), '../imgs/label_dir_noniid2_test.jpg', self.test_targets)
        else:
            raise NotImplemented
        
        data_num_dict = DataAllocator.cal_data_num(train_slice_dct, test_slice_dct, self.train_targets, self.test_targets)
        logging.info(data_num_dict)

        return train_slice_dct, test_slice_dct
    
    @staticmethod
    def iid(targets, client_num):
        np.random.seed(1993)
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        dict_users = {}
        cls_indx = {}
        classes = np.unique(targets)
        for i in classes:
            indx = np.random.permutation(select(targets, i, i+1))
            cls_indx[i] = list(indx)
        
        for i in range(client_num):
            temp = []
            for c in classes:
                temp += cls_indx[c][(len(cls_indx[c])//client_num)*i:(len(cls_indx[c])//client_num)*(i+1)] 
            dict_users[i] = np.array(temp, dtype='int64')
        return dict_users
        
    def fedavg_noniid(self, targets, test_targets, num_shards=None)->dict:
        '''
        will return a list, each element represent one client's data and target 's index
        '''
        if num_shards == None:
            num_shards = self.client_num*2
        assert num_shards % len(np.unique(targets)) == 0
        def get_infor(targ, num_shards, client_num):
            total_sample_nums = len(targ)
            size_of_shards = int(total_sample_nums / num_shards)
            if total_sample_nums % num_shards != 0:
                warnings.warn(
                    "warning: the length of dataset isn't divided exactly by num_shard.some samples will be dropped."
                )
            
            if num_shards % client_num != 0:
                warnings.warn(
                    "warning: num_shard isn't divided exactly by num_clients. some samples will be dropped."
                )
            dict_users = {i: np.array([], dtype='int64') for i in range(client_num)}
            idxs = np.arange(total_sample_nums)

            return dict_users, size_of_shards, idxs

        # the number of shards that each one of clients can get
        shard_pc = int(num_shards / self.client_num)
        train_dict_users, train_size_of_shards, train_idxs = get_infor(targets, num_shards, self.client_num)        
        test_dict_users, test_size_of_shards, test_idxs = get_infor(test_targets, num_shards, self.client_num)        
        
        # assign
        idx_shard = [i for i in range(num_shards)]
        for i in range(self.client_num):
            rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
            idx_shard = list(set(idx_shard) - rand_set) # 因为这个地方是减去了，所以在后面的随机抽取中并不会发生一些问题
            for rand in rand_set:
                train_dict_users[i] = np.concatenate(
                    (train_dict_users[i], train_idxs[rand * train_size_of_shards:(rand + 1) * train_size_of_shards]),
                    axis=0
                )
                test_dict_users[i] = np.concatenate(
                    (test_dict_users[i], test_idxs[rand * test_size_of_shards:(rand + 1) * test_size_of_shards]),
                    axis=0
                )
        return train_dict_users, test_dict_users
    
    @staticmethod
    def circle_noniid(targets, client_num, overlap_class_num, save_path=None):
        '''
        The circle non-iid, devide all the class to clients by circle extend, which
        allowing overlap.
        '''
        classes = np.unique(targets)
        base_per_client = len(classes) // client_num
        assert overlap_class_num%2 == 0
        assert len(classes)%client_num == 0 and overlap_class_num<=len(classes)//client_num
        dict_users = {i: np.array([], dtype='int64') for i in range(client_num)}
        class_idx = []
        times = [0 for c in classes]
        for i in range(client_num):
            base_class = list(range(i*base_per_client, (i+1)*base_per_client))
            left_start = (i*base_per_client-overlap_class_num//2+len(classes))%len(classes)
            left_overlap = list(range(left_start, left_start+overlap_class_num//2))
            right_overlap = list(range(((i+1)*base_per_client)%len(classes), ((i+1)*base_per_client+overlap_class_num//2)%len(classes)))
            clinet_cls = left_overlap+base_class+right_overlap
            class_idx.append(clinet_cls)
            for c in clinet_cls:
                times[c]+=1
        for i in classes:
            indx = np.where(targets==i)[0]
            indx = np.random.permutation(indx)
            splits = np.split(indx, times[i])
            allocate_time = 0
            for client, j in enumerate(class_idx):
                if i in j:
                    dict_users[client] = np.concatenate((dict_users[client], splits[allocate_time]),
                                                        axis=0)
                    allocate_time += 1
                    
            assert allocate_time == len(splits)
        if save_path:
            DataAllocator.show_partition(copy.deepcopy(dict_users), len(classes), save_path, targets)
        return dict_users
    
    @staticmethod
    def noverlap_noniid(targets, client_num, cls_num, base_class=None)->dict:
        '''
        commonly used by  continue learning
        '''
        assert cls_num >= client_num
        indices = list(range(cls_num+1))
        dict_users = {i: np.array([], dtype='int64') for i in range(client_num)}

        if base_class is None:
            client_class_num = cls_num // client_num
            for i in range(client_num):
                dict_users[i] = select(targets, low_range=indices[i*client_class_num], high_range=indices[(i+1)*client_class_num])
        else:
            client_class_num = (cls_num-base_class) // (client_num-1)
            dict_users[0] = select(targets, low_range=indices[0], high_range=indices[base_class])
            for i in range(client_num-1):
                dict_users[i+1] = select(targets, \
                    low_range=indices[base_class + i*client_class_num], high_range=indices[base_class + (i+1)*client_class_num])
        return dict_users

    #-------------------label skew-----------------
    @staticmethod
    def hetero_dir_partition(targets, test_targets, num_clients, num_classes, dir_alpha, min_require_size=None):
        '''
        Use the dir distribution to split each class's samples to every client
        The result is: for each client, different class has different num of samples
        '''
        if min_require_size is None:
            min_require_size = num_classes

        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)
            test_targets = np.array(test_targets)
        num_samples = targets.shape[0]

        min_size = 0
        np.random.seed(1993) #make the allocatation same for same setting 
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            test_idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(num_classes):
                idx_k = np.where(targets == k)[0]
                test_idx_k = np.where(test_targets==k)[0]
                np.random.shuffle(idx_k)
                np.random.shuffle(test_idx_k)
                proportions = np.random.dirichlet( #这个地方不应该叫alpha，而应该叫beta
                    np.repeat(dir_alpha, num_clients))
                # Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in
                    zip(proportions, idx_batch)]) #如果某一个client的数据分配 已经到达了平均值的话，那么就不必再分配了，但是这样仍然会不均衡
                real_proportions = proportions / proportions.sum()
                
                proportions = (np.cumsum(real_proportions) * len(idx_k)).astype(int)[:-1]
                test_proportions = (np.cumsum(real_proportions) * len(test_idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                            zip(idx_batch, np.split(idx_k, proportions))]
                test_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                            zip(test_idx_batch, np.split(test_idx_k, test_proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        #print(test_idx_batch)
        client_dict = dict()
        test_client_dict = dict()
        for cid in range(num_clients):
            np.random.shuffle(idx_batch[cid])
            np.random.shuffle(test_idx_batch[cid])
            client_dict[cid] = np.array(idx_batch[cid])
            test_client_dict[cid] = np.array(test_idx_batch[cid])

        return client_dict, test_client_dict
    
    @staticmethod
    def label_skew_quantity_based_partition(targets, test_targets, num_clients, num_classes, major_classes_num):
        """Label-skew:quantity-based partition. This has correlation with shard non-iid split

        For details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

        Args:
            targets (List or np.ndarray): Labels od dataset.
            test_targets (List or np.ndarray): Lables for test dataset
            num_clients (int): Number of clients.
            num_classes (int): Number of unique classes.
            major_classes_num (int): Number of classes for each client, should be less then ``num_classes``.

        Returns:
            dict: ``{ client_id: indices}``.
        """
        if not isinstance(targets, np.ndarray):
            targets = np.array(targets)

        idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
        test_idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
        # only for major_classes_num < num_classes.
        # if major_classes_num = num_classes, it equals to IID partition
        times = [0 for _ in range(num_classes)]
        contain = [] #存储不同client的class分配数据

        np.random.seed(1993) #make the allocatation same for same setting 
        
        assert num_clients*major_classes_num >= num_classes and (num_clients>=num_classes or num_classes%num_clients==0)
        if num_classes>num_clients:
            pre_allocate = np.random.permutation(np.arange(num_classes))
            pre_allocate = np.split(pre_allocate, num_clients)

        for cid in range(num_clients):#先分配class id
            if num_clients>=num_classes:
                current = [cid % num_classes] #保证了当client num >= class num的时候，所有的class都可以被分配至少一次
                times[cid % num_classes] += 1
                j = 1
            else:
                current = []
                j=0
                current.extend(pre_allocate[cid].tolist())
                for c in pre_allocate[cid]:
                    times[c] += 1
                    j+=1
        
            while j < major_classes_num:
                ind = np.random.randint(num_classes)
                if ind not in current:
                    j += 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        #print(times)
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            test_idx_k = np.where(test_targets==k)[0]
            np.random.shuffle(idx_k)
            np.random.shuffle(test_idx_k)
            split = np.array_split(idx_k, times[k])
            test_split = np.array_split(test_idx_k, times[k])
            ids = 0
            for cid in range(num_clients):
                if k in contain[cid]:
                    idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                    test_idx_batch[cid] = np.append(test_idx_batch[cid], test_split[ids])
                    ids += 1

        client_dict = {cid: idx_batch[cid] for cid in range(num_clients)}
        test_client_dict = {cid: test_idx_batch[cid] for cid in range(num_clients)}

        #data_num_dict = DataAllocator.cal_data_num(client_dict, test_client_dict)
        #print(data_num_dict)
        return client_dict, test_client_dict
    
    @staticmethod
    def show_partition(dict_indx:dict, class_num, save_path, targets):
        '''
        Useful when you want to check the partition work well
        Accept a dict from the above functions result
        '''
        for i in dict_indx.keys():
            idx = dict_indx[i]
            cls_len = []
            for c in range(class_num):
                cls_len.append(len(list(filter(lambda x: targets[x]==c, idx))))
            dict_indx[i] = cls_len

        data = pd.DataFrame.from_dict(dict_indx, orient='index',columns=[f'class {i}' for i in range(class_num)])
        #columns=['client']+[f'class {i}' for i in range(class_num)]
        data.plot.barh(stacked=True)
        plt.xlabel('sample num')
        plt.ylabel('client')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(save_path)
    
    @staticmethod
    def cal_data_num(train_dict, test_dict, targets, test_targets):
        data_num = {'train_dict':None, 'test_dict':None}
        for name, d in {'train_dict':train_dict, 'test_dict':test_dict}.items():
            temp_dct = {}
            for k,v in d.items():
                if name == 'train_dict':
                    temp_dct[k] = {f'class {i}':sum(targets[v]==i) for i in np.unique(targets[v])}
                else:
                    temp_dct[k] = {f'class {i}':sum(test_targets[v]==i) for i in np.unique(test_targets[v])}
            data_num[name] = temp_dct
        return data_num
    
def select(y, low_range, high_range):
        '''
        返回 y 的指定范围 (low_range, high_range) 中的数据
        '''
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return idxes

##test
if __name__ == '__main__':
    class_num = 10
    targets = np.arange(class_num).repeat(5000) #模拟数据集10个类，每类有500个样本
    test_targets = np.arange(class_num).repeat(1000)
    