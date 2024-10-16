import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from model.incrementNet import IncrementNet, IL2ANet, DERNet, AdaptiveNet, SplitNet, GrowNet, CosNormNet
from clients import ClientsGroup
from utils.tools import set_mode
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class Serial_Server(object):
    def __init__(self, args, logger, dev) -> None:
        self.args = args
        self.logger = logger
        self.dev = dev

        self.net:IncrementNet = self.get_initNet()

        self.oldKnowledge = {}

        self.myClients = None # 所有clients构成的set

        self.all_testData = []
        self.all_testLoader = None

        self.cnum_in_comm = int(max(args.num_of_clients * args.cfraction, 1))
        self.roundAcc = []
        
        self._final_train_accs = [] #当前round 所有client的最终train acc
        self._previous_class_lists = []
        self.class_names = None #the class_name with repespect to class index
    
    def get_initNet(self):
        if self.args.net_type == 'semantic':
            from model.incrementalExtend import SemanticNet
            net = SemanticNet(self.args, self.logger, self.args.model_name, self.dev)
        net = net.to(self.dev)
        
        return net

    def get_clientsNum(self):
        return self.args.num_of_clients

    def get_clientsGroup(self):
        self.myClients = ClientsGroup(self.args, self.logger, \
            self.args.dataset, self.args.allocate_type, self.args.num_of_clients, self.dev)
        self.myClients.dataSetBalanceAllocation()
        
        if getattr(self.args, 'full_classifier', False):
            self.logger.info('Set the full classifier before train')
            self.total_class_num = self.myClients.total_class_num
            self.net.update_fc(range(self.total_class_num))
        self.class_names = self.myClients.class_name
        self.logger.info('Get the class names:{}'.format(self.class_names))
        self.args.class_names = self.class_names
    
    def get_allClientsTestData(self):
        if self.args.is_inctest is None or self.args.is_inctest == False:
            for id in range(self.get_clientsNum()):
                client_name = f'client({id})'
                client_test_data = self.myClients.clients_set[client_name].test_dataset
                self.all_testData.append(client_test_data)
            self.all_testData = ConcatDataset(self.all_testData)
        else:
            client_name = f'client({self.get_clientsNum()-1})'
            self.all_testData = self.myClients.clients_set[client_name].test_dataset
        self.all_testLoader = DataLoader(self.all_testData, batch_size=128, shuffle=False, num_workers=self.args.num_workers)
        self.logger.info('get all clients test data in order: {}'.format(len(self.all_testData)))

    def roundTrainAndEval(self):
        for round_id in range(self.args.num_comm):
            self.logger.info('='*20+" communicate round {} ".format(round_id+1)+'='*20)

            client_order = self.sample_clients()
            
            clients_in_comm = ['client({})'.format(i) for i in client_order]
            self.logger.info('client in this communication: '+str(client_order))

            for pos, one_client in enumerate(clients_in_comm):
                if round_id == 1:
                    self.myClients.clients_set[one_client].test_dataloader = self.all_testLoader

                self.myClients.clients_set[one_client].prepare_localUpdate(round_id, pos)

                self.preprocess_knowledge(round_id, pos)
                self.myClients.clients_set[one_client].pullOldKnowledge(self.oldKnowledge)

                self.logger.info('update {} now...'.format(self.myClients.clients_set[one_client].get_clientName()))
                self.net = self.myClients.clients_set[one_client].localUpdate(self.net)
            
                self.oldKnowledge = self.myClients.clients_set[one_client].pushOldKnowledge()
                self.process_knowledge(round_id, pos)

            if (round_id + 1) % self.args.val_freq == 0:
                self.evaluate_round(round_id)
                self.logger.info('Train acc Matrix: \n{}'.format(self._final_train_accs))

    def preprocess_knowledge(self, round_id, pos):
        '''
        对knowledge进行预处理，预处理之后再发送出去
        '''
        pass
        
    def process_knowledge(self, round_id, pos):
        '''
        对old knowledge进行处理
        ''' 
        if pos == 0: 
            self._final_train_accs.append([])

        if 'final_train_acc' in self.oldKnowledge:
            self._final_train_accs[-1].append(self.oldKnowledge['final_train_acc'])
        
        if 'class_list' in self.oldKnowledge:
            self._previous_class_lists.append(self.oldKnowledge['class_list'])
            if len(self._previous_class_lists) > self.get_clientsNum():
                self._previous_class_lists.pop(0)
            #self.oldKnowledge.pop('class_list')
            self.oldKnowledge['previous_class_lists'] = self._previous_class_lists

    def sample_clients(self):
        '''
        在一个round之前，采样client，返回list
        '''
        if not hasattr(self, 'sample_generator'):
            self.sample_generator = np.random.default_rng(seed=1993) #可以让采样过程具备稳定性

        if self.cnum_in_comm == self.get_clientsNum():
            if getattr(self.args, 'client_shuffle', False)==True:
                client_order = self.sample_generator.permutation(self.get_clientsNum())
            elif getattr(self.args, 'client_sample', '') == 'train_acc_lower' and len(self._final_train_accs) != 0:#对于train acc较小的client将其排在前面
                client_order = np.argsort(self._final_train_accs[-1])
            else:
                client_order = np.arange(self.get_clientsNum())
        else:
            client_order = self.sample_generator.permutation(self.get_clientsNum())
        return client_order[0:self.cnum_in_comm]
    
    @set_mode('net.state', 'evaluate_round', 'normal')
    def evaluate_round(self, round_id):
        '''
        在round结束之后，评估全局模型
        '''
        # 指定数量的comm_round结束后进行一次测试，需使用所有类别的数据测试
        all_preds, all_targets = [], [] ; all_scores = []
        self.net.eval()
        with torch.no_grad():
            sum_accu = 0
            train_num = 0
            for data, targets in self.all_testLoader:
                data, targets = data.to(self.dev), targets.to(self.dev)
                output, _ = self.net(data)
                preds = torch.argmax(output, dim=1)
                sum_accu += (preds == targets).cpu().data.numpy().sum() # 这里虽然sum_accu可能是在gpu上，也可能为parameters，但是可以正常进行计算，且在format输出的时候也可以正常输出
                train_num += len(targets)
                all_preds.append(preds.data.cpu())
                all_targets.append(targets.cpu())
                all_scores.append(torch.softmax(output.data.cpu(), dim=1))
            self.logger.info('communication round: {}/{} ==> '.format(round_id+1, self.args.num_comm)+'accuracy: {:.4f}'.format(sum_accu / train_num))
            self.roundAcc.append(sum_accu / train_num)
            self.logger.info('round acc: {}'.format(self.roundAcc))
            verbose = []
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            all_scores = torch.cat(all_scores)
            class_indx = torch.unique(all_targets)
            all_preds = all_preds.numpy()
            all_targets = all_targets.numpy()
            all_scores = all_scores.numpy()

            client_cnum = int(self.myClients.total_class_num / self.myClients.num_of_clients)
            if client_cnum <= 0:
                client_cnum = 1
            for i in range(0, len(class_indx), client_cnum):
                indx = np.where(np.logical_and(all_targets >= i, all_targets < i + client_cnum))[0]
                label = '{}-{}'.format(str(i).rjust(2, '0'), str(i+client_cnum-1).rjust(2, '0'))
                verbose.append(label)
                verbose.append(np.around((all_targets[indx] == all_preds[indx]).sum()*100 / len(indx), decimals=4))
            self.logger.info('verbose acc: {}'.format(verbose))

            self.vverbose_evaluate(round_id, all_targets, all_preds, all_scores)
            
    def vverbose_evaluate(self, round_id, all_targets, all_preds, all_scores, additional_dict={}):
        if getattr(self.args, 'vverbose_round_acc', False): #this is important for class imbalance situation
            self.logger.info('Getting the detailed round acc...')
            vverbose = {'type':'vverbose_round_acc', 'round':round_id}
            vverbose.update(additional_dict)
            vverbose['confusion_matrix'] = confusion_matrix(all_targets, all_preds).tolist()
            vverbose['balance_acc'] = balanced_accuracy_score(all_targets, all_preds)
            self.logger.dumps_dict(vverbose)

    def broadcast_send(self, obj, client_list:list):
        '''
        模拟广播通信，将信息发送给各个client
        '''
        for client in client_list:
            self.myClients.clients_set[client].receive(obj)