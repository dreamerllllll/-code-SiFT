from .serial_server import Serial_Server
import copy
import numpy as np
import torch
import torch.nn.functional as F

class Ark_Server(Serial_Server):
    '''
    Using the global model to accumulate the knowledge
    '''
    def __init__(self, args, logger, dev) -> None:
        super().__init__(args, logger, dev)
        self.global_model = None
        self.global_roundAcc = []
        self.global_matrial = [] #avg the matrials to create global

        self.ark_start_round = getattr(self.args, 'ark_start_round', 0)
        self.ark_start_pos = getattr(self.args, 'ark_start_pos', 1)

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
                
                #### Ark the global model
                if getattr(self.args, 'ark_global', False) and round_id>=self.ark_start_round and pos>=self.ark_start_pos:
                    self.ark_start_pos = -1
                    self.ark_global_model()

                self.preprocess_knowledge(round_id, pos)
                self.myClients.clients_set[one_client].pullOldKnowledge(self.oldKnowledge)
                
                self.logger.info('update {} now...'.format(self.myClients.clients_set[one_client].get_clientName()))
                self.net = self.myClients.clients_set[one_client].localUpdate(self.net)

                self.oldKnowledge = self.myClients.clients_set[one_client].pushOldKnowledge()
                self.process_knowledge(round_id, pos)
        
            if (round_id + 1) % self.args.val_freq == 0:
                self.evaluate_round(round_id, self.net, 'net')
                if self.global_model is not None:
                    self.evaluate_round(round_id, self.global_model, 'global_net')

    def ark_global_model(self):
        if self.global_model is None:
            self.global_model = copy.deepcopy(self.net)
            self.logger.info('Init the global model')
        else:
            assert hasattr(self.args, 'ark_decay')
            decay = self.args.ark_decay #e.g. 0.8
            params = self.net.get_flatten_param_wbn()
            global_params = self.global_model.get_flatten_param_wbn()
            global_params = decay*global_params + (1-decay)*params
            self.global_model.load_flatten_param_wbn(global_params)
            self.logger.info('Ark the global model')
    
    def evaluate_round(self, round_id, net, type):
        '''
        在round结束之后，评估全局模型
        '''
        # 指定数量的comm_round结束后进行一次测试，需使用所有类别的数据测试
        all_preds, all_targets = [], [] ; all_scores = []
        net.eval()
        with torch.no_grad():
            sum_accu = 0
            train_num = 0
            for data, targets in self.all_testLoader:
                data, targets = data.to(self.dev), targets.to(self.dev)
                output, _ = net(data)
                preds = torch.argmax(output, dim=1)
                sum_accu += (preds == targets).cpu().data.numpy().sum()
                train_num += len(targets)
                all_preds.append(preds.data.cpu())
                all_targets.append(targets.cpu())
                all_scores.append(torch.softmax(output.data.cpu(), dim=1))
            self.logger.info('communication round: {}/{} ==> '.format(round_id+1, self.args.num_comm)+'accuracy: {:.4f}'.format(sum_accu / train_num))
            
            if type=='net':
                self.roundAcc.append(sum_accu / train_num)
                self.logger.info('round acc: {}'.format(self.roundAcc))
            elif type == 'global_net':
                self.global_roundAcc.append(sum_accu / train_num)
                self.logger.info('global round acc: {}'.format(self.global_roundAcc))
            else:
                raise NotImplemented
            
            verbose = []
            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            all_scores = torch.cat(all_scores)
            class_indx = torch.unique(all_targets)
            all_preds = all_preds.numpy()
            all_targets = all_targets.numpy()
            all_scores = all_scores.numpy()

            ## not support the stupid test

            client_cnum = int(self.myClients.total_class_num / self.myClients.num_of_clients)
            if client_cnum <= 0:
                client_cnum = 1

            for i in range(0, len(class_indx), client_cnum):
                indx = np.where(np.logical_and(all_targets >= i, all_targets < i + client_cnum))[0]
                label = '{}-{}'.format(str(i).rjust(2, '0'), str(i+client_cnum-1).rjust(2, '0'))
                verbose.append(label)
                verbose.append(np.around((all_targets[indx] == all_preds[indx]).sum()*100 / len(indx), decimals=4))
            self.logger.info('verbose acc: {}'.format(verbose))

            self.vverbose_evaluate(round_id, all_targets, all_preds, all_scores, additional_dict={'model_type':type})
    
    def preprocess_knowledge(self, round_id, pos):
        '''
        Add the global model, if not none
        '''
        super().preprocess_knowledge(round_id, pos)
        if self.global_model is not None:
            self.oldKnowledge['global_model'] = self.global_model

    
        