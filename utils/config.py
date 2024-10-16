import argparse
from utils.tools import check_dir
import yaml
import os
import datetime
import copy

class Config(object):
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
        parser.add_argument('-cff', '--config_f', type=str, default=None, help='config file')

        # self.project_name = 'FedIncLearn-'+str(datetime.date.today())
        self.project_name = 'FedIncLearnPro'

        parser.add_argument('--log_type', type=str, default=None, help='visualize logger type, e.g. tensorboard, wandb, none')

        parser.add_argument('-g', '--gpu', type=str, default=None, help='gpu id to use(e.g. 0,1,2,3)')
        parser.add_argument('-lp', '--log_dir', type=str, default=None, help='the saving root dir of logs, including the text logs, pic logs and code or model stat_dict saved')
        parser.add_argument('-ll', '--log_level', type=str, default=None, choices=['info', 'debug'], help='the log level to output')
        parser.add_argument('-sd','--seed', type=int, default=None, help='the seed for all random process')
        parser.add_argument('-pre', '--prefix', type=str, default=None, help='the prefix about the log name')

        # 方法参数
        parser.add_argument('-sv', '--server', type=str, default=None, help='which kind of server you want use')
        parser.add_argument('-m', '--method', type=str, default=None, help='the startegy to use for client train')
        parser.add_argument('-mn', '--model_name', type=str, default=None, help='the backbone to train')
        parser.add_argument('-ng', '--num_groups', type=int, default=None, help='the num of groups for gn')
        parser.add_argument('-nt', '--net_type', type=str, default=None, help='the increment net for different method')
        parser.add_argument('-ncomm', '--num_comm', type=int, default=None, help='number of communications')
        parser.add_argument('-nc', '--num_of_clients', type=int, default=None, help='numer of the clients')
        parser.add_argument('-cf', '--cfraction', type=float, default=None, help='C fraction, 0 means 1 client, 1 means total clients')
        
        # 数据相关
        parser.add_argument('-ds', '--dataset', type=str, default=None, help='which dataset to use')
        parser.add_argument('-is', '--image_size', type=int, default=None, help='the image size')
        parser.add_argument('-at', '--allocate_type', type=str, default=None, choices=['iid', 'noverlap', 'poverlap'],help='the way to allocate the client')
        parser.add_argument('-bc', '--base_class', type=int, default=None, help='use to valid incremental method')
        parser.add_argument('-rp', '--root_path', type=str, default=None, help='the dataset path to load or download')
        parser.add_argument('--fixed_memory', type=bool, default=None, help='whether the memory space is fixed')
        parser.add_argument('--memory_per_class', type=int, default=None, help='memory num per class')

        # 训练相关
        parser.add_argument('--client_init_epoch', type=int, default=None, help='first client of first round train epoch')
        parser.add_argument('--client_init_lr', type=float, default=None, help="first client of first round learning rate, \
                            use value from origin paper as default")
        parser.add_argument('--client_init_weight_decay', type=float, default=None, help='first client of the first round weight decay use for sgd')
        parser.add_argument('--client_init_milestones', type=int, nargs=None, default=None, help='first client of the first round milestones, only useful when scheduler is multi_step')

        parser.add_argument('--round_init_epoch', type=int, default=None, help='first round train epoch')
        parser.add_argument('--round_init_lr', type=float, default=None, help="first round learning rate, \
                            use value from origin paper as default")
        parser.add_argument('--round_init_weight_decay', type=float, default=None, help='the first round weight decay use for sgd')
        parser.add_argument('--round_init_milestones', type=int, nargs=None, default=None, help='the first round milestones, only useful when scheduler is multi_step')
        parser.add_argument('--round_init_momentum', type=float, nargs=None, default=None, help='first round momentum')

        parser.add_argument('-E', '--epoch', type=int, default=None, help='local train epoch')
        parser.add_argument('-B', '--batchsize', type=int, default=None, help='local train batch size')
        parser.add_argument('-nw', '--num_workers', type=int, default=None, help='local train num_workers')
        parser.add_argument('-eb', '--eval_batchsize', type=int, default=None, help='the batchsize for evaluating')
        parser.add_argument('-lr', '--lr', type=float, default=None, help="learning rate, \
                            use value from origin paper as default")
        parser.add_argument('-ot', '--opt_type', type=str, default=None, choices=['sgd', 'adam'], help='the optimizer to use')
        parser.add_argument('-mo', '--momentum', type=float, default=None, help='the momentum use for sgd')
        parser.add_argument('-wd', '--weight_decay', type=float, default=None, help='the weight decay use for sgd')
        parser.add_argument('-sc', '--scheduler', type=str, default=None, choices=['multi_step', 'cos'], help='which scheduler to use')
        parser.add_argument('-lrd', '--lrate_decay', type=float, default=None, help='the learning rate decay rate, only useful when scheduler is multi_step')
        parser.add_argument('-ms', '--milestones', type=int, nargs=None, default=None, help='the milestones, only useful when scheduler is multi_step')
        parser.add_argument('-vf', '--val_freq', type=int, default=None, help="model validation frequency(of communications)")
        parser.add_argument('-lv', '--local_val_freq', type=int, default=None, help='the local validation frequence')
        parser.add_argument('-sf', '--save_freq', type=int, default=None, help='global model save frequency(of communication)')

        parser.add_argument('--is_inctest', type=bool, default=None, help='client test_data include old data?')
        parser.add_argument('--round_milestones', type=int, nargs=None, default=None, help='the round milestones, in some communication round decay lr')

        # 具体方法相关
        parser.add_argument('-T', '--temperature', type=float, default=None, help='the temperature to use')
        parser.add_argument('--verbose_acc', type=bool,default=None, help='verbose test acc')
        
        args = parser.parse_args()
        args = args.__dict__
        
        for name, value in args.items():
            setattr(self, name, value)
        
        # 引入config文件参数
        if self.config_f != None:
            with open(self.config_f, encoding='UTF-8') as config_f:
                temp_args = yaml.load(config_f, Loader=yaml.FullLoader)
                config_args = {}
                config_args.update(temp_args['global'])
                config_args.update(temp_args['data'])
                config_args.update(temp_args['train'])
            for key, val in config_args.items():
                if getattr(self, key, None) == None: #option文件中可以有上面没有定义的参数
                    setattr(self, key, val)

        # 对部分参数进行修正，比如self.log_path
        dataset_options = self.dataset+'_'+self.allocate_type+'_'+'nc'+str(self.num_of_clients)
        train_options = '' if self.prefix is None else self.prefix+'_'
        train_options = train_options+'cf'+str(self.cfraction)+'_'+'ncom'+str(self.num_comm)+'_'+\
                        'E'+str(self.epoch)+'_'+'B'+str(self.batchsize)+'_'+'lr'+str(self.lr)
        
        
        if getattr(self, 'second_dir', ''):
            log_dir = '{}/{}/{}/{}/{}/{}'.format(self.log_dir, self.method, dataset_options, self.model_name, self.second_dir, train_options)
        else:
            log_dir = '{}/{}/{}/{}/{}'.format(self.log_dir, self.method, dataset_options, self.model_name, train_options)
        log_dir = log_dir+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S') #统一加上时间，便于查看
        check_dir(log_dir)
        self.log_dir = log_dir
        self.repair_arg()

        self.pid = os.getpid()
        self.start_timestamp = datetime.datetime.now().timestamp()

    def repair_arg(self):
        '''
        对部分参数进行一些动态的设置和修正
        '''
        if self.round_init_epoch == None:
            self.round_init_epoch = self.epoch
        if self.round_init_lr == None:
            self.round_init_lr = self.lr
        if self.round_init_weight_decay == None:
            self.round_init_weight_decay = self.weight_decay
        if self.round_init_milestones == None:
            self.round_init_milestones = self.milestones
        if self.round_init_momentum == None:
            self.round_init_momentum = self.momentum
        
        if self.client_init_epoch == None:
            self.client_init_epoch = self.round_init_epoch
        if self.client_init_lr == None:
            self.client_init_lr = self.round_init_lr
        if self.client_init_weight_decay == None:
            self.client_init_weight_decay = self.round_init_weight_decay
        if self.client_init_milestones == None:
            self.client_init_milestones = self.round_init_milestones

            
    def update(self, update_dict:dict):
        for key, val in update_dict.items():
            setattr(self, key, val)
    
    def get_parameters_dict(self):
        parameter_dict = copy.deepcopy(vars(self))
        for name, value in vars(self).items():
            if value == None:
                parameter_dict.pop(name)
        return parameter_dict
    
    def print_config(self, logger) -> None:
        logger.info(20*"-")
        logger.info("log hyperparameters in seed {}".format(self.seed))
        logger.info(20*"-")
        for name, value in vars(self).items():
            if not getattr(self, name) == None:
                logger.info('{}: {}'.format(name, value))
        logger.info(20*"-")
        # logger.info('Inital configs overwrited by yaml config file: {}'.format(self.init_overwrite_names))
        # logger.info('Inital configs overwrited by loaded pkl configs: {}'.format(self.load_overwrite_names))
    
    