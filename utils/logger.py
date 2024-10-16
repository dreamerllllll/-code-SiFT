import logging
from tensorboardX import SummaryWriter
import os
import wandb
import datetime
import json
import numpy as np

class Logger(object):
    '''
    整合了用于文本记录的logging模块的logger以及用于数据图像记录的logger，使用tensor board来实现
    '''
    def __init__(self, args) -> None:
        self.args = args
        if args.log_level=='info':
            self.log_level = logging.INFO
        elif args.log_level=='debug':
            self.log_level = logging.DEBUG
        else:
            raise ValueError('not support')

        self._textlogger = logging.getLogger() # 使用root logger
        self._textlogger.setLevel(self.log_level)
        self._data_json_file = os.path.join(args.log_dir, 'data.json')

        format = '%(asctime)s => %(message)s'
        formatter = logging.Formatter(format)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(filename=os.path.join(args.log_dir, 'train.log'), mode='a')
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        
        self._textlogger.addHandler(console_handler)
        self._textlogger.addHandler(file_handler)
        self._textlogger.propagate = False
        self._textlogger.info('text Logger is ready')

        self._logger_type = None
        self._tblog = None

    def info(self, msg):
        self._textlogger.info(msg=msg)
    
    def error(self, msg, **kwargs):
        self._textlogger.error(msg=msg, **kwargs)
    
    def debug(self, msg):
        self._textlogger.debug(msg=msg)
    
    def warning(self, msg):
        self._textlogger.warning(msg=msg)

    def init_visual_log(self, args, log_name=None):
        # prepare visualize log
        self._logger_type = args.log_type
        if 'tensorboard' in self._logger_type and self._tblog == None:
            self._tblog = SummaryWriter(os.path.join(args.log_dir, 'tb'))
            self.info('applying tensorboard as visual log')
        if 'wandb' in self._logger_type:
            os.environ['WANDB_DIR'] = args.log_dir
            name = log_name if log_name is not None else str(datetime.datetime.now().strftime('%Y%m%d%H%M%S')+f'_seed({args.seed})')
            wandb.init(
                project=args.project_name,
                name=name,
                config=args.get_parameters_dict(),
            )
            self.info('applying wandb as visual log')
        if 'none' in self._logger_type:
            self.info('applying nothing as visual log')
        elif 'tensorboard' not in self._logger_type and 'wandb' not in self._logger_type:
            raise ValueError('unknown logger_type: {}'.format(self._logger_type))

    def visual_logging(self, phase:str, client:str, msg_dict:dict, step:int):
        '''
        phase = 'train', 'valid', 'test'
        '''
        if 'wandb' in self._logger_type:
            wandb.log(msg_dict)
        if 'tensorboard' in self._logger_type:
            if phase == 'train':
                for key, value in msg_dict.items():
                    self._tblog.add_scalar('seed{}_{}/{}/{}'.format(self.args.seed, phase, client, key), value, step) # 名字可以是分级的，这样在相应的生成中也是分级的，每一个二级目录对应于一个card
            elif phase == 'test':
                for key, value in msg_dict.items():
                    self._tblog.add_scalar('seed{}_{}/{}'.format(self.args.seed, phase, key), value, step)
        if 'none' in self._logger_type:
            pass

    def end_visual_log(self):
        if 'wandb' in self._logger_type:
            wandb.finish()
        if 'tensorboard' in self._logger_type:
            self._tblog.close()
        if 'none' in self._logger_type:
            pass

        ## log the time delta
        now = datetime.datetime.now().timestamp()
        self.info('The time used: {:.3f}s = {:.3f}h'.format(now-self.args.start_timestamp, (now-self.args.start_timestamp)/3600))
        
    def dumps_dict(self, dict_inf, path=None):
        '''
        将dict转换为string并写入文件中
        '''
        if path == None:
            path= self._data_json_file
        for k,v in dict_inf.items():
            if isinstance(v, np.float32):
                dict_inf[k] = np.float64(v)
        with open(path, 'a') as f: 
            d = json.dumps(dict_inf)
            f.write(d)
            # f.write(str(dict_inf))
            f.write('\n')
