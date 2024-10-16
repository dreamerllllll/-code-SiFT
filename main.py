import os
import numpy as np
from utils.config import Config
from utils.logger import Logger
from utils.factory import get_server
import torch
import copy

def set_random(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

if __name__ == "__main__":
    torch.set_num_threads(2)

    args = Config()
    logger = Logger(args)
    seed_list = copy.deepcopy(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for seed in seed_list:
        tmp_args = copy.deepcopy(args)
        tmp_args.seed = seed
        set_random(tmp_args.seed)

        logger.init_visual_log(tmp_args)

        tmp_args.print_config(logger)
        
        server = get_server(tmp_args, logger, dev)
        server.get_clientsGroup()
        server.get_allClientsTestData()
        server.roundTrainAndEval()

        logger.end_visual_log()
        