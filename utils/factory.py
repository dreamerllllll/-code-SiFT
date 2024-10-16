from methods.sift import SiFT

from utils.myData import MnistData, Cifar100, Cifar10, CifarSubset


def get_learner(args, logger, client):
    method = args.method.lower()
    if method == 'sift':
        return SiFT(args, client, logger)
    else:
        raise NotImplemented

def getData(args, datasetName):
    name = datasetName.lower()
    if name == 'mnist':
        return MnistData(args)
    elif name == 'cifar100':
        return Cifar100(args, args.root_path)
    elif name == 'cifar10':
        return Cifar10(args, args.root_path)
    elif name == 'cifarsubset':
        return CifarSubset(args, args.root_path, cnum=args.cnum)
    elif name == 'ham10k':
        from utils.myData import HAM10K
        return HAM10K(args, args.root_path)
    elif name == 'organcmnist':
        from utils.myData import OrganCMnist
        return OrganCMnist(args, args.root_path)
    elif name == 'organsmnist':
        from utils.myData import OrganSMnist
        return OrganSMnist(args, args.root_path)
    else:
        raise NotImplemented

def get_server(args, logger, dev):
    if args.server == 'ark_server':
        from server.ark_server import Ark_Server
        return Ark_Server(args, logger, dev)
    else:
        raise ValueError