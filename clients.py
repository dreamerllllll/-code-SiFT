import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.tools import SimpleDataset
import utils.factory as factory
from utils.data_allocator import DataAllocator

class Client(object):
    def __init__(self, args, logger, trainDataSet, testDataSet, class_list, name, dev):
        self.class_list = class_list
        self.dev = dev
        self.name = name
        self.args = args
        self.logger = logger

        self.use_path = False
        self.trainData = None # 原始训练数据, 未经过SimpleDataset封装
        self.trainTarget = None
        self.train_tfs = None
        self.test_tfs = None
        self.own_testDataSet = None # 只包含当前client的类别的测试数据

        self.train_dataset = trainDataSet
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
        self.test_dataset = testDataSet
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

        self.learner = factory.get_learner(args, logger, self) # 负责训练的learner
        self.history_epoch = 0 # 代表总的训练轮次，方便画图像

    def get_clientName(self):
        return self.name

    def get_client_class_num(self):
        return len(self.class_list)

    def prepare_localUpdate(self, round_id, pos):
        self.learner._round_id = round_id + 1
        self.learner._pos_current_round = pos + 1

    def localUpdate(self, Net):
        # 进行本地更新
        self.learner.prepare_model(Net)
        Net = self.learner.client_train(Net)
        self.learner.client_eval(Net)
        Net = self.learner.after_train(Net)
        return Net
    
    def pushOldKnowledge(self): # 上传当前client的训练信息
        oldKnowledge = self.learner.client_upload()
        return oldKnowledge

    def pullOldKnowledge(self, oldKnowledge): # 下载之前client的训练信息
        self.learner.client_download(oldKnowledge)

    def local_val(self):
        pass

    def receive(self, obj):
        '''
        Receive something from Server and pass it to learner
        '''
        self.learner.receive(obj)
    
    def get_random_memory(self, size_per_class):
        '''
        Get some random samples from self.trainData as memory
        
        Args:
            size_per_class: the size of samples as memory from each class
        '''
        data, targets = [], []
        for c in self.class_list:
            indx = np.random.permutation(np.where(self.trainTarget==c)[0])
            assert len(indx)>=size_per_class
            data.extend(self.trainData[indx[:size_per_class]])
            targets.extend(self.trainTarget[indx[:size_per_class]])
        return np.array(data), np.array(targets)
        

class ClientsGroup(object):
    def __init__(self, args, logger, dataSetName, allocate_type, numOfClients, dev):
        self.dataset_name = dataSetName
        self.allocate_type = allocate_type
        self.num_of_clients = numOfClients
        self.dev = dev
        self.args = args
        self.logger = logger
        self.total_class_num = None

        data = factory.getData(self.args, self.dataset_name) # 得到包含train_data，test_data的对象
        self.total_class_num = data.total_class_num

        self.train_data = data.train_data
        self.train_targets = data.train_targets
        self.test_data = data.test_data
        self.test_targets = data.test_targets

        self.class_order = data.class_order
        self.class_name = np.array(data.class_name)
        # self.logger.info('class order is {}'.format(self.class_order))
        self.train_targets = _map_new_class_index(self.train_targets, self.class_order)
        self.test_targets = _map_new_class_index(self.test_targets, self.class_order)
        self.class_name = self.class_name[self.class_order]

        self.use_path = data.use_path

        self.train_tfs = transforms.Compose([*data.train_tfs, *data.common_tfs])
        self.test_tfs = transforms.Compose([*data.test_tfs, *data.common_tfs])

        self.clients_set = {}

    def dataSetBalanceAllocation(self):
        '''
        给每个client分配训练数据和测试数据
        '''

        # 对数据进行划分 划分方法: 按照target排好序(由SourceData来提供保证)  iid:全部打乱，然后进行划分  noverlap:直接进行划分，类别不重复 poverlap:类别部分重复 
        allocator = DataAllocator(self.args, self.train_targets, self.test_targets, self.total_class_num)
        train_slice_dct, test_slice_dct = allocator.getAllocDict()

        incre_test_data = []
        incre_test_targets = []
        client2classlist = {}

        for i in range(self.num_of_clients):
            client_train_data, client_train_targets = self.train_data[train_slice_dct[i]], self.train_targets[train_slice_dct[i]]
            client_test_data, client_test_targets = self.test_data[test_slice_dct[i]], self.test_targets[test_slice_dct[i]]
            incre_test_data.append(client_test_data)
            incre_test_targets.append(client_test_targets)

            client_class_list = np.unique(client_train_targets)
            name = 'client({})'.format(i)

            own_testDataSet = None
            if self.args.is_inctest is None or self.args.is_inctest == False:
                own_testDataSet = None
                testDataSet = SimpleDataset(client_test_data, client_test_targets, self.test_tfs, self.use_path)
            else:
                own_testDataSet = SimpleDataset(client_test_data, client_test_targets, self.test_tfs, self.use_path)
                testDataSet = SimpleDataset(np.concatenate(incre_test_data), np.concatenate(incre_test_targets), self.test_tfs, self.use_path)
            one_client = Client(
                self.args, 
                self.logger, 
                SimpleDataset(client_train_data, client_train_targets, self.train_tfs, self.use_path), 
                testDataSet,
                client_class_list, 
                name, 
                self.dev
            )
            one_client.use_path = self.use_path
            one_client.trainData = client_train_data
            one_client.trainTarget = client_train_targets
            one_client.train_tfs = self.train_tfs
            one_client.test_tfs = self.test_tfs
            one_client.own_testDataSet = own_testDataSet if own_testDataSet else testDataSet
            self.clients_set[name] = one_client

            client2classlist[i] = list(client_class_list)
        setattr(self.args, 'client2classlist', client2classlist) #log this public information

# 按指定的order重新分配label
def _map_new_class_index(label, order):
    return np.array(list(map(lambda x: order.index(x), label)))


if __name__=="__main__":
    MyClients = ClientsGroup('mnist', False, 100, 1)
    print(MyClients.clients_set['client(10)'].train_ds[:])
    print(MyClients.clients_set['client(11)'].train_ds[400:500])
    
