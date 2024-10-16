import numpy as np
import gzip
import os
import abc
import torchvision.transforms as tfs
import torchvision.datasets as datasets
import torch
import json
from tqdm import *
import logging
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import medmnist
import PIL.Image as Image

class SourceData(abc.ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.name = None
        self.train_data = None
        self.train_targets = None
        self.train_data_size = None
        self.test_data = None
        self.test_targets = None
        self.test_data_size = None
        
        self.use_path = None
        self.class_order = None
        self.image_size = args.image_size

        self.train_tfs = []
        self.test_tfs = []
        self.common_tfs = []

        self.get_data()
        self.sort_data()
        self.set_transform()

    @abc.abstractclassmethod
    def get_data(self):
        '''
        To get the data
        Fullfill the self.train_data, self.train_targets, self.test_data, self.test_targets, 
        self.train_data_size, self.test_data_size 
        '''
        pass
    
    @abc.abstractclassmethod
    def sort_data(self):
        '''
        Sort the data according to the class
        '''
        pass
    
    @abc.abstractclassmethod
    def set_transform(self):
        '''
        set the transforms
        Fullfill the self.train_tfs,self.test_tfs, self.common_tfs
        '''
        pass

    @property
    def total_class_num(self):
        return len(self.class_order)
    
    def show_class_distribution(self):
        ##calculate the class distribution info
        def cal_class_distribution(targets):
            for c in np.unique(targets):
                logging.info(f'{c}: {(targets==c).sum()} => {(targets==c).sum() / len(targets)}')
        logging.info('=====train dataset')
        cal_class_distribution(self.train_targets)
        logging.info('=====test dataset')
        cal_class_distribution(self.test_targets)
        logging.info('=====overall')
        logging.info(f'train dataset: {len(self.train_targets)}=>{len(self.train_targets)/(len(self.train_targets)+len(self.test_targets))}')
        logging.info(f'test dataset: {len(self.test_targets)}=>{len(self.test_targets)/(len(self.train_targets)+len(self.test_targets))}')
    

class Cifar10(SourceData):
    def __init__(self, args, root_dir, download=False) -> None:
        self.root_dir = root_dir
        self.download = download
        super().__init__(args)
        
        self.use_path = False
        self.class_order = list(range(10))
        # self.class_order = list([3, 6, 1, 9, 0, 4, 7, 5, 2, 8])
        self.class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    def get_data(self):
        train = datasets.CIFAR10(self.root_dir, True, download=self.download)
        test = datasets.CIFAR10(self.root_dir, False, download=self.download)
        self.train_data, self.train_targets = train.data, np.array(train.targets)
        self.test_data, self.test_targets = test.data, np.array(test.targets)
        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)
    
    def sort_data(self): # 将数据进行排序
        index = np.argsort(self.train_targets)
        self.train_targets = self.train_targets[index]
        self.train_data = self.train_data[index]

        index = np.argsort(self.test_targets)
        self.test_targets = self.test_targets[index]
        self.test_data = self.test_data[index]

    def set_transform(self):
        self.train_tfs = [
            tfs.ToPILImage(),
            tfs.Resize((self.image_size, self.image_size)), 
            tfs.RandomCrop(self.image_size, padding=int(self.image_size/8)), # 32, 4
            tfs.RandomHorizontalFlip(p=0.5),
            tfs.ColorJitter(brightness=63/255)
        ]
        self.test_tfs = [tfs.ToPILImage(), tfs.Resize((self.image_size, self.image_size))]
        self.common_tfs = [
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ]

class CifarSubset(SourceData):
    def __init__(self, args, root_dir, download=False, cnum=25) -> None:
        self.root_dir = root_dir
        self.download = download
        self.cnum = cnum
        super().__init__(args)
        
        self.use_path = False
        self.class_order = list(range(cnum))

    def get_data(self):
        train = datasets.CIFAR100(self.root_dir, True, download=self.download)
        test = datasets.CIFAR100(self.root_dir, False, download=self.download)
        self.train_data, self.train_targets = train.data, np.array(train.targets)
        self.test_data, self.test_targets = test.data, np.array(test.targets)

        def select(x, y, low_range, high_range):
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            return x[idxes], y[idxes]
        
        self.train_data, self.train_targets = select(self.train_data, self.train_targets, 0, self.cnum)
        self.test_data, self.test_targets = select(self.test_data, self.test_targets, 0, self.cnum)

        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)
    
    def sort_data(self): # 将数据进行排序
        index = np.argsort(self.train_targets)
        self.train_targets = self.train_targets[index]
        self.train_data = self.train_data[index]

        index = np.argsort(self.test_targets)
        self.test_targets = self.test_targets[index]
        self.test_data = self.test_data[index]

    def set_transform(self):
        self.train_tfs = [
            tfs.ToPILImage(),
            tfs.Resize((self.image_size, self.image_size)), 
            tfs.RandomCrop(self.image_size, padding=int(self.image_size/8)), # 32, 4
            tfs.RandomHorizontalFlip(),
            tfs.ColorJitter(brightness=63/255)
        ]
        self.test_tfs = [tfs.ToPILImage(),tfs.Resize((self.image_size, self.image_size))]
        self.common_tfs = [
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        ]
    
class Cifar100(SourceData):
    def __init__(self, args, root_dir, download=False) -> None:
        self.root_dir = root_dir
        self.download = download
        super().__init__(args)
        
        self.use_path = False
        self.class_order = list(range(100))
        ## class_name should be correspond with class order
        self.class_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    

    def get_data(self):
        train = datasets.CIFAR100(self.root_dir, True, download=self.download)
        test = datasets.CIFAR100(self.root_dir, False, download=self.download)
        self.train_data, self.train_targets = train.data, np.array(train.targets)
        self.test_data, self.test_targets = test.data, np.array(test.targets)
        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)
    
    def sort_data(self): # 将数据进行排序
        index = np.argsort(self.train_targets)
        self.train_targets = self.train_targets[index]
        self.train_data = self.train_data[index]

        index = np.argsort(self.test_targets)
        self.test_targets = self.test_targets[index]
        self.test_data = self.test_data[index]

    def set_transform(self):
        transform_t = getattr(self.args, 'transform_type', None)
        if transform_t is None: #default
            logging.info('use the default transform')
            self.train_tfs = [
                tfs.ToPILImage(),
                tfs.Resize((self.image_size, self.image_size)), 
                tfs.RandomCrop(self.image_size, padding=int(self.image_size/8)), # 32, 4
                tfs.RandomHorizontalFlip(),
                tfs.ColorJitter(brightness=63/255)
            ]
            if self.image_size>32: #center crop
                size = int((256 / 224) * self.image_size)
                self.test_tfs = [
                    tfs.ToPILImage(),
                    tfs.Resize(size, interpolation=3),
                    tfs.CenterCrop(self.image_size),
                ]
            else:
                self.test_tfs = [tfs.ToPILImage(), tfs.Resize((self.image_size, self.image_size))]
            self.common_tfs = [
                tfs.ToTensor(),
                tfs.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
            ]

class HAM10K(SourceData):
    '''
    HAM10000 dataset, also used by ISIC2018
    '''
    def __init__(self, args, root_dir) -> None:
        self.root_dir = root_dir
        self.lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular skin lesions',
        'df': 'Dermatofibroma'
        }
        
        super().__init__(args)

        self.use_path = True
        self.class_order = list(np.arange(7))
        self.class_name = [self.idx2cls[i] for i in range(7)]
    
    def get_data(self):
        meta_csv = 'data/ISIC/HAM10000_metadata.csv'
        self.all_img_path = glob.glob(os.path.join(self.root_dir, '*.jpg'))
        self.img_id2path = {os.path.splitext(os.path.basename(i))[0]:i for i in self.all_img_path}

        df_original = pd.read_csv(meta_csv)
        df_original['path'] = df_original['image_id'].map(self.img_id2path.get)
        df_original['cell_type'] = df_original['dx'].map(self.lesion_type_dict.get)
        df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes #will encode In alphabetical order

        self.cls2idx = {n:idx for idx, n in enumerate(pd.Categorical(df_original['cell_type']).categories)}
        self.idx2cls = {v:k for k,v in self.cls2idx.items()}

        # Get number of images associated with each lesion_id
        df_undup = df_original.groupby('lesion_id').count()
        # Filter out lesion_id's that have only one image associated with it
        df_undup = df_undup[df_undup['image_id'] == 1]
        df_undup.reset_index(inplace=True)

        # Identify lesion_id's that have duplicate images and those that have only one image.
        def get_duplicates(x):
            unique_list = list(df_undup['lesion_id'])
            if x in unique_list:
                return 'unduplicated'
            else:
                return 'duplicated'
            
        # Create a new colum that is a copy of the lesion_id column
        df_original['duplicates'] = df_original['lesion_id']
        # apply the function to this new column
        df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
        # Filter out images that don't have duplicates
        df_undup = df_original[df_original['duplicates'] == 'unduplicated']
        # Create a val set using df because we are sure that none of these images have augmented duplicates in the train set
        y = df_undup['cell_type_idx']
        _, df_val = train_test_split(df_undup, test_size=0.5, random_state=101, stratify=y)

        ### the rest images construct train set
        def get_val_rows(x):
            # create a list of all the lesion_id's in the val set
            val_list = list(df_val['image_id'])
            if str(x) in val_list:
                return 'val'
            else:
                return 'train'

        # Identify train and val rows
        # Create a new colum that is a copy of the image_id column
        df_original['train_or_val'] = df_original['image_id']
        # Apply the function to this new column
        df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
        # Filter out train rows
        df_train = df_original[df_original['train_or_val'] == 'train']

        df_train = df_train.reset_index()
        df_val = df_val.reset_index()

        self.train_data = df_train['path'].values
        self.train_targets = df_train['cell_type_idx'].values

        ## argument the train_data to balance the classification
        if getattr(self.args, 'oversample_balance',False):
            np.random.seed(1993)
            class_num = [sum(self.train_targets==c) for c in np.unique(self.train_targets)]
            for c in np.unique(self.train_targets):
                idx = np.where(self.train_targets==c)[0]
                sample_num = max(class_num) - class_num[c]
                sample_idx = np.random.choice(idx, size=sample_num)
                self.train_data = np.concatenate((self.train_data, self.train_data[sample_idx]))
                self.train_targets = np.concatenate((self.train_targets, self.train_targets[sample_idx]))

        self.train_data_size = len(self.train_data)
        self.test_data = df_val['path'].values
        self.test_targets = df_val['cell_type_idx'].values
        self.test_data_size =  len(self.test_data)
        
    def sort_data(self):
        index = np.argsort(self.train_targets)
        self.train_targets = self.train_targets[index]
        self.train_data = self.train_data[index]

        index = np.argsort(self.test_targets)
        self.test_targets = self.test_targets[index]
        self.test_data = self.test_data[index]

    def set_transform(self):
        self.train_tfs = [
            tfs.RandomResizedCrop(self.image_size),
            tfs.RandomHorizontalFlip(),
            tfs.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)
        ]
        self.test_tfs = [
            tfs.Resize((self.image_size,self.image_size)) #use tuple, not int
        ]
        self.common_tfs = [
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.76303697, 0.54564005, 0.57004493], std=[0.14092775, 0.15261292, 0.16997]),
        ]

class OrganCMnist(SourceData):
    def __init__(self, args, root_dir, download=False) -> None:
        self.root_dir = root_dir
        self.download = download
        self.n_channels = 1

        super().__init__(args)
        self.use_path = False
        self.class_order = list(np.arange(11))
        self.class_name = ["bladder", "femur-left", "femur-right", "heart", "kidney-left","kidney-right","liver", "lung-left","lung-right","pancreas","spleen",]
    
    def get_data(self):
        train_d = medmnist.OrganCMNIST('train', download=self.download, root=self.root_dir, size=self.image_size)
        test_d = medmnist.OrganCMNIST('test', download=self.download, root=self.root_dir, size=self.image_size)
        self.train_data = train_d.imgs ; self.train_targets = np.squeeze(np.array(train_d.labels, dtype=np.int64))
        self.test_data = test_d.imgs ; self.test_targets = np.squeeze(np.array(test_d.labels, dtype=np.int64))
       
        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)
    
    def sort_data(self):
        index = np.argsort(self.train_targets)
        self.train_targets = self.train_targets[index]
        self.train_data = self.train_data[index]

        index = np.argsort(self.test_targets)
        self.test_targets = self.test_targets[index]
        self.test_data = self.test_data[index]

    
    def set_transform(self):
        self.train_tfs = [
            lambda i:Image.fromarray(i).convert('RGB'),
            tfs.Resize((self.image_size, self.image_size)), 
            tfs.RandomCrop(self.image_size, padding=int(self.image_size/8)), # 32, 4
            tfs.RandomHorizontalFlip(p=0.5),
            tfs.ColorJitter(brightness=63/255)
        ]
        self.test_tfs = [lambda i:Image.fromarray(i).convert('RGB'), tfs.Resize((self.image_size, self.image_size))]
        self.common_tfs = [
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.4942, 0.4942, 0.4942), std=(0.2776, 0.2776, 0.2776)), #calculated
        ]
    
class OrganSMnist(OrganCMnist):
    '''
    This is harder than OrganCMnist
    '''
    def get_data(self):
        train_d = medmnist.OrganSMNIST('train', download=self.download, root=self.root_dir, size=self.image_size)
        test_d = medmnist.OrganSMNIST('test', download=self.download, root=self.root_dir, size=self.image_size)
        self.train_data = train_d.imgs ; self.train_targets = np.squeeze(np.array(train_d.labels, dtype=np.int64))
        self.test_data = test_d.imgs ; self.test_targets = np.squeeze(np.array(test_d.labels, dtype=np.int64))
       
        self.train_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)

    def set_transform(self):
        self.train_tfs = [
            lambda i:Image.fromarray(i).convert('RGB'),
            tfs.Resize((self.image_size, self.image_size)), 
            tfs.RandomCrop(self.image_size, padding=int(self.image_size/8)), # 32, 4
            tfs.RandomHorizontalFlip(p=0.5),
            tfs.ColorJitter(brightness=63/255)
        ]
        self.test_tfs = [lambda i:Image.fromarray(i).convert('RGB'), tfs.Resize((self.image_size, self.image_size))]
        self.common_tfs = [
            tfs.ToTensor(),
            tfs.Normalize(mean=(0.4952, 0.4952, 0.4952), std=(0.2769, 0.2769, 0.2769)), #calculated
        ]
    

class MnistData(SourceData):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.use_path = False
        self.class_order = list(range(10))

    def get_data(self):
        data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_targets_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_targets_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_targets = extract_labels(train_targets_path)
        test_images = extract_images(test_images_path)
        test_targets = extract_labels(test_targets_path)

        assert train_images.shape[0] == train_targets.shape[0]
        assert test_images.shape[0] == test_targets.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0) # 逐个元素相乘
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        self.train_data = train_images
        self.train_targets = train_targets
        self.test_data = test_images
        self.test_targets = test_targets
    
    def sort_data(self): # 数据按0-9排序
        index = np.argsort(self.train_targets)
        self.train_targets = self.train_targets[index]
        self.train_data = self.train_data[index]

        index = np.argsort(self.test_targets)
        self.test_targets = self.test_targets[index]
        self.test_data = self.test_data[index]
    
    def set_transform(self):
        self.common_tfs = [torch.tensor]

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')   # 创建一个新的data type
    return np.frombuffer(bytestream.read(4), dtype=dt)[0] # np.frombuffer 可以返回一个真实的数字


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream: # gzip默认会按照字节流的方式进行读取
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1) # 这里的shape中有一个1，但是之后会将其取消掉
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels