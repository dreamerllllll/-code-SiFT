import os
import numpy as np
import torch
import torch.utils.data as data
import PIL.Image as Image
from typing import Any, List, Union
import torch
from functools import wraps

def check_dir(dir):
    if not os.path.exists(dir): 
        os.makedirs(dir)

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.)
    return onehot

def get_dataset(class_idx, data, targets, tfs):
    def select(x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
    
    out_data, out_targets = [], []
    for idx in class_idx:
        class_data, class_targets = select(data, targets, low_range=idx, high_range=idx+1)
        out_data.append(class_data)
        out_targets.append(class_targets)
    out_data, out_targets = np.concatenate(out_data), np.concatenate(out_targets)
    return SimpleDataset(out_data, out_targets, tfs, False)
        
class SimpleDataset(data.Dataset):
    def __init__(self, data, targets, transform, use_path=False) -> None:
        super().__init__()

        assert len(data) == len(targets), 'Data size error!'
        self.data = data
        self.targets = targets
        self.transform = transform # resize等操作放到transform里面
        self.use_path = use_path
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        if self.use_path:
            image = self.transform(pil_loader(self.data[index]))
        else:
            # image = self.transform(Image.fromarray(self.data[index]))
            image = self.transform(self.data[index])
        label = self.targets[index]

        return image, label

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def split_logits(logits, targets):
        # input_dim: batch_size*class_num, batch_size*1
        preds = torch.max(logits, dim=1)[1] # batch_size*1
        true_mask = (preds == targets)
        error_mask = (preds != targets)
        true_logits = logits[true_mask]
        error_logits = logits[error_mask]
        return true_logits, error_logits, true_mask, error_mask

def get_class_name(args, dataset_name):
    from .factory import getData
    data_obj = getData(args, dataset_name)
    assert hasattr(data_obj, 'class_order') and hasattr(data_obj, 'class_name')
    class_order = data_obj.class_order ; class_name = data_obj.class_name
    return list(np.array(class_name)[class_order])

def set_mode(name, inv, outv):
    def wrapper(func):
        @wraps(func)
        def sub_wrapper(self, *args, **kwargs):
            if len(name.split('.'))>1:
                obj_name, att_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
                setattr(eval('self.'+obj_name), att_name, inv)
            else:
                setattr(self, name, inv)
            res = func(self, *args, **kwargs)
            if len(name.split('.'))>1:
                obj_name, att_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
                setattr(eval('self.'+obj_name), att_name, outv)
            else:
                setattr(self, name, outv)
            return res
        return sub_wrapper
    return wrapper

def get_medmnist_img():
    import medmnist
    train_d = medmnist.OrganCMNIST('train', download=False, root='/data/xuyang/work/datasets/medmnist', size=224).imgs
    print(train_d.shape, type(train_d))
    
    for i in np.random.choice(range(len(train_d)), 10):
        img = train_d[i]
        img = Image.fromarray(img).convert('RGB')
        img.save('outputs/imgs/organc/{}.png'.format(str(i)))

if __name__ == '__main__':
    pass