import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
import torchvision.models
import model.backbones as bb
from model.backbones.FedAvg_Models import CIFAR10_CNN, Mnist_2NN, Mnist_CNN
from model.backbones.cifar10_resnet import resnet32
from model.backbones.resnet_cbam import resnet18_cbam

import logging

UseGn = False
KeepBNInit = False
def get_backbone(args, backbone_type):
    '''
    返回合适的backbone(没有fc), 且返回feature_dim(需要涉及到的backbone拥有num_features属性)
    '''
    model_name = backbone_type.lower()
    backbone = None

    if model_name == 'mnist_2nn':
        backbone = Mnist_2NN()
        backbone.fc3 = nn.Identity()
    elif model_name == 'mnist_cnn':
        backbone = Mnist_CNN()
        backbone.fc2 = nn.Identity()
    elif model_name == 'cifar_cnn':
        backbone = CIFAR10_CNN()
        backbone.fc3 = nn.Identity()
    elif model_name == 'resnet32':
        backbone = resnet32()
    elif model_name == 'resnet18':
        backbone = bb.resnet18(**args.model_cfg)
    elif model_name == 'resnet34':
        backbone = bb.resnet34(**args.model_cfg)
    elif model_name == 'resnet18_cbam':
        backbone = resnet18_cbam(pretrained=False, args=args)
    elif model_name == 'resnet18_pretrained':
        backbone = torchvision.models.resnet18(pretrained=True)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif model_name == 'efficientnet_b0':
        backbone = timm.create_model('efficientnet_b0', pretrained=False)
        num_features = backbone.num_features
        backbone.classifier = nn.Identity()
        return backbone, num_features
    elif model_name in ['vit_tiny_patch16_224','vit_base_patch16_224']:
        backbone = timm.create_model(model_name)
        num_features = backbone.num_features
        backbone.head = nn.Identity()
        return backbone, num_features
    else:
        raise NotImplementedError(backbone)
    
    return backbone, backbone.num_features

class IncrementNet(torch.nn.Module):
    '''
    提供backbone的上层包装，完成一些内容，比如更新fc
    '''
    def __init__(self, args, logger, backbone_type, dev) -> None:
        super().__init__()
        self.args = args
        self.backbone_type = backbone_type
        self.feature_extractor, self.feature_dim = get_backbone(self.args, backbone_type)
        self._logger = logger
        self.fc = None
        self.dev = dev
        self.known_class_list = []
        self.last_class_list = [] #上一个client中的class list

    def update_fc(self, classes_list): 
        '''
        传入的是class list,而不是一个标量。如果known_class_list已经包含了传入的classes_list的话，那么没有必要进行fc更新，否则进行更新
        '''
        if set(classes_list) & set(self.known_class_list) == set(classes_list):
            self._logger.info('No need to update classifier head')
            return
        elif len(set(classes_list) & set(self.known_class_list)) == 0: # 完全不相交
            self.known_class_list.extend(classes_list)
            nb_classes = len(self.known_class_list)
        else:
            raise NotImplemented

        fc = nn.Linear(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight # 注意：wight的shape的形状问题。第一个维度，也就是每一行代表一个类别的对应权重向量。因此这里才可以这样用
            fc.bias.data[:nb_output] = bias
            self._logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self._logger.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.fc
        self.fc = fc.to(self.dev)
    
    @property
    def known_class_num(self):
        return len(self.known_class_list)
    
    def freeze(self):
        self.eval()
        for i in self.parameters():
            i.requires_grad = False
    def unfreeze(self):
        for i in self.parameters():
            i.requires_grad = True

    def extract_features(self, x):
        return self.feature_extractor(x)
            
    def forward(self,x):
        feature = self.feature_extractor(x)
        out = self.fc(feature)
        return out, {'features':feature}
    
    def get_model_parameters(self, deep_copy=True):
        '''
        deep_copy=True: 会得到完全脱离于计算图的模型参数,这对于需要单纯保留当前参数的场景是有用的
        deep_copy=False: 得到没有脱离当前计算图的模型参数,这对于一些辅助loss是很重要的
            (当其需要当前模型的参数，但是计算的结果不应该脱离计算图的时候。因为如果参数脱离计算图的话，在反向传播的时候
            不会对参数的更新起到作用)
        '''
        if deep_copy:
            return [i.data.clone() for i in self.parameters()]
        else:
            return [i for i in self.parameters()]
        
    @property
    def model_parameters(self):
        return self.get_model_parameters(deep_copy=True)       

    def load_param(self, param_list:list):
        '''
        更新参数，将参数更新为传进来的param_list中的值
        '''
        for indx, param in enumerate(self.parameters()):
            param.data.copy_(param_list[indx])
    
    def get_flatten_param(self, deep_copy=True):
        '''
        deep_copy=True: 会得到完全脱离于计算图的模型参数,这对于需要单纯保留当前参数的场景是有用的
        deep_copy=False: 得到没有脱离当前计算图的模型参数,这对于一些辅助loss是很重要的
            (当其需要当前模型的参数，但是计算的结果不应该脱离计算图的时候。因为如果参数脱离计算图的话，在反向传播的时候
            不会对参数的更新起到作用)
        '''
        param = self.get_model_parameters(deep_copy)
        return torch.cat([i.flatten() for i in param]) #flatten和cat都不会让一个本来在计算图中的tensor脱离计算图

    def get_flatten_param_wbn(self):
        '''
        get the param list with bn running param
        只能为deep_copy模式,因为不需要不需要依靠这个函数的返回值用于训练
        '''
        vec = []
        for name, param in self.state_dict().items(): #use state_dict not the parameters
            vec.append(torch.flatten(param.clone().cpu()))
        return torch.cat(vec)
    
    def load_flatten_param_wbn(self, params):
        idx = 0
        for name, param in self.state_dict().items(): #修改state_dict所返回的参数会使得原本的参数也发生改变
            length = param.numel()
            param.data.copy_(params[idx:idx + length].reshape(param.shape))
            idx += length

    def get_feature_extractor_param(self, deep_copy=True):
        '''
        只得到feature extractor的flatten param
        
        deep_copy=True: 会得到完全脱离于计算图的模型参数,这对于需要单纯保留当前参数的场景是有用的
        deep_copy=False: 得到没有脱离当前计算图的模型参数,这对于一些辅助loss是很重要的
            (当其需要当前模型的参数，但是计算的结果不应该脱离计算图的时候。因为如果参数脱离计算图的话，在反向传播的时候
            不会对参数的更新起到作用)
        '''
        if deep_copy:
            param = [i.data.clone() for i in self.feature_extractor.parameters()] #完全和计算图分离了
        else:
            param = [i for i in  self.feature_extractor.parameters()] #这里不能使用detach，因为使用detach会导致和原始的计算图分离，导致没有辅助loss没有效果
        return self.flatten_param(param)
    
    @staticmethod
    def flatten_param(param):
        '''
        将某一个参数列表进行flatten，这样比较适合后续处理
        '''
        return torch.cat([i.flatten() for i in param])

    def load_flatten_param(self, flatten_param):
        '''
        load一个经过flatten的param参数
        首先要恢复其shape, 然后进行load
        '''
        param = self.parse_flatten_param(flatten_param)
        self.load_param(param)

    def parse_flatten_param(self, flatten_param:torch.Tensor):
        assert len(self.get_flatten_param(False)) == len(flatten_param)
        res = []
        start, end = 0, 0
        for indx, param in enumerate(self.parameters()):
            end = start + param.numel()
            res.append(flatten_param[start:end].reshape(param.shape))
            start = end
        return res
    
    @property
    def flatten_gradient(self):
        '''
        得到本模型的梯度信息 得到的梯度信息和原来的模型梯度没有什么关联性，是新的tensor
        '''
        return torch.cat([i.grad.data.flatten() for i in self.parameters() if i.grad is not None])

    @property
    def flatten_fe_gradient(self):
        '''
        得到本模型的梯度信息 得到的梯度信息和原来的模型梯度没有什么关联性，是新的tensor
        '''
        return torch.cat([i.grad.data.flatten() for i in self.feature_extractor.parameters() if i.grad is not None])

    def reinit(self):
        '''
        reinit the model, which will fully forget the knowledge 
        '''
        temp_bb,_ = get_backbone(self.args, self.backbone_type)
        self.feature_extractor.load_state_dict(temp_bb.state_dict())
        nn.init.uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

class CosNormNet(IncrementNet):
    '''
    The version of Using cosine normalization
    '''
    def __init__(self, args, logger, backbone_type, dev) -> None:
        super().__init__(args, logger, backbone_type, dev)
        self.learnable_T = nn.Parameter(torch.ones(1))

    def forward(self, x):
        feature = self.feature_extractor(x)
        feature = feature / torch.norm(feature, p=2, dim=1, keepdim=True)
        self.fc.weight.data.copy_(self.fc.weight.data / torch.norm(self.fc.weight.data, p=2, dim=1, keepdim=True))
        out = torch.mm(feature, self.fc.weight.T)
        return out, {'features':feature}
    
class SplitNet(IncrementNet):
    def __init__(self, args, logger, backbone_type, dev) -> None:
        from model.backbones.vit import MyAttention, MySimpleAttention,MyChannelAttention
        super().__init__(args, logger, backbone_type, dev)

        self.feature_extractor.stage_3.register_forward_hook(self.feature_hook)
        self.classifiers = nn.ModuleDict()
        self.prompt_len = 1
        self.prompts = nn.ParameterDict() #一个class一个

        #self.atten = MyAttention(64, num_heads=2, attn_drop=0.2, proj_drop=0.2)
        #self.atten = MySimpleAttention(64)
        #self.atten = MyChannelAttention(64)
        self.limit_fc = None
        self.feature_map = None
        self.aux_cls = None

    def feature_hook(self, model, input, output):
        self.feature_map = output

    def extract_features(self, x):
        feature = self.feature_extractor(x)
        feature_map = self.feature_map
        return feature, feature_map

    def update_fc(self, classes_list): 
        '''
        如果known_class_list已经包含了传入的classes_list的话，那么没有必要进行fc更新，否则进行更新
        '''
        str_classes_list = str(tuple(classes_list))
        if str_classes_list not in self.classifiers:
            #classifier = nn.Linear(self.feature_dim, len(classes_list)) #这里如果置换为同一的长度?
            classifier = nn.Sequential(nn.Linear(self.feature_dim, len(classes_list)))
            classifier.to(self.dev)
            self.classifiers.add_module(str_classes_list, classifier)

            self.prompts.update({str_classes_list: nn.Parameter(torch.randn(1, self.prompt_len, 64))})

            self.class_set = set()
            for k in self.classifiers.keys():
                self.class_set = self.class_set|set(eval(k))
            self._logger.info('Update class set to {}'.format(list(self.class_set)))

            # limit_fc = nn.Linear(self.feature_dim, len(self.class_set)).to(self.dev)
            # if self.limit_fc is not None:
            #     nb_output = self.limit_fc.out_features
            #     weight = copy.deepcopy(self.limit_fc.weight.data)
            #     bias = copy.deepcopy(self.limit_fc.bias.data)
            #     limit_fc.weight.data[:nb_output] = weight # 注意：wight的shape的形状问题。第一个维度，也就是每一行代表一个类别的对应权重向量。因此这里才可以这样用
            #     limit_fc.bias.data[:nb_output] = bias
            # self.limit_fc = limit_fc
        else:
            self._logger.info('Classifier already exist')

    def create_aux_classifier(self, class_list):
        '''
        创建一个辅助分类头
        '''
        self.aux_cls = nn.Linear(self.feature_dim, len(class_list)).to(self.dev)

    def forward(self, x):
        '''
        对其进行预测, 采用集成预测
        '''
        feature, feature_map = self.extract_features(x)
        b,c,h,w = feature_map.shape

        preds = torch.zeros(len(x), len(self.class_set), device=x.device)
        for i, k in enumerate(self.classifiers.keys()):
            cls_l = eval(k)
            #input = torch.cat((self.prompts[k].expand(b,-1,-1).to(self.dev), feature_map.reshape(b,c,-1).permute(0,2,1)),dim=1)
            #feature, atten = self.atten(input)
            logit = self.classifiers[k](feature)
            preds[:, cls_l] = (preds[:, cls_l]*i+logit)/(i+1) #均值

        if self.aux_cls is not None:
            elim_logit = self.aux_cls(feature)
            #preds[:,:len(elim_logit[0])] = preds[:,:len(elim_logit[0])]  #融合
            eval_logit = (preds[:,:len(elim_logit[0])]+elim_logit)/2
            return preds, {'features':feature, 'feature_map':feature_map, 'elim_logits':elim_logit, 'eval_logits':eval_logit}
        return preds, {'features':feature, 'feature_map':feature_map, 'eval_logits':preds}

    def freeze_other_classifiers(self, class_list):
        assert str(tuple(class_list)) in self.classifiers
        for i in self.classifiers.keys():
            if i != str(tuple(class_list)):
                for p in self.classifiers[i].parameters():
                    p.requires_grad = False

    def freeze_other_prompts(self, class_list):
        assert str(tuple(class_list)) in self.prompts
        for i in self.prompts.keys():
            if i != str(tuple(class_list)):
                self.prompts[i].requires_grad = False
    
    def get_bn_param_dict(self):
        res = {}
        for n,v in self.state_dict().items():
            if 'bn' in n and 'num_batches' not in n:
                res[n] = v.data.clone()
        return res
    
    def load_bn_param_dict(self, d):
        state = self.state_dict()
        for k,v in d.items():
            state[k].data.copy_(v)
    

class GrowNet(IncrementNet):
    def __init__(self, args, logger, backbone_type, dev) -> None:
        from model.backbones.vit import MyAttention, MySimpleAttention, MyChannelAttention
        super().__init__(args, logger, backbone_type, dev)
        
        self.feature_extractors = nn.ModuleDict() #一个client一个feature extractor
        self.classifiers = nn.ModuleDict()
        self.p2_classifiers = nn.ModuleDict()
        self.p2_prototype = nn.ModuleDict()
        self.client2class = {}
        self.class_set = set()
        self.p2_class_set = set()
        self.phase = None
        self.cur_client, self.last_client = None, None #name of client
        self.last_clients = [] #之前的client名字
        self.round_id = -1
        
        if self.args.dataset == 'cifar100':
            self.all_class_num = 100
        elif self.args.dataset == 'cifar10':
            self.all_class_num = 10

    def extract_features(self, x, name):
        if type(name) == str:
            feature = self.feature_extractors[name](x)
        elif type(name) == list:
            feature = []
            for n in name:
                f = self.feature_extractors[n](x)
                feature.append(f)
            feature = torch.cat(feature, dim=-1) #B x N*D
        return feature
    
    def update_fc(self, name, classes_list, phase, round_id): 
        self.last_client = self.cur_client
        self.cur_client = name
        self.last_clients.append(name)
        self.round_id = round_id

        self.phase = phase
        assert phase in ['first', 'second']
        if name not in self.feature_extractors.keys():
            if phase == 'first':
                backbone = get_backbone(self.args, self.backbone_type)[0].to(self.dev)
                # if self.last_client is not None:
                #     backbone.load_state_dict(self.get_avg_state_dict(self.cur_client))
                #     self._logger.info('Load state dicts')
                self.feature_extractors.add_module(name, backbone)
                self.classifiers.add_module(name, nn.Linear(self.feature_dim, len(classes_list)).to(self.dev))
                #self.classifiers.add_module(name, nn.Linear(self.feature_dim, self.all_class_num).to(self.dev))
                #classes_list = list(range(min(classes_list), self.all_class_num)) #修改为最小 ~ 最后

                self.client2class[name] = classes_list
                self.class_set = self.class_set | set(classes_list)
                self._logger.info('Create classifier for {}'.format(name))
        else: #phase second
            assert self.phase == 'second'
            ## construct a classifier that use all features
            # if name not in self.p2_classifiers:
            #     self.p2_class_set = self.p2_class_set | set(classes_list)
            #     self.p2_classifiers.add_module(name, nn.Linear(self.feature_dim*len(self.feature_extractors), len(classes_list)).to(self.dev))
            #     self.p2_prototype.add_module(name, nn.Linear(self.feature_dim*len(self.feature_extractors), 1).to(self.dev))

            ## redefine the classifiers from the first phase
            if round_id == 2: #只是在第二轮进行替代，之后只是微调
                old_classifier = self.classifiers[name]
                if old_classifier.in_features == self.feature_dim: #说明没有进行过更新，此时才进行更新
                    new_classifier = nn.Linear(self.feature_dim*len(self.feature_extractors), self.all_class_num).to(self.dev)
                    
                    # nb_in = old_classifier.in_features
                    # nb_out = old_classifier.out_features
                    # weight = copy.deepcopy(old_classifier.weight.data)
                    # bias = copy.deepcopy(old_classifier.bias.data)
                    # new_classifier.weight.data[:nb_out, :nb_in] = weight # 注意：wight的shape的形状问题。第一个维度，也就是每一行代表一个类别的对应权重向量。因此这里才可以这样用
                    # new_classifier.bias.data[:nb_in] = bias

                    self.classifiers.update({name:new_classifier})
                    #self.client2class[name] = list(range(0, self.all_class_num))
                    self._logger.info('Updated classifier for {}'.format(name))
                
        self._logger.info('Trainable parameter num:{}'.format(sum([i.numel() for i in self.parameters() if i.requires_grad==True])))
    
    def forward(self, x, name=None, phase=None, raw_features=None):
        '''
        name: clientX or None
        clientX means we will use the clientX's parameters, used for training. None means we don't know the 
        client name, used for test
        Phase means the first or the second phase
        '''
        if phase is None:
            phase = self.phase #这样设计在测试的时候就不需要传进来phase参数了

        assert phase in ['first', 'second'] 
        if name is not None and self.phase == 'first':
            feature = self.extract_features(x, name)
            
            logits = self.classifiers[name](feature)

            ##To return features and old logits
            if self.last_client is not None:
                #old_features = self.feature_extractors[self.last_client](x)
                old_features = []
                old_logits = []
                for i in self.feature_extractors.keys():
                    old_feature = self.feature_extractors[i](x)
                    old_features.append(old_feature)
                    if i != name: #一定不能包括自己的logit
                       old_logit = self.classifiers[i](old_feature)
                       old_logits.append(old_logit)
                    # if i == self.last_client:
                    #     old_logit = self.classifiers[i](old_feature)
                    #     old_logits.append(old_logit)
                old_features = torch.cat(old_features,dim=0) #B*N x D
                old_logits = torch.stack(old_logits) #N x B x D
                
                # logits = torch.zeros(len(x), len(self.class_set), device=x.device)
                # cls_num = torch.zeros(len(self.class_set),device=x.device)
                # for i, k in enumerate(self.classifiers.keys()):
                #     cls_l = self.client2class[k]
                #     cls_num[cls_l] = cls_num[cls_l]+1
                #     old_feature = self.feature_extractors[k](x)
                #     if k != name:
                #         logit = self.classifiers[k](old_feature)
                    
                #     logit = torch.softmax(logit/2, dim=-1)*weight #归一化 使用温度系数
                #     logits[:, cls_l] = logits[:,cls_l]+logit
                return logits, {'features':feature, 'old_features':old_features}  
        
        # if name is not None and self.phase == 'second':
        #     features = self.extract_features(x, list(self.feature_extractors.keys()))
        #     feature = self.extract_features(x, name)
        #     logits = self.classifiers[name](feature)
        #     logits += self.p2_classifiers[name](features)    #相加修正式
        #     #logits *= self.p2_classifiers[name](features)   #相乘修正式
        #     proto_logits = F.cosine_similarity(self.p2_prototype[name].weight.repeat(feature.shape[0], 1), features) #prototype
        #     return logits, {'proto_logits':proto_logits}
        elif name is not None and self.phase == 'second':
            if raw_features is not None:
                features = raw_features
            else:
                features = self.extract_features(x, list(self.feature_extractors.keys()))
                
            ##进行微调分类头
            logits = self.classifiers[name](features)
            old_logits = []
            for i in self.feature_extractors.keys():
                if i != name: #一定不能包括自己的logit
                    old_logit = self.classifiers[i](features)
                    old_logits.append(old_logit)
            old_logits = torch.stack(old_logits) #N x B x D
            return logits, {'old_logits':old_logits, 'features':features}
        else: #test
            #features = []
            if self.phase == 'second':
                if raw_features is not None:
                    features = raw_features
                else:
                    features = self.extract_features(x, list(self.feature_extractors.keys()))
            logits = torch.zeros(len(x), len(self.class_set), device=x.device)
            cls_num = torch.zeros(len(self.class_set),device=x.device)
            energys, temp_logits, cls_list = [], [], []
            
            for i, k in enumerate(self.classifiers.keys()):
                cls_l = self.client2class[k]
                cls_list.append(cls_l)
                cls_num[cls_l] = cls_num[cls_l]+1
                if self.phase == 'first' or self.classifiers[k].in_features==self.feature_dim:
                    feature = self.feature_extractors[k](x)
                    logit = self.classifiers[k](feature)
                elif self.phase == 'second':
                    logit = self.classifiers[k](features)

                #features.append(feature)
                # if self.phase == 'second' and k in self.p2_classifiers.keys():
                #     p2logit = self.p2_classifiers[k](features)
                #     logit = logit+p2logit #相加修正式
                #     #logit *= p2logit #相乘修正式

                    ### 利用prototype logit
                    #proto_sim = F.cosine_similarity(self.p2_prototype[k].weight.repeat(features.shape[0],1),features)
                    #proto_weight = torch.max(proto_logit,dim=-1,keepdim=True)[0]
                    #logit *= proto_sim[:,None]*2 #这里乘以2，使得总体分布在0~2之间，因为0~1之间的话会导致无论怎样都会更小

                    #self._logger.info(proto_sim)

                #std = logit.std(dim=-1, unbiased=False, keepdim=True) #std越大的，就给予其更大的优先级
                #weight = torch.sigmoid(std)

                # energy = torch.logsumexp(logit, dim=-1, keepdim=True)
                # energys.append(energy)
                temp_logits.append(logit)

            ## 对energy进行处理
            #energys = torch.cat(energys, dim=-1) # B x N

            # sot = torch.argsort(energys, dim=-1)
            # weight = torch.zeros(sot.shape,device=self.dev)
            # for i in range(len(weight)):
            #     weight[i,sot[i]] = torch.arange(0, len(energys[0]),dtype=torch.float32,device=self.dev)
            # weight = weight+1
            #self._logger.info(weight)
            
            if self.phase == 'first':
                for idx, cls_l in enumerate(cls_list):
                    #self._logger.info(f'{cls_l}: {torch.softmax(temp_logits[idx]/2, dim=-1)[:,len(cls_l):]}')

                    #logit = torch.softmax(temp_logits[idx]/2, dim=-1)*energys[:,idx][:,None]
                    logit = torch.softmax(temp_logits[idx]/2, dim=-1)
                    #logit = temp_logits[idx]
                    #logit = temp_logits[idx]*energys[:,idx][:,None]
                    logits[:, cls_l] = logits[:, cls_l]+logit
            else:
                cls_num = torch.ones(len(self.class_set),device=x.device)*len(cls_list) #每一个类别都预测了相同的次数
                for idx, cls_l in enumerate(cls_list):
                    for i in range(len(temp_logits)):
                        logit = torch.softmax(temp_logits[i][:,cls_l]/2, dim=-1)
                        logits[:, cls_l] = logits[:, cls_l]+logit
            logits = logits/cls_num
            
            # if self.phase == 'second':
            #     ## Add the p2classifier output
            #     p2logits = torch.zeros(len(x), len(self.class_set), device=x.device)
            #     p2cls_num = torch.ones(len(self.class_set),device=x.device)
            #     features = torch.cat(features, dim=-1) #输入为统一的features
            #     for i, k in enumerate(self.p2_classifiers.keys()):
            #         cls_l = self.client2class[k]
            #         p2cls_num[cls_l] = p2cls_num[cls_l]+1
            #         p2logit = self.p2_classifiers[k](features)

            #         p2logit = torch.softmax(p2logit/2, dim=-1)
            #         p2logits[:, cls_l] = p2logits[:,cls_l]+p2logit
            #     logits = logits+p2logits
            #     logits = logits/p2cls_num
        return logits, {}

    def freeze_feature_extractor(self, name):
        assert name in self.feature_extractors
        for i in self.feature_extractors.keys():
            if i == name:
                for p in self.feature_extractors[i].parameters():
                    p.requires_grad=False
                
    def freeze_classifier(self, name):
        assert name in self.classifiers
        for i in self.classifiers.keys():
            if i == name:
                for p in self.classifiers[i].parameters():
                    p.requires_grad=False

    def freeze_p2_classifier(self, name):
        assert name in self.p2_classifiers
        for i in self.p2_classifiers.keys():
            if i == name:
                for p in self.p2_classifiers[i].parameters():
                    p.requires_grad=False

        for i in self.p2_prototype.keys():
            if i == name:
                for p in self.p2_prototype[i].parameters():
                    p.requires_grad = False
    
    #possibly bug
    def get_avg_state_dict(self, name):
        res = []
        for n in self.feature_extractors.keys():
            if n != name:
                res.append(self.feature_extractors[n].state_dict())
        
        stat = {}
        for i in res[0].keys():
            for j in range(len(res)):
                if i not in stat:
                    stat[i] = res[j][i]
                else:
                    stat[i] += res[j][i]
            stat[i] = stat[i]/len(res)
        return stat

class DERNet(torch.nn.Module):  
    def __init__(self, args, logger, backbone_type, dev) -> None:
        super().__init__()
        self.args = args
        self._logger = logger
        self.backbone_type = backbone_type
        self.feature_extractor = nn.ModuleList()
        self.feature_dim = None
        self.fc = None
        self.aux_fc = None
        self.dev = dev
        self.known_class_list = []
        self.last_class_list = []
    
    def extract_features(self, x):
        features = [fe(x) for fe in self.feature_extractor]
        features = torch.cat(features, 1)
        return features
    
    def forward(self, x):
        features = [fe(x) for fe in self.feature_extractor]
        all_features = torch.cat(features, 1)

        out = self.fc(all_features)
        aux_logits = self.aux_fc(features[-1])

        return out, {"aux_logits":aux_logits, "features":all_features}
    
    def update_fc(self, classes_list):
        if set(classes_list) & set(self.known_class_list) == set(classes_list):
            self._logger.info('No need to update feature extractor and classifier head')
            return
        elif len(set(classes_list) & set(self.known_class_list)) == 0: # 完全不相交
            self.known_class_list.extend(classes_list)
            nb_classes = len(self.known_class_list)
        else:
            raise NotImplemented
        
        ft, feature_dim = get_backbone(self.args, self.backbone_type)
        if len(self.feature_extractor) == 0:
            self.feature_extractor.append(ft)
        else:
            self.feature_extractor.append(ft)
            self.feature_extractor[-1].load_state_dict(self.feature_extractor[-2].state_dict())
        self.feature_dim = feature_dim * len(self.feature_extractor)

        self.aux_fc = nn.Linear(feature_dim, len(classes_list)+1) # 只学当前task数据的头，保存的所有旧类视为一类

        fc = nn.Linear(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :-feature_dim] = weight
            fc.bias.data[:nb_output] = bias
            self._logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self._logger.info('Created classifier head with output dim {}'.format(nb_classes))
        del self.fc
        self.fc = fc

    def freeze(self):
        self.eval()
        for i in self.parameters():
            i.requires_grad = False
    
    def reset_fc_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)


class AdaptiveNet(nn.Module):
    def __init__(self, args, logger, backbone_type, dev, pretrained=False):
        super(AdaptiveNet, self).__init__()
        self.backbone_type = backbone_type
        self.TaskAgnosticExtractor, _ = get_backbone(args, backbone_type) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained = pretrained
        self.dev = dev
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args
        self.logger = logger

        self.known_class_list = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_features(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        all_features = torch.cat(features, 1)
        out = self.fc(all_features) #{logits: self.fc(features)}

        aux_logits = self.aux_fc(features[-1])

        return out, {"aux_logits":aux_logits, "features":all_features, "base_features":base_feature_map}
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self, classes_list):
        if set(classes_list) & set(self.known_class_list) == set(classes_list):
            self.logger.info('No need to update feature extractor and classifier head')
            return
        elif len(set(classes_list) & set(self.known_class_list)) == 0: # 完全不相交
            self.known_class_list.extend(classes_list)
            nb_classes = len(self.known_class_list)
        else:
            raise NotImplemented

        _ , _new_extractor = get_backbone(self.args, self.backbone_type)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.AdaptiveExtractors[-1].feature_dim
        fc = nn.Linear(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, :self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.logger.info('Updated classifier head output dim from {} to {}'.format(nb_output, nb_classes))
        else:
            self.logger.info('Created classifier head with output dim {}'.format(nb_classes))

        del self.fc
        self.fc = fc

        self.aux_fc = nn.Linear(self.out_dim, len(classes_list)+1)

    def copy(self):
        return copy.deepcopy(self)
    
    def reset_fc_parameters(self):
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

class IL2ANet(IncrementNet):
    '''
    IL2Net中需要有辅助类，因此这些辅助类也是需要更新的
    '''
    def update_fc(self, classes_list, num_aux):
        if set(classes_list) & set(self.known_class_list) == set(classes_list):
            num_old = self.known_class_num
            num_total = self.known_class_num
        elif len(set(classes_list) & set(self.known_class_list)) == 0: # 完全不相交
            num_old = self.known_class_num
            self.known_class_list.extend(classes_list)
            num_total = len(self.known_class_list)
        else:
            raise NotImplemented

        fc = nn.Linear(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            # nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
            self._logger.info('Updated classifier head output dim from {} to {} (aux: {})' \
                              .format(num_old, num_total+num_aux, num_aux))
        else:
            self._logger.info('Created classifier head with output dim {} (aux: {})' \
                              .format(num_total+num_aux, num_aux))
        del self.fc
        self.fc = fc.to(self.dev)
