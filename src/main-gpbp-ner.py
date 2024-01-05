import torch
import json
import sys
import numpy as np
import torch.nn as nn
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader,Dataset
import configparser
# from torch.utils.tensorboard import SummaryWriter
from utils.bert_optimization import BertAdam
import utils.target_calculate
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='0'
#处理函数
def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            d={}
            d["text"]=line["text"]
            d["spo_list"]=[(spo["subject"],spo["subject_head"],spo["subject_tail"], spo["predicate"], spo["object"],spo["object_head"],spo["object_tail"], spo["subject_type"], spo["object_type"])
                            for spo in line["spo_list"]]
            d["entity_list"]=[]
            for k, v in line['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            d["entity_list"].append((start, end, k))
            D.append(d)
        return D
def load_schema(schema_name):
    with open(schema_name, 'r', encoding='utf-8') as f:
        for schema in f:
            schema = json.loads(schema)
    id2schema=[]
    for sch in schema:
        id2schema.append(str(sch))
    return schema,id2schema
def load_en2id(en2id_name):
    with open(en2id_name, 'r', encoding='utf-8') as f:
        for l in f:
            en2id = json.loads(l)
    id2en=[]
    for en in en2id:
        id2en.append(str(en))
    return en2id,id2en
def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1
def search_list(pattern, sequence):
    """从sequence中寻找子串pattern
    找到返回list
    """
    ilist = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            ilist.append(i)
    return ilist
def load_true_data(filename):
    with open(filename, encoding="utf-8") as f:
        test_text_list=[]
        test_true_spo=[]
        test_true_ent=[]
        for text in f.readlines():
            test_text_list.append(json.loads(text.rstrip())["text"])
            test_true_spo.append(json.loads(text.rstrip())["spo_list"])
            ent=[]
            for k, v in json.loads(text.rstrip())['label'].items():
                    for spans in v.values():
                        for start, end in spans:
                            ent.append({"predicate":k, "head":start,"tail":end})
            test_true_ent.append(ent)
    return test_text_list,test_true_spo,test_true_ent
class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema ,en2id):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema
        self.en2id = en2id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)
    
    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] 
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s,s_head,s_tail,p,o, o_head,o_tail, s_type, o_type in item["spo_list"]:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[p]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            sh = search(s, input_ids)
            oh = search(o, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))#头实体head，头实体tail，关系，尾实体head，尾实体tail
        entity_labels = [set() for i in range(len(self.en2id))]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for ehead,etail,etype in item["entity_list"]:
            p = self.en2id[etype]
            e = self.tokenizer.encode(text[ehead:etail+1], add_special_tokens=False)
            ehlist = search_list(e,input_ids)
            if len(ehlist):
                for eh in ehlist:
                    entity_labels[p].add((eh, eh+len(e)-1)) 
        for sh, st, p, oh, ot in spoes:
            head_labels[p].add((sh, oh)) #类似TP-Linker#p位置有头实体和尾实体的head
            tail_labels[p].add((st, ot))#p位置有头实体和尾实体的tail

        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))#空位补0
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels
    
#初始化配置
con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf8')
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
encoder = BertModel.from_pretrained(args_path["model_path"])

#数据导入
schema,id2schema = load_schema(args_path["schema_data"])
en2id,id2en = load_en2id(args_path["ent2id_data"])
test_text_list,test_true_spo,test_true_ent = load_true_data(args_path["test_file"])
train_text_list,train_true_spo,train_true_ent = load_true_data(args_path["train_file"])
train_rawdata = load_data(args_path["train_file"])
maxlen = con.getint("para", "maxlen")
batch_size = con.getint("para", "batch_size")
train_data = data_generator(train_rawdata, tokenizer, max_len=maxlen, schema=schema,en2id=en2id)
train_loader = DataLoader(train_data , batch_size=batch_size, shuffle=True, collate_fn=train_data.collate)

#模型设计
device = torch.device("cuda:0")
class  GPNet1(nn.Module):
    #主要实现ner分类
    def __init__(self, encoder, ent_type_size):
        super(GPNet1, self).__init__()
        self.encoder = encoder
        self.entlayer = RawGlobalPointer(hiddensize=1024, ent_type_size=ent_type_size, inner_dim=64)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        entoutputs = self.entlayer(outputs, batch_mask_ids)
        return entoutputs
        
class GPNet2(nn.Module):
    #主要实现rel分类
    def __init__(self, encoder, ent_type_size):
        super(GPNet2, self).__init__()
        self.encoder = encoder
        self.so_head = RawGlobalPointer(hiddensize=1024, ent_type_size=ent_type_size, inner_dim=64, RoPE=False, tril_mask=False)
        self.so_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=ent_type_size, inner_dim=64, RoPE=False, tril_mask=False)

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        so_head_outputs = self.so_head(outputs, batch_mask_ids)
        so_tail_outputs = self.so_tail(outputs, batch_mask_ids)
        return so_head_outputs, so_tail_outputs

gpnet1 = GPNet1(encoder,len(en2id)).to(device)
gpnet2 = GPNet2(encoder,len(schema)).to(device)

#优化器设计
def set_optimizer(model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer

optimizer = set_optimizer(gpnet1, train_steps= (int(len(train_data) / con.getint("para", "batch_size")) + 1) * con.getint("para", "epochs"))
# optimizer = torch.optim.AdamW(
# 	net.parameters(),
#     lr=1e-5
# )

train_eval_size = 50
train_text_list=train_text_list[0:train_eval_size]
train_true_spo= train_true_spo[0:train_eval_size]
train_true_ent= train_true_ent[0:train_eval_size]

test_info = {}
train_info = {}

test_info['ner_precision'] = []
test_info['ner_recall'] = []
test_info['ner_f1'] = []
test_info['rel_precision'] = []
test_info['rel_recall'] = []
test_info['rel_f1'] = []  
train_info['ner_precision'] = []
train_info['ner_recall'] = []
train_info['ner_f1'] = []
train_info['rel_precision'] = []
train_info['rel_recall'] = []
train_info['rel_f1'] = []  
train_info['loss'] = [] 
# gpnet1.load_state_dict(torch.load('./erenet_1.pth'))
best_ner_f1 = 0.0
best_rel_f1 = 0.0
for eo in range(con.getint("para", "epochs")):
    gpnet1.train()
    total_loss = 0.0
    for idx, batch in enumerate(train_loader):
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
            batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
        logits = gpnet1(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits, mask_zero=True)#改成5类
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f] [idx:%f]"%(eo, con.getint("para", "epochs"), loss.item(),idx))
    train_info['loss'].append(total_loss)
    gpnet1.eval()
    print('\n')
    #train
    ner_precision,ner_recall,ner_f1=\
        utils.target_calculate.gpnet1_test_printInfo(train_text_list,train_true_ent,tokenizer,gpnet1,device,id2en)
    print('train ner|precision:',ner_precision,'|recall:',ner_recall,'|f1:',ner_f1)
    train_info['ner_precision'].append(ner_precision)
    train_info['ner_recall'].append(ner_recall)
    train_info['ner_f1'].append(ner_f1)
    with open('logtrain.json', 'w') as f:
        json.dump(train_info, f)
    #test
    ner_precision,ner_recall,ner_f1=\
        utils.target_calculate.gpnet1_test_printInfo(test_text_list,test_true_ent,tokenizer,gpnet1,device,id2en)
    print('test ner|precision:',ner_precision,'|recall:',ner_recall,'|f1:',ner_f1)
    test_info['ner_precision'].append(ner_precision)
    test_info['ner_recall'].append(ner_recall)
    test_info['ner_f1'].append(ner_f1)
    with open('logtest.json', 'w') as f:
        json.dump(test_info, f)
    if ner_f1>best_ner_f1:
        best_ner_f1 = ner_f1
        torch.save(gpnet1.state_dict(), './gb.pth')
        print('save best ner model')
    # if rel_f1>best_rel_f1:
    #     best_rel_f1 = rel_f1
    #     torch.save(gpnet1.state_dict(), './erenet_rel_best.pth')
    #     print('save best rel model')
    torch.save(gpnet1.state_dict(), './gpnet1.pth')
