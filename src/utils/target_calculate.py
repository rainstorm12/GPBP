import json
import torch
import numpy as np
#对json查询所有的entity，实例化
class dataset_entity():
    def __init__(self, spo):
        self.spo = spo
        self.data_entity,self.length =self.search_entity()
    def search_entity(self):
        length = 0
        data_entity=[]
        for i in range(0,len(self.spo)):
            spo_list = self.spo[i]
            entitylist=[]
            for tsp in spo_list:
                e={'name':tsp['subject'],'type':tsp['subject_type']}
                if e not in entitylist:
                    entitylist.append(e)
                e={'name':tsp['object']['@value'],'type':tsp['object_type']['@value']}
                if e not in entitylist:
                    entitylist.append(e)
            length += len(entitylist)
            data_entity.append(entitylist)
        return data_entity,length
#判断ner的tp，fp，fn
def cal_nerscore(true_ent,pred_ent):
    nerscore_tps = 0
    nerscore_fps = 0
    nerscore_fns = 0
    for i in range(0,len(true_ent)):
        true_entity_list = true_ent[i]
        pred_entity_list = pred_ent[i]
        #ner_tp 预测的里面有多少对的
        #ner_fp 预测的里面有多少错的
        for pe in pred_entity_list:
            thispe = False
            for te in true_entity_list:
                if pe['predicate']==te['predicate'] and pe['head']==te['head'] and pe['tail']==te['tail']:
                    thispe = True#只要能查到对应的一个，这个pred就是对的
            if thispe:
                nerscore_tps+=1
            else:
                nerscore_fps+=1
    for i in range(0,len(true_ent)):
        true_entity_list = true_ent[i]
        pred_entity_list = pred_ent[i]
        for te in true_entity_list:
            thispe = False
            for pe in pred_entity_list:
                if pe['predicate']==te['predicate'] and pe['head']==te['head'] and pe['tail']==te['tail']:
                    thispe = True#只要能查到对应的一个，这个pred就是对的
            if not thispe:
                nerscore_fns+=1
    return nerscore_tps,nerscore_fps,nerscore_fns
#判断两个字典相同不相同
def arr_same(psp,truesp,ptype):
    if ptype=='strict':
        objectrue = 0
        subjectrue = 0
        if psp['object']==truesp['object'] and psp['object_type']==truesp['object_type']:
            objectrue = 1
        if psp['subject']==truesp['subject'] and psp['subject_type']==truesp['subject_type']:
            subjectrue = 1
        if objectrue==1 and subjectrue ==1 and psp['predicate']==truesp['predicate']:
            return True
        else:
            return False
    elif ptype=='relaxed':
        objectrue = 0
        subjectrue = 0
        if psp['object']==truesp['object']:
            objectrue = 1
        if psp['subject']==truesp['subject']:
            subjectrue = 1
        if objectrue==1 and subjectrue ==1 and psp['predicate']==truesp['predicate']:
            return True
        else:
            return False
#判断rel的tp，fp，fn
def cal_relscore(true_spo,pred_spo,ptype):
    relscore_tps = 0
    relscore_fps = 0
    relscore_fns = 0
    for i in range(0,len(true_spo)):
        true_spo_list = true_spo[i]
        pred_spo_list = pred_spo[i]
        #rel_tp 预测的里面有多少对的
        #rel_fp 预测的里面有多少错的
        for psp in pred_spo_list:
            thispsp = False
            for truesp in true_spo_list:
                if arr_same(psp,truesp,ptype):
                    thispsp = True#只要能查到对应的一个，这个predspo就是对的
            if thispsp:
                relscore_tps+=1
            else:
                relscore_fps+=1
    for i in range(0,len(true_spo)):
        true_spo_list = true_spo[i]
        pred_spo_list = pred_spo[i]
        #rel_fn 对的里面有多少预测错的
        for truesp in true_spo_list:
            thispsp = False
            for psp in pred_spo_list:
                if arr_same(psp,truesp,ptype):
                    thispsp = True#只要能查到对应的一个，这个predspo就是对的
            if not thispsp:
                relscore_fns+=1
    return relscore_tps,relscore_fps,relscore_fns
#计算准确率
def cal_precision(tps,fps,fns):
    if tps+fps==0:
        precision = 0
    else:
        precision = tps/(tps+fps)
    return precision
#计算召回率
def cal_recall(tps,fps,fns):
    if tps + fns==0:
        recall = 0
    else:
        recall = tps/(tps+fns)
    return recall
#计算f1
def cal_fl(tps,fps,fns):
    precision = cal_precision(tps,fps,fns)
    recall = cal_recall(tps,fps,fns)
    if precision+recall == 0:
        f1 = 0.0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return f1
#计算ner的百分比指标
def cal_rel_percentage(true_spo,pred_spo,ptype):
    relscore_tps,relscore_fps,relscore_fns = cal_relscore(true_spo,pred_spo,ptype)
    precision = cal_precision(relscore_tps,relscore_fps,relscore_fns )
    recall = cal_recall(relscore_tps,relscore_fps,relscore_fns)
    if precision+recall == 0:
        f1 = 0.0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision,recall,f1
#计算rel的百分比指标
def cal_ner_percentage(true_ent,pred_ent):
    nerscore_tps,nerscore_fps,nerscore_fns = cal_nerscore(true_ent,pred_ent)
    precision = cal_precision(nerscore_tps,nerscore_fps,nerscore_fns )
    recall = cal_recall(nerscore_tps,nerscore_fps,nerscore_fns)
    if precision+recall == 0:
        f1 = 0.0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision,recall,f1
#测试模型时输出指标
def model_test_printInfo(text_list,true_spo,true_ent,tokenizer,net,device,id2schema,id2en,relptype='strict'):
    pred_spo = []
    pred_ent = []
    for text in text_list:
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=256)["offset_mapping"]
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text, max_length=256)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        scores = net(input_ids, attention_mask, token_type_ids)
        outputs = [o[0].data.cpu().numpy() for o in scores]
        # subjects, objects = set(), set()
        entitys = set()
        outputs[0][:, [0, -1]] -= np.inf
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > 0)):
            entitys.add((h, t, l)) #这里在调试的时候考虑一下会不会出现多个解的情况
        ent_list = []
        for h, t, l in list(entitys):
            ent_list.append({"predicate":id2en[l], "head":new_span[h][0],"tail":new_span[t][-1],"entity":text[new_span[h][0]:new_span[t][-1] + 1]})
        pred_ent.append(ent_list)
        spoes = set()
        # subjects = set(list(subjects)[0:20])
        # objects = set(list(objects)[0:20])
        entitys = set(list(entitys)[0:50])
        # for sh, st in subjects:
        #     for oh, ot in objects:
        for sh,st,sl in entitys:
            for oh,ot,ol in entitys:
                if sh!=oh and ot!=st:   #排除矩阵对角线 
                    p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                    p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        spoes.add((
                            text[new_span[sh][0]:new_span[st][-1] + 1], id2en[sl],id2schema[p],
                            text[new_span[oh][0]:new_span[ot][-1] + 1], id2en[ol]
                        ))
        spo_list = []
        for spo in list(spoes):
            spo_list.append({"predicate":spo[2], "object":spo[3], "object_type": spo[4],
                             "subject":spo[0], "subject_type":spo[1]
                             })
        pred_spo.append(spo_list)
    #测试结果
    # ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_spo,pred_spo)
    ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_ent,pred_ent)
    rel_precision,rel_recall,rel_f1 = cal_rel_percentage(true_spo,pred_spo,relptype)
    return ner_precision,ner_recall,ner_f1,rel_precision,rel_recall,rel_f1

#测试模型时输出指标
def gpnet1_test_printInfo(text_list,true_ent,tokenizer,net,device,id2en):
    pred_ent = []
    for text in text_list:
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=256)["offset_mapping"]
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text, max_length=256)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        scores = net(input_ids, attention_mask, token_type_ids)
        # outputs = [o[0].data.cpu().numpy() for o in scores]
        # subjects, objects = set(), set()
        outputs = scores.data.cpu().numpy()
        entitys = set()
        outputs[0][:, [0, -1]] -= np.inf
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > 0)):
            entitys.add((h, t, l)) #这里在调试的时候考虑一下会不会出现多个解的情况
        ent_list = []
        for h, t, l in list(entitys):
            ent_list.append({"predicate":id2en[l], "head":new_span[h][0],"tail":new_span[t][-1],"entity":text[new_span[h][0]:new_span[t][-1] + 1]})
        pred_ent.append(ent_list)
    #测试结果
    # ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_spo,pred_spo)
    ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_ent,pred_ent)
    return ner_precision,ner_recall,ner_f1


#测试模型时输出指标
def gpnet2_test_printInfo(text_list,true_spo,true_ent,tokenizer,gpnet1,gpnet2,device,id2schema,id2en,relptype='strict'):
    pred_spo = []
    pred_ent = []
    for text in text_list:
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=256)["offset_mapping"]
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text, max_length=256)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        scores1 = gpnet1(input_ids, attention_mask, token_type_ids)
        outputs1 = scores1.data.cpu().numpy()
        scores2 = gpnet2(input_ids, attention_mask, token_type_ids)
        outputs2 = [o[0].data.cpu().numpy() for o in scores2]
        # outputs2 = scores.data.cpu().numpy()
        # subjects, objects = set(), set()
        entitys = set()
        outputs1[0][:, [0, -1]] -= np.inf
        outputs1[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs1[0] > 0)):
            entitys.add((h, t, l)) #这里在调试的时候考虑一下会不会出现多个解的情况
        ent_list = []
        for h, t, l in list(entitys):
            ent_list.append({"predicate":id2en[l], "head":new_span[h][0],"tail":new_span[t][-1],"entity":text[new_span[h][0]:new_span[t][-1] + 1]})
        pred_ent.append(ent_list)
        spoes = set()
        # subjects = set(list(subjects)[0:20])
        # objects = set(list(objects)[0:20])
        entitys = set(list(entitys)[0:50])
        # for sh, st in subjects:
        #     for oh, ot in objects:
        for sh,st,sl in entitys:
            for oh,ot,ol in entitys:
                if sh!=oh and ot!=st:   #排除矩阵对角线 
                    p1s = np.where(outputs2[0][:, sh, oh] > threshold)[0]
                    p2s = np.where(outputs2[1][:, st, ot] > threshold)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        spoes.add((
                            text[new_span[sh][0]:new_span[st][-1] + 1], id2en[sl],id2schema[p],
                            text[new_span[oh][0]:new_span[ot][-1] + 1], id2en[ol]
                        ))
        spo_list = []
        for spo in list(spoes):
            spo_list.append({"predicate":spo[2], "object":spo[3], "object_type": spo[4],
                             "subject":spo[0], "subject_type":spo[1]
                             })
        pred_spo.append(spo_list)
    #测试结果
    # ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_spo,pred_spo)
    ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_ent,pred_ent)
    rel_precision,rel_recall,rel_f1 = cal_rel_percentage(true_spo,pred_spo,relptype)
    return ner_precision,ner_recall,ner_f1,rel_precision,rel_recall,rel_f1,pred_spo

#测试模型时输出指标
def gpnet2_test_printInfo2(true_entl,text_list,true_spo,true_ent,tokenizer,gpnet2,device,id2schema,id2en,relptype='strict'):
    pred_spo = []
    pred_ent = []
    for j in range(0,len(text_list)):
        text = text_list[j]
        entl = true_entl[j]
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=256)["offset_mapping"]
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text, max_length=256)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)

        scores2 = gpnet2(input_ids, attention_mask, token_type_ids)
        outputs2 = [o[0].data.cpu().numpy() for o in scores2]
        # outputs2 = scores.data.cpu().numpy()
        # subjects, objects = set(), set()
        entitys = set()
        for i in range(0,len(entl)):
            for h,t in entl[i]:
                entitys.add((h,t,i))
        # outputs1[0][:, [0, -1]] -= np.inf
        # outputs1[0][:, :, [0, -1]] -= np.inf
        # for l, h, t in zip(*np.where(outputs1[0] > 0)):
        #     entitys.add((h, t, l)) #这里在调试的时候考虑一下会不会出现多个解的情况
        ent_list = []
        for h, t, l in list(entitys):
            ent_list.append({"predicate":id2en[l], "head":new_span[h][0],"tail":new_span[t][-1],"entity":text[new_span[h][0]:new_span[t][-1] + 1]})
        pred_ent.append(ent_list)
        spoes = set()
        # subjects = set(list(subjects)[0:20])
        # objects = set(list(objects)[0:20])
        entitys = set(list(entitys)[0:50])
        # for sh, st in subjects:
        #     for oh, ot in objects:
        for sh,st,sl in entitys:
            for oh,ot,ol in entitys:
                if sh!=oh and ot!=st:   #排除矩阵对角线 
                    p1s = np.where(outputs2[0][:, sh, oh] > threshold)[0]
                    p2s = np.where(outputs2[1][:, st, ot] > threshold)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        spoes.add((
                            text[new_span[sh][0]:new_span[st][-1] + 1], id2en[sl],id2schema[p],
                            text[new_span[oh][0]:new_span[ot][-1] + 1], id2en[ol]
                        ))
        spo_list = []
        for spo in list(spoes):
            spo_list.append({"predicate":spo[2], "object":spo[3], "object_type": spo[4],
                             "subject":spo[0], "subject_type":spo[1]
                             })
        pred_spo.append(spo_list)
    #测试结果
    # ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_spo,pred_spo)
    ner_precision,ner_recall,ner_f1 = cal_ner_percentage(true_ent,pred_ent)
    rel_precision,rel_recall,rel_f1 = cal_rel_percentage(true_spo,pred_spo,relptype)
    return ner_precision,ner_recall,ner_f1,rel_precision,rel_recall,rel_f1
if __name__ == '__main__':
    pass
    