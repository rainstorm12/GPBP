
import json

#example
input_data = {
    "senid": 1010,
    "text": "综合管廊主体是指构成综合管廊的承重结构体以及与综合管廊结构相连成为整体的附属建（构）筑物；",
    "spo_list": [
        {"predicate": "contain", "subject": "综合管廊主体", "subject_type": "PRO", "subject_head": 0, "subject_tail": 5, "object": "附属建（构）筑物", "object_type": "PRO", "object_head": 36, "object_tail": 43},
        {"predicate": "need", "subject": "附属建（构）筑物", "subject_type": "PRO", "subject_head": 36, "subject_tail": 43, "object": "与综合管廊结构相连成为整体", "object_type": "CON", "object_head": 22, "object_tail": 34},
        {"predicate": "need", "subject": "附属建（构）筑物", "subject_type": "PRO", "subject_head": 36, "subject_tail": 43, "object": "构成综合管廊的承重结构体", "object_type": "CON", "object_head": 8, "object_tail": 19}
    ],
    "label": {
        "PRO": {"综合管廊主体": [[0, 5]], "附属建（构）筑物": [[36, 43]]},
        "CON": {"构成综合管廊的承重结构体": [[8, 19]], "与综合管廊结构相连成为整体": [[22, 34]]}
    }
}

#convert 1 stage: format convert
def convert(input_data,senid):
    output_data = {
        "senid" : senid,
        "text": input_data["text"],
        "spo_list": [
            {
                "predicate": spo["predicate"],
                "object": spo["object"],
                "object_type": spo["object_type"],
                "subject": spo["subject"],
                "subject_type": spo["subject_type"]
            }
            for spo in input_data["spo_list"]
        ]
    }
    return output_data

# convert 2 stage: content convert by rule
import copy
class preprocess_triple():
    #预处理三元组输入
    def __init__(self) -> None:
        self.shemas = {"CON":"内容","PRO":"巡查项目","MAI":"维护方法","PAT":"巡查方法","PER":"巡查周期"}
        self.systems = ["综合及其他系统","通信软件系统","消防系统","管廊本体结构","供电及照明系统","通风系统","监控与报警系统","排水系统"]#全部的系统名称
        self.rules = [  #包含关系单独罗列
                        {"subject_type": "内容", "predicate": "contain", "object_type": "内容","rule_predicate":"包含"},
                        {"subject_type": "系统", "predicate": "contain", "object_type": "系统","rule_predicate":"子系统"},
                        {"subject_type": "系统", "predicate": "contain", "object_type": "巡查项目","rule_predicate":"巡查项目"},
                        {"subject_type": "巡查项目", "predicate": "contain", "object_type": "系统","rule_predicate":"巡查子项目"},
                        {"subject_type": "巡查项目", "predicate": "contain", "object_type": "巡查项目","rule_predicate":"巡查子项目"},
                        {"subject_type": "巡查方法", "predicate": "contain", "object_type": "巡查方法","rule_predicate":"巡查子方法"},
                        {"subject_type": "维护方法", "predicate": "contain", "object_type": "维护方法","rule_predicate":"维护子方法"},
                        #巡查项目/内容的维护方法，维护方法对应的内容/周期
                        {"subject_type": "系统", "predicate": "need", "object_type": "维护方法","rule_predicate":"维护方法"},
                        {"subject_type": "巡查项目", "predicate": "need", "object_type": "维护方法","rule_predicate":"维护方法"},
                        {"subject_type": "内容", "predicate": "need", "object_type": "维护方法","rule_predicate":"维护方法"},
                        {"subject_type": "维护方法", "predicate": "need", "object_type": "内容","rule_predicate":"维护内容"},
                        {"subject_type": "维护方法", "predicate": "need", "object_type": "巡查周期","rule_predicate":"维护周期"},
                        #巡查项目/内容的巡查方法，巡查项目/方法对应的巡查内容，巡查项目/方法/内容对应的周期
                        {"subject_type": "内容", "predicate": "need", "object_type": "巡查方法","rule_predicate":"巡查方法"},
                        {"subject_type": "系统", "predicate": "need", "object_type": "巡查方法","rule_predicate":"巡查方法"},
                        {"subject_type": "巡查项目", "predicate": "need", "object_type": "巡查方法","rule_predicate":"巡查方法"},
                        {"subject_type": "系统", "predicate": "need", "object_type": "内容","rule_predicate":"巡查内容"},
                        {"subject_type": "巡查项目", "predicate": "need", "object_type": "内容","rule_predicate":"巡查内容"},
                        {"subject_type": "巡查方法", "predicate": "need", "object_type": "内容","rule_predicate":"巡查内容"},
                        {"subject_type": "系统", "predicate": "need", "object_type": "巡查周期","rule_predicate":"巡查周期"},
                        {"subject_type": "内容", "predicate": "need", "object_type": "巡查周期","rule_predicate":"巡查周期"},
                        {"subject_type": "巡查项目", "predicate": "need", "object_type": "巡查周期","rule_predicate":"巡查周期"},
                        {"subject_type": "巡查方法", "predicate": "need", "object_type": "巡查周期","rule_predicate":"巡查周期"},
                        #同种内容/维护方法/巡查项目/巡查方法存在递进式的需要关系
                        {"subject_type": "内容", "predicate": "need", "object_type": "内容","rule_predicate":"需要"},
                        {"subject_type": "维护方法", "predicate": "need", "object_type": "维护方法","rule_predicate":"需要"},
                        {"subject_type": "巡查项目", "predicate": "need", "object_type": "巡查项目","rule_predicate":"需要"},
                        {"subject_type": "巡查方法", "predicate": "need", "object_type": "巡查方法","rule_predicate":"需要"},
                        #特殊的巡查方法需要维护方法支持，维护方法需要巡查项目支持
                        {"subject_type": "巡查方法", "predicate": "need", "object_type": "维护方法","rule_predicate":"需要"},
                        {"subject_type": "维护方法", "predicate": "need", "object_type": "巡查项目","rule_predicate":"需要"},]

    def rule_process(self,source_triple,triple_copy=False):
        if triple_copy==True:
            triple = copy.deepcopy(source_triple)
        else:
            triple = source_triple
        triple['object_type'] = self.shemas[triple['object_type']]
        triple['subject_type'] = self.shemas[triple['subject_type']]

        for s in self.systems:
            if triple['object'] == s:
                triple['object_type'] = "系统"
            if triple['subject'] == s:
                triple['subject_type'] = "系统"
        
        for rule in self.rules:
            if triple['predicate'] == rule['predicate'] and triple['subject_type']==rule['subject_type'] and triple['object_type']==rule['object_type']:
                triple['predicate'] = rule['rule_predicate']
                break
        return triple

if __name__=="__main__":
    
    #资源读取
    with open('./data/text/source_data.json', 'r',encoding = 'utf-8') as f:
        text_list = [json.loads(text.rstrip()) for text in f.readlines()]

    output_data = [convert(text_list[i],i+1) for i in range(0,len(text_list))]

    #source somuut
    with open('./data/text/source_somuut.json', 'w',encoding = 'utf-8') as file_obj:
        json.dump(output_data, file_obj, ensure_ascii=False, indent=2)

    #process somuut
    pt = preprocess_triple()
    for sen in output_data:
        for triple in sen["spo_list"]:
            pt.rule_process(triple)

    #processed somuut
    with open('./data/text/somuut.json', 'w',encoding = 'utf-8') as file_obj:
        json.dump(output_data, file_obj, ensure_ascii=False, indent=2)
    
