
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
        self.shemas = {"CON":"内容","PRO":"巡查项目","MAI":"维护方法","PAT":"巡检方法","PER":"巡检周期"}
        self.systems = ["综合及其他系统","通信软件系统","消防系统","管廊本体结构","供电及照明系统","通风系统","监控与报警系统","排水系统"]
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

        if triple['predicate'] == 'contain' and triple['object_type']=="内容" and triple['subject_type']=="内容":
            triple['predicate'] = '子内容'
        elif triple['predicate'] == 'contain' and triple['object_type']=="巡检项目" and triple['subject_type']=="巡检项目":
            triple['predicate'] = '巡检子项目'
        elif triple['predicate'] == 'contain' and triple['object_type']=="巡检方法" and triple['subject_type']=="巡检方法":
            triple['predicate'] = '巡检子方法'
        elif triple['predicate'] == 'contain' and triple['object_type']=="维护方法" and triple['subject_type']=="维护方法":
            triple['predicate'] = '维护子方法'
        elif triple['predicate'] == 'contain' and triple['object_type']=="系统" and triple['subject_type']=="系统":
            triple['predicate'] = '子系统'
        else:
            triple['predicate'] = triple['object_type']
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
    
