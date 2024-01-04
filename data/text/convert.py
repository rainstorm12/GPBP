
import json
#资源读取
with open('./data/text/source_data.json', 'r',encoding = 'utf-8') as f:
    text_list = [json.loads(text.rstrip()) for text in f.readlines()]

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

#convert
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

output_data = [convert(text_list[i],i+1) for i in range(0,len(text_list))]

with open('./data/text/somuut.json', 'w',encoding = 'utf-8') as file_obj:
    json.dump(output_data, file_obj, ensure_ascii=False, indent=2)