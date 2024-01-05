from py2neo import Node, Relationship,Graph,NodeMatcher,RelationshipMatcher,Subgraph,Path
import pandas as pd
import math
import re
import time
from tqdm import tqdm
import json
#依赖函数
#node列表内查重函数
def node_duplicate_checking(node,node_list):
    num_attribute = len(node)
    list_attribute = list(dict(node).keys())
    for nd in node_list:
        if len(nd)==num_attribute:#检查属性数目是否相同
            flag=True#只要有属性不相同就会置位
            for attribute in list_attribute:
                if node[attribute]!=nd[attribute]:
                    flag=False
                    break
            if flag==True:#所有属性都相同说明有重复，返回重复节点
                return nd
    #如果不存在重复则返回0
    return 0
#relation列表内查重函数
def relation_duplicate_checking(relation,relation_list):
    num_attribute = len(relation)
    list_attribute = list(dict(relation).keys())
    for nd in relation_list:
        if len(nd)==num_attribute:#检查属性数目是否相同
            flag=True#只要有属性不相同就会置位
            for attribute in list_attribute:
                if relation[attribute]!=nd[attribute]:
                    flag=False
                    break
            if flag==True:#所有属性都相同说明有重复，返回重复节点
                return nd
    #如果不存在重复则返回0
    return 0
def search_graph(node_matcher,node,nodetype,nodename):
    anode = node_matcher.match(nodetype).where(name=nodename)
    return node_duplicate_checking(node,anode)

if __name__=="__main__":
    #资源读取
    with open('./data/text/somuut.json', 'r',encoding = 'utf-8') as f:
        text_list = json.load(f)
    
    #整合全部的triple
    triple = []
    for t in text_list:
        triple = triple + t["spo_list"]

    ##图谱此时已经不是空集合
    test_graph = Graph(
            "http://localhost:7474", 
            auth=('neo4j','neo4j')
        )
    # test_graph.delete_all()  # 删除已有的所有内容
    node_matcher = NodeMatcher(test_graph)
    relationship_matcher = RelationshipMatcher(test_graph)
    #建图
    for t in tqdm(triple, desc="Creating Nodes and Relationships", unit="triple"):
        node1 = Node (t['subject_type'], name = t['subject'])
        node2 = Node (t['object_type'], name = t['object'])
        
        nodegraph = search_graph(node_matcher,node1,t['subject_type'],t['subject'])
        if not nodegraph:
            test_graph.create(node1)
        else:
            node1 = nodegraph

        nodegraph = search_graph(node_matcher,node2,t['object_type'],t['object'])
        if not nodegraph:
            test_graph.create(node2)
        else:
            node2 = nodegraph
        
        if len(list(relationship_matcher.match((node2,node1), r_type=None)))==0:
            if len(list(relationship_matcher.match((node1,node2), r_type=None)))==0:
                relation = Relationship(node1,t['predicate'],node2)
                test_graph.create(relation)