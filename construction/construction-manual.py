from py2neo import Node, Relationship,Graph,NodeMatcher,RelationshipMatcher,Subgraph,Path
import pandas as pd
import math
import re
import time
from tqdm import tqdm

# https://py2neo.org/v4/
# neo4j console
# 项目 无属性 
# 巡检内容 
# 检测方法 维护方法 检测周期 特殊情况

#列表查重复
def node_include(node,nodelist):
    if node in nodelist:
        return True
    return False

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

#半结构化数据处理：断句
def Brok(context):
    scl=[]
    while '[' in context and ']' in context:
        prefix_id = context.index('[')
        suffix_id = context.index(']')
        subcontext = context[prefix_id+1:suffix_id]
        if prefix_id>0:
            context = context[:prefix_id-1]+context[suffix_id+2:]
        else:
            context = context[suffix_id+2:]
        scl.append(subcontext)
    scldict={}#子项目下属集合
    if len(scl)>0:
        for subc in scl:
            sid = subc.index(':')
            sname = subc[:sid]
            scontext = subc[sid+1:]
            scontextlist = re.split('、',scontext)
            scldict[sname] = scontextlist
    context_list = re.split('、',context)
    return context_list, scldict

def main(test_graph,dataset,EntityNodedict):
    object_num = dataset.shape[0]#表格总行数

    # exceltitle = ['系统','巡查项目','Unnamed: 2','内容','方法','巡检周期']
    exceltitle = dataset.columns.values

    nodetype = ['系统','巡查项目','巡查子项目','内容','方法','巡检周期']

    #初始化relation表
    relationdict = {}
    for type in nodetype:
        relationdict[type] = []
    #内容中存在的子项目集合列表
    contextsubdict = []

    for j in range(len(exceltitle)):
        for i in range(object_num):
            et = exceltitle[j]#表头
            nt = nodetype[j]#节点类别
            data_context = dataset[et][i]#节点名称
            # 如果不是nan，去重复之后加入到节点列表中
            if not pd.isna(data_context):
                if nt == '内容':
                    node_list=[]
                    cl,scldict = Brok(data_context)
                    #存在子项目集合
                    sclnodedict = {}
                    if len(scldict)>0:
                        for csd in scldict.keys():
                            sclnodedict[csd]=[]
                            #创建节点
                            node = Node ('巡查子项目', name = csd)
                            #判断不重复
                            if not node_duplicate_checking(node,EntityNodedict['巡查子项目']):
                                #存入节点表
                                EntityNodedict['巡查子项目'].append(node)
                                #存入图谱
                                test_graph.create(node)
                            else:#重复返回对应节点,不需要存入图谱或节点表
                                node = node_duplicate_checking(node,EntityNodedict['巡查子项目'])
                            sclnodedict[csd].append(node)
                            #内容部分重复创建节点
                            for csdcontext in scldict[csd]:
                                #创建节点
                                node = Node ('内容', name = csdcontext)
                                #判断不重复
                                if not node_duplicate_checking(node,EntityNodedict['内容']):
                                    #存入节点表
                                    EntityNodedict['内容'].append(node)
                                    #存入图谱
                                    test_graph.create(node)
                                else:#重复返回对应节点,不需要存入图谱或节点表
                                    node = node_duplicate_checking(node,EntityNodedict['内容'])
                                sclnodedict[csd].append(node)
                    #内容中存在的子项目集合列表
                    #每个key下属一个list，[0]为本身节点及巡查自项目，后面为内容节点
                    contextsubdict.append(sclnodedict)
                    #内容中其他数据
                    if not cl[0] == '': 
                        for data in cl:
                            #创建节点
                            node = Node (nt, name = data)
                            #判断不重复
                            if not node_duplicate_checking(node,EntityNodedict[nt]):
                                #存入节点表
                                EntityNodedict[nt].append(node)
                                #存入图谱
                                test_graph.create(node)
                            else:#重复返回对应节点,不需要存入图谱或节点表
                                node = node_duplicate_checking(node,EntityNodedict[nt]) 
                            node_list.append(node)  
                else:
                    node_list=[]
                    cl = re.split('、',data_context)
                    for data in cl:
                        #创建节点
                        node = Node (nt, name = data)
                        #判断不重复
                        if not node_duplicate_checking(node,EntityNodedict[nt]):
                            #存入节点表
                            EntityNodedict[nt].append(node)
                            #存入图谱
                            test_graph.create(node)
                        else:#重复返回对应节点,不需要存入图谱或节点表
                            node = node_duplicate_checking(node,EntityNodedict[nt]) 
                        node_list.append(node)             
            # 本行为nan则直接将上次的node作为本行的relation节点
            # 最后建立relation表
            relationdict[nt].append(node_list)

    # print(relationdict)
    #针对relation表建立关系表
    for i in range(object_num):
        for j in range(len(nodetype)-3):#排除最后两项（方法or巡检周期）
            nt0 = nodetype[j]#父节点类型
            nt1 = nodetype[j+1]#子节点类型
            for node0 in relationdict[nt0][i]:#父节点实例
                for node1 in relationdict[nt1][i]:#子节点实例
                    relation = Relationship(node0,nt1,node1)
                    test_graph.create(relation)
            if nt0 == '巡查子项目':
                if len(contextsubdict[i])>0:#存在子项目集合
                    for csd in contextsubdict[i].keys():
                        for node0 in relationdict[nt0][i]:#父节点实例
                            relation = Relationship(node0,nt0,contextsubdict[i][csd][0])#巡查子项目->巡查子项目
                            test_graph.create(relation)
                        for csdcontext in contextsubdict[i][csd][1:]:
                            relation = Relationship(contextsubdict[i][csd][0],nt1,csdcontext)#巡查子项目->内容
                            test_graph.create(relation)


    #方法or巡检周期
    for i in range(object_num):
        nt0 = nodetype[-3]#父节点-->内容
        for node0 in relationdict[nt0][i]:#父节点实例
            for j in range(1,3):#-1 or -2
                nt1 = nodetype[-j]#子节点-->方法or巡检周期
                for node1 in relationdict[nt1][i]:#子节点实例
                    relation = Relationship(node0,nt1,node1)
                    test_graph.create(relation)
        if len(contextsubdict[i])>0:#存在子项目集合
            for csd in contextsubdict[i].keys():
                for node0 in contextsubdict[i][csd][1:]:
                    for j in range(1,3):#-1 or -2
                        nt1 = nodetype[-j]#子节点-->方法or巡检周期
                        for node1 in relationdict[nt1][i]:#子节点实例
                            relation = Relationship(node0,nt1,node1)
                            test_graph.create(relation)

if __name__ == '__main__':
    #建图
    from py2neo import Graph

    test_graph = Graph("http://localhost:7474/", auth=("neo4j", "neo4j"))  # 连接neo4j图数据库
    test_graph.delete_all()  # 删除已有的所有内容

    #表格信息读取
    #D:\design\neo4j-visal\manual.xlsx
    xlsx = pd.ExcelFile(r'./data/manual/preprocessed_manual.xlsx')

    #初始化实例化node表（存储节点）
    nodetype = ['系统','巡查项目','巡查子项目','内容','方法','巡检周期']
    EntityNodedict = {}
    for type in nodetype:
        EntityNodedict[type] = []

    # 构建一份
    # dataset = pd.read_excel(xlsx, '综合及其他')
    # main(test_graph,dataset,EntityNodedict)

    #构建多份
    sheetlist=['管廊本体','仪器仪表及自动化','综合及其他','通信软件','消防','电气供电']
    for sname in sheetlist:
        dataset = pd.read_excel(xlsx, sname)
        main(test_graph,dataset,EntityNodedict)
    
    # 可视化neo4j
    # MATCH (n) RETURN n LIMIT 10000
    # 查询父亲节点
    # MATCH (n)-[r]->(m) 
    # WHERE m.name='渗漏水' 
    # RETURN n.name AS name1,type(r) AS type,m.name AS name2 
    # 长路径查询
    # MATCH path=(m)-[r*1..4]->(n) 
    # WHERE m.name='主体结构'
    # RETURN path

    print("构建图谱结束")