from py2neo import Node, Relationship,Graph,NodeMatcher,RelationshipMatcher,Subgraph,Path
import pandas as pd
import math
import re
import time
from tqdm import tqdm

test_graph = Graph(
            "http://localhost:7474", 
            auth=('neo4j','neo4j')
        )


cypher_query = """
MATCH (n)-[r]->(m)
RETURN n.name AS name1, type(r) AS type, m.name AS name2,head(labels(n)) As type1, head(labels(m)) As type2
"""

# 执行查询并获取结果
result = test_graph.run(cypher_query)

# 遍历结果
data_to_write = ""
for record in result:
    data_to_write += record['name1']+"("+record['type1']+")"+" ||| "+record['type']+" ||| "+record['name2']+"("+record['type2']+")"+"\n"

# 指定文件路径
file_path = "./data/text/triple.txt"

# 将字符串按行写入文件
with open(file_path, "w", encoding="utf-8") as file:
    file.write(data_to_write)

print(f"Data has been written to {file_path}")