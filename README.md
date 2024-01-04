# GPBP
This repo provides the source code & data of our paper "[GPBP: Pipeline Extraction of Entities and Relations for Construction of Urban Utility Tunnel Knowledge Graph](https://ieeexplore.ieee.org/document/10295733)" (SAFEPROCESS 2023).


## Framework Illustration

the main contributions of this paper are as follows: 

• We construct, clean, and annotate corpus datasets from maintenance manuals and official documents of urban utility tunnel, which contains 800 sentences and nearly 30000 characters. We name the datasets Standards for Operation and Maintenance of Urban Utility Tunnel (SOMUUT). 

• According to the specific application environment of utility tunnel, we proposed a Global Pointer Based Pipeline approach (GPBP) for entity and relation extraction, which has a better performance on SOMUUT than mainstream models like GPLinker

## Urban Utility Tunnel Knowledge Graph

Start service of Neo4j

```shell
neo4j console
```

Build a knowledge graph on the platform through maintenance manuals (expert experience)

```bash
python construction/construction-manual.py 
```

### our Knowledge Graph

<img src="img/graph.jpg" alt="our graph" style="zoom:50%;" />

## Model Structure

<img src="img/framework.jpg" alt="our framework" style="zoom: 50%;" />

## Other Tools

[Web demo](https://github.com/rainstorm12/pipe-graph) written by Vue framework

[Server demo](https://github.com/rainstorm12/pipesite) written by Django framework

## Citation
