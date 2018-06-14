import json
import os
import numpy as np

bs = json.load(open("baselines_map_mrr.json"))
bs_e2v = json.load(open("combined_map_mrr.json"))
table = json.load(open("table_ndcg_cut.json"))
single = ["BL1", "BL2", "BL3"]#, "entity"]
row_2 = ["5"]
single_3 = ["map"] #"recip_rank"
double = ["BL1+t2vE*", "BL2+t2vE*", "BL3+t2vE*"] #, "column+l"]
double_3 = ["map"] 
col_2 = ["1", "2","3"]
table_1 = ["b+word2vec(table)", "b+entity2vec(table)", "b+graph2vec", "baseline", "b+word2vec"]
table_2 = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_15", "ndcg_cut_20"]
entity = bs["t2vE*"]["5"]["map"]
entity = list(map(float, entity))
entity =[entity,entity,entity]
with open("filename", "w") as f:
    r1 = []
    r2 = []
    for term in double:
        print("double",term)
        for iterm in row_2:
            for element in double_3:
                data = bs_e2v[term][iterm][element]
                data = list(map(float, data))
                r2.append(data)
    r2 = np.array(r2)
    r = r2 - entity
    r = r.tolist()
    json.dump(r, f, separators=(',', ':'), indent=4)

   
