from scipy import spatial
import json
import os
from operator import itemgetter
from collections import OrderedDict
import re
dictionary = json.load(open("/home/stud/lideng/emb_1st_col/1st_emb_50.json"))
test_ids = json.load(open("/home/stud/lideng/master_project/data/entity_id_test.json"))
seed_entities = json.load(open("/home/stud/lideng/gt_with_order/seed_entity_order.json"))
#rel_entity = json.load(open("/home/stud/lideng/PEE_n_exclude.json"))
rel_entity = json.load(open("/home/stud/lideng/PEE_jaccard.json"))
#rel_entity = json.load(open("/home/stud/lideng/PEE_wlm.json"))

with open("filename.tsv", "w") as f:  #\alpha = 0.0
    index = 0
    for table_id in test_ids:
        index += 1
        seed_en_1 = seed_entities[table_id][0]
        seed_emb_1 = dictionary[seed_en_1]
        rel_ens = rel_entity[table_id]["1"]
        rel_ens.update(rel_ens) # remove duplicate keys
        dic1 = dict()
        for term in rel_ens:
            element = str.lower(term) #"term" is Capital, "element" is LowerCase
            element = re.sub(r" ", "_", element)
            smi_1 = 0
            prob_1 = rel_ens[term]
            try:
                rel_emb_1 = dictionary[element]
                smi_1 = 1 - spatial.distance.cosine(seed_emb_1, rel_emb_1)
            except KeyError:
                pass
            if element == "" or element ==" ":
                continue
            else:
                smi = 0.2 * prob_1 + 0.8 * smi_1
                dic1[element] = smi
        dic1 = OrderedDict(sorted(dic1.items(), key=itemgetter(1), reverse = True))
        index2 = 0
        for k, v in dic1.items():
            index2 += 1
            f.write("\t".join([str(index), "Q0", k, str(index2), str(v), "run1st1"]) + "\n")

 
