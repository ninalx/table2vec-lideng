"""
This script is used to generate ground truth file for Wikipedia tables,
e.g., the relevant attributes.
"""

TABLE_INDEX = "wiki_tables_index_1"

from elastic import Elastic
import json

def WP_gt(input_file=None):
    es = Elastic(TABLE_INDEX)
    test_table_ids = json.load(open(input_file))
    seed_entity = json.load(open("seed_file"))
    
    with open("gt_file","w") as f:
        index = 0
        for table_id in test_table_ids:
            
            entities = es.get_doc(doc_id=table_id).get("_source").get("category_on_elasticsearch")
            index += 1 
            for entity in set(entities):
                
                entity = str.lower(entity)
                
                if entity in [seed_entity[table_id][0], seed_entity[table_id][1], seed_entity[table_id][2], seed_entity[table_id][3],seed_entity[table_id][4]]:
                    print(index, entity)
                    continue
                else:
                    f.write("\t".join([str(index), "0", entity, "1"]) + "\n")

if __name__ == "__main__":
    WP_gt(input_file = "file path for test table ids")
