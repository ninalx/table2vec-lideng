"""
This script is used to generate ground truth file for Wikipedia tables,
e.g., the relevant attributes.
"""

TABLE_INDEX = "wiki_tables_index_1"

from elastic import Elastic
import json

def WP_gt(input_file=None):
    es = Elastic(TABLE_INDEX)
    dic1 = dict()
    test_table_ids = json.load(open(input_file))
    with open("seed_entity_order.json","w") as f:
        for table_id in test_table_ids:
            table_entity = []
            index =0
            entities = es.get_doc(doc_id=table_id).get("_source").get("entity_1st_col")
            for entity in entities:
                index += 1
                if index <6:         
                    entity = str.lower(entity)
                    table_entity.append(entity)
                else:
                    break
                
                dic1[table_id] = table_entity
        print(dic1)
        json.dump(dic1,f, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    WP_gt(input_file = "/home/stud/lideng/master_project/data/entity_id_test.json")
