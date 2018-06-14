"""
This is used to index tables in tables_redi2
 @ elasticsearch: 2.4.6

"""
from elastic import Elastic
import os
import json

log = {}

def table_index():
    test_file = open("data/entity_id_test.json", "r")
    test_id = json.load(test_file)
    validation_file = open("data/table_id_validation.json", "r")
    validation_id = json.load(validation_file)
    index_name = "wiki_tables_index_1"

    mappings = {
        "type": Elastic.notanalyzed_field(),
        "entity": Elastic.notanalyzed_field(),
        "entity_1st_col": Elastic.analyzed_field(),
        "secondColumn": Elastic.analyzed_field(),
        "data": Elastic.analyzed_field(),
        "caption": Elastic.analyzed_field(),
        "headings": Elastic.analyzed_field(),
        "pgTitle": Elastic.analyzed_field(),
        "secondTitle": Elastic.analyzed_field(), # Elastic.analyzed_field() contains textual data
        "numCols": Elastic.notanalyzed_field(),
        "numDataRows": Elastic.notanalyzed_field(),
        "numHeaderRows": Elastic.notanalyzed_field(),
        "numericColumns": Elastic.notanalyzed_field(),
        "catchall": Elastic.analyzed_field(),  # Elastic.FIELD_ELASTIC_CATCHALL

    }

    elastic = Elastic(index_name)
    elastic.create_index(mappings, force=True)

    maindir = "/home/stud/lideng/master_project/tables_redi2_1"
    count = 1
    type_count = 1
    type_count2 = 1

    for d in os.listdir(maindir):
        print("filename ------ ",d)
        if d == '.DS_Store':
            continue
        docs = {}
        inpath = os.path.join(maindir, d)
        infile = open(inpath, "r")
        tables = json.load(infile)
        for key in tables.keys():
            if key in test_id:
                type = "test"
                type_count += 1
            elif key in validation_id:
                type = "validation"
                type_count2 += 1
            else:
                type = "all"
            table = tables[key]
            first_column = []
            second_column = []
            all_entities = []

            caption = table["caption"]
            headings = label_replace(table["title"])
            data = table["data"]
            pgTitle = table["pgTitle"]
            secondTitle = table["secondTitle"]
            #pcs = caption + " " + pgTitle + " " + secondTitle
            numCols = table["numCols"]
            numDataRows = table["numDataRows"]
            numHeaderRows = table["numHeaderRows"]
            numericColumns = table["numericColumns"]
            all = caption + " " + json.dumps(headings) + json.dumps(data) + ' ' + pgTitle + ' ' + secondTitle

#            cell_entities = [table["title"].replace(" ","_")]

            for row in data:
                for element in row:
                    if "[" in element and "|" in element and "]" in element:
                        en = element.split("[")[1].split("|")[0]
                        all_entities.append(en)

            for element in table["title"]:
                if "[" in element and "|" in element and "]" in element:
                    en = element.split("[")[1].split("|")[0]
                    all_entities.append(en)

            for element in table["pgTitle"]:
                if "[" in element and "|" in element and "]" in element:
                    en = element.split("[")[1].split("|")[0]
                    all_entities.append(en)

            for row in data:
                if len(row) == 0:
                    continue
                if "[" in row[0] and "|" in row[0] and "]" in row[0]:
                    en_1st = row[0].split("[")[1].split("|")[0]
                    first_column.append(en_1st)
                try:
                    second_column.append(row[1])
                except:
                    second_column.append("")

            docs[key] = {"type": type, "entity": all_entities, "data": data,
                         "caption": caption, "headings": headings, "pgTitle": pgTitle,  "secondTitle": secondTitle,
                         "numCols": numCols, "entity_1st_col":first_column, "secondColumn":second_column,
                         "numDataRows": numDataRows, "numHeaderRows": numHeaderRows, "numericColumns": numericColumns,
                         "catchall": all}

        try:
            elastic.add_docs_bulk(docs)
            print("-------- --------", len(docs))
            docs = {}
        except:
            for key, value in docs.items():
                try:
                    elastic.add_doc(key, value)
                except:
                    print("*****************")
                    log[key] = value
            docs = {}
        print("--------", count, "--------", len(docs))
        count += 1
    f = open("indexed_table.json", "w")
    json.dump(log, f, ensure_ascii=False, indent=2)


def label_replace(caption):
    caption_return = []
    for cap in caption:
        if "[" in cap and "|" in cap and "]" in cap:
            entity_str = cap.split("|")[1].split("]")[0]
        elif "*" in cap:
            entity_str = cap.replace("*", "")
        else:
            entity_str = cap
        caption_return.append(entity_str)
    return caption_return


if __name__ == "__main__":
    table_index()
    # caption = [ "Athlete", "Country", "1", "2", "3", "4", "Total", "Notes" ]
    # print(caption_replace(caption))
