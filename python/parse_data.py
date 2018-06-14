'''
#parse the wikiTable dataset (json format)#
The following works has been done:
-- extract words from each json file into a list of string separately
-- filter usless data: stopwords(english), empty string, short string(len<3), numbers, non-alphabetic characters,
        html tags, and url link.
-- save parsed data of each wikiTable .json file separately into a new .txt file
'''
# ==============================================================================

import tensorflow as tf
import numpy as np
import os
import sys

import json
import re
import glob
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
stops = stopwords.words('english')

def get_all_strings_from_list(list):
    strings = []

    for obj in list:
        if not isinstance(obj, str):
            strings += get_all_strings_from_list(obj)
        else:
            strings.append(obj)

    return strings

# read data into a list of strings
def read_data(dataset):
    data = []

    for tablename in dataset:
        table = dataset[tablename]

        # get words from the what is under the keys ["title", "pgTitle", "secondTitle", "caption", "data"]
        keys = ["title", "pgTitle", "secondTitle", "caption", "data"]

        for key in keys:
            # if table does not have key, skip this key
            if not key in table:
                continue

            strings = table[key]

            data += get_all_strings_from_list([strings])

    # filter data
    #print(data)
    data = filter(data)

    return data


def filter(strings):
    data = []


    for string in strings:

        # make every string lowercase
        string = str.lower(string)


        # filter out html tag span style
        #string = re.sub(r'<span.*?/span>', '', string, re.DOTALL)
        string = re.sub(r'<span style.*?/span>', '', string)
        string = re.sub(r'<.*?>','', string)

        # filter out url
        string = re.sub(r'(?P<url>https?://[^\s]+)', '', string)

        # remove whitespace on the sides
        string = str.strip(string)

        # skip empty words
        if string == '':
            continue

        # skip words that only contain numbers

        if re.match(r'^[0-9]+$', string):
            continue

        # replace symbols with a space
        string = re.sub(r'[^a-z\.]+', ' ', string)

        # remove whitespace on the sides again
        # (the symbol replacement might have added spaces on the sides)
        string = str.strip(string)
        # skip useless strings
        if string == '' or string == '.' or string == 'none' or len(string)<3 or string in stops:
            continue

        # split on whitespace
        words = string.split(' ')
        data += words

    return data

#data = []
index=0
for filename in sorted(glob.glob('/Users/InSung/Desktop/tensor-flow/tables_redi2_1/re_tables_0007.json')):
    index += 1
    print('---','Processing ' + filename + '...')
    input = json.load(open(filename))

    #print(input)
    input_data = read_data(input)
    #data += input_data
    data = filter(input_data)
    #print(data)
    with open('/Users/InSung/repos/msc-thesis-li-deng/parsed_words/%i.txt' %index, 'w') as f:
        #print(data)
        print(len(data))

        data = ' '.join(data)

        f.write(data)
# save parsed data as a json file for later use

