#=============================================================================
#  the purpose of this script is to read embeddings of input words
#  you are able to use read_emb.py for reading term embeddings:
#       - given input a word,
#       - it will output its corresponding embedding(200-D vector in our experiment).
#
#==============================================================================

import json
import sys
import os

dictionary = json.load(open("filename"))

def get_embedding():
    def main():
        word = input("embedding for word: ")
        try:
            print(dictionary[word])
            main()

        except KeyError:
            print("word not in vocabulary! try again!")
            main()

    main()

get_embedding()