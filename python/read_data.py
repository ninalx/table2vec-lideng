# *======================================================================*
#  *read all parsed files into one single file for training embeddings*
# ---------------------------------------------------
#   *python 3.6.4*
# *======================================================================*

import glob
def read_data(filename):
  """Extract the data as a list of words."""
  with open(filename, 'r') as f:
    data = f.read()
  #print(data)
  #data = json.loads(json_data) #for loading .json file
  return data
vocabulary = ''
for filename in sorted(glob.glob('filenames')):
    print('---', 'Processing ' + filename + '...')
    input_data = read_data(filename)
    if filename == "parsed_entities/1504.txt":
        print(input_data)
    vocabulary += input_data

with open('filename', 'w') as f:
    #print(vocabulary)
    print(len(vocabulary))
    f.write(vocabulary)