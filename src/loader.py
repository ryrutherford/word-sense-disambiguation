'''
@author: jcheung
'''
import xml.etree.cElementTree as ET
import codecs
from wsd import run_algorithms, supervised_wsd#, get_most_frequent_sense_accuracy, find_most_common_words_semcor
from nltk.corpus import stopwords
import re

class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context
    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma.decode("utf-8"), b' '.join(self.context).decode("utf-8"), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances

def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        #print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key

def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')

"""
def find_most_common_words(dev_key, test_key, num_words):
    freq_dict = {}
    #getting the most frequent terms from the dev_key
    for value in dev_key.values():
        for lemma in value:
            match_obj = re.search(r"[a-zA-Z0-9_\-]+", lemma)
            lemma = match_obj.group(0)
            freq_dict[lemma] = 1 if lemma not in freq_dict else freq_dict[lemma] + 1

    #getting the most frequent terms from the test_key
    for value in test_key.values():
        for lemma in value:
            match_obj = re.search(r"[a-zA-Z0-9_\-]+", lemma)
            lemma = match_obj.group(0)
            freq_dict[lemma] = 1 if lemma not in freq_dict else freq_dict[lemma] + 1

    #sorting the dictionary
    freq_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}
    index = 0
    for key, val in freq_dict.items():
        if(index == num_words):
            break
        index += 1
        print((key, val))
"""

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}

    #find_most_common_words(dev_key, test_key, 15)

    #algorithms = ["wn_lesk"] #["wn_lesk", "most_frequent_wsd"]
    #run_algorithms(test_instances, test_key, algorithms)

    #find_most_common_words_semcor(100)
    #get_most_frequent_sense_accuracy()
    supervised_wsd()