'''
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs
import nltk
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
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


#helper function to apply the porter stemmer to sentences
def stem_sentence(sentence):
    porter = PorterStemmer()
    words = word_tokenize(sentence)
    stemmed_sentence=[]
    for word in words:
        stemmed_sentence.append(porter.stem(word))
        stemmed_sentence.append(" ")
    return "".join(stemmed_sentence)

#helper function to stem the stopwords list
def stem_stopwords(stop_words):
    porter = PorterStemmer()
    stemmed_stopwords=[]
    for word in stop_words:
        stemmed_stopwords.append(porter.stem(word))
    return stemmed_stopwords

#helper function to return a wordnet pos tag based on the nltk tag identified
def wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    elif nltk_tag.startswith('S'):
        return wn.ADJ_SAT
    else:
        return None

#a helper function to lemmatize the words in a sentence and return the lemmatized sentence
def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(words)
    lemmatized_sentence=[]
    for word_and_pos in pos_tagged:
        tag = wordnet_pos(word_and_pos[1])
        if(tag == None):
            lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0]))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0], wordnet_pos(word_and_pos[1])))
        lemmatized_sentence.append(" ")
    return "".join(lemmatized_sentence)

def lemmatize_stopwords(stop_words):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(stop_words)
    lemmatized_stopwords =[]
    for word_and_pos in pos_tagged:
        tag = wordnet_pos(word_and_pos[1])
        if(tag == None):
            lemmatized_stopwords.append(lemmatizer.lemmatize(word_and_pos[0]))
        else:
            lemmatized_stopwords.append(lemmatizer.lemmatize(word_and_pos[0], wordnet_pos(word_and_pos[1])))
    return lemmatized_stopwords

def compute_overlap(signature, context):
    stop_words = stopwords.words("english")
    overlap = 0
    #for each word in the signature we check if the word is in the context (and not a stopword)
    #if it is then we increment the overlap count
    for word in signature:
        #the re.match checks if the "word" is a non alpha numeric character (we don't count these as overlaps)
        if word in context and word not in stop_words and re.match("\W", word) == None:
            overlap += 1
            #removing the word from the context so we don't count it again
            context = list(filter(lambda w: w != word, context))
    return overlap

def most_frequent_wsd(instances):
    wsd_result = {}
    for wsd in instances.values():
        #getting the context as a contiguous
        context = b" ".join(wsd.context).decode("utf-8").lower()
        #converting the lemma (word to be disambiguated) to a string
        #at this point we are just replacing _ with spaces
        lemma = wsd.lemma.decode("utf-8").lower().replace("_", " ")
        #getting the synsets for this lemma (based on the pos that nltk.pos_tag determines)
        try:
            senses = wn.synsets(lemma, pos=wordnet_pos(nltk.pos_tag(word_tokenize(lemma))[0][1]))
            best_sense = senses[0]
        except:
            #some words need the "_" but some don't for some reason
            lemma = lemma.replace(" ", "_")
            senses = wn.synsets(lemma)
            best_sense = senses[0]
        max_overlap = 0
        for sense in senses:
            #tokenizing the signature
            signature = word_tokenize(sense.definition().lower())
            #numbers have lemma @card@ which we don't care about
            context = re.sub("@card@", "", context)
            #some words in the context will have underscores in them, we need to remove the underscores for comparison
            context = context.replace("_", " ")
            #computing the overlap between the signature and the context
            overlap = compute_overlap(signature, word_tokenize(context))
            if(overlap > max_overlap):
                max_overlap = overlap
                best_sense = sense
        wsd_result[wsd.id] = best_sense
    return wsd_result 

def get_accuracy(predicted_sysnset, actual_lemmasense):
    total_predicted = 0
    total_correct = 0
    for wn_id, synset in predicted_sysnset.items():
        #synset_name = synset.name()
        lemmasense_key = actual_lemmasense[wn_id]
        #lex_file_sense = []
        #for each correct lemmasense key for this term we extract the corresponding synset
        for key in lemmasense_key:
            synset_from_lsk = wn.synset_from_sense_key(key)
            if(synset_from_lsk == synset):
                total_correct += 1
                break
        total_predicted += 1
    accuracy = round((total_correct/total_predicted)*100, 2)
    return accuracy

if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    
    wsd_result = most_frequent_wsd(dev_instances)
    get_accuracy(wsd_result, dev_key)
    #print(len(dev_instances)) # number of dev instances
    #print(len(test_instances)) # number of test instances
    