"""
this file contains the code to run 4 different wsd algorithms
 - 2 implementations of Lesk's algorithm
 - 2 supervised approaches to WSD (one using MNB and one using bootstrapping and Decision Trees)
 - The implementations for Lesk's algorithm uses the  SemEval 2013 Shared Task #12 dataset
 - The implementaions for the supervised models use SemCor
 - Lesk's algorithm attempts to disambiguate each word
 - The supervised classifiers only attempt to disambiguate 1 word (I trained 12 classifiers to disambiguate 12 words)

to run the algorithms run loader.py
"""

import re
import json
import nltk
from nltk.corpus import stopwords, semcor, wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#dictionary used in methods for supervised wsd
supervised_wsd_lemmas_and_pos = {"say": "v", "make": "v", "know": "v", "take": "v", "use": "v", "find": "v", "man": "n", "time": "n", "year": "n", "day": "n", "thing": "n", "way": "n"}

#helper function to apply the porter stemmer to sentences
def stem(tokenized_sentence):
    porter = PorterStemmer()
    stemmed_tokens=[]
    for word in tokenized_sentence:
        stemmed_tokens.append(porter.stem(word))
    #we remove duplicates using the dict.fromkeys method
    return list(dict.fromkeys(stemmed_tokens))

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
def lemmatize(tokenized_sentence):
    lemmatizer = WordNetLemmatizer()
    pos_tagged = nltk.pos_tag(tokenized_sentence)
    lemmatized_sentence=[]
    for word_and_pos in pos_tagged:
        tag = wordnet_pos(word_and_pos[1])
        if(tag == None):
            lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0]))
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word_and_pos[0], wordnet_pos(word_and_pos[1])))
    #we remove duplicates using the dict.fromkeys method
    return list(dict.fromkeys(lemmatized_sentence))

#helper function to compute the overlap of a signature (definition of lemma synset) and context
#by default we don't ignore stopwords
def compute_overlap(signature, context, stop_words=[], punctuation_counts=True):
    overlap = 0
    #for each word in the signature we check if the word is in the context (and not a stopword)
    #if it is then we increment the overlap count
    for word in signature:
        #the re.match checks if the "word" is a non alpha numeric character (we don't count these as overlaps)
        if word in context and word not in stop_words and (punctuation_counts or re.match(r"\W", word) == None):
            overlap += 1
            #removing the word from the context so we don't count it again
            context = list(filter(lambda w: w != word, context))
    return overlap

#function that computes the most_frequent_wsd algorithm
def most_frequent_wsd(wsd_result, wsd, context, pre_processing=None, sw=False, punctuation_counts=True):
    #converting the lemma (word to be disambiguated) to a string
    #at this point we are just replacing _ with spaces
    lemma = wsd.lemma.decode("utf-8").lower().replace("_", " ")

    stop_words = [] if sw == False else stopwords.words("english")

    #preprocessing the context and stopwords
    if(pre_processing == "lemmatize"):
        stop_words = [] if sw == False else lemmatize(stop_words)
        context = lemmatize(context)
    elif(pre_processing == "stem"):
        stop_words = [] if sw == False else stem(stop_words)
        context = stem(context)
    
    #getting the synsets for this lemma (based on the pos that nltk.pos_tag determines)
    try:
        senses = wn.synsets(lemma, pos=wordnet_pos(nltk.pos_tag(word_tokenize(lemma))[0][1]))
        best_sense = senses[0]
    #if there were no synsets returned based on the pos we find all synsets
    except:
        #some words need the "_" but some don't for some reason
        lemma = lemma.replace(" ", "_")
        senses = wn.synsets(lemma)
        best_sense = senses[0]
    max_overlap = 0
    for sense in senses:
        #tokenizing the signature
        signature = word_tokenize(sense.definition().lower())

        #lemmatizing the signature
        if(pre_processing == "lemmatize"):
            signature = lemmatize(signature)
        elif(pre_processing == "stem"):
            signature = stem(signature)

        #computing the overlap between the signature and the context
        overlap = compute_overlap(signature, context, stop_words, punctuation_counts)
        #if the new overlap is greater than max overlap then we use this sense as our best sense
        if(overlap > max_overlap):
            max_overlap = overlap
            best_sense = sense
    wsd_result[wsd.id] = best_sense

#wordnet's lesk algorithm
def wn_lesk(wsd_result, wsd, context):
    #converting the lemma (word to be disambiguated) to a string
    lemma = wsd.lemma.decode("utf-8").lower()

    wsd_result[wsd.id] = nltk.wsd.lesk(context, lemma)

#Lesk wsd algorithm using most_frequent sense premise
def lesk(instances, algorithm, pre_processing=None, sw=False, punctuation_counts=True):
    wsd_result = {}
    for wsd in instances.values():
        #getting the context as a utf-8 string (not byte sequence))
        context = b" ".join(wsd.context).decode("utf-8").lower()
        #numbers have lemma @card@ which we don't care about
        context = re.sub("@card@", "", context)
        
        #removing underscores led to greater performance in most_frequent_wsd but worse performance in wn_lesk
        if(algorithm == "most_frequent_wsd"):
            #some words in the context will have underscores in them, we need to remove the underscores for comparison
            context = context.replace("_", " ")
        context = word_tokenize(context)
        if(algorithm == "wn_lesk"):
            wn_lesk(wsd_result, wsd, context)
        elif(algorithm == "most_frequent_wsd"):
            most_frequent_wsd(wsd_result, wsd, context, pre_processing, sw, punctuation_counts)          
    return wsd_result

#method to find the num_words most common lemmas in the semcor corpus
def find_most_common_words_semcor(num_words):
    freq_dict = {}
    #tagged_sents contains a list of sentences (lists) which are broken down into chunks (lists)
    #chunks contain semantic information
    tagged_sents = semcor.tagged_sents(tag="both")
    for sentence in tagged_sents:
        for chunk in sentence:
            if(isinstance(chunk, nltk.tree.Tree)):
                if(isinstance(chunk.label(), nltk.corpus.reader.wordnet.Lemma)):
                    #lemma has form: x.pp.d.y where x.pp.d will map to a synset for the word y
                    lemma = chunk.label()
                    #extract the y part from the lemma
                    try:
                        #some are not stored as lemmas so the lemma.name() call will fail and we'll need to do a regex operation to extract the word
                        lemma_name = lemma.name()
                    except:
                        lemma_name = re.search(r"[a-zA-Z_\-]+", lemma)
                    finally:
                        #get the pos
                        pos = wordnet_pos(chunk[0].label())
                        lemma_name = f"{lemma_name}, {pos}"
                        #update frequency count for lemma_name
                        freq_dict[lemma_name] = 1 if lemma_name not in freq_dict else freq_dict[lemma_name] + 1
    
    #sorting the frequency dict and outputting the most frequent
    freq_dict = {k: v for k, v in sorted(freq_dict.items(), key=lambda item: item[1], reverse=True)}
    index = 0
    with open("most_frequent_words.txt", "w") as fp:
        for key, val in freq_dict.items():
            if(index == num_words):
                break
            index += 1
            #print((key, val))
            fp.write(f"Word: {key}, Count: {val}\n")

#method to get baseline stats if we used the most frequent sense to disambiguate the words passed as a parameter
def get_most_frequent_sense_accuracy(words=supervised_wsd_lemmas_and_pos):
    all_freq = {}
    most_common_freq = {}

    tagged_sents = semcor.tagged_sents(tag="sem")
    for sentence in tagged_sents:
        for chunk in sentence:
            if(isinstance(chunk, nltk.tree.Tree)):
                if(isinstance(chunk[0], nltk.tree.Tree)):
                    continue
                else:
                    #lemma has form: x.pp.d.y where x.pp.d will map to a synset for the word y
                    lemma = chunk.label()
                    #extract the y part from the lemma
                    try:
                        #some are not stored as lemmas so the lemma.name() call will fail and we'll need to do a regex operation to extract the word
                        lemma_name = lemma.name()
                        sense = lemma.synset().name()
                    except:
                        lemma_name = re.search(r"[a-zA-Z_\-]+", lemma)
                        sense = lemma
                    finally:
                        if(lemma_name in words):
                            most_frequent_sense = wn.synsets(lemma_name, pos=words[lemma_name])[0].name()
                            if(sense == most_frequent_sense):
                                most_common_freq[lemma_name] = 1 if lemma_name not in most_common_freq else most_common_freq[lemma_name] + 1
                            else:
                                if(lemma_name not in most_common_freq):
                                    most_common_freq[lemma_name] = 0
                            all_freq[lemma_name] = 1 if lemma_name not in all_freq else all_freq[lemma_name] + 1
    accuracy_dict = {}
    for key in all_freq.keys():
        accuracy_dict[key] = most_common_freq[key]/all_freq[key]
    
    write_dict = {"accuracy": accuracy_dict, "total_frequency": all_freq, "most_common_freq": most_common_freq}

    with open("baseline_accuracy_supervisedwsd.json", "w") as fp:
        json.dump(write_dict, fp, indent=2)

#helper method to find context words "index" locations away from the word we're looking for
#the direction tells us whether to look up from the index or down if the first context word found is not valid
#a context word is not valid if it's punctuation (it's tree label is None)
def find_context_word(index, sentence, direction):
    context_word = "_" if index < 0 or index >= len(sentence) else sentence[index]
    
    #if the features label is None then the feature is punctuation which we don't care for
    additional_index = 1
    while(context_word != "_" and context_word.label() is None):
        if(direction == "-"):
            index = index - additional_index
            context_word = "_" if index < 0 else sentence[index]
        else:
            index = index + additional_index
            context_word = "_" if index >= len(sentence) else sentence[index]
        additional_index += 1
    
    if(context_word == "_"):
        return context_word, index

    try:
        lemma = context_word.label()
        #if the label is a lemma then we will use the lemma name as the context otherwise we'll use the value of the tree
        if(isinstance(lemma, nltk.corpus.reader.wordnet.Lemma)):
            context_word = lemma.name()
        else:
            try:
                #if this fails then that means that the label had the lemma format but was not a lemma object
                context_word = context_word[0].lower()
            except:
                #we get the lemma from the string
                context_word = re.search(r"[a-zA-Z_\-]+", lemma).group(0)
    except:
        print(context_word)
    
    return context_word.lower(), index

#helper method used by supervised learning models to extract features and labels for a given lemma in the SemCor corpus                                                   
def extract_features_and_labels_by_lemma(lemma_of_interest):
    features = []
    labels = []
    tagged_sents = semcor.tagged_sents(tag="both")
    for sentence in tagged_sents:
        index = 0
        for chunk in sentence:
            if(isinstance(chunk, nltk.tree.Tree)):
                if(isinstance(chunk.label(), nltk.corpus.reader.wordnet.Lemma)):
                    #lemma has form: x.pp.d.y where x.pp.d will map to a synset for the word y
                    lemma = chunk.label()
                    #extract the y part from the lemma
                    try:
                        #some are not stored as lemmas so the lemma.name() call will fail and we'll need to do a regex operation to extract the word
                        lemma_name = lemma.name()
                    except:
                        lemma_name = re.search(r"[a-zA-Z_\-]+", lemma)
                    finally:
                        if(lemma_name == lemma_of_interest):
                            #extract the features w-2, w-1, w+1, w+2, (w-2, w-1), (w-1, w+1), (w+1, w+2)
                            #where w is the index of the current word in the sentence
                            one_back, ob_index = find_context_word(index - 1, sentence, "-")
                            two_back, _ = find_context_word(ob_index - 1, sentence, "-")
                            one_forward, of_index = find_context_word(index + 1, sentence, "+")
                            two_forward, _ = find_context_word(of_index + 1, sentence, "+")
                            two_back_one_back = two_back + "_" + one_back
                            one_back_one_forward = one_back + "_" + one_forward
                            one_forward_two_forward = one_forward + "_" + two_forward

                            new_feature = {"w-2": two_back, "w-1": one_back, "w+1": one_forward, "w+2": two_forward, "w-2, w-1": two_back_one_back, "w-1, w+1": one_back_one_forward, "w+1, w+2": one_forward_two_forward}
                            new_label = lemma.synset().name()

                            features.append(new_feature)
                            labels.append(new_label)
                            
            index += 1
    return features, labels

#method that will train and predict for the supervised learning models. bootstrap=False runs MNB, bootstrap=True runs DT
def supervised_wsd(words=supervised_wsd_lemmas_and_pos, bootstrap=False):
    stats_dict = {}
    most_frequent_stats_dict = {}
    for lemma, pos in words.items():
        
        #extract features and labels from semcor corpus
        features, labels = extract_features_and_labels_by_lemma(lemma)
        
        #we use a pipeline to vectorize the features and then do a tfidf transformation on them
        features = Pipeline([("dict", DictVectorizer()), ("tfid", TfidfTransformer())]).fit_transform(features).toarray()

        #split data into training and test set
        context_train, context_test, senses_train, senses_test = train_test_split(features, labels, test_size=0.3, random_state=42)

        #if bootstrap is false then we run basic supervised wsd with Multinomial Naive Bayes
        if(bootstrap == False):

            #train mnb using gridsearchcv
            #parameters to be tested in GridSearchCV
            #parameters were chosen based off research on stackoverflow (and out of interest of saving computation time)
            parameters = {
                    'alpha': [1e-1, 1e-3, 1],
                    'fit_prior': [True, False]
                }
            gscv = GridSearchCV(MultinomialNB(), parameters, n_jobs=-1, verbose=10 )
            gscv = gscv.fit(context_train, senses_train)
        else:
            parameters = {
                'base_estimator__max_depth' : [1, 2, 3, 4, 5],
                'max_samples' : [0.05, 0.1, 0.2, 0.5]
            }
            
            #BaggingClassifier with the default base_estimator will use a decisiont tree
            gscv = GridSearchCV(BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42), parameters, n_jobs=-1, verbose=10)
            gscv = gscv.fit(context_train, senses_train)

        #predict on test set
        predicted_senses = gscv.predict(context_test)
        accuracy = accuracy_score(senses_test, predicted_senses)
        stats_dict[lemma] = accuracy

        #getting the baseline accuracy for if we predicted the most frequent sense
        predicted_senses_most_frequent = []
        most_frequent_sense = wn.synsets(lemma, pos=pos)[0].name()
        for _ in range(len(predicted_senses)):
            predicted_senses_most_frequent.append(most_frequent_sense)

        accuracy = accuracy_score(senses_test, predicted_senses_most_frequent)
        most_frequent_stats_dict[lemma] = accuracy

    output_dict = {"supervised_wsd_accuracy": stats_dict, "most_frequent_sense_wsd_accuracy": most_frequent_stats_dict}

    if(bootstrap == False):
        #output results to json file
        with open("supervised_wsd_accuracies.json", "w") as fp:
            json.dump(output_dict, fp, indent=2)
    else:
        #output results to json file
        with open("bootstrap_supervised_wsd_accuracies.json", "w") as fp:
            json.dump(output_dict, fp, indent=2)

#function that will calculate the accuracy based on the predicted synsets
def get_accuracy(predicted_sysnset, actual_lemmasense):
    total_predicted = 0
    total_correct = 0
    num_exceptions = 0
    incorrect_predictions = {}
    correct_predictions = {}
    for wn_id, synset in predicted_sysnset.items():
        lemmasense_key = actual_lemmasense[wn_id]
        #for each correct lemmasense key for this term we extract the corresponding synset
        for key in lemmasense_key:
            #synset_from_sense_key(key) throws a WordNetError for some keys
            try:
                synset_from_lsk = wn.synset_from_sense_key(key)
            except nltk.corpus.reader.wordnet.WordNetError:
                num_exceptions += 1
                print(f"WordNet failed to get synset from key: {key}. Number of exceptions: {num_exceptions}")
                continue
            
            both_synsets = f"{str(synset_from_lsk)}, {str(synset)}"
            if(synset_from_lsk == synset):
                total_correct += 1
                correct_predictions[both_synsets] = 1 if both_synsets not in correct_predictions else correct_predictions[both_synsets] + 1
                break
            else:
                incorrect_predictions[both_synsets] = 1 if both_synsets not in incorrect_predictions else incorrect_predictions[both_synsets] + 1
        total_predicted += 1
    accuracy = round((total_correct/total_predicted)*100, 2)

    print("Most common incorrect predictions and their actual value")
    for key, value in incorrect_predictions.items():
        if(value > 5):
            print(f"key: {key}, value: {value}")
    print()
    print("Most common correct predictions and their actual value")
    for key, value in correct_predictions.items():
        if(value > 5):
            print(f"key: {key}, value: {value}")
    print()
    return accuracy

def run_algorithms(instances, keys, algorithms):
    for algorithm in algorithms:
        """
        Most frequently used synset heuristic lesk algorithm
        """
        if(algorithm == "most_frequent_wsd"):
            with open("most_frequent_wsd.txt", "w") as fp:
                punctuation_counts_possibilities = [True]
                stopword_possibilities = [False]
                pre_processing_possibilities = ["lemmatize"]
                for punctuation_count in punctuation_counts_possibilities:
                    for sw in stopword_possibilities:
                        for pp in pre_processing_possibilities:
                            fp.write(f"#### Running {algorithm} with pre-processing: {pp}, using stopwords: {sw}, counting punctuation as part of overlap: {punctuation_count} ####\n")
                            wsd_result_no_pp = lesk(instances, algorithm, pre_processing=pp, sw=sw, punctuation_counts=punctuation_count)
                            accuracy = get_accuracy(wsd_result_no_pp, keys)
                            fp.write(f"Accuracy for {algorithm} with pre-processing: {pp}, using stopwords: {sw}, counting punctuation as part of overlap: {punctuation_count}: {accuracy}\n\n")
        elif(algorithm == "wn_lesk"):
            """
            Wordnet's Lesk algorithm
            """
            with open("wn_lesk.txt", "w") as fp:
                fp.write("#### Running wn_lesk ####\n")
                wn_lesk_result = lesk(instances, algorithm)
                accuracy = get_accuracy(wn_lesk_result, keys)
                fp.write(f"Accuracy for wn_lesk: {accuracy}\n\n")
        #multinomial naive bayes
        elif(algorithm == "MNB"):
            supervised_wsd()
        #decision tree with bootstrapping
        elif(algorithm == "DT"):
            supervised_wsd(bootstrap=True)