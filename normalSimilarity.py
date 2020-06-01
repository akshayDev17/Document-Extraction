import sys
from math import log
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.cluster import cosine_distance

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

def preprocess(raw):
    word_list = word_tokenize(raw)
    stopwords_en = stopwords.words('english')
    imp_words = [word.lower() for word in word_list \
        if word not in stopwords_en]
    return imp_words

def term_freq(t1, t2):
    '''TF calculations
    score a word with a high value if it occurs frequently'''
    word_set = set(t1).union(set(t2))

    freq_t1 = FreqDist(t1)
    t1_len = len(t1)
    t1_count_dict = dict.fromkeys(word_set, 0)
    for word in t1:
        t1_count_dict[word] = freq_t1[word]/t1_len

    freq_t2 = FreqDist(t2)
    t2_len = len(t2)
    t2_count_dict = dict.fromkeys(word_set, 0)
    for word in t2:
        t2_count_dict[word] = freq_t2[word]/t2_len
    return (t1_count_dict, t2_count_dict)

def inverse_doc_freq(t1, t2):
    '''IDF calculations
    score a word low, if it occurs across multiple documents'''
    word_set = set(t1).union(set(t2))

    t12_idf_dict = dict.fromkeys(word_set, 0)
    txt_len = 2 # 2 documents, t1 and t2
    for word in t12_idf_dict.keys():
        if word in t1:
            t12_idf_dict[word] += 1
        if word in t2:
            t12_idf_dict[word] += 1

    for word, val in t12_idf_dict.items():
        t12_idf_dict[word] = 1 + log(txt_len/(float(val)))
    return t12_idf_dict

def tfidf(t1, t2):
    '''compute tfidf values for each word in the documents'''
    word_set = set(t1).union(set(t2))
    t1_tfidf_dict = dict.fromkeys(word_set, 0)
    t2_tfidf_dict = dict.fromkeys(word_set, 0)
    
    t1_tf_dict, t2_tf_dict = term_freq(t1, t2)
    t12_idf_dict = inverse_doc_freq(t1, t2)

    for word in t1:
        t1_tfidf_dict[word] = t1_tf_dict[word]*t12_idf_dict[word]

    for word in t2:
        t2_tfidf_dict[word] = t2_tf_dict[word]*t12_idf_dict[word]

    return (t1_tfidf_dict, t2_tfidf_dict)

def word_embed(t_1, t_2):
    tagged_docs = []
    doc1 = TaggedDocument(words=t_1, tags=[u'text-1'])
    tagged_docs.append(doc1)
    
    doc2 = TaggedDocument(words=t_2, tags=[u'text-2'])
    tagged_docs.append(doc2)

    model = Doc2Vec(tagged_docs, dm=0, alpha=0.025, vector_size=20\
        , min_alpha=0.0125, min_count=0)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=100)
    
    # # manually train for each epoch, with reduction in 
    # # the learning rate value for each epoch
    # for epoch in range(80):
    #     if epoch % 10 == 0:
    #         print("Current training epoch is", epoch)
    #     # model.train(tagged_docs)
    #     model.train(tagged_docs, total_examples=model.corpus_count, epochs=1)
    #     model.alpha -= 0.002    # decrease the learning rate
    #     model.min_alpha = model.alpha

    similarity_val = model.n_similarity(t_1, t_2)
    print("Similarity from embedding is", round(similarity_val*100, 3))

def similarity(text_1, text_2):
    ''' compute cosine distance between document vectors'''
    t1_tfidf_dict, t2_tfidf_dict = tfidf(text_1, text_2)
    v1, v2 = list(t1_tfidf_dict.values()), list(t2_tfidf_dict.values())
    similarity_val = 1 - cosine_distance(v1, v2)
    print("Cosine similarity from TF-IDF was", round(similarity_val*100, 3))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 normalSimilarity.py <x1> <x2>")
    else:
        f1, f2 = open(sys.argv[1], 'r'), open(sys.argv[2], 'r')
        text1, text2 = preprocess(f1.read()), preprocess(f2.read())
        word_embed(text1, text2)
        similarity(text1, text2)
        