from flask import *
import pandas as pd
import numpy as np
from nltk import sent_tokenize
import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx

import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_colwidth', 200)

app = Flask(__name__)

@app.route('/get/knowledgegraph', methods=['POST'])
def getKnowledgeGraph():

    #Request parameters
    #textparagraph = request.form['textparagraph']
    textparagraph = "Natural language processing (NLP) is an interdisciplinary subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation"

    print("### START text paragraph ###\n")
    print(textparagraph)
    print("### END text paragraph ###\n")

    #Create sentences
    np_sentences = np.array(getSentences(textparagraph))

    #Convert sentences to pandas dataframe
    pd_sentences = pd.DataFrame(np_sentences, columns = ['sentences'])
    print("### START Sentences ###\n")
    print(pd_sentences.head())
    print("### END Sentences ###\n")


    #check subject and object for sample sentence
    doc = nlp("Natural language processing has its roots in the 1950s")
    print("### START Subjects/Objects ###\n")
    for tok in doc:
        print(tok.text, "...", tok.dep_)
    print("### END Subjects/Objects ###\n")

    #Check entities from a sample sentence
    print("### START Entities ###\n")
    print(get_entities("Natural language processing has its roots in the 1950s"))
    print("### END Entities ###\n")

    entity_pairs = []
    for i in tqdm(pd_sentences["sentences"]):
        entity_pairs.append(get_entities(i))

    print("### START Entity Pairs ###\n")
    print(entity_pairs[10:20])
    print("### START Entity Pairs ###\n")

    print("### START get relation ###\n")
    get_relation("Natural language processing has its roots in the 1950s")
    print("### END get relation ###\n")

    relations = [get_relation(i) for i in tqdm(pd_sentences['sentences'])]
    print(pd.Series(relations).value_counts()[:5])

    # extract subject
    source = [i[0] for i in entity_pairs]
    # extract object
    target = [i[1] for i in entity_pairs]
    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

    print("### START subject/object/relation ###\n")
    print(kg_df.head())
    print("### START subject/object/relation ###\n")

    # create a directed-graph from a dataframe

    #G=nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    #plt.figure(figsize=(12,12))
    #pos = nx.spring_layout(G)
    #nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    #plt.show()

    G=nx.from_pandas_edgelist(kg_df, "source", "target", edge_key="edge", edge_attr=True, create_using=nx.MultiDiGraph())
    plt.figure(figsize=(12,12))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, arrows=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
    plt.show()

    data = {
        'status' : "SUCCESS"
        }

    json_dump = json.dumps(data)
    return json_dump

def getSentences(text):
    return sent_tokenize(text)

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text

  return [ent1.strip(), ent2.strip()]

def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}] 

#  pattern = [{'DEP':'ROOT'},
#            {'DEP':'prep','OP':"*"},
#            {'DEP':'agent','OP':"*"},
#            {'POS':'ADJ','OP':"*"},
#            {'POS':'aux','OP':"*"},
#            {'POS':'nsubj','OP':"*"},
#            {'POS':'pcomp','OP':"*"},
#            {'POS':'compound','OP':"*"},
#            {'POS':'dobj','OP':"*"},
#            {'POS':'quantmod','OP':"*"},
#            {'POS':'pobj','OP':"*"}]

  matcher.add("matching_1", [pattern]) 

  matches = matcher(doc)
  k = len(matches) - 1

  span = doc[matches[k][1]:matches[k][2]] 

  return(span.text)

if __name__ == '__main__':
    getKnowledgeGraph()

