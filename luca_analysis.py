import pandas as pd
import pyLDAvis as pyLDAvis

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import numpy as np

import gensim
import pyLDAvis.gensim
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents


df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')

df_train = df = df.loc[df['productid'] == "B000DZFMEQ"]
#dfclentext = nltk.clean_html(temp)

# compile sample documents into a list
doc_set = df_train['cleanedtext']
temp = []
for i in doc_set:
    temp.append(i)

doc_set2 = temp
# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set2:
    # clean and tokenize document string
    if not isinstance(i, float):
        raw = i.lower()

    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)

print(ldamodel.print_topics(num_topics=3, num_words=3))
for i in ldamodel.show_topics(num_words=4):
    print(i[0], i[1])

# Get Per-topic word probability matrix:
K = ldamodel.num_topics

topicWordProbMat = ldamodel.print_topics(K)
print(topicWordProbMat)

for t in texts:
    vec = dictionary.doc2bow(t)
    print(ldamodel[vec])


columns = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10']
df = pd.DataFrame(columns=columns)
pd.set_option('display.width', 1000)

# 40 will be resized later to match number of words in DC
zz = np.zeros(shape=(40, K))

last_number = 0
DC = {}

for x in range(10):
    data = pd.DataFrame({columns[0]: "",
                         columns[1]: "",
                         columns[2]: "",
                         columns[3]: "",
                         columns[4]: "",
                         columns[5]: "",
                         columns[6]: "",
                         columns[7]: "",
                         columns[8]: "",
                         columns[9]: "",
                         }, index=[0])
    df = df.append(data, ignore_index=True)

for line in topicWordProbMat:

    tp, w = line
    probs = w.split("+")
    y = 0
    for pr in probs:

        a = pr.split("*")
        df.iloc[y, tp] = a[1]

        if a[1] in DC:
            zz[DC[a[1]]][tp] = a[0]
        else:
            zz[last_number][tp] = a[0]
            DC[a[1]] = last_number
            last_number = last_number + 1
        y = y + 1

print(df)
print(zz)
import matplotlib.pyplot as plt

zz = np.resize(zz, (len(DC.keys()), zz.shape[1]))

for val, key in enumerate(DC.keys()):
    plt.text(-2.5, val + 0.5, key,
             horizontalalignment='center',
             verticalalignment='center'
             )
plt.imshow(zz, cmap='hot', interpolation='nearest')
plt.show()
pyLDAvis.display(ldamodel)
