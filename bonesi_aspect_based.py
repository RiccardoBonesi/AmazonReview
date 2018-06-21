import pandas as pd
from gensim import corpora
import pickle
import gensim
import pyLDAvis.gensim

from spacy.lang.en import English
parser = English()

import nltk

# nltk.download('wordnet')
from nltk.corpus import wordnet as wn


import os
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path





def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens



# nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))



def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens




def aspect2(df):
    # https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

    reviews = df.text.values

    text_data = []
    for r in reviews:
        tokens = prepare_text_for_lda(r)
        text_data.append(tokens)

    # LDA with Gensim
    # First, we are creating a dictionary from the data,
    # then convert to bag-of-words corpus and save the dictionary and corpus for future use.
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


    # We are asking LDA to find 20 topics in the data
    # NUM_TOPICS = 20
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    # ldamodel.save('model5.gensim')
    # topics = ldamodel.print_topics(num_words=4)
    # for topic in topics:
    #     print(topic)

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

    lda10 = gensim.models.ldamodel.LdaModel.load('model5.gensim')
    lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
    # pyLDAvis.show(lda_display10)
    pyLDAvis.save_html(lda_display10, 'lda_display10')



    print("END ASPECT2")




if __name__ == "__main__":
    # df = pd.read_csv("Dataset/food.tsv", sep="\t", encoding='latin-1')
    df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    df = df.loc[df['productid'] == "B000DZFMEQ"]
    aspect2(df)