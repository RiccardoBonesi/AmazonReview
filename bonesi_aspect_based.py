from pyLDAvis import sklearn

from dataset_utils import *

import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import seaborn as sns
from gensim import corpora
from spacy.lang.en import English
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib
import warnings
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import os
from sklearn.metrics import precision_recall_fscore_support as score
import itertools


import sklearn
import sklearn.metrics
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyLDAvis.gensim
from sklearn import svm
import seaborn as sns
from gensim import corpora
from spacy.lang.en import English
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud,STOPWORDS
import matplotlib

from nltk.stem.wordnet import WordNetLemmatizer




warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

parser = English()

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
    # print([token for token in tokens])
    tokens = [token for token in tokens if len(token) > 4]

    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def join_bigram(l):
    return " ".join([i.split()[0] for i in l])

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = gensim.models.ldamulticore.LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                                                     iterations=500)
        lm_list.append(lm)
        cm = gensim.models.ldamodel.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list, c_v


def reviews_sentiment():
    try:
        df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    except:
        df = generate_df()

    df = df.dropna()

    stop = stopwords.words('english')
    # df1 = df["cleanedtext"].str.lower().str.split().combine_first(pd.Series([[]], index=df.index))

    for index, row in df.iterrows():
        word_tokens = word_tokenize(row.cleanedtext)

        filtered_sentence = [w for w in word_tokens if not w in stop]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop:
                filtered_sentence.append(w)

        # print(word_tokens)
        # print(filtered_sentence)

        df.set_value(index, 'cleanedtext', " ".join(filtered_sentence))

    sentiment_scores = list()
    i = 0
    for sentence in df.cleanedtext:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        # print(sentence + ": POLARITY=" + str(line.sentiment.polarity))

    # df['polarity'] = sentiment_scores
    # normalized_polarity = 2*(df['polarity'] - df['polarity'].min()) / (df['polarity'].max() - df['polarity'].min())-1
    # normalized_score = 2*(df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())-1
    # sns.distplot(normalized_polarity)
    # sns.distplot(normalized_score)
    # plt.show()

    # PLOT POSITIVE
    normalized_polarity = df[df['PosNeg'] == 'positive'].polarity
    normalized_score = (df[df['PosNeg'] == 'positive'].score - df[df['PosNeg'] == 'positive'].score.min()) / (
            df[df['PosNeg'] == 'positive'].score.max() - df[df['PosNeg'] == 'positive'].score.min()) + 0.5

    sns.distplot(normalized_polarity, kde=False)
    sns.distplot(normalized_score, kde=False)
    #
    normalized_polarity = df[df['PosNeg'] == 'negative'].polarity
    normalized_score = (df[df['PosNeg'] == 'negative'].score - df[df['PosNeg'] == 'negative'].score.min()) / (
            df[df['PosNeg'] == 'negative'].score.max() - df[df['PosNeg'] == 'negative'].score.min()) - 1

    sns.distplot(normalized_polarity, kde=False)
    sns.distplot(normalized_score, kde=False)

    plt.show()
    # TODO FINE parte negativa e positiva, qua sotto correlazione
    print(np.corrcoef(df.score, df.polarity))
    matplotlib.style.use('ggplot')

    plt.scatter(df.score, df.polarity)
    plt.show()
    # train, test = train_test_split(df, test_size=0.1)
    # train_pos = train[train['sentiment'] == 'positive']
    # train_pos = train_pos['text']
    # train_neg = train[train['sentiment'] == 'negative']
    # train_neg = train_neg['text']


def generate_topic_wordclouds(NUM_TOPICS, ldamodel, productId, productList):
    if NUM_TOPICS == 1:
        index = 111
    elif NUM_TOPICS == 2:
        index = 121
    elif NUM_TOPICS == 3:
        index = 311
    elif NUM_TOPICS == 4:
        index = 221
    elif NUM_TOPICS == 5:
        index = 321
    elif NUM_TOPICS == 6:
        index = 231
    elif NUM_TOPICS == 7:
        index = 241
    elif NUM_TOPICS == 8:
        index = 241
    elif NUM_TOPICS == 9:
        index = 331
    elif NUM_TOPICS == 10:
        index = 251

    fig = plt.figure(figsize=(60, 30))

    for t in range(NUM_TOPICS):
        ax = plt.subplot(index)
        wordcloud = WordCloud(width=800, height=400).generate(ldamodel.print_topic(t, 10))
        ax.imshow(wordcloud, aspect="equal")
        ax.axis("off")
        index += 1

    plt.suptitle(productList[productId])
    plt.tight_layout(pad=0)
    plt.show()



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def classification_train_test(df, on_update=None):
    stop = stopwords.words('english')
    from sklearn.model_selection import train_test_split

    x =10

    y = on_update(x)

    for index, row in df.iterrows():
        word_tokens = word_tokenize(row.cleanedtext)

        filtered_sentence = [w for w in word_tokens if not w in stop]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop:
                filtered_sentence.append(w)


        x = x+0.0003
        on_update(x)


        # print(word_tokens)
        # print(filtered_sentence)

        df.set_value(index, 'cleanedtext', " ".join(filtered_sentence))



    sentiment_scores = list()
    i = 0

    on_update(25)

    for sentence in df.cleanedtext:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        # print(sentence + ": POLARITY=" + str(line.sentiment.polarity))

    on_update(30)

    df['polarity'] = sentiment_scores
    df['positive'] = df.PosNeg.apply(lambda x: 1 if x == 'positive' else 0)
    X_train, X_test, y_train, y_test = train_test_split(df['cleanedtext'], df['positive'], random_state=0)
    # print('X_train first entry: \n\n', X_train.first)
    print('\n\nX_train shape: ', X_train.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df['cleanedtext'], df['positive'], random_state=0)
    # print('X_train first entry: \n\n', X_train[0])
    class_names = df['positive']
    print('\n\nX_train shape: ', X_train.shape)
    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer().fit(X_train)
    vect
    vect.get_feature_names()[::2000]

    on_update(35)

    len(vect.get_feature_names())
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized
    X_train_vectorized.toarray()
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    from sklearn.metrics import roc_auc_score
    predictions = model.predict(vect.transform(X_test))
    from sklearn.metrics import confusion_matrix
    precision, recall, fscore, support = score(y_test, predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
    print("accuracy: " + str(accuracy))
    print('AUC: ', roc_auc_score(y_test, predictions))
    conf_mat_a = sklearn.metrics.confusion_matrix(y_test, predictions)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(y_test)))),
        normalize=True
    )

    on_update(40)

    plt.savefig('confusionMatrix.png')
    plt.show()
    feature_names = np.array(vect.get_feature_names())
    sorted_coef_index = model.coef_[0].argsort()
    print('Smallest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
    print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    len(vect.get_feature_names())
    X_train_vectorized = vect.transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    precision, recall, fscore, support = score(y_test, predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
    print("accuracy: " + str(accuracy))
    print('AUC: ', roc_auc_score(y_test, predictions))
    conf_mat_a = sklearn.metrics.confusion_matrix(y_test, predictions)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(y_test)))),
        normalize=True
    )
    plt.savefig('confusionMatrix.png')
    plt.show()
    print('AUC: ', roc_auc_score(y_test, predictions))
    feature_names = np.array(vect.get_feature_names())
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    print('Smallest Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
    print('Largest Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))
    vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    len(vect.get_feature_names())
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    precision, recall, fscore, support = score(y_test, predictions)
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
    print("accuracy: " + str(accuracy))
    print('AUC: ', roc_auc_score(y_test, predictions))

    on_update(45)

    conf_mat_a = sklearn.metrics.confusion_matrix(y_test, predictions)
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    plot_confusion_matrix(
        conf_mat_a,
        list(map(str, range(max(y_test)))),
        normalize=True
    )
    plt.savefig('confusionMatrix.png')
    plt.show()
    print('AUC: ', roc_auc_score(y_test, predictions))
    feature_names = np.array(vect.get_feature_names())
    sorted_coef_index = model.coef_[0].argsort()
    print('Smallest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
    print('Largest Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))
    print("ciao")


    on_update(50)


def polarity_score_confronto(df, on_update = 50):
    stop = stopwords.words('english')
    # df1 = df["cleanedtext"].str.lower().str.split().combine_first(pd.Series([[]], index=df.index))

    on_update(55)

    for index, row in df.iterrows():
        word_tokens = word_tokenize(row.cleanedtext)

        filtered_sentence = [w for w in word_tokens if not w in stop]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop:
                filtered_sentence.append(w)

        # print(word_tokens)
        # print(filtered_sentence)

        df.set_value(index, 'cleanedtext', " ".join(filtered_sentence))
    sentiment_scores = list()

    on_update(65)


    i = 0
    for sentence in df.cleanedtext:
        line = TextBlob(sentence)
        sentiment_scores.append(line.sentiment.polarity)
        # print(sentence + ": POLARITY=" + str(line.sentiment.polarity))
    df['polarity'] = sentiment_scores
    normalized_polarity = 2 * (df['polarity'] - df['polarity'].min()) / (
                df['polarity'].max() - df['polarity'].min()) - 1
    normalized_score = 2 * (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min()) - 1
    sns.distplot(normalized_polarity)
    sns.distplot(normalized_score)

    on_update(70)

    plt.show()
    # PLOT POSITIVE
    normalized_polarity = df[df['PosNeg'] == 'positive'].polarity
    normalized_score = (df[df['PosNeg'] == 'positive'].score - df[df['PosNeg'] == 'positive'].score.min()) / (
            df[df['PosNeg'] == 'positive'].score.max() - df[df['PosNeg'] == 'positive'].score.min()) + 0.5
    sns.distplot(normalized_polarity, kde=False)
    sns.distplot(normalized_score, kde=False)
    #

    on_update(80)

    normalized_polarity = df[df['PosNeg'] == 'negative'].polarity
    normalized_score = (df[df['PosNeg'] == 'negative'].score - df[df['PosNeg'] == 'negative'].score.min()) / (
            df[df['PosNeg'] == 'negative'].score.max() - df[df['PosNeg'] == 'negative'].score.min()) - 1
    sns.distplot(normalized_polarity, kde=False)
    sns.distplot(normalized_score, kde=False)
    plt.show()

    on_update(90)

    # TODO FINE parte negativa e positiva, qua sotto correlazione
    print(np.corrcoef(df.score, df.polarity))
    matplotlib.style.use('ggplot')
    plt.scatter(df.score, df.polarity)
    plt.show()
    df.reset_index()

    on_update(100)

    return stop



def reviews_sentiment(**parameters):
    try:
        df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    except:
        df = generate_df()

    df = df.dropna()

    classification_train_test(df, **parameters)
    polarity_score_confronto(df, **parameters)





def reviews_absa(productId, on_update=None):
    # https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

    # provo ad importare il df o lo genero
    try:
        df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    except:
        df = generate_df()

    df = df.dropna()

    # aggiorna il valore della progress bar
    on_update(5)

    # B002QWP89S    629
    # B007M83302    564
    # B0013NUGDE    564
    # B000KV61FC    554
    # B000PDY3P0    486
    # B006N3IG4K    455
    # B003VXFK44    455
    # B001LG945O    347
    # B001LGGH40    338
    # B004ZIER34    330

    productList = ["B002QWP89S", "B007M83302", "B0013NUGDE", "B000KV61FC", "B000PDY3P0", "B006N3IG4K", "B003VXFK44",
                   "B001LG945O", "B001LGGH40", "B004ZIER34" , "B00141UC9I", "B001AJ1ULS" ,"B000KV61FC"]

    df = df.loc[df['productid'] == productList[productId]]

    on_update(10)

    reviews = df.cleanedtext.values

    # monogrammi
    # text_data = []
    # for r in reviews:
    #     tokens = prepare_text_for_lda(r)
    #     # print(tokens)
    #     text_data.append(tokens)

    # birammi
    text_data = []
    for r in reviews:
        tokens = prepare_text_for_lda(r)
        print(tokens)
        bigram = list(nltk.bigrams(tokens))
        tokens = []
        for i in bigram:
            tokens.append((''.join([w + ' ' for w in i])).strip())
        text_data.append(tokens)

    wholetext = list(itertools.chain.from_iterable(text_data))
    text = nltk.Text(wholetext)
    # Calculate Frequency distribution
    freq = nltk.FreqDist(text)
    metadb = df.shape[0]*0.4
    # Print and plot most common words
    lenfreq = freq.most_common(20)
    freqword = []
    for i in lenfreq:
        if i[1] > metadb:
            freqword.append(i[0])

    freq.plot(10)

    # birammi
    # text_data = []
    # for r in reviews:
    #     tokens = prepare_text_for_lda(r)
    #     print(tokens)
    #     bigram = list(nltk.bigrams(tokens))
    #     tokens = []
    #     for i in bigram:
    #         tokens.append((''.join([w + ' ' for w in i])).strip())
    #     text_data.append(tokens)

    # LDA with Gensim
    # First, we are creating a dictionary from the data,
    # then convert to bag-of-words corpus and save the dictionary and corpus for future use.

    on_update(20)

    dictionary = corpora.Dictionary(text_data)
    # dictionary.filter_n_most_frequent(len(freqword))
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    # pickle.dump(corpus, open('corpus.pkl', 'wb'))
    # dictionary.save('dictionary.gensim')

    on_update(30)

    # Finding out the optimal number of topics
    np.random.seed(50)
    lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=text_data, limit=10)
    max_value = max(c_v)
    max_index = c_v.index(max_value)
    NUM_TOPICS = max_index + 1
    # NUM_TOPICS = 4

    print("NUM TOPICS: {}".format(NUM_TOPICS))

    on_update(50)

    # creo il modello con il NUM_TOPICS ottimale
    ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, num_topics=NUM_TOPICS, id2word=dictionary,
                                                       iterations=500)

    on_update(70)

    # ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=6)

    # calcolo coherence value
    # value = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=text_data, dictionary=dictionary,
    #                                                     coherence='c_v')
    # coherence_lda = value.get_coherence()
    # print('\nCoherence Score: ', coherence_lda)

    x = ldamodel.show_topics(num_topics=NUM_TOPICS, num_words=15, formatted=False)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    # print(ldamodel.print_topic(2, 100))

    # Compute Coherence Score using c_v
    r_list=[]
    prob_list=[]
    topic_list=[]
    for r in text_data:
        bow = dictionary.doc2bow(r)
        t = ldamodel.get_document_topics(bow)
        maxval = list(max(t, key=lambda i: i[1]))
        minval = list(min(t, key=lambda i: i[1]))
        print(t)
        print(maxval)
        print(minval)
        if(maxval[1]==minval[1]):
            maxval[0] = 0
        else:
            maxval[0] = maxval[0]+1
        r_list.append(r)
        prob_list.append(maxval[1])
        topic_list.append(maxval[0])

    #TODO SENTIMENT PER TOPIC A DATAFRAME DFFINAL
    on_update(80)

    df_final = pd.DataFrame(data={'review':r_list,'probability':prob_list,'topic_no':topic_list})

    for current_topic in df_final.topic_no.unique():

        text_reviews = [join_bigram(i) for i in df_final.loc[df_final['topic_no'] == current_topic].review]
        sentiment_scores = list()
        for current_topic_reviews in text_reviews:
            # print(current_topic_reviews)
            if (current_topic_reviews != ''):
                line = TextBlob(current_topic_reviews)
                sentiment_scores.append(line.sentiment.polarity)
                # print(current_topic_reviews + ": POLARITY=" + str(line.sentiment.polarity))
        #TODO per bonesi: qui ci sono le polarity di ogni topic: la prima polarity è quella generale
        print(np.mean(sentiment_scores))
        print("Current Topic = {}".format(current_topic))
    # from IPython import embed; embed()
    # text_reviews = [join_bigram(i) for i in df_final.review]
    # calcolo la polarity del topic
    # sentiment_scores = list()
    # for topic, words in topics_words:
    #     print(" ".join(words))
    #     line = TextBlob(" ".join(words))
    #     sentiment_scores.append(line.sentiment.polarity)
    #     print(" ".join(words) + ": POLARITY=" + str(line.sentiment.polarity))

    # calcolo la polarity del topic
    # sentiment_scores = list()
    # for topic, words in topics_words:
    #     print(" ".join(words))
    #     line = TextBlob(" ".join(words))
    #     sentiment_scores.append(line.sentiment.polarity)
    #     # print(" ".join(words) + ": POLARITY=" + str(line.sentiment.polarity))

    generate_topic_wordclouds(NUM_TOPICS, ldamodel, productId, productList)

    print("TOPICS")
    print(ldamodel.print_topics(num_topics=NUM_TOPICS, num_words=3))

    # dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    # corpus = pickle.load(open('corpus.pkl', 'rb'))
    # lda10 = gensim.models.ldamodel.LdaModel.load('model5.gensim')
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=True)

    on_update(100)

    # plot lda
    pyLDAvis.show(lda_display)

    # print("saving LDA...")
    # pyLDAvis.save_html(lda_display10, 'LDA/lda_display10' + productId )
    # print("LDA saved: " + productId)

    print("END ABSA")


if __name__ == "__main__":
    # reviews_absa(0)

    reviews_sentiment()

    # for a in range(9):
    #     reviews_absa(a)
    #
    # reviews_sentiment()
