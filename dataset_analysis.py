import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
java_path = "C:/Program Files/Java/jdk1.8.0_161/bin/java.exe"
os.environ['JAVAHOME'] = java_path


def exploratory_data_analysis(df):
    print("exploratory_data_analysis")
    print("Number of reviews:", len(df))

    # Score Distribution
    ax = plt.axes()
    sns.countplot(df.score, ax=ax)
    ax.set_title('Score Distribution')
    plt.show()

    print("Average Score: ", np.mean(df.score))
    print("Median Score: ", np.median(df.score))

    # creo un nuovo attributo opinion
    # negative = review(1-3), positive=(4-5)
    df.ix[df.score == 3, 'opinion'] = "neutral"
    df.ix[df.score > 3, 'opinion'] = "positive"
    df.ix[df.score < 3, 'opinion'] = "negative"

    ## opinion distribution
    ax = plt.axes()
    sns.countplot(df.opinion, ax=ax)
    ax.set_title('Sentiment Positive vs Negative Distribution')
    plt.show()

    ## userid distribution
    # ax = plt.axes()
    # sns.countplot(df.userid, ax=ax)
    # ax.set_title('UserId Distribution')
    # plt.show()
    #
    # ## productid distribution
    # ax = plt.axes()
    # sns.countplot(df.userid, ax=ax)
    # ax.set_title('ProductId Distribution')
    # plt.show()

    print("Proportion of positive review:", len(df[df.opinion == "positive"]) / len(df))
    print("Proportion of neutral review:", len(df[df.opinion == "neutral"]) / len(df))
    print("Proportion of negative review:", len(df[df.opinion == "negative"]) / len(df))
    # 77% of the fine food reviews are considered as positive
    # and 23% of them are considered as negative.

    reviews = df.text.values
    labels = df.opinion.values

    ### TEXT REVIEWS ###

    if df.opinion[1] == "positive":
        print("positive" + "\t" + reviews[1][:90] + "...")
    else:
        print("negative" + "\t " + reviews[1][:90] + "...")

    ### Exploratory Visualization ###
    # mi creo un vocabolario di parole positive e negative

    positive_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "positive"]
    negative_reviews = [reviews[i] for i in range(len(reviews)) if labels[i] == "negative"]

    ### positive
    cnt_positve = Counter()

    for row in positive_reviews:
        cnt_positve.update(row.split(" "))
    print("Vocabulary size for positve reviews:", len(cnt_positve.keys()))

    ### negative
    cnt_negative = Counter()

    for row in negative_reviews:
        cnt_negative.update(row.split(" "))
    print("Vocabulary size for positve reviews:", len(cnt_negative.keys()))

    cnt_total = Counter()

    for row in reviews:
        cnt_total.update(row.split(" "))

    pos_neg_ratio = Counter()
    vocab_pos_neg = (set(cnt_positve.keys())).intersection(set(cnt_negative.keys()))
    for word in vocab_pos_neg:
        if cnt_total[word] > 100:
            ratio = cnt_positve[word] / float(cnt_negative[word] + 1)
            if ratio > 1:
                pos_neg_ratio[word] = np.log(ratio)
            else:
                pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

    positive_dict = {}
    for word, cnt in pos_neg_ratio.items():
        if (cnt > 1):
            positive_dict[word] = cnt

    #### positive WORDCLOUD #####
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=positive_dict)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    ax = plt.axes()
    ax.set_title('Word Cloud with the Highest Positive/Negative Ratio')
    plt.show()

    negative_dict = {}
    for word, cnt in pos_neg_ratio.items():
        if (cnt < 1) & (cnt > 0):
            negative_dict[word] = -np.log(cnt)

    #### negative WORDCLOUD #####
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=negative_dict)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    ax = plt.axes()
    ax.set_title('Word Cloud with the Lowest Positive/Negative Ratio')
    plt.show()

    print("END EXPLORATORY ANALYSIS")


# vocabulary reduction function to reduce
# the vocabulary based on min frequency or polarity.
def vocabulary_reduction(reviews, labels, min_freq=10, polarity_cut_off=0.1):
    pos_count = Counter()
    neg_count = Counter()
    tot_count = Counter()

    for i in range(len(reviews)):
        for word in reviews[i].split():
            tot_count[word] += 1
            if labels[i] == 1:
                pos_count[word] += 1
            else:
                neg_count[word] += 1

                # Identify words with frequency greater than min_freq
    vocab_freq = []
    for word in tot_count.keys():
        if tot_count[word] > min_freq:
            vocab_freq.append(word)

            # Use polarity to reduce vocab
    pos_neg_ratio = Counter()
    vocab_pos_neg = (set(pos_count.keys())).intersection(set(neg_count.keys()))
    for word in vocab_pos_neg:
        if tot_count[word] > 100:
            ratio = pos_count[word] / float(neg_count[word] + 1)
            if ratio > 1:
                pos_neg_ratio[word] = np.log(ratio)
            else:
                pos_neg_ratio[word] = -np.log(1 / (ratio + 0.01))

    mean_ratio = np.mean(list(pos_neg_ratio.values()))

    vocab_polarity = []
    for word in pos_neg_ratio.keys():
        if (pos_neg_ratio[word] < (mean_ratio - polarity_cut_off)) or (
                pos_neg_ratio[word] > (mean_ratio + polarity_cut_off)):
            vocab_polarity.append(word)

    vocab_rm_polarity = set(pos_neg_ratio.keys()).difference(vocab_polarity)
    vocab_reduced = (set(vocab_freq)).difference(set(vocab_rm_polarity))

    reviews_cleaned = []

    for review in reviews:
        review_temp = [word for word in review.split() if word in vocab_reduced]
        reviews_cleaned.append(' '.join(review_temp))

    return reviews_cleaned


if __name__ == "__main__":
    df = pd.read_csv("Dataset/food.tsv", sep="\t", encoding='latin-1')
    # df = pd.read_csv("cleanedTextCSV.csv", sep="\t", encoding='latin-1')
    exploratory_data_analysis(df)
    # data_preprocessing(df)
    # df_train = df = df.loc[df['productid'] == "B000DZFMEQ"]
    # df_test = df[201:250]
    # # aspect_based(df_train, df_test)
    # aspect2(df_train, df_test)
