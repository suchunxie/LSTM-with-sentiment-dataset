import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from .clean import *

def process_review_for_train(train):
    vocab = []
    train_list = []
    for review in train:
        review = to_clean_sentence_for_word2vector(review, vocab)
        train_list.append(review)
    return train_list, vocab

def process_review_for_test(test):
    test_list = []
    for review in test:
        review = to_clean_sentence(review)
        test_list.append(review)
    return test_list


def get_xy_train_test(train_df, test_df, train, test):
    train_list, _ = process_review_for_train(train)
    test_list = process_review_for_test(test)

    label = train_df.iloc[:, -1].values.tolist()  # labels = df["Label"],tolist()
    test_label = test_df.iloc[:, -1].values.tolist()  # labels = df["Label"],tolist()

    x_train= np.array(train_list)
    y_train = np.array(label)
    x_test  = np.array(test_list)
    y_test = np.array(test_label)

    return x_train, y_train , x_test, y_test


def save_word2vec(train):
    _, vocab = process_review_for_train(train)
    dimension_size = 100
    model = Word2Vec(vocab, vector_size= dimension_size, workers = 4, min_count= 1, window = 3, max_vocab_size= 15000 )
    #model.save("word2vec.model")
    #model = Word2Vec(vocab1, vector_size= dimension_size, workers = 4, min_count= 1, window = 3, max_vocab_size= 15000 )

    print(model.wv.most_similar("good"))
    print(model.wv.most_similar("man"))

    #print(model.wv.index_to_key)
    print(len(model.wv.index_to_key))
    print( model.wv["good"]) # 查看具体某个词的100维向量表示
    model.wv.save_word2vec_format("2_Myword2vec.txt", binary = False)


def show(y_train, y_test, x_train, x_test):
    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print(x_train.shape)

    unique,counts = np.unique(y_train,return_counts=True)
    print("Y train distribution: ", dict(zip(unique,counts)))

    unique,counts = np.unique(y_test,return_counts=True)
    print("Y test distribution: ", dict(zip(unique,counts)))

    plt.figure()
    sns.countplot(y_train)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Y Train")
    plt.show()

    plt.figure()
    sns.countplot(y_test)
    plt.xlabel("Classes")
    plt.ylabel("Frequency")
    plt.title("Y Test")
    plt.show()