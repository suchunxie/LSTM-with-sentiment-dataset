"""
CNN with Movie Review dataset

Reference:
Natual Language processing in action

dataset 
https://ai.stanford.edu/~amaas/data/sentiment/

预处理' word2vector
https://zhuanlan.zhihu.com/p/63852350?utm_source=wechat_session&utm_medium=social&utm_oi=743111959821430784&utm_content=first

"""
from data.utils import *
from data.visualize import *
from network.model import *
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer,PorterStemmer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 


if __name__ == '__main__':
    begin_training = True
    visullize = False
    use_adam = True

    train_path = r"C:\Users\bouseng\Downloads\aclImdb_v1\aclImdb\train"
    test_path = r"C:\Users\bouseng\Downloads\aclImdb_v1\aclImdb\test"
    word2Vec_file = r'E:\dataset\GoogleNews-vectors-negative300.bin'

    print('processing data...')
    train_df, test_df, train, test = get_train_test(train_path, test_path)
    x_train, y_train , x_test, y_test = get_xy_train_test(train_df, test_df, train, test)

    # save_word2vec(train)

    if visullize:
        show(y_train, y_test, x_train, x_test)

    if train:
        tokenizer, word_index, x_train, x_test = get_token_wordidx_train_test(x_train, x_test)
        ##TODO save as npy file
        embedding_matrix = get_emb_matrix(word2Vec_file, tokenizer, word_index)
        print("x_train", x_train.shape)
        print("x_test", x_test.shape)


        model = get_model(embedding_matrix)

        if use_adam:
            optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-3, amsgrad=False)
        else:
            optimizer = "rmsprop"

        model.compile(optimizer = optimizer, 
                    loss= "binary_crossentropy",
                    metrics = ["accuracy"])

        history = model.fit(x_train, y_train,
                batch_size = 128,
                epochs = 10,
                validation_data = (x_test, y_test), verbose =2)

        loss, accuracy = model.evaluate(x_test, y_test)
        print('Accuracy: %f' % (accuracy*100))

        plt.figure()
        plt.plot(history.history["accuracy"], label = "Train")
        plt.plot(history.history["val_accuracy"], label = "Test")
        plt.title("Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel ("Epochs")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(history.history["loss"],label="Train")
        plt.plot(history.history["val_loss"],label="Test")
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()

"""
写法1
有一个小问题是, word_index是整个训练集或者dataset的单词， 会很大，
如果要只选择比如频度最高的前2万个词的话， 需要对word index做限定,
做限定的方法：
# tokenizer时候设置的num words是15000， word index词库有87000
# 但是tokenizer的词顺序是根据频度来的，所以对word index截断就行

import copy
x = list(tokenizer.word_counts.items()) # 原先是字典，转化为list(creepers', 8),这样会有0，1位置
sorted = sorted(x, key = lambda p:p[1], reverse = True) # 对list x， list有woed和频度是[0,1]位置的位置分布,取第二个值为key去sort
small_word_index = copy.deepcopy(word_index) #防止原来的词典也被改变了
print("Removing less frequency words from word index")
from item in sorted[20000:]: # for words frequency which listed behind , pop
    small_word_index.pop(item[0]) # delete the word, index still remain???, but will not used in tokenize
print("Finished")
print(len(small_word_index))
    



写法1

embeddings_index = {}
f = open(pretrained_word2v, "r", encoding="UTF-8")
for line in f:
    values = line.split()
    word  = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word]= coefs #embeddings_index是词对应的维度列表
f.close()


#print("length of embeddings_index", embeddings_index)
num_words = 20000 # tokenizer时的设定一致
embedding_dim = 100
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < num_words:
        if embedding_vector is not None: #如果word index的词，不在预先训练的文档中，就是全0矩阵
            embedding_matrix[i]= embedding_vector
print( "embedding_matrix",embedding_matrix)
"""
