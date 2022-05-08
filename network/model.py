
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Dropout, Flatten, SimpleRNN, LSTM
#from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LayerNormalization
from gensim.models import KeyedVectors
import copy
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_model(embedding_matrix):
    #num_neurons = 32
    Embedding_layer = Embedding(15001, 300,  weights = [embedding_matrix], input_length = 180, trainable = False)
    # Embedding_layer = Embedding(15001, 300,  input_length = 180, trainable = False)
    maxlen = 180
    model = Sequential()
    #model.add(Embedding(15001, embedding_dim, input_length=maxlen))
    model.add(Embedding_layer)
    model.add(LayerNormalization())
    #model.add(Embedding_layer)
    model.add(LSTM(128)) # neuron = 128
    #model.add(SimpleRNN(num_neurons, 
    #                    input_shape = (180,300),
    #                  return_sequences = True,  # use True when the following layer is not output layer
    #   ))
    model.add(LayerNormalization())
    #model.add(keras.layers.normalization.BatchNormalization())
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.summary()

    return model


def get_emb_matrix(word2Vec_file, tokenizer, word_index):
    """
    vocab = []# vocab是足迹训练word2vec的时候用的
    train_list = []
    for review in train: 
        review = to_clean_sentence(review)
        train_list.append(review)
        tokenizer = RegexpTokenizer(r'\w+') #分词， 把每句话分词， 用逗号引起来那种， 这是给word2vector用的
        # tokenizer = RegexpTokenizer("[\w']+")
        tokens = tokenizer.tokenize(review) 
        #nocomma_tokens = " ".join(w for w in tokens)
        #print("tokens", tokens)
        vocab.append(tokens)
    print(vocab[0])
    print("vocab length",len(vocab)) #25000
    """
    # 用pretrained好的， 词库大
    #  https://www.heywhale.com/mw/project/5e05c4c42823a10036b04e1b
    #word2vec_model = gensim.models.KeyedVectors.load_word2vec_format
    #pretrained_word2v = "D:/Data/operate_Html/RNN/pretrained glove data_ word embedding/glove.6B.100d.txt"
    pretrained_word2v = KeyedVectors.load_word2vec_format(word2Vec_file, binary=True)

    print(pretrained_word2v.most_similar('man'))
    """
    embeddings_index = {}
    f = open(pretrained_word2v, "r", encoding="UTF-8")
    for line in f:
        values = line.split()
        word  = values[0]
        coefs = np.asarray(values[1:], dtype="float32") # 100 dimensions
        embeddings_index[word]= coefs #embeddings_index是词对应的维度列表
    f.close()
    """
    # 把word index的vector和 已经训练好的 对应起来
    #print("word index", word_index)
    print(type(word_index))
    #print("word_counts",tokenizer.word_counts.items())

    x = list(tokenizer.word_counts.items()) # 原先是字典，转化为list(creepers', 8),这样会有0，1位置
    sorted_ = sorted(x, key = lambda p:p[1], reverse = True) # 对list x， list有woed和频度是[0,1]位置的位置分布,取第二个值为key去sort
    print(sorted_)
    small_word_index = copy.deepcopy(word_index) #防止原来的词典也被改变了
    print("Removing less frequency words from word index")
    for item in sorted_[15000:]: # for words frequency which listed behind , pop
        small_word_index.pop(item[0]) # sorted里的item是(creepers', 8)这样的， 会一组一组都被删掉
    print("Finished")
    print(len(small_word_index))
    print(type(small_word_index))
    #print(small_word_index)# 字典只剩到20000

    # 写法2
    vocab_size = 15000
    #embedding_matrix  = np.random.uniform(size =(vocab_size+1, 300))
    embedding_matrix = np.zeros((vocab_size+1, 300))
    print("Transfering to embedding matrix")
    count = 0
    for word, index in small_word_index.items():
        try:
            word_vector = pretrained_word2v[word] #pretrained的单词，所对应的就是vector
            embedding_matrix[index] = word_vector # vector is word_index的index 所对应的值
        except:
            count += 1
            print("word: [", word,"] not in pretrained model!, use random embedding instead")
    print("Finished")
    print("count",count)
    print("Embedding matrix", embedding_matrix)

    return embedding_matrix


def get_token_wordidx_train_test(x_train, x_test):
    tokenizer = Tokenizer(num_words= 15000, filters = " ", ) #这里如果设置oov_token = "<OOV>"的话，本来0的位置就是预留的， 
    #再设置， 关联预先的embedding的时候0， 1 个词都会是全0
    tokenizer.fit_on_texts(x_train) # test list的文档已经被预先处理过，别再处理了
    #word_index = tokenizer.word_index
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(x_train)
    print(train_sequences[0])
    print(len(word_index))
    x_train = pad_sequences(train_sequences, maxlen =180)

    test_sequences = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(test_sequences, maxlen =180)

    return tokenizer,word_index,x_train,x_test

if __name__ == '__main__':
    model = get_model(1)
    print(model)
