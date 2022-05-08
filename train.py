
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
