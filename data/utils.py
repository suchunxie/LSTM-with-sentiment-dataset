import os
import glob
import pandas as pd

# 更简单的导入数据方法
#  https://www.tensorflow.org/text/guide/word_embeddings?hl=zh-cn
def pre_process_data(filepath):
    # Load pos and neg examples from seperate dirs then shuffle them together.
    positive_path = os.path.join(filepath, "pos")
    negative_path = os.path.join(filepath, "neg") # 因为这两个文档前面的filepath都是一样的
    dataset = []

    #print("positive_path:\n", positive_path)
    # print("negative_path:\n", negative_path)
    ## D:/Data/operate_Html/RNN/train/pos

    positive_path = positive_path + "/*.txt"
    negative_path = negative_path + "/*.txt"
    #print("positive_path:\n", positive_path) 

    pos_list = glob.glob(positive_path)
    neg_list = glob.glob(negative_path)

    for filename in pos_list:
        with open (filename, "r", encoding ="utf-8" ) as file :
            dataset.append(file.read())
            file.close()

    for filename in  neg_list:
        with open (filename, "r", encoding ="utf-8" ) as file :
            dataset.append(file.read())
            file.close()

    return pos_list, neg_list, dataset 

# Process label 
def process_label(pos_list, neg_list, dataset):
    pos_label = [1 for _ in range(len(pos_list))]
    neg_label = [0 for _ in range(len(neg_list), len(dataset))]
    label = pos_label + neg_label 
    return label 


def sum_preprocess(path):
    pos_list, neg_list, dataset = pre_process_data(path)
    label = process_label(pos_list, neg_list, dataset)
    return dataset, label


def get_train_test(train_path, test_path):
    
    """
    # Process data
    #Train
    pos_list, neg_list, dataset = pre_process_data("D:/Data/operate_Html/RNN/train/") 
    label = process_label(pos_list, neg_list, dataset)
    print(len(label))  # #25000,  # pos, neg is 12500 each.

    # Test
    test_pos, test_neg, test_dataset = pre_process_data("D:/Data/operate_Html/RNN/test/") 
    test_label = process_label(test_pos, test_neg, test_dataset)
    print(len(test_label)) 
    """

    """
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = np.array(dataset)
    test_dataset = np.array(test_dataset)
    dataset  = dataset.cache().prefetch(buffer_size = AUTOTUNE)
    test_dataset  = test_dataset.cache().prefetch(buffer_size = AUTOTUNE)
    """
    dataset, label = sum_preprocess(train_path)
    test_dataset, test_label = sum_preprocess(test_path)

    # Train dataframe
    train_df = pd.DataFrame(data = dataset, columns=["Review"]) # 只先构建一个列, review, 列里的data是dataset
    train_df["Lable"] = label
    train_df = train_df.sample(frac=1).reset_index(drop=True) # shuffle(dataset)
    print(train_df.head())

    ### Test dataframe
    test_df = pd.DataFrame(data = test_dataset, columns=["test_Review"]) 
    test_df ["Label"]= test_label 
    test_df  = test_df .sample(frac=1).reset_index(drop=True) # shuffle(dataset)
    print(test_df .head())

    # Create a vocabulary
    train = train_df["Review"].tolist()
    test = test_df["test_Review"].tolist()

    return train_df, test_df, train, test