import pandas as pd
import tensorflow as tf

#训练集
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"

#测试集
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# 解析csv文件列，前四个特征值，最后为所属标签
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

# 标签
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def maybe_download():
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

'''
返回两个 (feature,label) 对，分别对应训练集和测试集
'''
def load_data(y_name='Species'):

    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""
    train_path, test_path = maybe_download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)

    '''
    从DataFrame中丢弃指定表头名称的列，并返回该列
    下方传入的y_name = Species,即标签分类,把元数据拆分为两个DataFrame：..x和..y；..x为特征数据集；..y为对应标签数据集
    '''
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


#

'''
训练模型时调用，用于提供训练数据

train_feature 是 Python 字典，其中：每个键都是特征的名称。每个值都是包含训练集中每个样本的值的数组。
train_label 是包含训练集中每个样本的标签值的数组。
args.batch_size 是一个定义批次大小的整数。
'''
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.

    # 输入特征和标签转换为 tf.data.Dataset 对象，固定语法
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    # Shuffle：如果训练样本是随机排列的，则训练效果最好。随机化处理
    # repeat: 训练方法通常会多次处理样本。在不使用任何参数的情况下调用 tf.data.Dataset.repeat 方法可确保 train 方法拥有无限量的训练集样本
    # batch: 一次处理一批样本;默认批次大小设置为 100，意味着 batch 方法将组合多个包含 100 个样本的组。较小的批次大小通常会使 train 方法（有时）以牺牲准确率为代价来加快训练模型。
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    # 1. 将测试集中的特征和标签转换为 tf.dataset 对象。
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    # 2.创建一批测试集样本。（无需随机化处理或重复使用测试集样本。）
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    # 3.将该批次的测试集样本返回 classifier.evaluate。
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
