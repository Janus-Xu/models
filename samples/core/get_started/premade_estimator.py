#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    print(type(args.train_steps))

    # Fetch the data

    '''
    ---1. 导入和解析数据集---
    '''
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    # 通过调用 tf.feature_column 模块中的函数来构建 feature_column 对象列表
    '''
    ---2. 描述数据---
    
    [_NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
     _NumericColumn(key='SepalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), 
     _NumericColumn(key='PetalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), 
     _NumericColumn(key='PetalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
    '''
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.

    '''
    ---3. 选择模型类型---
    
    创建训练模型：选择神经网络->要指定模型类型，请实例化一个 Estimator 类。
    此处使用预创建的 Estimator tf.estimator.DNNClassifier
    此 Estimator 会构建一个对样本进行分类的神经网络。
    以下为实例化DNNClassifier
    '''
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        # hidden_units 定义神经网络内每个隐藏层中的神经元数量。（本例中第一个隐藏层中有 10 个，第二个隐藏层中有 10 个）
        # 要更改隐藏层或神经元的数量，只需为 hidden_units 参数分配另一个列表即可。
        hidden_units=[10, 10],
        # The model must choose between 3 classes.
        # 指定了神经网络可以预测的潜在值的数量，当前问题需要把结果分为3类
        n_classes=3)
    # 还可以指定优化器，优化器和学习速率也非常重要

    # Train the Model.
    '''
    ---4. 训练模型---
    
    steps 指示 train 在完成指定的迭代次数后停止训练
    增加 steps 会延长模型训练的时间,args.train_steps 的默认值是 1000;
    训练的步数是一个可以调整的超参数。选择正确的步数通常需要一定的经验和实验基础。
    
    input_fn,在iris_data中存在。会确定提供训练数据的函数
    '''
    classifier.train(
        # args.batch_size 是一个定义批次大小的整数
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.

    '''
    ---5. 评估模型---
    
    每个 Estimator 都提供了 evaluate 方法
    classifier.evaluate 必须从测试集（而非训练集）中获取样本
    eval_input_fn 函数负责提供来自测试集的一批样本。
    '''
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y,
                                                 args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size))

    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
