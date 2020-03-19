#!/usr/bin/env python3
# 建立泰坦尼克号人物存活条件的决策树

from math import log

import pandas as pd
import numpy as np


# 属性集
attributes = {0: 'Pclass', 1: 'Sex', 2: 'Age', 3: 'SibSp', 4: 'Parch', 5: 'Fare', 6: 'Embarked'}
# 每个属性所有可能的取值
attributes_values = {
    'Pclass': [0, 1, 2, 3],
    'Sex': [0, 1],
    'Age': [0, 1, 2, 3, 4, 5, 6],
    'SibSp': [0, 1, 2, 3, 4, 5, 8],
    'Parch': [0, 1, 2, 3, 4, 5, 6, 9],
    'Fare': [0, 1, 2, 3],
    'Embarked': [0, 1, 2]
}


def process_data(data, type=0):
    """预处理数据

    对数据进行填充、变换、选择

    Arg:
        data: 原始数据，DataFrame
        type: 数据集类型，训练集为0（默认），测试集为1
    
    Returns:
        特征属性集合X的向量, 二维向量；
        决策属性Y的向量，一维向量
    """
    # 年龄填充中位数
    data['Age'] = data['Age'].fillna(data['Age'].median())
    # 工资填充中位数
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    # 上船地点填充数据量最多的S
    data['Embarked'] = data['Embarked'].fillna('S')

    # 性别用男0，女1表示
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1
    # 年龄按0-7、8-13、14-25、26-35、36-45、46-55、>55分成七个区间（考虑年龄段对生存几率的影响，按照生命科学标准划分）
    data.loc[data['Age'] <= 7, 'Age'] = 0
    data.loc[(data['Age'] > 7) & (data['Age'] <= 13), 'Age'] = 1
    data.loc[(data['Age'] > 13) & (data['Age'] <= 25), 'Age'] = 2
    data.loc[(data['Age'] > 25) & (data['Age'] <= 35), 'Age'] = 3
    data.loc[(data['Age'] > 35) & (data['Age'] <= 45), 'Age'] = 4
    data.loc[(data['Age'] > 45) & (data['Age'] <= 55), 'Age'] = 5
    data.loc[data['Age'] > 55, 'Age'] = 6
    # 费用按0-8，9-15，16-31，>31分成4个区间（数据均匀分布）
    data.loc[data['Fare'] <= 8] = 0
    data.loc[(data['Fare'] > 8) & (data['Fare'] <= 15), 'Fare'] = 1
    data.loc[(data['Fare'] > 15) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[data['Fare'] > 31, 'Fare'] = 3
    # 搭乘点用S：0，C：1，Q：2表示
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

    # 将年龄和工资转化为整数
    data['Age'] = data['Age'].astype(int)
    data['Fare'] = data['Fare'].astype(int)

    # 选取特征属性
    # PassengerId，Name，Ticket明显对生存率没有影响，不做考虑
    # Cabin数据严重缺失，不做考虑
    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values
    if type == 0:
        y = data['Survived'].values
    else:
        y = None

    return x, y


def cal_entropy(examples_y):
    """计算熵

    计算给定决策属性集的熵

    Arg:
        examples_y: 样例决策属性集，一维向量

    Return:
        熵的值entropy，小数
    """
    examples_num = examples_y.size
    negative_num = examples_y[examples_y == 0].size
    positive_num = examples_y[examples_y == 1].size

    entropy = 0
    # 加0.01防止计算log(0)
    entropy -= (negative_num / examples_num) * log(negative_num + 0.01 / examples_num, 2)
    entropy -= (positive_num / examples_num) * log(positive_num + 0.01 / examples_num, 2)

    return entropy


def cal_info_gain(examples_x, examples_y, attribute):
    """计算信息增益比率

    计算指定目标特征属性相对于当前样例集的信息增益比率

    Args:
        examples_x: 样例特征属性集，二维向量
        examples_y: 样例决策属性集，一维向量
        attribute: 目标特征属性的index，整数

    Return:
        信息增益比率的值gain，小数
    """
    gain = cal_entropy(examples_y)
    target_x = np.array([example[attribute] for example in examples_x])
    for value in np.unique(target_x):
        # 筛选出决策属性
        indexes = np.where(target_x == value)[0]
        current_y = examples_y[indexes]
        gain -= abs(current_y.size / examples_y.size) * cal_entropy(current_y)
    return gain


def get_best_attribute(examples_x, examples_y, attributes):
    """获取最佳属性

    获取目标属性集中分类样例能力最好的属性

    Args:
        examples_x: 样例特征属性集，二维向量
        examples_y: 样例决策属性集，一维向量
        attributes: 目标属性集，dict

    Return:
        最佳属性的index，整数
    """
    max_gain = cal_info_gain(examples_x, examples_y, list(attributes.keys())[0])
    best_attribute = list(attributes.keys())[0]
    for key in attributes.keys():
        if cal_info_gain(examples_x, examples_y, key) > max_gain:
            max_gain = cal_info_gain(examples_x, examples_y, key)
            best_attribute = key
    return best_attribute


def id3(examples_x, examples_y, attributes):
    """id3算法

    递归实现决策树构建id3算法

    Args:
        examples_x: 样例特征属性集，二维向量
        examples_y: 样例决策属性集，一维向量
        attributes: 当前属性集，dict
    Return:
        构建好的决策树，dict
    """
    # 创建root结点
    root = ''

    # 返回单节点root
    if np.all(examples_y == 0):
        return '-'
    if np.all(examples_y == 1):
        return '+'
    if attributes == {}:
        if np.argmax(np.bincount(examples_y)) == 1:
            return '+'
        else:
            return '-'

    # 开始递归
    best_attribute = get_best_attribute(examples_x, examples_y, attributes)
    root = attributes[best_attribute]
    attributes.pop(best_attribute)
    sub_tree = {}
    target_x = np.array([example[best_attribute] for example in examples_x])
    for value in attributes_values[root]:
        indexes = np.where(target_x == value)[0]
        if indexes.size == 0:
            if np.argmax(np.bincount(examples_y)) == 1:
                sub_tree[value] = '+'
            else:
                sub_tree[value] = '-'
        else:
            sub_tree[value] = id3(examples_x[indexes], examples_y[indexes], attributes.copy())

    return {root: sub_tree}


def is_survived(x, decision_tree):
    """生存判断函数

    判断个体是否能成功生存

    Args:
        x: 个体特征属性，一维向量
        decision_tree: 决策树，dict

    Return:
        个体是否生存，是1，否0
    """
    tree = decision_tree.copy()
    while True:
        if tree == '+':
            return 1
        if tree == '-':
            return 0
        label = list(tree.keys())[0]
        index = list(attributes.keys())[list(attributes.values()).index(label)]
        tree = tree[label][x[index]]


def main():
    # 读取并处理数据
    data_train = pd.read_csv('titanic_train.csv')
    data_test = pd.read_csv('test.csv')
    x_train, y_train = process_data(data_train)
    x_test, y_test = process_data(data_test, 1)

    # 构造决策树
    decision_tree = id3(x_train, y_train, attributes.copy())
    print("决策树：", decision_tree)

    # 利用决策树去预测测试集
    y_test = []
    for x in x_test:
        y_test.append(is_survived(x, decision_tree))
    y_test = np.array(y_test)
    print("测试集决策属性结果：", y_test)
    print("测试集中生存人数：", np.where(y_test == 1)[0].size)


if __name__ == '__main__':
    main()
