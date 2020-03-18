#!/usr/bin/env python3
# 建立泰坦尼克号人物存活条件的决策树

import pandas as pd


def process_data(data, type=0):
    """预处理数据

    对数据进行填充、变换、选择

    Arg:
        data: 原始数据，DataFrame
        type: 数据集类型，训练集为0（默认），测试集为1
    
    Returns:
        特征属性集合X的向量, 二维向量；
        决策属性Y的向量，二维向量
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
    # 年龄按0-7、8-13、14-25、26-35、36-45、46-55、>55分成五个区间（考虑年龄段对生存几率的影响，按照生命科学标准划分）
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
        y = data[['Survived']].values
    else:
        y = None

    return x, y
    

def main():
    # 读取并处理数据
    data_train = pd.read_csv('titanic_train.csv')
    data_test = pd.read_csv('test.csv')
    x_train, y_train = process_data(data_train)
    x_test, y_test = process_data(data_test, 1)


if __name__ == '__main__':
    main()
