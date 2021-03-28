from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree


inputfile = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/消费决策数据.txt'
# 读取数据，并去掉表头
with open(inputfile, encoding='utf-8') as f:
    sample = []
    samples = []
    for line in f:
        for l in line.strip().split('\t'):
            sample.append(l)
        samples.append(sample)
        sample = []
feature_name = samples[0][1:]
target_names = ['Not Buy', 'Buy']
samples = samples[1:]


# 分割label以及属性
y = []
X = []
for sample in samples:
    y.append(float(sample[0]))
    X.append([float(sample[1]), float(sample[2]), float(sample[3])])

ss_X = StandardScaler()
X = ss_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# print("训练集标签为1的分布比例为：", sum(y_train)/len(y_train), "%")
# print("测试集标签为1的分布比例为：", sum(y_test)/len(y_test), "%")
# print("训练集的长度为：", len(y_test))
# plt.figure()
# plt.subplot(211)
# plt.scatter(range(0, len(y_train)), y_train)
# plt.subplot(212)
# plt.scatter(range(0, len(y_test)), y_test)
# plt.suptitle('y_[train-test]_distribution')
# plt.show()
#
#
# clf = tree.DecisionTreeClassifier(max_depth=30,
#                                   min_samples_leaf=20,
#                                   max_leaf_nodes=17,
#                                   random_state=0,
#                                   )
#
# clf = clf.fit(X_train, y_train)
# predictions = clf.predict(X_test)
# print(predictions)
# labels = list(set(y_test))
# conf_mat = confusion_matrix(y_test, predictions, labels=labels)
# print("混淆矩阵如下：\n", conf_mat)
# acc = clf.score(X_test, y_test, sample_weight=None) * 100
# print('预测精度为:%.2f' % acc, '%')
#
# import graphviz
#
# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 feature_names=feature_name,
#                                 class_names=target_names,
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.view()

acc_list = []
for i in range(10, 25):
    clf = tree.DecisionTreeClassifier(max_depth=30,
                                      min_samples_leaf=60,
                                      max_leaf_nodes=i,
                                      random_state=0,
                                      )

    clf = clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    labels = list(set(y_test))
    conf_mat = confusion_matrix(y_test, predictions, labels=labels)
    print("混淆矩阵如下：\n", conf_mat)
    acc = clf.score(X_test, y_test, sample_weight=None) * 100
    print('预测精度为:%.2f' % acc, '%')
    acc_list.append(acc)
plt.figure()
plt.plot(range(10, 25), acc_list, 'bo', range(10, 25), acc_list, 'k')
plt.xlabel('min_samples_leaf')
plt.ylabel('Accuracy%')
plt.show()


