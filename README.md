# 购物决策---基于决策树模型的模型训练与调优

## 前言

本实验是基于决策树模型的训练与调优记录，旨在通过学习与总结，不存在其他用途与目的。数据采用薛微出版的《R语言数据挖掘方法及应用》教材中提供的数据集---【购物决策数据集】；为简单起见，应用调包SKlearn进行数据预处理与模型训练；后半部分将包含对训练结果的分析、参数调优与最佳模型与参数选择。本实验采用python语言进行模型构建。

模型理解：训练一决策树模型。得到一个新的数据（至少包含性别年龄收入），只要输入性别、年龄和收入，我们可以预测它是否购买。

## 数据预览

**购物决策数据集如图所示：**

![image-20210325161855113](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\分类决策树\README.assets\image-20210325161855113.png)

**数据说明：**

1，一共包含四个字段；

2，第一个字段为label标签，其他三个字段分别为属性标签：年龄、性别、收入；

3，label字段中，0表示未购买，1表示购买。

## 数据预处理

由于数据存放在‘消费决策数据.txt’文本文件中，每个样本中各个字段之间以‘\t’作为分隔符，因此数据预处理阶段将包含以下几个阶段：

1，读入数据，将样本放入Python列表变量中；

2，将数据由字符型变量转换为浮点型变量；

3，对数据进行标准化或者归一化。

Let's start!

**第一步骤：读入数据，将样本放入Python列表变量中**

引入数据路径

```python
inputfile = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/消费决策数据.txt'
```

读取数据，去掉表头

```python
with open(inputfile, encoding='utf-8') as f:
    sample = []
    samples = []
    for line in f:
        for l in line.strip().split('\t'):
            sample.append(l)
        samples.append(sample)
        sample = []
samples = samples[1:]
```

**第二步骤：将数据由字符型变量转换为浮点型变量**

将数据格式转换为浮点型

```python
new_samples = []
for sample in samples:
    sample = [float(x) for x in sample]
    new_samples.append(sample)
```

分割label及属性，方便使用sklearn进行数据集划分

```python
y = []
X = []
for sample in new_samples:
    y.append(sample[0])
    X.append([sample[1], sample[2], sample[3]])
```

第三步骤：对数据进行标准化

先引入对应的标准化类

```python
from sklearn.preprocessing import StandardScaler
```

数据标准化

```python
ss = StandardScaler()
X = ss.fit_transform(X)
```

**打印看看标准化前和标准化后的数据对比**

标准化前：                       标准化后：                                              

[[41.  2.  1.]                       [[ 0.35766699  0.89209491 -1.29362056]
  [47.  2.  1.]                        [ 1.41714792  0.89209491 -1.29362056]
  [41.  2.  1.]                        [ 0.35766699  0.89209491 -1.29362056]
  ...                                        ...
  [32.  1.  3.]                        [-1.23155439 -1.12095696  1.16254887]
  [34.  1.  3.]                        [-0.87839408 -1.12095696  1.16254887]
   [34.  1.  3.]]                      [-0.87839408 -1.12095696  1.16254887]]

**对三个属性字段进行可视化**

<img src="C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325164457778.png" alt="image-20210325164457778" style="zoom:50%;" />

第一幅图是Age字段，往下依次是Gender性别和Income收入.

## 数据集分割与格式转换

对数据集进行分割，用Sklearn会非常方便

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
```

我们去y_train和y_test进行可视化看看数据分布：

```python
print("训练集标签为1的分布比例为：", sum(y_train)/len(y_train), "%")
print("测试集标签为1的分布比例为：", sum(y_test)/len(y_test), "%")
plt.figure()
plt.subplot(211)
plt.scatter(range(0, len(y_train)), y_train)
plt.suptitle('y_train_distribution')
plt.subplot(212)
plt.scatter(range(0, len(y_test)), y_test)
plt.suptitle('y_test_distribution')
plt.show()
```

控制台输出：

训练集标签为1的分布比例为： 0.3746130030959752 %
测试集标签为1的分布比例为： 0.37962962962962965 %

图表输出：

<img src="C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\分类决策树\README.assets\Figure_1.png" alt="Figure_1" style="zoom:72%;" />

可见训练集与测试集的分布大致是相同的，这对于之后的进一步模型训练是有利的。

## 模型训练

### 模型重要参数选择与设定

在神经网络模型中，最重要的参数无非包括：树的最大深度，叶节点的最大个数，每个节点要继续分裂所保证的最少的数量

**最大迭代次数：**

为了能让树进行充分地学习，我们先把树地最大深度设为一个比较大的值，设为30

**叶节点的最大个数：**

考虑一种情况：假设平均每个叶节点的样本数量最少保证20个，那么叶节点的个数最多不超过17个（样本树为323左右），所以此处暂时设置叶节点的个数为17

**每个节点要继续分裂所保证的最少的数量：**

如上述，设置为20；

**代码如下：**

```python
clf = tree.DecisionTreeClassifier(max_depth=5,
                                  min_samples_leaf=20,
                                  max_leaf_nodes=17,
                                  random_state=0,
                                  )
```

### 开始模型训练

```python
clf.fit(X_train, y_train)
```

## 模型评估

1，经过上一步骤的拟合训练，直接调用训练结果对测试集的数据进行预测；

2，输出预测结果，与真实结果进行比对，生成混淆矩阵(此模型为二分类问题)

3，输出模型预测精准度。

```python
# 模型预测
predictions = clf.predict(X_test)
```

**输出结果：**

[0. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0. 1. 1. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0.
 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 1. 0. 1. 0.
 1. 1. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1.
 0. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 0.]

```python
labels = list(set(y_test))
conf_mat = confusion_matrix(y_test, predictions, labels=labels)
print("混淆矩阵如下：\n", conf_mat)
f = clf.score(X_test, y_test, sample_weight=None)*100
print('预测精度为:%.2f' % f, '%')
```

**输出结果：**

混淆矩阵如下：
 [[50 17]
 [27 14]]
预测精度为:59.26 %

**可视化决策树：**

<img src="C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\分类决策树\README.assets\image-20210328182450936.png" alt="image-20210328182450936" style="zoom:50%;" />

**结果分析：**

1，预测精度为59.26 %，一个比较bad的结果；

2，回归整个模型创建过程，找出不合理之处：

a. 重要参数设置，最大树深度为30，对于300左右的小样本量，足以让学习充分；

b. 最大叶节点个数17，是根据每个节点要继续分裂需保证节点所包含样本最小数量而得，待后续模型参数调优斟酌.

## 模型参数调优

从整棵决策树细枝末节出发，我们先考虑每个节点所要包含的最小的样本数量出发，剪掉细枝末节，再考虑整棵树的深度。

**首先，节点最小样本数量：**

我们参数范围限定在1到100之间，看模型准确率随节点最小样本数量的变化。对节点最小样本数量调参，先注释掉以上单个模型训练步骤，其他模型参数保持不变，用for循环改变最大迭代次数，依次输出结果。并用列表变量接收每次训练的结果。

代码如下：

```pyton
acc_list = []
for i in range(1, 100):
    clf = tree.DecisionTreeClassifier(max_depth=30,
                                      min_samples_leaf=i,
                                      max_leaf_nodes=17,
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
plt.plot(range(1, 100), acc_list, 'bo', range(1, 100), acc_list, 'k')
plt.xlabel('max_depth')
plt.ylabel('Accuracy%')
plt.show()
```

**部分控制台输出结果：**

混淆矩阵如下：
 [[65  2]
 [36  5]]
预测精度为:64.81 %
混淆矩阵如下：
 [[65  2]
 [36  5]]
预测精度为:64.81 %
混淆矩阵如下：
 [[63  4]
 [32  9]]
预测精度为:66.67 %
混淆矩阵如下：
 [[65  2]
 [35  6]]
预测精度为:65.74 %
混淆矩阵如下：
 [[61  6]
 [32  9]]
预测精度为:64.81 %
混淆矩阵如下：
 [[61  6]
 [32  9]]
预测精度为:64.81 %
混淆矩阵如下：
 [[55 12]
 [27 14]]
预测精度为:63.89 %
混淆矩阵如下：
 [[55 12]
 [27 14]]

**图标输出：**

![Figure_2](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\分类决策树\README.assets\Figure_2.png)

由图可看到，在i迭代至i = 60左右时，模型预测精度收敛。但是，另一方面又看到节点最小数量在0-20之间预测精度达到峰值，精度比收敛时高出4%左右；但是在0-20范围内取值，因为参数偏小，模型充分生长，导致过拟合，可能会导致对未来加入预测的数据集预测精度波动很大，即模型泛化能力较差。因此，我们在节点最小数量这一参数上暂时先取优为65.

**其次，最大叶节点数目：**

最大叶节点数目参数我们把调优范围限制在10-25之间，观察参数在哪个范围内，模型的精准度较高。代码如下：

```python
acc_list = []
for i in range(10, 25):
    clf = tree.DecisionTreeClassifier(max_depth=30,
                                      min_samples_leaf=65,
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
```

**控制台部分输出如下：**

混淆矩阵如下：
 [[67  0]
 [41  0]]
预测精度为:62.04 %
混淆矩阵如下：
 [[67  0]
 [41  0]]
预测精度为:62.04 %
混淆矩阵如下：
 [[67  0]
 [41  0]]
预测精度为:62.04 %
混淆矩阵如下：
 [[67  0]
 [41  0]]
预测精度为:62.04 %......

**图表输出：**

![Figure_3](C:\Users\DELL\PycharmProjects\数据结构与算法\机器学习算法手写与实践\分类决策树\README.assets\Figure_3.png)

分析：

图表看起来是一条恒定常数，说明上一参数调优中，我们已经完全地控制了决策树地过分生长；我们把上一参数调为60，输出一下图表，得到的结果是跟上述一样的图。

重新分析下几个参数调优的含义：其实无论是树的深度、节点所含样本的最少数目抑或是叶节点的最大个数，都是为了控制树的过分生长。因此，既然在节点所含样本的最少数目调优中已得到最优值，那在其他参数上的调优预测精准度显然都是一恒定的常数。因此，调优部分到此结束。



## 交流

问题可联系2393946194@qq.com，欢迎互相交流与学习！





