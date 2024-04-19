# Machine Learning
机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法（要防止错误累积）。很多推论问题属于非程序化決策，所以部分的机器学习研究是开发容易处理的近似算法。
![](2024-04-11-14-12-07.png)
## Supervised Machine Learning

Supervised machine learning 是一种人工智能(AI)和机器学习(ML)的方法，通过使用带有标签的数据来训练算法，以便它能够学习如何预测或分类新的、未见过的数据。这里的“监督”指的是学习过程中使用的数据集包含了输入数据和对应的正确输出（即标签），模型通过这些数据进行学习。

这个过程可以分解为以下几个步骤：

    收集数据：首先，需要收集相关的数据。这些数据可能来源于各种渠道，比如日志文件、数据库、在线资源等。

    预处理和清洗数据：数据通常需要经过清洗和预处理，以去除噪声、处理缺失值、格式化以及其他必要的步骤，使其适合用于模型训练。

    标注数据：在监督学习中，每个样本数据都需要有一个对应的标签或输出。例如，在图片识别任务中，每张图片都需要标注为“狗”、“猫”或其他类别。

    选择模型：根据问题的性质，选择适当的机器学习模型。常见的模型包括决策树、随机森林、神经网络等。

    训练模型：使用收集到的带标签的数据训练选定的模型。在这个过程中，算法会尝试学习数据之间的关系，以便能够根据输入预测出正确的输出。

    评估模型：使用一部分之前未参与训练的数据来测试模型的性能。评估指标可能包括准确率、召回率、F1分数等，具体取决于任务的性质。

    参数调优和优化：根据模型的初步表现，调整模型参数或使用不同的模型结构来提高性能。

    部署模型：一旦模型表现令人满意，它就可以被部署到生产环境中，对新数据进行预测或分类。


### Regression
回归旨在预测一个连续的数值。使用回归模型，我们可以基于输入数据预测一个实数值的输出。这类问题的关键是预测的目标变量是连续的。回归分析广泛应用于预测和预测分析领域，例如：

预测房价：根据房屋的位置、大小、年龄等特征来预测房价。
股票价格预测：根据历史数据和其他经济指标预测股票的未来价格。
气温预测：根据过去的天气数据来预测未来的气温。
常见的回归算法包括线性回归、多项式回归、岭回归（Ridge Regression）、套索回归（Lasso Regression）和弹性网回归（Elastic Net Regression）等。

### Classification

分类旨在预测一个离散的标签。使用分类模型，我们可以将输入数据分配到两个或多个类别中。这类问题的关键是预测的目标变量是离散的，或者说是类别的。分类广泛应用于许多领域，如：

邮件过滤：将电子邮件分类为“垃圾邮件”或“非垃圾邮件”。
客户流失预测：预测客户是否会在未来一段时间内离开或继续使用服务。
图像识别：识别图像中的对象属于哪个类别，例如狗、猫或汽车。
分类问题可以是二分类（如判断邮件是否为垃圾邮件）或多分类（如识别图片中的物体属于哪个类别）。常见的分类算法包括逻辑回归、决策树、随机森林、支持向量机（SVM）、神经网络等。

总结来说，回归和分类的主要区别在于输出变量的类型：回归预测连续变量，而分类预测离散变量。选择哪种类型的监督学习方法取决于具体问题的需求和数据的性质。


## Unsupervised Machine Learning

无监督学习（Unsupervised Machine Learning）是机器学习的一种方法，它与监督学习不同，不依赖于带有标签的数据。在无监督学习中，算法被训练在没有任何先验标签的情况下，仅仅基于数据的内在结构和模式来分析和推理数据。这种学习方法主要用于探索性数据分析、发现隐藏模式或数据压缩。无监督学习的主要类型包括聚类、降维和关联规则学习。

### Clustering

聚类（Clustering）
聚类是一种将数据点分组的无监督学习方法，使得同一组内的数据点彼此相似，而不同组的数据点相异。它用于数据挖掘和统计数据分析，以发现数据中的隐藏模式或分组。常见的聚类算法包括K-均值（K-means）、层次聚类（Hierarchical Clustering）和DBSCAN等。


## Decomposition
try to group data(very large), and use maybe two or three col to representation all of the col

![](2024-04-11-14-19-16.png)
train you function


![](2024-04-11-14-18-31.png)
使用大概20%的数据去测试你model


![](2024-04-11-14-20-57.png)
use model to predict data


![](2024-04-11-14-21-21.png)


![](2024-04-11-14-21-53.png)


![](2024-04-11-14-24-22.png)


![](2024-04-11-14-24-31.png)


![](2024-04-11-14-26-57.png)


![](2024-04-11-14-28-08.png)

 
![](2024-04-11-14-33-15.png)


![](2024-04-11-14-34-40.png)


![](2024-04-11-14-35-53.png)


![](2024-04-11-14-38-25.png)



### Regression

```python

import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
# new import statements
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
```



```python
url = "https://data.cityofchicago.org/resource/xhc6-88s9.geojson"
df = gpd.read_file(url)
df.to_file("ccvi.geojson")  # save it as local file
```

![](2024-04-11-16-14-51.png)

现在可以优化一下上面的代码, 不需要在每次重新运行的时候重新下载

```python
# Do not reptitivitely download datasets
# Save a local copy instead
if os.path.exists("ccvi.geojson"): 
    print("read local file")
    df = gpd.read_file("ccvi.geojson")
else: 
    print("download the dataset")
    url = "https://data.cityofchicago.org/resource/xhc6-88s9.geojson"
    df = gpd.read_file(url)
    df.to_file("ccvi.geojson")

```


在一些dataframe中有一些None， 或者一些missing value，我们不想处理以下是改如何操作。

```python
# How many rows have missing values?
df['rank_covid_19_incidence_rate'].isna().value_counts()
```
![](2024-04-11-16-23-01.png)

```python
# row has missing values if its geography_type is ZIP
df['geography_type'].value_counts()
```
![](2024-04-11-16-25-03.png)

```python
# remove rows that have missing values
df = df[df['geography_type'] == 'CA']
```

double check

```python
df['rank_covid_19_incidence_rate'].isna().value_counts()
```

![](2024-04-11-16-26-07.png)


```python
# list the type of each column
df.dtypes
```
![](2024-04-11-16-30-38.png)
因为数据都是object type，会存在一些missing value， 因此画图是不准确的。我们需要typecasting。

```python

# extract the columns we want to type caste
columns_typecasting = list(df.columns)
columns_typecasting.remove('community_area_name')
columns_typecasting.remove('geometry')
columns_typecasting.remove('ccvi_category')
columns_typecasting.remove('geography_type')
columns_typecasting

df[columns_typecasting] = df[columns_typecasting].apply(pd.to_numeric)

df.dtypes
```

![](2024-04-11-16-32-22.png)

apply函数对DataFrame的每一列（或行）应用一个函数。在这个例子中，应用的函数是pd.to_numeric。pd.to_numeric是Pandas库中的一个函数，用于将一个列或Series中的数据转换成数值类型。默认情况下，它会将不能转换为数值的值转换成NaN（代表“非数字”）。


## How can we train/fit models to known data to predict unknowns?
- Feature(s) => Predictions
    - Population => Deaths
    - Cases => Deaths
    - Cases by Age => Deaths
    
- General structure for fitting models:
    ```python
    model = <some model>
    model.fit(X, y) # X stands for feature matrix, y stands for prediction label
    y = model.predict(X)
    ```
    where `X` needs to be a matrix or a `DataFrame` and `y` needs to be an array (vector) or a `Series`
```python
# We must specify a list of columns to make sure we extract a DataFrame and not a Series
# Feature DataFrame
df[["rank_socioeconomic_status"]].head()
# 两个[[]]可以取出一个data frame
```
![](2024-04-11-16-42-36.png)
![](2024-04-11-16-42-49.png)


训练machine

```python
xcols = ["rank_socioeconomic_status"]
ycol = "rank_covid_19_crude_mortality_rate"

model = LinearRegression()
model.fit(df[xcols], df[ycol])
# less interesting because we are predicting what we already know
y = model.predict(df[xcols])
y
# df[xcols]是自变量的数据，df[ycol]是因变量的数据。这个方法计算了最佳的拟合线，使得预测值和实际值之间的差异最小。
```
![](2024-04-11-16-46-53.png)

Predicting for new values of x

```python
predict_df = pd.DataFrame({"rank_socioeconomic_status":[10,20,30]})
predict_df

```

![](2024-04-11-16-51-08.png)


```python

predict_df = pd.DataFrame({"rank_socioeconomic_status": range(0, 81, 10)})

predict_df["predicted_mortality_rate_rank"] = model.predict(predict_df) 
predict_df
```

![](2024-04-11-16-52-29.png)

```python

# Create a line plot to visualize relationship between "rank_socioeconomic_status" and "predicted_mortality_rate_rank"
ax = predict_df.plot.line(x="rank_socioeconomic_status", y="predicted_mortality_rate_rank", color="r")
# Create a scatter plot to visualize relationship between "rank_socioeconomic_status" and "predicted_mortality_rate_rank"
df.plot.scatter(x="rank_socioeconomic_status", y="rank_covid_19_crude_mortality_rate", ax=ax, color="k")

```

![](2024-04-11-16-54-24.png)



```python
# Model coefficients
model.coef_
# numpy array
# array([0.66152681])


# Slope of the line
model.coef_[0]

# 0.6615268089620371


# Intercept of the line
model.intercept_

# 12.565883265292989



print(f"mortality_rate_rank ~= {round(model.coef_[0], 4)} * socioeconomic_status_rank + {round(model.intercept_, 4)}")
```


![](2024-04-11-16-58-41.png)



### How well does our model fit the data?
- explained variance score
- R^2 ("r squared")

#### `explained_variance_score(y_true, y_pred)`
- `from sklearn.metrics import explained_variance_score`
- calculates the explained variance score given:
    - y_true: actual death values in our example
    - y_pred: prediction of deaths in our example
- documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html

![](2024-04-11-18-42-15.png)

![](2024-04-11-17-00-46.png)

![](2024-04-17-13-46-33.png)
- explained_variance try to explain how many variance(方差) of you prediction are with respective to actual variance


#### Explained variance score
- `explained_variance = (known_var - explained_variance) / known_var`
    - where `known_var = y_true.var()` and `explained_variance = (y_true - y_pred).var()`

```python
# Compute variance of "predicted_mortality_rate_rank" column
known_var = df[ycol].var()
known_var


# explained_variance
explained_variance = (df[ycol] - predictions).var()   
explained_variance


# explained_variance score
explained_variance_score = (known_var - explained_variance) / known_var
explained_variance_score

# 1 => best score - infty => lowest score

```
![](2024-04-11-18-46-48.png)

#### `r2_score(y_true, y_pred)`

- - `from sklearn.metrics import r2_score`
- calculates the explained variance score given:
    - y_true: actual death values in our example
    - y_pred: prediction of deaths in our example
- documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html 


#### R^2 score (aka coefficient of determination) approximation

- `r2_score = (known_var - r2_val) / known_var`
    - where `known_var = y_true.var()` and `r2_val = ((y_true - y_pred) ** 2).mean()`
![](2024-04-11-18-48-30.png)
```python
r2_score(df[ycol], predictions) #defalut is explain variance score


# r2_val
r2_val = ((df[ycol] - predictions) ** 2).mean()
r2_val


```
![](2024-04-11-18-46-57.png)




- Split data into train and test sets
- Use the test sets to tell

```python
# Split the data into two equal parts
len(df) // 2

# Manual way of splitting train and test data
train, test = df.iloc[:len(df)//2], df.iloc[len(df)//2:]
len(train), len(test)
```

Problem with manual splitting is, we need to make sure that the data is not sorted in some way.


#### `train_test_split(<dataframe>, test_size=<val>)`

- requires `from sklearn.model_selection import train_test_split`
- shuffles the data and then splits based on 75%-25% split between train and test
    - produces new train and test data every single time
- `test_size` parameter can take two kind of values:
    - actual number of rows that we want in test data
    - fractional number representing the ratio of train versus test data
    - default value is `0.25`
- documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


```python
train, test = train_test_split(df)
len(train), len(test) # splits based on 75%-25%

# (75% data, 25% data)
#(57,20)


# Test size using row count
train, test = train_test_split(df, test_size=30)
len(train), len(test)
#(47,30)


# Test size using fraction
train, test = train_test_split(df, test_size=0.5)
len(train), len(test)
#(38, 39)
```



```python

train, test = train_test_split(df)

# Let's use the train and the test data
model = LinearRegression()
# Fit using training data
model.fit(train[xcols], train[ycol])
# Predict using test data
y = model.predict(test[xcols])
# We can use score directly as it automatically invokes predict
model.score(test[xcols], test[ycol])

```
![](2024-04-11-18-57-49.png)

因为 test data是在之前从未见过的因此分数很低。这也是检测是否overfit
<b>overfit</b> model on particular data set and it's not perform as good in new data set.

![](2024-04-18-17-36-36.png)

还有一种检测overfiting的方式

#### Cross Validation:
`cross_val_score(estimator, X, y)`
- requires `from sklearn.model_selection import cross_val_score`
-  do many different train/test splits of the values, fitting and scoring the model across each combination
-  
它更具有计算成本， 通常你的model有很多数据，但是不知道哪个配置是最好的，但是可以通过这个方式来确定最好的配置。
功能：
如果有两个model，选择一个最好的
有有一些参数我们想调整model， 哪一个set of param可以给我们更好的表现

![](2024-04-18-17-40-25.png)
默认组合是5个set，但是可以设置

下面是个例子，从两个model中选择一个最好的

![](2024-04-18-17-54-49.png)

因此在这个情况下我们会选择第二个
第二个model高是因为shuffling会对其造成一些影响，
如果数据量很小的话shuffling会对结果造成影响。
cross validation, 减少了shuffling

因此需要计算standard deviation来确定shuffling并没有对数据造成很大的影响

![](![alt%20text](image-1.png).png)

在这种情况下sd越小说明model表现得越好。因此我们会选择model 2


![](2024-04-18-18-01-37.png)

有负的coef是正常的，因为一些因素的增长可能会造成一些负面的影响


## Pipeline

![](2024-04-18-18-05-47.png)

![](2024-04-18-18-07-23.png)

![](2024-04-18-18-21-04.png)


![](2024-04-18-18-25-53.png)


#### `PolynomialFeatures(degree=<val>, include_bias=False)`


```python3

from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer

```

- `degree` enables us to mention how many degrees we need
- 参数用来设置多项式的最高次数
- 
- `include_bias` default value is True, which will add a column of 1s - we typically don't use that. （Constant trend）
- returns an object instance on which we can invoke `fit` and 
- 如果 include_bias=True，那么在生成的多项式特征中会包括一个值恒为1的特征。这个常数1可以在模型中充当  b0

​
  的角色，允许模型在不依赖其他变量的情况下，有一个非零的起点。
- `transform`:
    - `transform(X, columns=<col names>)`: transform data to polynomial features`
    - `fit_transform(X[, y])`: fit to data, then transform it.
- documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

![](2024-04-18-18-35-42.png)

![](2024-04-18-18-36-13.png)

如果 include_bias = True,会多一列1：
![](2024-04-18-18-37-17.png)


`fit_transform(X[, y])`: fit to data, then transform it.
会有相同的结果

![](2024-04-18-18-38-12.png)


#### `Pipeline(...)`
Pipeline allows you to sequentially apply a list of transformers to preprocess the data and, if desired, conclude the sequence with a final predictor for predictive modeling.


Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit. The transformers in the pipeline can be cached using memory argument.

- Argument: list of steps in the pipeline:
    - each step represented as a tuple with name of the step and the object instance
    - last step will be the estimator

![](2024-04-18-18-41-20.png)



![](2024-04-18-18-48-06.png)

#### `OneHotEncoder()`
One-Hot Encoding 是一种处理分类变量的方法，常用于数据预处理阶段，以适应那些只能处理数值数据的机器学习模型。这种编码方法通过将每个类别变量转换为一个或多个二进制列来表示，每个类别对应一个列。当某个类别存在时，其对应的列值为 1，其余列值为 0。这种方法主要用于将分类数据转换为格式化的数值数据，以便机器学习算法能够更好地处理。
- encodes categorical features as a one-hot numeric array
- returns a "sparse matrix", which needs to be explicitly converted into an `array` using `to_array()` method, before `DataFrame` conversion
- documention: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

![](2024-04-18-18-50-38.png)

每一行只有一个1，意思是，第一行来自 Ohio Street Beach
第二行来自 Calumet Beach

![](2024-04-18-19-16-28.png)


#### `make_column_transformer(...)`
make_column_transformer 是 scikit-learn 库中的一个函数，用于构建一个“列转换器”，它能够同时对数据集中的多个列应用不同的预处理步骤。这使得在单个步骤中，针对不同类型的数据（如数值数据和分类数据）执行不同的转换成为可能。这对于处理含有多种数据类型的复杂数据集非常有用，且常用于机器学习的数据预处理阶段。
- Argument: transformations
    - each transformer argument will be a `tuple` with object instance as first item and list of feature columns as the second
- documention: https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html

make_column_transformer 通过创建一个能够对数据集的不同列执行不同转换操作的管道，简化了数据预处理的过程。比如，你可以为分类数据设置 One-Hot Encoding，同时为数值数据设置标准化或归一化处理。


可以在不同的col上做不同的transformation
![](2024-04-18-19-21-48.png)


![](2024-04-18-19-26-53.png)
这样这个model在不overfit的情况下预测test结果的表现是非常好的


#### Numpy

```python
import numpy as np
```


NumPy（Numerical Python 的缩写）是 Python 编程语言的一个非常重要的库，主要用于进行高效的数值计算。它提供了一个强大的N维数组对象，以及用于操作这些数组的广泛工具和函数。NumPy 是科学计算中最核心的 Python 包之一，广泛应用于数据分析、机器学习、工程科学、图像处理等领域。

核心功能
多维数组对象（ndarray）：
NumPy 数组是一个多维数组对象，称为 ndarray。它是一个具有统一类型（通常是数值类型）元素的集合，可以进行高效的向量化运算。
广播功能：
广播是 NumPy 的一个强大功能，它允许不同形状的数组进行算术运算的能力。较小的数组会在运算过程中自动扩展以匹配较大数组的形状。
数组索引和切片：
NumPy 提供了多种方式来索引和切割数组，允许高效地访问和修改数组的部分内容。
数学函数库：
NumPy 提供了大量的数学函数，如三角函数、指数函数、对数函数等，它们都可以直接作用于数组级别，从而实现快速的向量运算。
线性代数支持：
NumPy 包含了基本的线性代数函数，包括矩阵乘法、矩阵分解、行列式、求解线性系统等。
随机数生成：
NumPy 也提供了生成各种概率分布的随机数的工具。


![](2024-04-18-20-00-52.png)

![](2024-04-18-20-01-26.png)
这些都是numpy array


#### How does `predict` actually work?

- Matrix multiplication with coefficients (`@`) and add intercept
![](2024-04-18-20-03-12.png)


```python
# create numpy array
np.array([7,8,9])
# array([7, 8, 9])

# Creating numpy array of 8 1's
np.ones(8)
# array([1., 1., 1., 1., 1., 1., 1., 1.])

# Creating numpy array of 8 0's
np.zeros(8)
```


![](2024-04-18-20-07-27.png)


#### Back to `numpy`
- `np.arange([start, ]stop, [step, ])`: gives us an array based on range; documentation: https://numpy.org/doc/stable/reference/generated/numpy.arange.html
![](2024-04-18-20-08-08.png)

#### Review: Slicing

- `seq_object[<START>:<exclusive_END>:<STEP>]`
    - `<START>` is optional; default is index 0
    - `<END>` is optional; default is `len` of the sequence
- slicing creates a brand new object instance


python list
```python
# create copy, 因此change b并不会影响a的内容
a = [7, 8, 9, 10]
# slice out 8 and 10
b =  a[1::2] # 1 is index, 2 is step 1+2=3, 10 index is 3
b
#[8, 10]   ##这里其实做了一个deep copy


b[1] = 100
#[8,100]

```

version in numpy

```python
a = np.array([7, 8, 9, 10])
b = a[1::2]
b
# array([ 8, 10])

b[1] = 100
a

# array([  7,   8,   9, 100])

```

<font color = "red">注意：</font>这里的a发生了改变，是因为在numpy中，b依旧是referencing a，因此对b进行改变，a也会改变。

How can you ensure that changes to a slice don't affect original `numpy.array`? Use `copy` method.


```python
a = np.array([7, 8, 9, 10])
b = a.copy() # copy everything instead of sharing
b = a[1::2] 
b[1] = 100
b, a
# (array([  8, 100]), array([  7,   8,   9, 100]))
```



#### Creating Multi-Dimensional Arrays

- using nested data structures like list of lists
- `shape` gives us the dimension of the `numpy.array`
- `len()` gives the first dimension, that is `shape[0]`

![](2024-04-18-20-18-09.png)

![](2024-04-18-20-19-06.png)


#### How to reshape a `numpy.array`?

- `<obj>.reshape(<newshape>)`: reshapes the dimension of the array; documentation: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

```python
b.reshape(3,2)


# array([[1, 2],
    #    [3, 4],
    #    [5, 6]])


b.reshape(6)
# array([1, 2, 3, 4, 5, 6])

b.reshape(5)
# 这个会出现报错，因为reshape的规则是，必须可以被total number 整除。
cannot reshape array of size 6 into shape (5,)

```

-1 means whatever size is the necessary shape for the remaining values. Enables us to just control one of the dimensions.

-1表示剩余部分的大小。

```python
# Use .reshape to change the dimensions to 3 x something valid
b.reshape(3,-1)

# array([[1, 2],
#        [3, 4],
#        [5, 6]])

b.reshape(-1)
# array([1, 2, 3, 4, 5, 6]) calculate dimensions size of only 1 dimensions using
```


### Vocabulary
- scalar: 0 dimensional array
- vector: 1 dimensional array
- matrix: 2 dimensional array
- tensor: n dimensional (0, 1, 2, 3, ...) array 


### Images as Tensors

- `wget` command:
    - `wget <url> -O <local file name>`


![](2024-04-18-20-34-52.png)


#### How to read an image file?

- required `import matplotlib.pyplot as plt`
    - `plt.imread(<fname>)`: reads an image file into a 3-dimensional array --- rows(pixels), columns(pixels), colors (red/green/blue)
    - `plt.imshow(<array>, cmap=<color map>)`: displays the image

![](2024-04-18-20-41-30.png)


![](2024-04-18-20-42-16.png)

a is using number to represent image


#### GOAL: crop down just to the bug using slicing

- `<array>[ROW SLICE, COLUMN SLICE, COLOR SLICE]`

![](2024-04-18-20-45-16.png)

![](2024-04-18-20-45-32.png)

Wherever there was red, the image is bright. The bug is very bright because of that. There are other places in the image that are bright but were not red. This is because when we mix RGB, we get white. Any color that was light will also have a lot of RED.

This could be a pre-processing step for some ML algorithm that can identify RED bugs. 

![](2024-04-18-20-46-44.png)


#### GOAL: show a grayscale that considers the average of all colors

- `<array>.mean(axis=<val>)`:
    - `axis` should be 0 for 1st dimension, 1 for 2nd dimension, 2 for 3rd dimension

```python
# average over all the numbers
# gives a measure of how bright the image is overall
a.mean()

# 91.74619781513016

a.shape
# (1688, 2521, 3)
# average over each column and color combination
a.mean(axis=0).shape  
# (2521, 3)

# 指定的 axis=0 表示沿着第一个维度（即沿着形状中的1688）进行计算。这意味着你将对所有1688个元素（在这个上下文中可能是图像）的相同位置的值取平均。计算后，这个维度会被压缩或者消除，结果中不再包含这个维度。
# average over each row and color combination
a.mean(axis=1).shape  
# (1688, 3)


# average over each row and column combination
a.mean(axis=2).shape
# (1688, 2521)
```

using average of all color

![](2024-04-18-20-53-56.png)

注意这里与上面的图像还有所区别，这是平均值，但是在上面是用0，也就是red，所以高亮更明显。


### Vector Multiplication: Overview

#### Elementwise Multiplication

$\begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix}
*
\begin{bmatrix}
4 \\ 5 \\ 6
\end{bmatrix}$

$\begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix}
*
\begin{bmatrix}
4 & 5 & 6
\end{bmatrix}$


```python
# Use .reshape to change the dimensions to something valid x 1 
# vertical shape
v1 = np.array([1, 2, 3])
v1
#array([1, 2, 3])

v2 = np.array([4, 5, 6]).reshape(-1, 1)
v2
# array([[4],
    #    [5],
    #    [6]])
```

![](2024-04-18-21-13-07.png)

```python
v1 * v2   # [1*4, 2*5, 3*6]  
# 注意 * element wise multiplication
# @ 是matrix multiplication
# 因此会获得这样的结果
# array([[ 4,  8, 12],
#        [ 5, 10, 15],
#        [ 6, 12, 18]])

v1 @ v2

#array([32])
```


直接进行dot production
$\begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix}
*
\begin{bmatrix}
4 \\ 5 \\ 6
\end{bmatrix}$

这是没有办法进行matrix multiplication的
但是可以dot production

```python
v1 = np.array([1, 2, 3])
v1
#array([1, 2, 3])

v2 = np.array([4, 5, 6])
v2
#array([4, 5, 6])

v1 * v2
#array([ 4, 10, 18])
```


#### Transpose

- flips the x and y

![](2024-04-18-21-26-02.png)


Transpose是如何运作的

![](2024-04-18-21-26-43.png)


```python
v1 = np.array([1, 2, 3]).reshape(-1, 1)
v1


# array([[1],
#        [2],
#        [3]])



v2 = np.array([4, 5, 6]).reshape(-1, 1)
v2


# array([[4],
#        [5],
#        [6]])


v1 * v2 # 1*4, 2*5, 3*6
# array([[ 4],
#        [10],
#        [18]])
```

$\begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix}
*
\begin{bmatrix}
4 \\ 5 \\ 6
\end{bmatrix}$
\=
$\begin{bmatrix}
4 \\ 10 \\ 18
\end{bmatrix}$



```python
v2.T
#array([[4, 5, 6]])
v1 * v2.T  # how is this working?  会boardcast
# array([[ 4,  5,  6],
#        [ 8, 10, 12],
#        [12, 15, 18]])
```

$\begin{bmatrix}
1 \\ 2 \\ 3
\end{bmatrix}
*
\begin{bmatrix}
4 & 5 & 6
\end{bmatrix}$
\=
?



## Broadcast

When compute A * B:
- If A and B have the same number of dimensions: 
    - Match the size of any dim by stretching 1 => N (rule 1)
- else: 
    - add dimensions of size 1 to the beginning of a shape (rule 2)


#### `np.concatenate([a1, a2, ...], axis=0)`.
- `a1, a2, …`: sequence of arrays
- `axis`: the dimension along with we want to join the arrays
    - default value is 0, which is for row dimension (vertically)
    - value of 1 is for column dimension (horizontally)



np.concatenate() 是 NumPy 库中的一个函数，用于沿指定轴连接数组序列。这个函数可以处理不同维度的数组，但要求除了指定的轴之外，所有其他轴的长度必须相同。

参数解释
a1, a2, ... : 这些是需要连接的数组，可以是任意数量，但它们必须具有相同的形状，除了在指定的轴上。
axis=0 : 这指定了数组将要沿着哪个轴进行连接。默认情况下，axis=0，意味着连接操作是沿着第一个轴（即行方向，对于2D数组）进行的。

![](2024-04-19-12-11-41.png)
![](2024-04-19-12-12-09.png)


![](2024-04-19-12-43-34.png)



## Predicting with Matrix Multiplication

1. use case for matrix multiplication:
    - `y = Xc + b`
2. one's column
3. matrix multiply vector

$\begin{bmatrix}
1 & 2 \\ 3 & 4\\
\end{bmatrix}
\cdot
\begin{bmatrix}
10 \\ 1 \\
\end{bmatrix}$


```python
houses = pd.DataFrame([[2, 1, 1985],
                       [3, 1, 1998],
                       [4, 3, 2005],
                       [4, 2, 2020]],
                      columns=["beds", "baths", "year"])
houses

```

![](2024-04-19-12-51-08.png)

```python
def predict_price(house):
    """
    Takes row (as Series) as argument,
    returns estimated price (in thousands)
    """
    return ((house["beds"]*42.3) + (house["baths"]*10) + 
            (house["year"]*1.67) - 3213)

predict_price(houses.iloc[0])


# 196.54999999999973

```
以上这个过程比较繁琐，因此可以通过matrix来简化步骤，

以下是优化

```python
# How do we convert a DataFrame into a numpy array?
X = houses.values
X

# 结果会转换为numpy array

# array([[   2,    1, 1985],
#        [   3,    1, 1998],
#        [   4,    3, 2005],
#        [   4,    2, 2020]])


house0 = X[0:1,:]
house0

#array([[   2,    1, 1985]])


# Create a vertical array (3 x 1) with the co-efficients
c = np.array([42.3, 10, 1.67]).reshape(-1,1)
c

# 这里是将上面×的数转换为一个np array

# array([[42.3 ],
#        [10.  ],
#        [ 1.67]])

# 并且将其reshape 为（3，1）
# 因此可以做matrix multiplication


# horizontal @ vertical
house0 @ c

# array([[3409.55]])

```
更具这个公式
`y = Xc + b`
house0 @ c - 3213    # 3213是intercept


另一种更简便的方式，可以将-3213作为一列放入c中

```python
c = np.array([42.3, 10, 1.67, -3213]).reshape(-1, 1)
c

# array([[ 4.230e+01],
#        [ 1.000e+01],
#        [ 1.670e+00],
#        [-3.213e+03]])

# 注意这里没有办法直接使用house0 @ c， 因为dimension不同
因此我们需要在X中多加一列为1的元素

one_col = np.ones(4).reshape(-1,1) 
# array([[1.],
#        [1.],
#        [1.],
#        [1.]])
X = np.concatenate([X, ones_column], axis=1)
X
# 这的axis = 1 可以理解为增加宽度，
# axis = 0是增加长度

# array([[2.000e+00, 1.000e+00, 1.985e+03, 1.000e+00],
#        [3.000e+00, 1.000e+00, 1.998e+03, 1.000e+00],
#        [4.000e+00, 3.000e+00, 2.005e+03, 1.000e+00],
#        [4.000e+00, 2.000e+00, 2.020e+03, 1.000e+00]])


# 这个时候可以进行matrix multiplication


house0 = X[0:1, :]
house0

#array([[2.000e+00, 1.000e+00, 1.985e+03, 1.000e+00]])

house0 @ c

# array([[196.55]])
# 结果是一样的


# Extracting each house and doing the prediction
# Cumbersome
house0 = X[0:1, :]
print(house0 @ c)
house1 = X[1:2, :]
print(house1 @ c)
house2 = X[2:3, :]
print(house2 @ c)
house3 = X[3:4, :]
print(house3 @ c)

# [[196.55]]
# [[260.56]]
# [[334.55]]
# [[349.6]]

以上这些代码等价于下面的代码

X @ c

# array([[196.55],
#        [260.56],
#        [334.55],
#        [349.6 ]])

```

### Fitting with `np.linalg.solve`


**Above:** we estimated house prices using a linear model based on the matrix multiplication as follows:

$Xc = y$

* $X$ (known) is a matrix with house features (from DataFrame)
* $c$ (known) is a vector of coefficients (our model parameters)
* $y$ (computed) are the prices

**Below:** what if X and y are know, and we want to find c?

使用model来train function 从而获取最好的coef


![](2024-04-19-13-11-46.png)

If we assume price is linearly based on the features, with this equation:

* $beds*c_0 + baths*c_1 + year*c_2 + 1*c_3 = price$

Then we get four equations:

* $2*c_0 + 1*c_1 + 1985*c_2 + 1*c_3 = 196.55$
* $3*c_0 + 1*c_1 + 1998*c_2 + 1*c_3 = 260.56$
* $4*c_0 + 3*c_1 + 2005*c_2 + 1*c_3 = 334.55$
* $4*c_0 + 2*c_1 + 2020*c_2 + 1*c_3 = 349.60$



#### `c = np.linalg.solve(X, y)`

- documentation: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html



```python
houses['ones'] = 1
houses

```
![](2024-04-19-13-17-14.png)



```python
X = houses[["beds", "baths", "year", "ones"]].values
X



# array([[   2,    1, 1985,    1],
#        [   3,    1, 1998,    1],
#        [   4,    3, 2005,    1],
#        [   4,    2, 2020,    1]])



y = houses[["price"]].values
y


# array([[196.55],
#        [260.56],
#        [334.55],
#        [349.6 ]])
```
![](2024-04-19-13-16-36.png)



```python

c = np.linalg.solve(X,y)   # 通过模型计算出来的coef  -3.213e+03
c

# array([[ 4.230e+01],
#        [ 1.000e+01],
#        [ 1.670e+00],
#        [-3.213e+03]])


X @ c 

# array([[196.55],
#        [260.56],
#        [334.55],
#        [349.6 ]])

# 下面来做一个预测
dream_house = np.array([2,2,2024,1])
dream_house @ c

# array([271.68])



```



### Two Perspectives on `Matrix @ vector`

$\begin{bmatrix}
4&5\\6&7\\8&9\\
\end{bmatrix}
\cdot
\begin{bmatrix}
2\\3\\
\end{bmatrix}
= ????
$


```python
X = np.array([[4, 5], [6, 7], [8, 9]])
c = np.array([2, 3]).reshape(-1, 1)
X @ c


# array([[23],
#        [33],
#        [43]])


```


### Row Picture

![](2024-04-19-13-28-17.png)


```python
def matrix_multi_by_row(X, c):
    """
    function that performs same action as @ operator
    """
    res = []
    for row_idx in range(X.shape[0]):
        row = X[row_idx:row_idx+1,:]
        res.append((row @ c).item())
    return res
matrix_multi_by_row(X, c)

```



### Column Picture
![](2024-04-19-13-29-24.png)


```python

def matrix_multi_by_col(X, c):
    """
    same result as matrix_multi_by_row above, 
    but different approach
    """
    total = np.zeros(X.shape[0]).reshape(-1, 1)
    # loop over each col index of X
    for col_idx in range(X.shape[1]):
        # extract each column using slicing
        col = X[:, col_idx:col_idx+1]
        # extract weight for the column using indexing
        weight = c[col_idx, 0]
        # add weighted column to total
        total += col * weight
    return total
matrix_multi_by_col(X, c)

# 能和上面得到一样的结果


