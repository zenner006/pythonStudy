# 第四章 NumPy基础：数组和矢量运算

按照标准Numpy的约定，总是使用import numpy as np

## NumPy的ndarray：一种多维数组对象

N维数组对象ndarray，快速而灵活的大数据集容器；

ndarray是同构数据多维容器，其中数据的元素必须是相同类型的；

每个数组都有shape（数组维度大小）和dtype（数组数据类型）属性

### 创建ndarray

1.  `array`函数创建数组。接受一切序列型的对象；

   (1)   传入列表；

   (2)   传入嵌套列表

2. `zeros`（全0数组）和ones（全1数组）函数

3. `empty`（返回一个空数组，都是垃圾值）输入的`shape`是一个类似元组一样的括号括起来得。

4. `eye，identity`函数返回单位矩阵

5. `ones_like，zeros_like`；以另一个数组为参照，根据其形状，创建一个全0/1数组

6. `arange`:类似于内置的`arange`，但返回得是一个`ndarray`

### ndarray的数据类型

​      dtype是一个特殊的对象，含有ndarray将一块内存解释为特定数据类型所需的信息。

​      他所支持的类型：int8（16/32/64），uint8（16/32/64），float16(32/64/128)，complex(64/128/256)（复数），bool，Object，string_，unicode_；

​      通过arr.dtype来查看arr的数据类型；

​      通过ndarray的astype方法：float_arr=arr.astype(np.float64)显式地转换其dtype。（astype无论如何都会创建一个新的数组）

### 数组与标量之间的运算

​      大小相等的数组之间的任何算术运算都会在元素中进行运算。

​      两个数组相加/减/乘都是元素间的

### 基本的索引和切片

​      索引与python大致类似。切片有所不同的是可以对它赋值。

​      一个1*10的数组`arr ：arr[5:8] = 12`    把其中索引为5,6,7的值赋为12    。

​      多维数组的访问有两种：`arr[0][2]` 和 `arr[0,2]`

​      再者python中的数据也是对象。即一个数组的部分被一个变量指向后。在其他变量下改变其值。通过后来的变量访问也是会变化的。例：

​      

```python
old_values = arr[0].copy()     # 赋给它复制了的

another_value = arr[0] # 直接赋值

arr[0] = 45

old_values再查看是没变化的。another_values是有变化的

arr[0] = old_values     # 他们的shape是一样的，就可以直接赋值
```

 

​      **切片索引**：对高纬度切片时，可以对任何一个维度进行切片。通常用冒号“：”表示选取所有的项

### 布尔型索引

​      这是个很神奇的东西。貌似之前看numpy代码时有理解。但是这里把原理讲得更加清楚。

​      首先呢就是一个判断值。对一个数字进行判断正负可以得到Ture或者False。对一个数组判断是否符合某个条件可以得到一组True或False。

​      再根据这组True，False在其他同样长的数组中挑选出True的值。

​      即：

```python
In: names = np.array([‘Bob’,’joe’,’Will’,’Bob’,’Will’,’Joe’,’Joe’])

In: names == ‘Bob’

Out: array([True,False,False,True,False,False,False],dtype=bool)

In: data = randn(7,1)

In: data[names==’Bob’]

Out: 返回True位置索引上的值。
```

​	这个bool索引的意义同切片的意义是相似的。都是剥离一部分数据出来。同样的在多维数组中可以再同时用切片对另一维进行分析。

​	除了等于（==）还有不等于（！=）；<，>,等等

​	多个bool条件可以用“-”（负），“&”（和）和“|”（或）来衔接。

### 花式索引（Fancy indexing）

​	这里的花式索引是利用整数数组进行索引。一个数组变量arr可以用索引：`arr[i]`得到第i个元素。在加入一个整数数组后：`arr[[3,4,2]]`会将索引3,4,2的元素按顺序提取出来。

​	对于多维数组可以将它同样的用花式索引。不过他使用的方式会有些不同。

​	他要得到一个矩形的行列子集

​	`In: arr[[1,5,7,2]][:,[0,3,1,2]]`

​      `Out: 得到一个二维数组，即先将arr中的行挑出（第一个中括号），再后面全部选中行，从中选择列（第二个中括号）`

​	而`arr[[1,5,7,2],[0,3,1,2]]`得到的是一个一维数组

​	要得到矩阵的行列子阵另一个方法是`np.ix_`函数，可以将两个一维整数数组转换为一个用于选取方形区域的索引器。	

​	花式索引和切片不同的是它总是复制那部分数据到新数组当中。

### 数组转置和轴对换

转置（transpose）。有三种方法（不进行任何复制操作）：

1. 对于普通的二维数组的转置可以用特殊的`T`属性

   `arr.T`就是`arr`的转置

2. 对于高维数组，`transpose`函数要得到一个由轴编号组成的元组。（比如三维数组）

   `arr.transpose((1,0,2))`其中的元组都是由0,1,2,组成，表示0轴，1轴，2轴；

   `arr.transpose((0,1,2))`这种顺序的元组使得数组是不改变的。上面那一行表示0轴和1轴进行轴对换。

   这里也可以进一步思考更高维的数组转置的轴元组该怎么写。

3. swapaxes`方法

   它接收要进行对换的轴的编号。同样是返回源数据的视图



   这里是一个这部分问题的参考链接[数组转置和轴对换CSDN博客](https://www.cnblogs.com/sunshinewang/p/6893503.html)

### 通用函数：快速的元素级数组函数

​	在`numpy`中提供了很多的数学函数，他们有一元的，也有两元的`ufunc`。在需要的时候可以查阅。在kindle这本书的书签#1745处。

### 利用数组进行数据处理

​	Numpy数组可以将许多中数据处理任务表述为简洁的数组表达式。用数组表达式代替循环的做法称为“矢量化”。

​	`meshgrid` 函数：可以把两个一维数组 $ a = [x1,x2,...,xi]  ,b = [y1,y2,...,yj]`经过函数得到：
$$
xs = \begin{bmatrix} 
			x_1&x_2&\cdots&x_i\\
			x_1&x_2&\cdots&x_i\\
			\vdots&\vdots& &\vdots\\
			x_1&x_2&\cdots&x_i
			\end{bmatrix}  
			
ys = \begin{bmatrix} 
			y_1&y_1&\cdots&y_1\\
			y_2&y_2&\cdots&y_2\\
			\vdots&\vdots& &\vdots\\
			y_j&y_j&\cdots&y_j
			\end{bmatrix}
$$
​	xs的行数是b向量元素的个数。ys的列数是a向量元素的个数。

#### 将条件逻辑表述为数组运算

​	numpy.where函数是三元表达式 `x if condition else y`

​	`np.where(cond,x,y)`其中`cond`，`x`和`y`是数组的话必须大小(shape)相等，标量的话则在被选中时返回那个标量。

#### 数学和统计方法

|     方法      |            说明             |
| :-----------: | :-------------------------: |
|      sum      |   求和，axis求某轴上的和    |
|     mean      | 算数平均数。零长度数组为NaN |
|    std,var    |        标准差，方差         |
|    min,max    |       最大值，最小值        |
| argmin,argmax |      最大最小值得索引       |
|    cumsum     |      所有元素的累计和       |
|    cumprod    |      所有元素的累计积       |

#### 用于布尔型数组的方法

​	布尔型的True和False通常会被转化为1和0；sum通常来对布尔型数组中的True计数。

```python
In [1]: arr = np.random.randn(100)

In [2]: (arr>0).sum()
Out[2]: 52
```

​	`any`：测试数组中是否存在一个或多个True，返回布尔类型

​	`all`：测试数组中是否都是布尔类型。 返回布尔类型

​	后面两个也适用于非布尔类型。非0元素也会被当做True

#### 排序

​	这里进介绍了sort

1. sort是在原数组上进行的
2. sort可以把多维数组按轴排序括号里写0或1

#### 唯一化以及其他的集合逻辑

​	最常用的集合唯一化函数`np.unique` 

```python
In [1]: names = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
In [2]: np.uniques(names)

Out[2]: array(['Bob','Joe','Will',])
    # 相较于set它是已经排序过得，即：sorted(set(names))
```

​	`np.in1d`用于测试一个数组中的值在另一个数组中的成员资格，返回一个布尔型整数：

|               方法                |                        说明                        |
| :-------------------------------: | :------------------------------------------------: |
| intersect1d(x,y)     union1d(x,y) |                     交集  并集                     |
|          setdiff1d(x,y)           |                     集合差，在                     |
|           setxor1d(x,y)           | 集合对称差，存在于一个数组但不同时存在于两个数组中 |

## 用于数组的文件输入输出

### 将数组以二进制格式保存到磁盘

​	函数np.save,np.load,np.savez（可以保存多个数组，load后返回类似字典一样的对象）

### 存取文本文件

​	`np.savetxt`，`np.loadtxt`，`np.genfromtxt`是对文本文件进行操作的函数

## 线性代数

​	矩阵乘法：`object_1.dot(object_2)`或`np.dot(obect_1,object_2)`

​	转置：`object.T`

​	逆：`inv()`	

## 随机数生成

​	`np.random.normal(size=(i,j))`生成一个`i×j`的标准正态分布样本数组

## 范例：随机漫步

​	用`np.random`模块来进行：

```python 
In [1]: nsteps = 1000
In [2]: draws = np.random.randint(0,2,size = nsteps)
In [3]: steps = np.where(draws > 0,1,-1)
In [4]: walk = steps.cumsum()
```

### 一次模拟多个随机漫步



```
In [1]: nsteps = 1000
In [2]: nwalks = 5000
In [3]: draws = np.random.randint(0,2,size =(nwalks,nsteps))
In [4]: steps = np.where(draws > 0,1,-1)
In [5]: walks = steps.cumsum(1)
```











