[TOC]

# 第五章 pandas入门

引入规定：

```python
from pandas import Series,DataFrame
import pandas as pd
```

## 5.1 pandas的数据结构介绍

​	pandas首先要先学会它的两个数据结构：Series，DataFrame。

### 5.1.1 Series

​	Series可看做一个一维数组。她由一组数据和与之相关的数据标签（索引）组成

​	创建：`obj = Series([4,7,-6,3],index=['d','b','a','c']) `

​	通过`obj.value`和`obj.index`访问数据及其索引；

​	默认的`index`是从`0`到`N-1`的整数型索引

​	

​	实验：创建Series时value列表长度与index列表长度不同会报错

```python
In [1]: obj1 = Series([4,5,6,7],index=[1,2,3,3,4])
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    obj1 = Series([4,5,6,7],index=[1,2,3,3,4])
  File "F:\code_Toool\python\python\lib\site-packages\pandas\core\series.py", line 262, in __init__
    .format(val=len(data), ind=len(index)))
ValueError: Length of passed values is 4, index implies 5
```

​	实验：index中含有相同值不会报错

```python
>>> obj1 = Series([4,5,6,7],index=[1,2,4,4])
>>> obj1
1    4
2    5
4    6
4    7
>>> obj1[4]
4    6
4    7
dtype: int64
```

​	对Series对象的访问方式，单个的和一组值的（按照你输入的顺序）：

```python
>>> obj2
b    4
d    7
a   -5
c    6
dtype: int64
>>> obj2['a']
-5
>>> obj2[['c','a','d']]
c    6
a   -5
d    7
dtype: int64
```

​	可对Series对象进行运算（布尔型函数进行过滤，基本运算等）

```python
>>> obj2[obj2>0]
b    4
d    7
c    6
dtype: int64
>>> obj2+2
b    6
d    9
a   -3
c    8
```

​	若数据放在字典中，可直接通过字典来创建：

```python
>>> sdata = {'Ohio':35000,'texas':71000,'Oregon':16000,'Utah':5000}
>>> obj3 = Series(sdata)
>>> obj3
Ohio      35000
texas     71000
Oregon    16000
Utah       5000
dtype: int64
```

​	自行传入索引的情况：

```python
>>> states = ['California','Ohio','Oregon','Texas']
>>> obj4 = Series(sdata,index = states)
>>> obj4
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas             NaN
dtype: float64
```

​	改变索引的情况：

```python
>>> obj
state
0    4
1    5
2    6
3    2
Name: population, dtype: int64
>>> obj.index=['Bob','Steve','Jeff','Ryan']
>>> obj
Bob      4
Steve    5
Jeff     6
Ryan     2
Name: population, dtype: int64
```

​	其中不存在的值用NaN，判断这个可用两个函数：

```python
>>> pd.isnull(obj4)
California     True
Ohio          False
Oregon        False
Texas          True
dtype: bool
>>> pd.notnull(obj4)
California    False
Ohio           True
Oregon         True
Texas         False
dtype: bool
```

​	Series对象本身及其索引都有一个name索引：

```python
>>> obj4.name = 'population'
>>> obj4.index.name = 'state'
>>> obj4
state
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas             NaN
Name: population, dtype: float64
```

### 5.1.2 DataFrame

​	DataFrame表格型的数据结构，包含一组有序的列，每列可以是不同的值类型（数值，字符串，布尔值等）。

​	常用直接传入一个由等长列表或NumPy数组组成的字典：

```python
>>> data
{'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002], 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
>>> frame = DataFrame(data)
>>> frame		# 会自动加上索引
    state  year  pop
0    Ohio  2000  1.5
1    Ohio  2001  1.7
2    Ohio  2002  3.6
3  Nevada  2001  2.4
4  Nevada  2002  2.9
```

​	通过column属性指定列序列：

```python
>>> DataFrame(data,columns=['year','state','pop'])
   year   state  pop
0  2000    Ohio  1.5
1  2001    Ohio  1.7
2  2002    Ohio  3.6
3  2001  Nevada  2.4
4  2002  Nevada  2.9
```

​	若传入的列在数据中找不到，就会产生NA值

```python
>>> frame2 = DataFrame(data,columns=['year','state','pop','debt'],index=['one','two','three','four','five'])	# 也可自行指定索引
>>> frame2
       year   state  pop debt
one    2000    Ohio  1.5  NaN
two    2001    Ohio  1.7  NaN
three  2002    Ohio  3.6  NaN
four   2001  Nevada  2.4  NaN
five   2002  Nevada  2.9  NaN
>>> frame2['state']	# 字典的获取方式
one        Ohio
two        Ohio
three      Ohio
four     Nevada
five     Nevada
Name: state, dtype: object
>>> frame2.year	# 属性的获取方式
one      2000
two      2001
three    2002
four     2001
five     2002
Name: year, dtype: int64
>>> type(frame2)
<class 'pandas.core.frame.DataFrame'>
>>> type(frame2.year)
<class 'pandas.core.series.Series'>
>>> type(frame2['state'])
<class 'pandas.core.series.Series'>
```

​	DataFrame的行也可以通过为止或名称的方式进行获取，比如用索引字段`ix`

```python
>>> frame2.ix['three']
year     2002
state    Ohio
pop       3.6
debt      NaN
Name: three, dtype: object
>>> type(frame2.ix['three'])	# 这也是一个Series对象
<class 'pandas.core.series.Series'>
```

​	列可以通过赋值方式修改。

```python
>>> frame2['debt'] = 16.5
>>> frame2
       year   state  pop  debt
one    2000    Ohio  1.5  16.5
two    2001    Ohio  1.7  16.5
three  2002    Ohio  3.6  16.5
four   2001  Nevada  2.4  16.5
five   2002  Nevada  2.9  16.5
>>> frame2['debt'] = np.arange(5)
>>> frame2
       year   state  pop  debt
one    2000    Ohio  1.5     0
two    2001    Ohio  1.7     1
three  2002    Ohio  3.6     2
four   2001  Nevada  2.4     3
five   2002  Nevada  2.9     4
```

​	将列表或数组赋值给DataFrame某个列时，长度必须跟DataFrame的列长度一样。

​	若赋值时Series时，则会根据Series的索引进行匹配，匹配不上的填上缺失值

```python
>>> val = Series([-1.2,-1.5,-1.7],index=['two','four','five'])
>>> frame2['debt'] = val
>>> frame2
       year   state  pop  debt
one    2000    Ohio  1.5   NaN
two    2001    Ohio  1.7  -1.2
three  2002    Ohio  3.6   NaN
four   2001  Nevada  2.4  -1.5
five   2002  Nevada  2.9  -1.7
```

​	为不存在的列赋值会创建出新列。关键字del用于删除列：

```python
>>> frame2['eastern'] = frame2.state =='Ohio'
>>> frame2
       year   state  pop  debt  eastern
one    2000    Ohio  1.5   NaN     True
two    2001    Ohio  1.7  -1.2     True
three  2002    Ohio  3.6   NaN     True
four   2001  Nevada  2.4  -1.5    False
five   2002  Nevada  2.9  -1.7    False
>>> frame2.columns
Index(['year', 'state', 'pop', 'debt'], dtype='object')
```

​	__通过索引方式得到的是相应数据的视图,并不是副本__，对返回的Series进行修改需要进行复制，通过Series的copy方法

```python
>>> a = frame2.ix['one'].copy()
>>> a
year     2000
state    Ohio
pop       1.5
debt      NaN
Name: one, dtype: object
>>> a['year'] = 1003	# a改变了
>>> a
year     1003
state    Ohio
pop       1.5
debt      NaN
Name: one, dtype: object
>>> frame2	# frame2并未改变
       year   state  pop  debt
one    2000    Ohio  1.5   NaN
two    2001    Ohio  1.7  -1.2
three  2002    Ohio  3.6   NaN
four   2001  Nevada  2.4  -1.5
five   2002  Nevada  2.9  -1.7
```

​	把嵌套字典传给DataFrame：外层键作为列，内层键作为行（索引）

```python
>>> pop = {'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
>>> frame3 = DataFrame(pop)
>>> frame3
      Nevada  Ohio
2000     NaN   1.5
2001     2.4   1.7
2002     2.9   3.6
>>> frame3.T
        2000  2001  2002
Nevada   NaN   2.4   2.9
Ohio     1.5   1.7   3.6
```

​	DataFrame的index和columns都有name属性：

```python
>>> frame3.index.name = 'year'
>>> frame3.columns.name = 'state'
>>> frame3
state  Nevada  Ohio
year               
2000      NaN   1.5
2001      2.4   1.7
2002      2.9   3.6
```

​	DataFrame 也有values属性：返回一个二维ndarray数组

```python 
>>> frame3.values
array([[nan, 1.5],
       [2.4, 1.7],
       [2.9, 3.6]])
>>> frame2.values
array([[2000, 'Ohio', 1.5, nan],
       [2001, 'Ohio', 1.7, -1.2],
       [2002, 'Ohio', 3.6, nan],
       [2001, 'Nevada', 2.4, -1.5],
       [2002, 'Nevada', 2.9, -1.7]], dtype=object)
```

### 5.1.3 索引对象

​	索引对象负责管理标签和其他元数据。

​	索引对象不可修改，这样使得index对象在多个数据结构之间的安全共享

​	每个索引还有一些方法和属性。

## 5.2 基本功能

### 5.2.1 重新索引

​	pandas的reindex方法：创建一个适应新索引的新对象。根据新索引重排，不存在则引入缺失值。

```python
>>> obj = Series([4.5,7.2,-5.3,3.6],index = ['d','b','a','c'])
>>> obj
d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64
>>> obj2 = obj.reindex(['a','b','c','d','e'])
>>> obj2
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64
>>> obj.reindex(['a','b','c','d','e'],fill_value = 0)	# fill_value可确定缺失值的赋值。
a   -5.3
b    7.2
c    3.6
d    4.5
e    0.0
dtype: float64

```

​	重新索引时可能需要做一些插值处理。method选项可达到这个目的

```python
>>> obj3 = Series(['blue','purple','yellow'],index = [0,2,4])
>>> obj3
0      blue
2    purple
4    yellow
dtype: object
>>> obj3.reindex(range(6),method='ffill')	# ffill可以实现前向值填充
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
```

​	人的心的method选项

|      参数       |         说明         |
| :-------------: | :------------------: |
|   ffill或pad    | 前向填充（或搬运）值 |
| bfill或backfill | 后向填充（或搬运）值 |

​	对DataFrame，index可修改行，列索引或两个都修改。若仅传入一个序列，仅重新索引行。

```python
>>> frame = DataFrame(np.arange(9).reshape((3,3)),index = ['a','c','d'],columns=['Ohio','Texas','California'])
>>> frame
   Ohio  Texas  California
a     0      1           2
c     3      4           5
d     6      7           8
>>> frame2 = frame.reindex(['a','b','c','d'])
>>> frame2
   Ohio  Texas  California
a   0.0    1.0         2.0
b   NaN    NaN         NaN
c   3.0    4.0         5.0
d   6.0    7.0         8.0
>>> state = ['Texas','Utah','California']
>>> frame.reindex(columns = state)	# 使用columns关键字可重新索引列
   Texas  Utah  California
a      1   NaN           2
c      4   NaN           5
d      7   NaN           8
```

### 5.2.2 丢弃指定轴上的项

​	所需函数：`drop`方法；    操作：一个索引数组或列表；

```python
# 对于Series
>>> obj = Series(np.arange(5.),index = ['a','b','c','d','e'])
>>> obj
a    0.0
b    1.0
c    2.0
d    3.0
e    4.0
dtype: float64
>>> new_obj = obj.drop('c')
>>> new_obj
a    0.0
b    1.0
d    3.0
e    4.0
dtype: float64
>>> obj.drop(['d','c'])
a    0.0
b    1.0
e    4.0
dtype: float64
>>> obj
a    0.0
b    1.0
c    2.0
d    3.0
e    4.0
dtype: float64

# 对于DataFrame
>>> data = DataFrame(np.arange(16).reshape((4,4)),index = ['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
>>> data
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
>>> data.drop(['Colorado','Ohio'])	# 删除两个行
          one  two  three  four
Utah        8    9     10    11
New York   12   13     14    15
>>> data.drop('two',axis=1)	# 删除一列
          one  three  four
Ohio        0      2     3
Colorado    4      6     7
Utah        8     10    11
New York   12     14    15
```

### 5.2.3 索引、选取和过滤

​	Series的索引类似于Numpy的数组索引，只不过Series的索引值不只是整数。

```python
>>> obj = Series(np.arange(4.),index = ['a','b','c','d'])
>>> obj
a    0.0
b    1.0
c    2.0
d    3.0
dtype: float64
>>> obj['b']
1.0
>>> obj[1]
1.0
>>> obj[2:4]
c    2.0
d    3.0
dtype: float64
>>> obj[['b','a','d']]
b    1.0
a    0.0
d    3.0
dtype: float64
>>> obj['b':'d']	# 这个索引值是有先后的；并且标签切片运算是包含末端的（这里是'd')
b    1.0
c    2.0
d    3.0
>>> obj['b':'d']=5	# 设置值
>>> obj
a    0.0
b    5.0
c    5.0
d    5.0
dtype: float64
```

​	对DataFrame进行索引就是获取一个或多个列：	这里有比较重要的概念辨析：

```python
#直接用DataFrame对象获取行会报错
>>> data['Colorado']	# 不能得到行
Traceback (most recent call last):
    KeyError: 'Colorado'

>>> data = DataFrame(np.arange(16).reshape((4,4)),index=['Ohio','Colorado','Utah','New York'],columns=['one','two','three','four'])
>>> data
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
>>> data['two']	
Ohio         1
Colorado     5
Utah         9
New York    13
Name: two, dtype: int32
>>> data[['three','one']]
          three  one
Ohio          2    0
Colorado      6    4
Utah         10    8
New York     14   12
>>> data[:2]	# 通过切片选取行
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7

>>> data[data['three']>5]	# 通过布尔型数组过滤行
          one  two  three  four
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
# 举例：
>>> data<5
            one    two  three   four
Ohio       True   True   True   True
Colorado   True  False  False  False
Utah      False  False  False  False
New York  False  False  False  False
>>> data['three']>5
Ohio        False
Colorado     True
Utah         True
New York     True
Name: three, dtype: bool
```

​	通过索引字段ix可以在DataFrame上进行标签索引。可用来选取行和列的子集。

```python
>>> data.ix['Colorado',['two','three']]
two      5
three    6
Name: Colorado, dtype: int32
>>> data.ix[['Colorado','Utah'],[3,0,1]]
          four  one  two
Colorado     7    4    5
Utah        11    8    9
>>> data.ix[2]
one       8
two       9
three    10
four     11
Name: Utah, dtype: int32
>>> data.ix[:'Utah','two']
Ohio        1
Colorado    5
Utah        9
Name: two, dtype: int32
>>> data.ix[data.three>5,['two','three']]	# 选择列3中大于五的行，及相应的列2中的行
          two  three
Colorado    5      6
Utah        9     10
New York   13     14
```

​	`xs`方法：根据标签选取单行或单列：`xs(key,axis=0,level=None,drop_level=True)`

```python
>>> a1 = data.xs('Colorado')
>>> a1
one      4
two      5
three    6
four     7
Name: Colorado, dtype: int32
>>> a2 = data.xs('one',axis=1)
>>> a2
Ohio         0
Colorado     4
Utah         8
New York    12
Name: one, dtype: int32
```

​	`get_value`和`set_value`用于选取和设置：

```python
# 选取值
>>> data.get_value('Ohio','one')
0
>>> data.get_value('Ohio','two')   
1

# 设置值 报了警告
>>> data.set_value('Colorado','two',99)
		   

Warning (from warnings module):
  File "__main__", line 1
FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead
          one  two  three  four
Ohio        0    1      2     3
Colorado    4   99      6     7
Utah        8    9     10    11
New York   12   13     14    15

# iat[]和at[]更改值的新方式
>>> data.iat['Colorado','two']=100	# iat[]用于纯整数的索引
Traceback (most recent call last):
  File "<pyshell#101>", line 1, in <module>
    data.iat['Colorado','two']=100
  File "F:\code_Toool\python\python\lib\site-packages\pandas\core\indexing.py", line 2157, in __setitem__
    key = list(self._convert_key(key, is_setter=True))
  File "F:\code_Toool\python\python\lib\site-packages\pandas\core\indexing.py", line 2288, in _convert_key
    raise ValueError("iAt based indexing can only have integer "
ValueError: iAt based indexing can only have integer indexers、
>>> data.iat[1,1]=101
>>> data
		     
          one  two  three  four
Ohio        0    1      2     3
Colorado    4  101      6     7
Utah        8    9     10    11
New York   12   13     14    15
                     
                     
>>> data.at['Colorado','two']=100	# 用DataFrame中的标签来索引
>>> data
		     
          one  two  three  four
Ohio        0    1      2     3
Colorado    4  100      6     7
Utah        8    9     10    11
New York   12   13     14    15
```

​	`reindex`方法：将一个或多个轴匹配到新索引

```python
>>> data
          one  two  three  four
Ohio        0    1      2     3
Colorado    4  101      6     7
Utah        8    9     10    11
New York   12   13     14    15
>>> data.index
Index(['Ohio', 'Colorado', 'Utah', 'New York'], dtype='object')
>>> data.columns
Index(['one', 'two', 'three', 'four'], dtype='object')
>>> data.reindex(['Colorado','Ohio','Utah', 'New York','Shanghai'])
           one    two  three  four
Colorado   4.0  101.0    6.0   7.0
Ohio       0.0    1.0    2.0   3.0
Utah       8.0    9.0   10.0  11.0
New York  12.0   13.0   14.0  15.0
Shanghai   NaN    NaN    NaN   NaN
>>> data.reindex(['two','three','five','four','one'],axis=1)
          two  three  five  four  one
Ohio        1      2   NaN     3    0
Colorado  101      6   NaN     7    4
Utah        9     10   NaN    11    8
New York   13     14   NaN    15   12
```

### 5.2.4 算术运算和数据对齐

​	对不同索引的对象进行算术运算，有共同索引的相加（`+`操作），不同索引的求并集

```python
>>> s1 = Series([7.3,-2.5,3.4,1.5],index=['a','c','d','e'])
>>> s1
a    7.3
c   -2.5
d    3.4
e    1.5
dtype: float64
>>> s2 = Series([-2.1,3.6,-1.5,4,3.1],index=['a','c','e','f','g'])
>>> s2
a   -2.1
c    3.6
e   -1.5
f    4.0
g    3.1
dtype: float64
>>> s1+s2	# 两个对象中不重叠存在的记为缺失值NaN
a    5.2
c    1.1
d    NaN
e    0.0
f    NaN
g    NaN
dtype: float64
```

​	对于DataFrame进行相加运算同时考虑行和列：

```python
>>> df1 = DataFrame(np.arange(9.0).reshape((3,3)),columns = list('bcd'),index=['Ohio','Texas','Colorado'])
>>> df1
            b    c    d
Ohio      0.0  1.0  2.0
Texas     3.0  4.0  5.0
Colorado  6.0  7.0  8.0
>>> df2 = DataFrame(np.arange(12.0).reshape((4,3)),columns = list('bde'),index=['Utah','Ohio','Texas','Oregon'])
>>> df2
          b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregon  9.0  10.0  11.0
>>> df1+df2
            b   c     d   e
Colorado  NaN NaN   NaN NaN
Ohio      3.0 NaN   6.0 NaN
Oregon    NaN NaN   NaN NaN
Texas     9.0 NaN  12.0 NaN
Utah      NaN NaN   NaN NaN
```

#### 5.2.4.1 在算术方法中填充值

​	还是上一节的主题在算术运算中，一个对象中的某个轴标签再领一个对象中找不到时，给它填充一个特殊值。使用`add`方法传入要相加的对象，以及一个fill_value参数

```python
>>> df1 = DataFrame(np.arange(12.).reshape((3,4)),columns = list('abcd'))
>>> df1
     a    b     c     d
0  0.0  1.0   2.0   3.0
1  4.0  5.0   6.0   7.0
2  8.0  9.0  10.0  11.0
>>> df2 = DataFrame(np.arange(20.).reshape((4,5)),columns = list('abcde'))
>>> df2
      a     b     c     d     e
0   0.0   1.0   2.0   3.0   4.0
1   5.0   6.0   7.0   8.0   9.0
2  10.0  11.0  12.0  13.0  14.0
3  15.0  16.0  17.0  18.0  19.0
>>> df1 + df2		# 这种方式会产生NA值
      a     b     c     d   e
0   0.0   2.0   4.0   6.0 NaN
1   9.0  11.0  13.0  15.0 NaN
2  18.0  20.0  22.0  24.0 NaN
3   NaN   NaN   NaN   NaN NaN
>>> df1.add(df2,fill_value=0)	# 本节使用的方式，显然它的结果令人意外。仔细辨认是相应位置的下标。
      a     b     c     d     e
0   0.0   2.0   4.0   6.0   4.0
1   9.0  11.0  13.0  15.0   9.0
2  18.0  20.0  22.0  24.0  14.0
3  15.0  16.0  17.0  18.0  19.0
>>> df1.add(df2,fill_value=1)	# 值为1，缺失值的填充的是下标加1
      a     b     c     d     e
0   0.0   2.0   4.0   6.0   5.0
1   9.0  11.0  13.0  15.0  10.0
2  18.0  20.0  22.0  24.0  15.0
3  16.0  17.0  18.0  19.0  20.0

# 在reindex中也可以指定填充值，填充方式很能理解
>>> df1.reindex(columns=df2.columns,fill_value=0)
     a    b     c     d  e
0  0.0  1.0   2.0   3.0  0
1  4.0  5.0   6.0   7.0  0
2  8.0  9.0  10.0  11.0  0
```

| 方法 | 说明 |
| :--: | :--: |
| add  | 加法 |
| sub  | 减法 |
| div  | 除法 |
| mul  | 乘法 |

#### 5.2.4.2 DataFrame和Series之间的运算

​	进行广播后的结果；

```
>>> arr = np.arange(12.).reshape((3,4))
>>> arr
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]])
>>> arr[0]
array([0., 1., 2., 3.])
>>> arr-arr[0]
array([[0., 0., 0., 0.],
       [4., 4., 4., 4.],
       [8., 8., 8., 8.]])
```

​	DataFrame和Series之间的运算:

```python
>>> frame
          b     d     e
Utah    0.0   1.0   2.0
Ohio    3.0   4.0   5.0
Texas   6.0   7.0   8.0
Oregin  9.0  10.0  11.0
>>> series = frame.ix[0]
>>> series
b    0.0
d    1.0
e    2.0
Name: Utah, dtype: float64
>>> frame-series	# DataFrame和Series之间的运算会将Series的索引匹配到DataFrame的列，然后沿着行一直广播
          b    d    e
Utah    0.0  0.0  0.0
Ohio    3.0  3.0  3.0
Texas   6.0  6.0  6.0
Oregin  9.0  9.0  9.0
>>> series2 = frame.ix[1]
>>> series2
b    3.0
d    4.0
e    5.0
Name: Ohio, dtype: float64
>>> frame-series2
          b    d    e
Utah   -3.0 -3.0 -3.0
Ohio    0.0  0.0  0.0
Texas   3.0  3.0  3.0
Oregin  6.0  6.0  6.0
```

​	如果某个索引值在DataFrame的列或者Series的索引中找不到，则参与运算的两个对象会被重新索引形成并集。

```python
>>> series3 = Series(range(3),index = ['b','e','f'])
>>> frame + series3
          b   d     e   f
Utah    0.0 NaN   3.0 NaN
Ohio    3.0 NaN   6.0 NaN
Texas   6.0 NaN   9.0 NaN
Oregin  9.0 NaN  12.0 NaN
```

​	希望运算在列上广播，则必须使用算术运算方法：

```python
>>> series4 = frame['d']
>>> series4
Utah       1.0
Ohio       4.0
Texas      7.0
Oregin    10.0
Name: d, dtype: float64
>>> frame - series4
        Ohio  Oregin  Texas  Utah   b   d   e
Utah     NaN     NaN    NaN   NaN NaN NaN NaN
Ohio     NaN     NaN    NaN   NaN NaN NaN NaN
Texas    NaN     NaN    NaN   NaN NaN NaN NaN
Oregin   NaN     NaN    NaN   NaN NaN NaN NaN
>>> frame.sub(series4,axis=0)
          b    d    e
Utah   -1.0  0.0  1.0
Ohio   -1.0  0.0  1.0
Texas  -1.0  0.0  1.0
Oregin -1.0  0.0  1.0
```

### 5.2.5 函数应用和映射

​	numpy的ufuncs（元素级数组方法）也可用于操作pandas对象：

```python
>>> frame = DataFrame(np.random.randn(4,3),columns=list('bde'),index=['utah','Ohio','Texas','Oregon'])
>>> frame
               b         d         e
utah    1.028567 -0.709809  0.444396
Ohio    0.610814 -0.344337  0.334814
Texas  -0.124318  0.603154 -0.992892
Oregon  0.600432 -1.160766  0.082406
>>> np.abs(frame)
               b         d         e
utah    1.028567  0.709809  0.444396
Ohio    0.610814  0.344337  0.334814
Texas   0.124318  0.603154  0.992892
Oregon  0.600432  1.160766  0.082406
```

​	传给apply的函数还可以返回多个值组成的Series：

```python
>>> def f(x):
	return Series([x.min(),x.max()],index = ['min','max'])
>>> frame.apply(f)
            b         d         e
min -0.124318 -1.160766 -0.992892
max  1.028567  0.603154  0.444396
```

​	元素级的python函数可以用。

```python
>>> format = lambda x: '%.2f' %x
>>> frame.applymap(format)
            b      d      e
utah     1.03  -0.71   0.44
Ohio     0.61  -0.34   0.33
Texas   -0.12   0.60  -0.99
Oregon   0.60  -1.16   0.08
>>> frame['e'].map(format)	# Series有一个用于应用于元素级函数的map方法
utah       0.44
Ohio       0.33
Texas     -0.99
Oregon     0.08
Name: e, dtype: object
```

### 5.2.6 排序和排名

​	排序可以根据sort_index方法来排序：

```python
>>> obj = Series(range(4),index = ['d','a','b','c'])
>>> obj.sort_index()	# 对Series的index进行排序
a    1
b    2
c    3
d    0
dtype: int64
    
>>> frame = DataFrame(np.arange(8).reshape((2,4)),index=['three','one'],columns=['d','a','b','c'])
>>> frame
       d  a  b  c
three  0  1  2  3
one    4  5  6  7
>>> frame.sort_index()	# 对DataFrame的列进行排序
       d  a  b  c
one    4  5  6  7
three  0  1  2  3
>>> frame.sort_index(axis=1)	# 对DataFrame的行进行排序
       a  b  c  d
three  1  2  3  0
one    5  6  7  4

# 进行设置ascending属性可以是排序成为降序
>>> frame.sort_index(axis=1,ascending=False)
       d  c  b  a
three  0  3  2  1
one    4  7  6  5

# 通过sort_values()对值进行排序
>>> obj = Series([4,7,-3,2])
>>> obj.order()
Traceback (most recent call last):
    AttributeError: 'Series' object has no attribute 'order'
>>> obj.sort_values()	# sort_value方法可以代替order方法。order在3.6后不可用了
2   -3
3    2
0    4
1    7
dtype: int64

# 排序时任何缺失值默认放到Series的末尾
>>> obj = Series([4,np.nan,7,np.nan,-3,2])
>>> obj
0    4.0
1    NaN
2    7.0
3    NaN
4   -3.0
5    2.0
dtype: float64
>>> obj.sort_values()
4   -3.0
5    2.0
0    4.0
2    7.0
1    NaN
3    NaN
dtype: float64
    
# 排序时想要根据一个或多个列中的值进行排序，把列传递给by属性
>>> frame = DataFrame({'b':[4,7,-3,2],'a':[0,1,0,1]})
>>> frame
   b  a
0  4  0
1  7  1
2 -3  0
3  2  1
>>> frame.sort_index(by='b')

Warning (from warnings module):
  File "__main__", line 1
FutureWarning: by argument to sort_index is deprecated, please use .sort_values(by=...)
   b  a
2 -3  0
3  2  1
0  4  0
1  7  1
>>> frame.sort_values(by='b')
   b  a
2 -3  0
3  2  1
0  4  0
1  7  1

# 多个列的情况，传入列的名称
>>> frame.sort_index(by=['a','b'])
   b  a
2 -3  0
0  4  0
3  2  1
1  7  1

# 排名
	# 和排序紧密相连，会增设一个排名值，从1开始，不过可以根据某种规则破坏平级关系
		# rank() 破坏平级关系（表现为：排名中值相等的项就是平级关系）使用的是：“为各组分配一个平均排名
>>> obj = Series([7,-5,7,4,2,0,4])
>>> obj.rank()	#两个‘7’的排名分别为6和7，他们的平均排名是6.5； 两个‘4’的排名分别为4和5，他们的平均排名是4.5
0    6.5
1    1.0
2    6.5
3    4.5
4    3.0
5    2.0
6    4.5
dtype: float64

#不想要上面的方式，可根据平级关系出现的先后给出排名
>>> obj.rank(method = 'first')	#index=0 的 ‘7’比 index=2 的‘7’要早出现
0    6.0	
1    1.0
2    7.0
3    4.0
4    3.0
5    2.0
6    5.0
dtype: float64

# ascending属性可以按降序排；‘max’从最大排名开始排
>>> obj.rank(ascending = False,method = 'max')
0    2.0
1    7.0
2    2.0
3    4.0
4    5.0
5    6.0
6    4.0
dtype: float64

# DataFrame的排名
>>> frame = DataFrame({'b':[4.3,7,-3,2],'a':[0,1,0,1],'c':[-2,5,8,-2.5]})
>>> frame
     b  a    c
0  4.3  0 -2.0
1  7.0  1  5.0
2 -3.0  0  8.0
3  2.0  1 -2.5
>>> frame.rank(axis =1)
     b    a    c
0  3.0  2.0  1.0
1  3.0  1.0  2.0
2  1.0  2.0  3.0
3  3.0  2.0  1.0
>>> frame.rank(axis=0)
     b    a    c
0  3.0  1.5  2.0
1  4.0  3.5  3.0
2  1.0  1.5  4.0
3  2.0  3.5  1.0
```

|  method   |         说明         |
| :-------: | :------------------: |
| ‘average' | 为各个值分配平均排名 |
|   'min'   |     使用最小排名     |
|   'max'   |     使用最大排名     |
|  'first'  |    按出现顺序排名    |



### 5.2.7 带有重复值的轴索引

​	轴索引可以不唯一，虽然大部分见到的索引都是唯一的

```python
>>> obj = Series(range(5),index = ['a','a','b','b','c'])
>>> obj
a    0
a    1
b    2
b    3
c    4
>>> obj.index.is_unique	# 轴是否唯一
False
>>> obj.is_unique	# 值是否唯一
True

# 选取有重复值的索引，返回一个Series，仅有一个值的，返回一个标量值
>>> obj['a']
a    0
a    1
>>> obj['c']
4

# DataFrame 的情况
>>> df = DataFrame(np.random.randn(4,3),index=['a','a','b','b'])
>>> df
          0         1         2
a -0.329043  0.299399  0.170806
a -0.908796  0.173288  0.295355
b  1.038509 -0.246484  0.551983
b  0.626471  1.410364 -0.892615
>>> df.ix['a']
          0         1         2
a -0.329043  0.299399  0.170806
a -0.908796  0.173288  0.295355
```

## 5.3  汇总和计算描述统计

​	`sum() mean()`: 参数axis,skipna,level #轴，是否排除缺失值，层次化索引

```python
>>> df = DataFrame([[1.4,np.nan],[7.1,-4.5],[np.nan,np.nan],[0.75,-1.3]],index=['a','b','c','d'],columns=['one','two'])
>>> df
    one  two
a  1.40  NaN
b  7.10 -4.5
c   NaN  NaN
d  0.75 -1.3
>>> df.sum()
one    9.25
two   -5.80
dtype: float64
>>> df.su,(axis=1)
a    1.40
b    2.60
c    0.00
d   -0.55
dtype: float64
>>> df.sum(axis=1,skipna=False)
a     NaN
b    2.60
c     NaN
d   -0.55
dtype: float64
>>> df.mean(axis=1,skipna=False)	# 平均数
a      NaN
b    1.300
c      NaN
d   -0.275
dtype: float64
```

​	`idxmin,idxmax`:达到最小值或最大值的索引；这些方法是间接统计

```python
>>> df.idxmax()
one    b
two    d
dtype: object
>>> 
```

​	`cumsum()`：累计型的；

```python
>>> df.cumsum()
    one  two
a  1.40  NaN
b  8.50 -4.5
c   NaN  NaN
d  9.25 -5.8
>>> 
```

​	`discribe()`: 多个汇总统计；非数值型数据会产生另外一种汇总统计：

```python
>>> df.describe()
            one       two
count  3.000000  2.000000
mean   3.083333 -2.900000
std    3.493685  2.262742
min    0.750000 -4.500000
25%    1.075000 -3.700000
50%    1.400000 -2.900000
75%    4.250000 -2.100000
max    7.100000 -1.300000
>>> obj = Series(['a','a','b','c']*4)
>>> obj
0     a
1     a
2     b
3     c
4     a
5     a
6     b
7     c
8     a
9     a
10    b
11    c
12    a
13    a
14    b
15    c
dtype: object
>>> obj.describe()
count     16
unique     3
top        a
freq       8
dtype: object
```

​	`count`：非NA值的个数

​	`argmin,argmax`: 最小/大值的索引位置（整数）

​	`quantile`: 计算样本分位数（0~1）

​	`var`: 样本的方差

​	`std`: 样本的标准差

​	`mad`: 根据平均值计算平均绝对离差

### 5.3.1 唯一值、值计数以及成员资格

​	`isin()`	

​	`unique()`： 计算唯一值，返回数组，顺序是发现顺序

​	 `value_counts()`: 各个值出现的次数

## 5.4 处理缺失值

​	`dropna()`去除nan值

​	`fillna`填充nan值

​	`isnull`：是否为nan

​	`notnull`：isnull的否定

```python
>>> data = Series([1,np.nan,3.5,np.nan,7])
>>> data
0    1.0
1    NaN
2    3.5
3    NaN
4    7.0
dtype: float64
>>> data.dropna()
0    1.0
2    3.5
4    7.0
dtype: float64
>>> data[data.notnull()]	# notnull也可以到达这个目的
0    1.0
2    3.5
4    7.0
dtype: float64

# DataFrame 问题会复杂一些
>>> from numpy import nan as NA
>>> data = DataFrame([[1.,6.5,3.],[1.,NA,NA],[NA,NA,NA],[NA,6.5,3.]])
>>> data
     0    1    2
0  1.0  6.5  3.0
1  1.0  NaN  NaN
2  NaN  NaN  NaN
3  NaN  6.5  3.0
>>> data.dropna()
     0    1    2
0  1.0  6.5  3.0

# 传入how='all',丢弃全为NA的值
>>> data.dropna(how='all')
     0    1    2
0  1.0  6.5  3.0
1  1.0  NaN  NaN
3  NaN  6.5  3.0
# 传入axis='1',丢弃列有NA的值

```

​	填补`nan`值：

```python
>>> data
     0    1    2
0  1.0  6.5  3.0
1  1.0  NaN  NaN
2  NaN  NaN  NaN
3  NaN  6.5  3.0
>>> data.fillna(0)
     0    1    2
0  1.0  6.5  3.0
1  1.0  0.0  0.0
2  0.0  0.0  0.0
3  0.0  6.5  3.0
# 用字典调用fillna可以对不同列填充不同值：
>>> data.fillna({0:0.1,1:0.2,2:0.3})
     0    1    2
0  1.0  6.5  3.0
1  1.0  0.2  0.3
2  0.1  0.2  0.3
3  0.1  6.5  3.0
# fillna的参数：axis：轴；inplace：修改调用者对象而不产生副本；method=’ffill‘：填充方式；limit：可以连续填充的最大整数
```

## 5.5 层次化索引

