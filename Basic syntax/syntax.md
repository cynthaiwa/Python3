# Basic Syntax(Python)


## Json

JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，易于人阅读和编写，同时也易于机器解析和生成。它基于JavaScript的一个子集，但是JSON是完全独立于语言的文本格式，今天几乎所有的编程语言都有解析JSON数据和生成JSON数据的代码库。

<h5>JSON结构</h5>
对象（Object）：对象在JSON中表示为一组键值对，用大括号{}包围。每个键后面跟着一个冒号，键值对之间用逗号分隔。
数组（Array）：数组在JSON中是值的有序集合，用方括号[]表示。数组中的值之间用逗号分隔。
值（Value）：值可以是双引号内的字符串、数字、true、false、null、对象或数组。

example.json
```python
{
  "name": "John Doe",
  "age": 30,
  "isEmployed": true,
  "address": {
    "street": "123 Main St",
    "city": "Anytown"
  },
  "phoneNumbers": [
    {"type": "home", "number": "212 555-1234"},
    {"type": "office", "number": "646 555-4567"}
  ]
}


```


    json.dump(obj, fp, *, skipkeys=False, ensure_ascii=True, 
    check_circular=True, allow_nan=True, cls=None, indent=None, 
    separators=None, default=None, sort_keys=False, **kw)  
    #用于将 Python 对象编码成 JSON 格式的字符串，并将其写入到一个文件中。





## deque
Function definition

```python
from collections import deque


append(x) # =>  add x to right
appendleft(x) # =>  add x to left
pop() # => remove and return, right side
popleft() # => remove and return, left side
```
All this function complexity is <font color = "red">O(1)</font>


## heapq
Heap is a binary tree, the value of each node is less than or equal to the value of each of its child nodes. => min heap

**python中是没有内置的max heap的函数的。因此我们可以通过将min heap中的所有value变为negative，然后再将其heapify。就可以得到一个max heap。当我们需要用value的时候，需要将其转为positive。**

```python
heapq.heapify(x) # => change list to heap   Complexity is O(N)
# notice that the first element in heap is always smallest element
heapq.heappush(heap, item) # => add element to heap     Complexity is O(logN)
heapq.heappop # => remove the smallest element in heap     Complexity is O(logN)
# the first element always smallest

 

```
