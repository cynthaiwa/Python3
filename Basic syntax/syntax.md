# Basic Syntax(Python)

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
