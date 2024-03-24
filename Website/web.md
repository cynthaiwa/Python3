# Web in Python 
*<font size = 2>--Every Webpage is a Tree</font>*

## Introduce web
**HTML（HyperText Markup Language）**：一种创建网页的标准标记语言
```python
<h1> - <h6> # => header 其中 1 - 6 是控制header 字体的大小（从大到小）
#eg.
<h1></h1>


<p></p> # => 段落

<a href="https://www.runoob.com">URL</a> # => this is url

<img src="dashboard2.svg"><br><br>  # html image

<li></li>  # => list item

<ul></ul> # => unsorted list

#eg.

<ul>
    <li><a href="browse.html">Browse</a></li>
    <li><a href="donate.html?from=A" style = "color:red">Donate</a></li>
    <button onclick="subscribe()">Subscribe</button>
</ul>


```
**HTTP Response**: HTTP 响应是指在客户端（如浏览器）向服务器发送HTTP请求后，服务器返回给客户端的信息。

Status: 
404 : Not Found/n
429 : To Many Request
500 : There has some BUG
200 : OK

![](2024-03-24-13-31-51.png)


## Selenium
<font size = 2>Selenum is most commonly used for testing web, but iw works great for tricky scraping too</font>


-can fetch .html, .js, .etc file
-can run a .js file in browser
-can grab HTML version of DOM(Document Object Model) after JavaScript has modified it

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
```

```python
options = Options() # => 用于配置chrome浏览器的启动选项
options.add_argument("--headless=new") # don't use a GUI (necessary on a VM)
# 无头模式，浏览器不会显示图形界面。
b = webdriver.Chrome(options=options) # => 这行代码初始化了一个Chrome浏览器的实例
url = ....
b.get(url) # sends HTTP GET request to the URL
# get方法用于指示浏览器导航到指定的
print(b.page_source)  # 获取当前页面的全部html代码，不包括Javascript 生成的HTML
```
<h5>Finding Element</h5>

```python
b.find_elements(By.ID, "alpha") # search for a specific element that matches ID\
# first match
# eg.
# <h1>Welcome</h1>
#     <h3>Here's a table</h3>
#     <table border="1" id="alpha">
#       <tbody><tr><td>A</td><td>B</td><td>C</td></tr>
#       <tr><td>1</td><td>2</td><td>3</td></tr>
#       <tr><td>4</td><td>5</td><td>6</td></tr>
#     </tbody></table>


b.find_elements(By.TAG_name, "table") # 上面的table就是一个TAG name
# find all tables

# 一些时候Javascript没有办法立刻生成新的表格，因此需要等待才能获取全部的table
# Polling：
while True:
    tbls = b.find_elements(By.TAG_NAME, "table")
    print("Tables:", len(tbls))
        
    if len(tbls) == 2:
        print(tbls)
        break
     
    time.sleep(0.1) # sleep for 0.1 second


```


## matplotlib

```python
import matplotlib.pyplot as plt

```

## Regular Expression

<font color="red"> / </font> is sign for escape original meaning  
```python
print("A\tB")

output:
A   B
```

<font color="red"> r </font> is a raw string, which means it will return the original string.  
```python
print(r"A\tB")

output:
A\tB
```
same as  

```python
print("A\\tB")

output:
A\tB
```
"\\" and "r" cancel original mean of character of sign, "\t" mean tab, and those two sign cancel "\t" mean, hence, the code will print "\t".
![](2024-03-22-15-03-25.png)