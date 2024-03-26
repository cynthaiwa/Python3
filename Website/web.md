# Web in Python 
*<font size = 2>--Every Webpage is a Tree</font>*

## Introduce web
**HTML（HyperText Markup Language）**：一种创建网页的标准标记语言 TAG NAME
```python
<h1> - <h6> # => header 其中 1 - 6 是控制header 字体的大小（从大到小）
#eg.
<h1></h1>


<p></p> # => 段落

<a href="https://www.runoob.com">URL</a> # => this is url

<img src="dashboard2.svg"><br><br>  # html image

<li></li>  # => list item

<ul></ul> # => unsorted list

<div></div> # => 定义级块元素，通胀用作容器来组织页面上的其他元素
<table boarder = "1"></table>  #=> 标签创建一个带有边框的表格
<tr></tr> # => row
<td></td> # 定义了表格中的标准单元格（table data）。它通常用于在表格（<table>）内部创建一个单元格

#eg.

<ul>
    <li><a href="browse.html">Browse</a></li>
    <li><a href="donate.html?from=A" style = "color:red">Donate</a></li>
    <button onclick="subscribe()">Subscribe</button>
</ul>


<b></b> # => 文本加粗

```
**HTTP Response**: HTTP 响应是指在客户端（如浏览器）向服务器发送HTTP请求后，服务器返回给客户端的信息。

Status: 
404 : Not Found/n
429 : To Many Request
500 : There has some BUG
200 : OK

![](2024-03-24-13-31-51.png)


## Selenium
<font size = 2>Selenum is most commonly used for testing web, but it works great for tricky scraping too</font>


-can fetch .html, .js, .etc file
-can run a .js file in browser
-can grab HTML version of <font color = "red">DOM(Document Object Model)</font> after JavaScript has modified it

<font color = "blue">*Document Object Model（DOM）是一种跨平台和语言独立的接口，它允许程序和脚本动态地访问和更新文档的内容、结构和样式。DOM 将一个 XML 或 HTML 文档表示为一个树状结构，其中每个节点都是文档中的一个对象，比如文档本身、元素、属性和文本。*</font>

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
# keeping checking regularly until you get all the details you are looking for
while True:
    tbls = b.find_elements(By.TAG_NAME, "table")
    print("Tables:", len(tbls))
        
    if len(tbls) == 2:
        print(tbls)
        break
     
    time.sleep(0.1) # sleep for 0.1 second



# get attribute value
a_elements = b.find_elements(By.TAG_NAME, 'a')
a_elements
for a in a_elements:
    print(a.get_attribute("href"), a.text)


output
https://tyler.caraza-harter.com/cs320/crawl/practice1/2.html 2   # 这里的2 和 3 是text value
https://tyler.caraza-harter.com/cs320/crawl/practice1/3.html 3
```

获取表格中的内容
```python
tbl = tbls[-1]

# TODO: find all tr elements
trs = tbl.find_elements(By.TAG_NAME, 'tr')

# TODO: find all td elements
# TODO: extract text for all td elements into a list of list
rows = []

for tr in trs:
    tds = tr.find_elements(By.TAG_NAME, 'td')
    assert len(tds) == 2
    
    rows.append([tds[0].text, tds[1].text])
# .text =>  获取页面元素的可见文本
rows
```
VM通过以上操作可以模拟browser对网页的操作，但是是无法直接查看到网页的，因此可以通过截图来或许网页内容。
```python
b.save_screenshot("some_file.png")# : saves a screenshot of the rendered page
# 可以通过b.set_window_size(width, height) 来修改screenshot的大小， 在save screenshot 之前使用
# import statement: 
from IPython.display import display Image
# helps us show the screenshot as an image inside the notebook

display(Image("page1.png"))  # through this operation we can see screen shot in notebook
```


事件触发和模拟
```python
## click
button_object.click()  # enable us to click the button
# first we need ues id to find button
button = b.find_element(By.ID, 'more')
# for this we can check html
# at this case, html is look like this => <button id="more" onclick="show()">Show More!</button>

# then we can us .click() operation to simulate browsers click button
button.click()


## password
text_object.send_keys()  # enables us to send data to textbox
# find text box
text = b.find_element(By.ID, 'password')   # tag name is input
# find click button
button = b.find_element(By.ID, 'login_btn') # tag name is button

# TODO: send the password (plain text just for example purposes)
text.clear() # first clean text box
#.clear() operation enables us to clear the previous text
text.send_keys('fido')

# TODO: click the button
button.click()


```
<h4> Selenium operations 合集</h4>
    
    b.get(URL): sends HTTP GET request to the URL

    b.page_source: HTML source for the page

    b.find_elements(By.ID, <ID>): searches for a specific element that matches the "id"

    b.find_elements(By.TAG_NAME, <TAG>): searches for a specific element using corresponding tag name

    b.find_element versus b.find_elements:
        find_element gives first match
        find_elements gives all matches

    <element obj>.text: gives text associated with that element

    <element obj>.get_attribute(<attribute>): gives attribute value; for ex: <anchor_obj>.get_attribute("href")

    b.save_screenshot("some_file.png"): saves a screenshot of the rendered page

    b.set_window_size(<width>, <height>): controls size of the image

    import statement: 
    from IPython.display import display, Image : helps us show the screenshot as an image inside the notebook

    button_element.click(): enables us to click the button

    text_element.send_keys(): enables us to send data to textbox


## Recursive Crawl
就像之前所说的Web是Tree结构，因此我们可以通过recursive的方法来获取所有web中的webpage
<font size = 3>crawling: process of finding all the webpage inside the website</font>


<h5>找到当前web的所有webpage</h5>

```python
def get_children(url):
    b.get(url)
    children = []
    a_elements = b.find_elements(By.TAG_NAME,"a")
    for a in a_elements:
        children.append(a.get_attribute('href'))
    return children

```
<h5>可以使用BFS找到当前web的所有子webpage</h5>
也就是说找到当前node的所有children -- 结合上面的get_children function

```python
# example url
start_url = "https://tyler.caraza-harter.com/cs320/crawl/practice7/1.html"

to_visit = deque([start_url])
added = {start_url}

while len(to_visit) > 0:
    cur_page = to_visit.popleft()
    print("cur:", cur_page)
    #g.node(cur_page.split("/")[-1])  #diagraph

    children = get_children(cur_page)

    for child in children:
        if not child in added:
            #g.edge(cur_page.split("/")[-1], child.split("/")[-1])
            to_visit.append(child)
            added.add(child)


```
if we uncomment code, we can get this result
![](2024-03-25-17-41-30.png)


## Flask


Flask是一个用Python编写的轻量级Web应用框架，它允许开发者快速地开发Web应用。在开发过程中，开发者会在本地计算机上启动一个Web服务器来测试他们的应用
```python
import flask
# flask syntax
app = flask.Flask("my application") # name of the web application can be anything

if __name__ == "__main__":
# 这行代码检查是否是直接运行这个脚本，而不是从另一个脚本导入它。在Python中，如果一个文件被直接运行，那么它的__name__变量会被设置为"__main__"
    app.run(host="0.0.0.0", debug=True, threaded=False) # this line will running forever, until program kill.
    #这行代码启动Flask应用的Web服务器。
```
默认情况下index.html会被默认为web的Home Page
```python
@app.route("/")  # 默认将index.html 作为homepage， 当前目录里必须有index.html文件
def home():
    with open("index.html") as f:
        html = f.read()
    return html

```
inside index.html

    <h1> Welcome </h1>


开始服务器
```python
python3 -m http.server # 打开：8000 服务器上应用程序监听的端口号
```
HTTPS: Hypertext Transfer Protocol Secure(超文本传输安全协议)
<font size = 2>是HTTP的安全版本。它用于在互联网上安全地传输数据，确保数据传输过程中的加密和完整性，以及网络通信双方的身份验证。</font>



URL的格式大致为协议://主机地址:端口号/路径
<font color = "red">Part1:Part2/Part3</font>

    Part1 means get request to correct computer
    Part2 means get request to correct server process
    Part3 means specify what resource we want from the server



<h4>Add Dynamic Content to File</h4>

```python
@app.route("/ha.html")
def laugh():
    return "haha" * 1000
# but in real life we only need to change part of the webpage
# TEMPLATE semi-static / semi-dynamic

@app.route("/time.html")
def time():
    with open("time.html") as f:
        html = f.read()   #html will be read as string
    html = html.replace("REPLACE", str(time.time())) # import time
    return html
# .replace() operation can replace element in html file
```
inside time.html

    <h1>time</h1>
        <p> Current time is REPLACE seconds after 1970.</p>


<h4> Rate Limiting</h4>
limit user visit webpage -- eg. If the user accesses the Web page several times in a short period of time, 429 Error will appear

```python
last_visit = 0

@app.route("/slow.html")
def slow():
    # if allow to visit:
    #     return regular content
    # else:
    #     return some warning here
    global last_visit  # showing that this is from global
    if time.time() - last_visit > 3:
        last_visit = time.time()
        return "Welcome"
    else:
        return flask.Response("<b>go away</b>",
                             status = 429,
                             headers = {"Retry-After": "3"})


```
<h4>flask.request </h4>
封装了客户端发起的HTTP请求的所有细节。通过它，你可以访问请求相关的数据，如请求头（Headers）、查询字符串（Query String）、表单数据、JSON数据、文件等。

以下是一些常用的flask.request属性和方法：

    request.method：获取当前请求的HTTP方法，例如GET、POST等。
    request.args：一个包含所有查询字符串参数的不可变字典（ImmutableMultiDict）。
    request.form：一个包含所有表单数据的不可变字典，用于处理POST或PUT请求的表单数据。
    request.json：如果请求是JSON格式的数据，这个属性包含解析后的JSON数据。
    request.files：一个包含所有上传文件的字典。
    request.headers：一个包含所有HTTP请求头的字典。
    flask.request.remote_addr: 可以获取访问网页的用户ip 地址


<h4>flask.response </h4>
代表了服务器发给客户端的HTTP响应。Flask允许你通过多种方式来构建响应对象。最简单的方式是直接从视图函数返回一个字符串和一个状态码，Flask会自动将其转换为一个响应对象。
在header中你可以添加任何你需要的东西作为 meta data

创建新的route，告诉爬虫哪些网页是可以获取的，哪些是不可以的

```python
@app.route("/never.html")
def never():
    return "human only, bots are note allowed"

# TODO: create a robots.txt page
@app.route("/robots.txt")
def bot_rules():
    return flask.Response("""
    User-Agent: *
    Disallow: /never.html
    """, headers={"Content-Type": "text/plain"})
# flask.Response: enables us to create a response object instance
# 		  Arguments: str representing response, headers dict representing metadata


```


<h4>urllib.robotparser模块</h4>
目的是解析指定网站的robots.txt文件。robots.txt是一个位于网站根目录下的文本文件，它告诉网络爬虫（搜索引擎的爬虫或其他自动化服务）哪些页面可以被抓取，哪些不可以。通过解析这个文件，程序可以遵守网站的爬虫政策，只访问允许访问的资源。


```python
import urllib.robotparser
rp = urllib.robotparser.RobotFileParser()
rp.set_url(base_url + "robots.txt")
rp.read()


rp.can_fetch("cs320bot", base_url + "/slow.html")

# 使用can_fetch(user_agent, url)方法时，你需要提供两个参数：

# user_agent: 代表请求检查权限的用户代理的名称。
# url: 需要检查是否允许访问的完整URL。
# 方法会返回一个布尔值：

# 如果返回True，则表示根据robots.txt的规则，提供的用户代理被允许访问指定的URL。
# 如果返回False，则表示根据robots.txt的规则，提供的用户代理不被允许访问指定的URL。
```


<h4>Query String</h4>
<font size = 2>用于传递额外的参数给Web服务器</font>
example:

    http://example.com/page?key1=value1&key2=value2

    ? 符号用于标识查询字符串的开始。
    key1=value1 表示第一个参数，其中key1是参数名，value1是参数值。
    & 符号用于分隔多个参数。
    key2=value2 表示第二个参数，其中key2是参数名，value2是参数值。

How to use query string?
```python
@app.route("/add.html")
def adder():                # args is query string argument
    args = dict(flask.request.args) # request -> user send to flask server
    # args is arguments
    try:
        x = float(args["x"])
        y = float(args["y"])
    except KeyError:
        return "Please tell me x and y"
        
    return f"{x} + {y} = {x+y}"

```
如果少了一个参数，则会返回报错信息。

    http://34.134.230.194:5000/add.html?x=2&y=3


**<font color = "red">Query String is ImmutableMultiDict</font>**
它的数据结构是非常类似于dict的，但是不同的是他是immutable的，无法修改。可以通过转换为dict来使用query string.


```python

major_counts = {}


@app.route("/survey.html")
def survey():
    major = flask.request.args.get("major","unknow")
    
    if major not in major_counts:
        major_counts[major] = 0
    major_counts[major] += 1    
    client_ip = flask.request.remote_addr
    response = 'Your IP address is {}. '.format(client_ip)
    # format方法是Python中常用的字符串格式化方法之一，它允许你在字符串中插入一个
    # 或多个占位符{}，然后通过format方法提供的参数替换这些占位符。


    response += "Majors: " + json.dumps(major_counts)
    
    return response
# client flask.request => server
#.args => query string
# .remote_addr => IP address that sends this request
# server flask.Response(...) => client
# content, status, header(content-type, retry-after)
```



## A/B Testing

是一种用于比较两个版本的网页或应用（版本A和版本B），以确定哪一个版本表现更好的统计分析方法。

Example Metric: CTR(Click-Through Rate)
CTR = clicks / impressions
"impression" means user sas it: that is, Impression = click + no clidk
![](2024-03-25-22-16-18.png)

Pvalue smaller means more likely that A ≠ B

in this example threshold is 5% => 0.05
pvalue is 0.189
hence we have enough evidence that A ≠ B

A/B testing: 将用户随机分配到两个或者多个版本页面

A/A testing: 验证测试系统有效性

当在A/B testing 结束后，发现 B version更好，但是担心是novelty原因，因此 可以switch to B and do B/A testing。

**simple size 也会影响A/B testing的结果**
example：

![](2024-03-25-22-23-57.png)

----------------------------------------------------------

![](2024-03-25-22-24-08.png)









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