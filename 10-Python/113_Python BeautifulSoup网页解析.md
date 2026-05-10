# Python BeautifulSoup网页解析


## 🔍 BeautifulSoup 网页解析


BeautifulSoup 解析器、find/find_all 选择器、CSS 选择器 select、导航 DOM 树、数据提取实战。


## BeautifulSoup 基础


```
// ========== 安装与基本使用 ==========
# pip install beautifulsoup4 lxml

from bs4 import BeautifulSoup

# HTML 示例
html = """

测试页面

Hello World
第一段文字
第二段文字
链接1
链接2

版权信息


"""

# 解析 HTML
soup = BeautifulSoup(html, "html.parser")
# 或使用 lxml (更快):
# soup = BeautifulSoup(html, "lxml")

# 获取标签
print(soup.title)             # 测试页面
print(soup.title.string)      # "测试页面"
print(soup.title.text)        # "测试页面"

print(soup.h1)                # 第一个 h1
print(soup.h1["id"])          # "main-title"
print(soup.h1.text)           # "Hello World"

print(soup.a)                 # 第一个 a 标签
print(soup.a["href"])         # "https://example.com/1"

# prettify: 格式化输出
print(soup.prettify())
```


## find / find_all 选择器


```
// ========== 搜索方法 ==========
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")

# find_all: 查找所有匹配标签
all_p = soup.find_all("p")
print(len(all_p))             # 2
for p in all_p:
    print(p.text)

# 带 class 过滤
content_p = soup.find_all("p", class_="content")
print([p.text for p in content_p])

# 带 id 过滤
title = soup.find("h1", id="main-title")
print(title.text)

# 多个条件
links = soup.find_all("a", href=True)  # 有 href 属性的 a
for a in links:
    print(a["href"], a.text)

# 文本内容匹配
import re
p_with_文 = soup.find_all("p", string=re.compile("文"))
print([p.text for p in p_with_文])

# limit 限制数量
first_two = soup.find_all("a", limit=2)

# ========== find (单个) ==========
first_p = soup.find("p")
print(first_p.text)

# 不存在的返回 None
not_found = soup.find("nonexistent")
print(not_found)  # None

# ========== find_parent / find_next_sibling ==========
first_a = soup.find("a")
parent = first_a.find_parent("div")
print(parent)  # 父 div

next_p = first_a.find_next_sibling("p")
print(next_p)  # 下一个 p 兄弟
```


## CSS 选择器 (select)


```
// ========== CSS 选择器 ==========
from bs4 import BeautifulSoup

soup = BeautifulSoup(html, "html.parser")

# select: 使用 CSS 选择器语法 (返回列表)
# select_one: 返回单个元素

# 标签选择器
soup.select("p")              # 所有 p
soup.select("a")              # 所有 a

# 类选择器
soup.select(".content")       # class="content"

# ID 选择器
soup.select("#main-title")    # id="main-title"
soup.select("#footer")

# 属性选择器
soup.select("a[href]")        # 有 href 的 a
soup.select('a[href*="example"]')  # href 包含 "example"
soup.select('a[href^="https"]')    # href 以 https 开头
soup.select('a[href$=".pdf"]')     # href 以 .pdf 结尾

# 层级选择器
soup.select("div p")          # div 内的 p
soup.select("div > p")        # div 的直接子元素 p
soup.select("h1 + p")         # h1 紧邻的下一个 p

# 组合选择器
soup.select("p.content")      # p 且 class="content"
soup.select("h1, p")          # 所有 h1 和 p

# select_one: 单个元素
first_link = soup.select_one("a")
print(first_link["href"])
```


## 导航 DOM 树


```
// ========== DOM 导航 ==========
from bs4 import BeautifulSoup

html_nested = """


项目1
项目2
项目3


额外内容


"""
soup = BeautifulSoup(html_nested, "html.parser")
ul = soup.find("ul")

# 子元素
print(ul.contents)      # 子节点列表 (包括文本)
print(list(ul.children))  # 迭代器版本

# 子孙元素
print(list(ul.descendants))  # 所有后代

# 父元素
print(ul.parent)           # 直接父元素
print(ul.parents)          # 所有祖先 (生成器)

# 兄弟元素
li = ul.find("li")
print(li.next_sibling)      # 下一个兄弟 (包含文本节点)
print(li.previous_sibling)  # 上一个兄弟
print(li.next_element)      # 下一个元素 (深度优先)
print(li.previous_element)  # 上一个元素

# find_next / find_previous
print(li.find_next("li"))        # 下一个 li
print(li.find_next("div"))       # 下一个 div
print(li.find_previous("ul"))    # 上一个 ul

# find_all_next / find_all_previous
print(li.find_all_next("li"))    # 之后所有的 li
```


## 数据提取实战


```
// ========== 提取链接和图片 ==========
from bs4 import BeautifulSoup
import requests

def extract_links(url):
    """提取页面中所有链接"""
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        links.append({
            "text": a.text.strip(),
            "href": a["href"],
        })
    return links

def extract_images(url):
    """提取页面中所有图片"""
    resp = requests.get(url, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    images = []
    for img in soup.find_all("img"):
        images.append({
            "src": img.get("src"),
            "alt": img.get("alt", ""),
            "width": img.get("width"),
            "height": img.get("height"),
        })
    return images

# ========== 提取表格数据 ==========
html_table = """


姓名年龄城市


Alice25北京
Bob30上海
Charlie35广州


"""

soup = BeautifulSoup(html_table, "html.parser")

# 提取表头
headers = [th.text for th in soup.find_all("th")]
print("表头:", headers)

# 提取数据行
rows = []
for tr in soup.find_all("tr")[1:]:  # 跳过表头
    cells = [td.text for td in tr.find_all("td")]
    rows.append(dict(zip(headers, cells)))

for row in rows:
    print(row)

# ========== 提取 JSON-LD ==========
def extract_jsonld(soup):
    """提取页面中的结构化数据"""
    scripts = soup.find_all("script", type="application/ld+json")
    import json
    for script in scripts:
        try:
            data = json.loads(script.string)
            print(data)
        except (json.JSONDecodeError, TypeError):
            pass
```


## 完整示例: 爬取新闻标题


```
// ========== 新闻爬虫 ==========
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Article:
    title: str
    url: str
    summary: str = ""

class NewsScraper:
    """简单新闻爬虫"""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def fetch_articles(self, url: str = None) -> List[Article]:
        if url is None:
            url = self.base_url

        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []

        # 常见的文章链接模式
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = a_tag.text.strip()

            # 过滤: 至少有 10 个字符,看起来像文章标题
            if len(text) < 10:
                continue

            # 补全相对 URL
            if href.startswith("/"):
                href = self.base_url.rstrip("/") + href

            # 跳过非 HTTP 链接
            if not href.startswith("http"):
                continue

            articles.append(Article(title=text, url=href))

        return articles

    def close(self):
        self.session.close()

# 使用示例
scraper = NewsScraper("https://httpbin.org")
try:
    articles = scraper.fetch_articles()
    print(f"找到 {len(articles)} 个链接")
    for a in articles[:5]:
        print(f"  {a.title[:50]:50} {a.url}")
finally:
    scraper.close()
```


> **Note:** 💡 BeautifulSoup: find_all/find 按标签名/属性搜索; select 使用 CSS 选择器; .contents/.children/.parent 导航; 结合 requests/httpx 实现爬虫。


## 练习


<!-- Converted from: 113_Python BeautifulSoup网页解析.html -->
