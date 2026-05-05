# 26_万维网WWW概述

## 核心概念

- **WWW（World Wide Web）**：万维网，基于互联网的信息服务系统
- **WWW三要素**：
  - **URL（Uniform Resource Locator）**：统一资源定位符，标识资源位置
  - **HTTP（HyperText Transfer Protocol）**：超文本传输协议，传输Web文档
  - **HTML（HyperText Markup Language）**：超文本标记语言，描述Web文档
- **Web文档**：由URL定位的超文本文档
- **超链接（Hyperlink）**：文档之间的链接关系
- **408考试重点**：WWW三要素、URL格式、HTTP基本概念

## 原理分析

### WWW的工作原理

1. **用户输入URL**：
   - 浏览器解析URL，提取主机名和路径
   - 浏览器调用DNS解析主机名

2. **建立连接**：
   - 浏览器与Web服务器建立TCP连接（三次握手）
   - 端口80（HTTP）或443（HTTPS）

3. **发送请求**：
   - 浏览器发送HTTP请求报文
   - 包含方法（GET/POST等）、URL、版本、头部

4. **服务器处理**：
   - Web服务器解析请求
   - 查找或生成请求的资源
   - 发送HTTP响应报文

5. **渲染页面**：
   - 浏览器接收响应报文
   - 解析HTML，渲染页面
   - 可能需要额外请求（图片、CSS、JS等）

6. **关闭连接**：
   - 根据HTTP版本，可能关闭或保持连接

### URL格式

```
协议://主机名:端口/路径?查询#片段
```

示例：`https://www.example.com:8080/path/page.html?query=value#section`

- **协议**：http、https、ftp等
- **主机名**：域名或IP地址
- **端口**：默认HTTP 80，HTTPS 443
- **路径**：资源在服务器上的位置
- **查询**：参数，以`?`开头
- **片段**：页面内锚点，以`#`开头

### HTML基本结构

```html
<html>
  <head><title>标题</title></head>
  <body>
    <h1>标题</h1>
    <p>段落</p>
    <a href="http://example.com">链接</a>
  </body>
</html>
```

## 直观理解

**WWW就像一个巨大的图书馆**：
- **URL** = 书的索书号（告诉图书馆员去哪里找）
- **HTTP** = 借书的流程（怎么借、怎么还）
- **HTML** = 书的内容格式（怎么排版、怎么组织）
- **Web浏览器** = 图书馆的自助借书机
- **Web服务器** = 图书馆的书架

**记忆技巧**：
- WWW三要素 = URL + HTTP + HTML
- URL = "地址标签"
- HTTP = "传输规则"
- HTML = "内容格式"
- "URL找地址，HTTP传数据，HTML显内容"

## 协议关联

- **WWW与DNS**：URL中的主机名需要DNS解析为IP地址
- **WWW与TCP**：HTTP基于TCP，使用三次握手建立连接
- **WWW与HTTP**：HTTP是WWW的核心协议
- **WWW与HTML**：HTML是WWW的内容格式
- **408考点**：
  - WWW三要素：URL、HTTP、HTML
  - URL的组成部分
  - HTTP的基本工作过程
- **陷阱**：WWW不等于互联网，WWW只是互联网上的一种服务
