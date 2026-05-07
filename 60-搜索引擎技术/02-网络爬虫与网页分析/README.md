# 02-网络爬虫与网页分析

## 1. 爬虫架构

网络爬虫（Web Crawler）是从互联网上自动采集网页的程序。大规模爬虫的核心架构包含以下组件：

```
种子URL → URL Frontier → DNS解析 → 网页抓取 → 网页解析 → URL提取 → 去重 → URL Frontier
                                    ↓
                              网页存储
```

- **种子URL（Seed URLs）**：爬虫启动的初始URL集合，通常来自已知的高质量网站或站点地图

- **URL Frontier（URL待抓队列）**：
  - 存储待抓取的URL，是爬虫的核心调度组件
  - 通常按优先级组织（重要页面优先抓取）
  - 包含策略：广度优先、深度优先、PageRank优先
  - 需要考虑不同站点的抓取间隔限制

- **DNS解析**：
  - 大量URL需要DNS查询，是主要瓶颈之一
  - 优化手段：本地DNS缓存、批量DNS解析、预取DNS结果

- **网页抓取**：
  - 多线程/异步并发抓取
  - HTTP客户端处理重定向、压缩、超时
  - 需要处理各种异常（4xx、5xx错误、超时、连接重置）

- **网页解析**：
  - HTML解析，提取正文内容和链接
  - 处理不同编码（UTF-8、GBK等）
  - JavaScript渲染（对SPA应用）

---

## 2. 爬虫礼貌策略

爬虫必须遵守网络礼仪，避免对目标网站造成过大负担：

### 2.1 robots.txt

- 位于网站根目录的文本文件（如 `https://example.com/robots.txt`）
- 指定哪些路径允许/禁止爬虫访问
- 格式示例：
  ```
  User-agent: *
  Disallow: /private/
  Disallow: /admin/
  Allow: /public/
  Crawl-delay: 5
  ```
- `User-agent`：指定适用的爬虫
- `Disallow`：禁止访问的路径
- `Crawl-delay`：两次抓取之间的最小间隔（秒）
- robots.txt是 **自愿遵守** 的协议，恶意爬虫可能无视

### 2.2 抓取频率限制

- **同一站点限速**：对同一域名的请求间隔不小于指定秒数
- **自适应限速**：根据服务器响应速度动态调整抓取频率
- **总带宽控制**：限制爬虫整体的网络带宽占用
- **退避策略**：遇到429（Too Many Requests）或5xx错误时指数退避

---

## 3. 网页去重

网页去重是爬虫的核心挑战之一，互联网上存在大量重复和近似重复的页面：

### 3.1 SimHash

SimHash是一种 **局部敏感哈希（LSH）** 算法，适合大规模文档近似去重：

1. 对文档分词，提取特征词及其权重
2. 对每个特征词计算哈希值（如64位）
3. 对每个位，根据特征词权重进行加权投票（正/负）
4. 最终得到64位SimHash指纹
5. 两个文档的SimHash的 **汉明距离（Hamming Distance）** 越小，越相似
6. 实践中通常认为汉明距离 ≤ 3 的文档为近似重复

**去重方式**：对新文档计算SimHash，与已有文档比较汉明距离。可将64位分为多段，利用抽屉原理加速查找。

### 3.2 MinHash

MinHash也是一种LSH算法，用于估计文档集合的 **Jaccard相似度**：

1. 对文档的Shingle集合计算多个哈希函数的最小值
2. MinHash签名相同的比例近似等于Jaccard相似度
3. 配合LSH Banding技术将相似文档分到同一桶中
4. 适合检测部分重复的内容

### 3.3 其他去重方法

- **精确去重**：计算文档MD5/SHA256指纹，完全相同的文档只保留一份
- **URL去重**：通过URL规范化去除重复URL（去除参数排序、锚点等）
- **布隆过滤器（Bloom Filter）**：用于URL去重，高效但有假阳性

---

## 4. 网页解析与正文提取

网页解析的核心任务是从HTML中提取有用信息：

- **HTML解析器**：解析HTML DOM树（如BeautifulSoup、Jsoup、Cheerio）
- **正文提取算法**：
  - 基于DOM树分析：识别正文区域（通常文本密度最高）
  - 基于文本密度：计算每个DOM节点的文本与标签比
  - 基于机器学习：训练分类器识别正文块
  - 常用工具：Boilerpipe、Trafilatura、Readability

- **结构化数据提取**：
  - 元数据（title、meta description、keywords）
  - 发布时间、作者
  - 产品价格、评分等（针对电商页面）
  - 可通过CSS选择器、XPath、正则表达式实现

---

## 5. 链接分析

网页之间的链接结构蕴含重要的质量信号：

### 5.1 PageRank算法

PageRank由Google创始人Larry Page和Sergey Brin提出，基于"链接即投票"的思想：

**基本公式**：

$$PR(p) = \frac{1-d}{N} + d \sum_{q \in B(p)} \frac{PR(q)}{L(q)}$$

- PR(p)：页面p的PageRank值
- d：阻尼因子（Damping Factor），典型值0.85
- N：网页总数
- B(p)：所有指向页面p的页面集合
- L(q)：页面q的出链数

**迭代计算**：
1. 初始化所有页面PR值为 1/N
2. 迭代应用上述公式，直到收敛（变化量小于阈值）
3. 通常在50-100次迭代后收敛

**随机游走模型**：
- 假设一个"随机冲浪者"在网络上随机点击链接
- 以概率 d 点击当前页面的某个出链
- 以概率 1-d 随机跳转到任意页面
- PageRank值 = 随机冲浪者停留在该页面的稳态概率

**处理Dead Ends和Spider Traps**：
- Dead End（无出链页面）：导致PR值泄漏，解决方法是将无出链页面视为指向所有页面
- Spider Trap（自循环页面组）：导致PR值汇聚，阻尼因子可缓解此问题

### 5.2 HITS算法

HITS（Hyperlink-Induced Topic Search）由Jon Kleinberg提出，为每个页面计算两个分数：

- **Authority（权威值）**：
  - 该页面被多少高质量Hub页面指向
  - 代表页面作为信息来源的质量

- **Hub（枢纽值）**：
  - 该页面指向多少高质量Authority页面
  - 代表页面作为导航目录的质量

**迭代计算**：
1. 初始化所有页面的Hub和Authority值为1
2. Authority更新：`auth(p) = Σ hub(q)`，对所有指向p的页面q求和
3. Hub更新：`hub(p) = Σ auth(q)`，对所有p指向的页面q求和
4. 归一化（使所有值的平方和为1）
5. 迭代直到收敛

**与PageRank的区别**：
- PageRank是全局的、与查询无关的
- HITS是查询相关的，针对每个查询计算
- HITS区分Hub和Authority两种角色

---

## 6. 网页反作弊（Web Spam检测）

网页反作弊旨在检测人为操纵排名的行为：

**常见作弊手段**：
- **关键词堆砌（Keyword Stuffing）**：在页面中大量重复关键词
- **隐藏文本（Hidden Text）**：文字颜色与背景相同，用户不可见但搜索引擎可抓取
- **链接农场（Link Farm）**：大量网站互相链接以提高PageRank
- **门页（Doorway Pages）**：为搜索引擎优化的低质量页面，自动跳转到目标页面
- **内容伪装（Cloaking）**：向搜索引擎和用户展示不同内容

**检测方法**：
- 基于内容特征：关键词密度、重复内容比例、隐藏文本检测
- 基于链接特征：PageRank分布异常、互链比例、链接增长速度
- 基于机器学习：综合多维特征训练分类器
- TrustRank：从人工审核的高质量种子集合出发，通过链接传播信任度，距离作弊源越远信任度越高

---

## 7. 深度爬取与JavaScript渲染

### 7.1 深度爬取

- 现代网站大量使用JavaScript动态加载内容
- 传统HTTP抓取只能获取初始HTML，无法获取JavaScript渲染后的内容
- 需要使用 **无头浏览器（Headless Browser）** 执行JavaScript

### 7.2 JavaScript渲染方案

- **Selenium**：驱动真实浏览器（Chrome、Firefox），功能强大但资源开销大
- **Puppeteer**：Node.js库，控制Headless Chrome
- **Playwright**：微软开源，支持多浏览器
- **Splash**：轻量级JavaScript渲染服务

### 7.3 挑战

- 渲染成本高：每个页面需要启动浏览器进程
- 速度慢：需等待JavaScript执行完成
- 反爬对抗：许多网站检测无头浏览器特征
- 需要等待策略：等待特定元素出现、等待网络空闲

**实际策略**：
- 优先尝试静态抓取，仅对必要页面使用JS渲染
- 利用API接口直接获取数据（很多SPA的后端API可直接访问）
- 分析网络请求，找到数据的实际来源（XHR/Fetch请求）
