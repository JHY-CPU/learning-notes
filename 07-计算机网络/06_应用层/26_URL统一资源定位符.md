# 27_URL统一资源定位符

## 核心概念

- **URL（Uniform Resource Locator）**：统一资源定位符，标识互联网上资源的位置
- URL是**URI（统一资源标识符）**的一种
- URL格式：`协议://主机名[:端口]/路径[?查询][#片段]`
- **默认端口**：HTTP 80，HTTPS 443，FTP 21
- **URL编码**：特殊字符需要URL编码（如空格→%20）
- **408考试重点**：URL的组成部分、各部分的含义

## 原理分析

### URL完整格式

```
scheme://[userinfo@]host[:port]/path[?query][#fragment]
```

各部分详解：

| 部分 | 必选 | 说明 | 示例 |
|------|------|------|------|
| scheme | 是 | 协议 | http, https, ftp |
| host | 是 | 主机名或IP | www.example.com |
| port | 否 | 端口号 | 80, 443 |
| path | 否 | 资源路径 | /dir/page.html |
| query | 否 | 查询参数 | ?key=value&k2=v2 |
| fragment | 否 | 片段标识 | #section1 |

### URL示例分析

**示例1**：`http://www.example.com/index.html`
- scheme: http
- host: www.example.com
- port: 80（默认，省略）
- path: /index.html

**示例2**：`https://user:pass@example.com:8080/search?q=test#results`
- scheme: https
- userinfo: user:pass
- host: example.com
- port: 8080
- path: /search
- query: q=test
- fragment: results

### URL编码

- URL中某些字符有特殊含义（如`?`、`&`、`=`、`#`、`/`）
- 特殊字符需要URL编码（百分号编码）
- 编码格式：`%` + 两位十六进制ASCII码
- 常见编码：
  - 空格 → `%20` 或 `+`
  - ? → `%3F`
  - & → `%26`
  - = → `%3D`

### URI vs URL vs URN

- **URI**：统一资源标识符（上位概念）
- **URL**：统一资源定位符（通过位置定位资源）
- **URN**：统一资源名称（通过名称定位资源）
- 关系：URL ⊂ URI，URN ⊂ URI

## 直观理解

**URL就像快递地址**：
- `http://` = 快递公司（协议）
- `www.example.com` = 小区名称（主机名）
- `:8080` = 单元号（端口）
- `/path/page.html` = 楼栋+房间号（路径）
- `?key=value` = 附加信息（查询参数）
- `#section` = 具体位置（片段）

**记忆技巧**：
- URL = "互联网上的门牌号"
- 格式 = "协议://主机:端口/路径?查询#片段"
- 默认端口：HTTP 80，HTTPS 443
- 特殊字符要URL编码

## 协议关联

- **URL与DNS**：URL中的主机名需要DNS解析为IP地址
- **URL与HTTP**：HTTP请求中包含URL
- **URL与HTML**：HTML中的超链接使用URL
- **408考点**：
  - URL的组成部分
  - 默认端口号
  - URL编码
- **陷阱**：URL是URI的一种，不是所有URI都是URL
