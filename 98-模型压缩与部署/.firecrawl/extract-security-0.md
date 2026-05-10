# Web安全工程师面试（SQL、XSS、CSRF、SSRF、XXE） - 博客园

URL: https://www.cnblogs.com/Hardworking666/p/17374812.html

# [Web安全工程师面试（SQL、XSS、CSRF、SSRF、XXE）](https://www.cnblogs.com/Hardworking666/p/17374812.html "发布于 2021-12-21 20:03")

### 文章目录

- [一、Web 访问过程分析](https://www.cnblogs.com/Hardworking666/p/17374812.html#Web__3)
- [二、SQL注入](https://www.cnblogs.com/Hardworking666/p/17374812.html#SQL_7)
  - [1、SQL注入的危害](https://www.cnblogs.com/Hardworking666/p/17374812.html#1SQL_13)
  - [2、SQL注入思路](https://www.cnblogs.com/Hardworking666/p/17374812.html#2SQL_18)
  - [3、SQL注入的类型](https://www.cnblogs.com/Hardworking666/p/17374812.html#3SQL_23)
  - [4、SQL注入防护](https://www.cnblogs.com/Hardworking666/p/17374812.html#4SQL_29)
- [三、XSS跨站脚本](https://www.cnblogs.com/Hardworking666/p/17374812.html#XSS_37)
  - [1、反射型XSS漏洞原理](https://www.cnblogs.com/Hardworking666/p/17374812.html#1XSS_48)
  - [2、存储型XSS漏洞原理](https://www.cnblogs.com/Hardworking666/p/17374812.html#2XSS_61)
  - [3、基于DOM的XSS](https://www.cnblogs.com/Hardworking666/p/17374812.html#3DOMXSS_67)
  - [4、XSS未给出具体位置的解决](https://www.cnblogs.com/Hardworking666/p/17374812.html#4XSS_77)
  - [5、DOM 型和 XSS 自动化测试或人工测试](https://www.cnblogs.com/Hardworking666/p/17374812.html#5DOM__XSS__83)
- [四、CSRF跨站请求伪造](https://www.cnblogs.com/Hardworking666/p/17374812.html#CSRF_89)
  - [1、CSRF与XSS区别](https://www.cnblogs.com/Hardworking666/p/17374812.html#1CSRFXSS_92)
  - [2、CSRF的危害](https://www.cnblogs.com/Hardworking666/p/17374812.html#2CSRF_101)
  - [3、CSRF的防御](https://www.cnblogs.com/Hardworking666/p/17374812.html#3CSRF_104)
  - [4、CSRF护网面试](https://www.cnblogs.com/Hardworking666/p/17374812.html#4CSRF_109)
- [五、SSRF服务器端请求伪造](https://www.cnblogs.com/Hardworking666/p/17374812.html#SSRF_122)
  - [1、成因](https://www.cnblogs.com/Hardworking666/p/17374812.html#1_123)
  - [2、危害](https://www.cnblogs.com/Hardworking666/p/17374812.html#2_129)
  - [3、利用](https://www.cnblogs.com/Hardworking666/p/17374812.html#3_136)
  - [4、漏洞挖掘](https://www.cnblogs.com/Hardworking666/p/17374812.html#4_143)
  - [5、防御方法](https://www.cnblogs.com/Hardworking666/p/17374812.html#5_162)
  - [6、绕过方法](https://www.cnblogs.com/Hardworking666/p/17374812.html#6_169)
- [六、XXE（XML 外部实体注入）](https://www.cnblogs.com/Hardworking666/p/17374812.html#XXEXML__180)
  - [1、xml介绍](https://www.cnblogs.com/Hardworking666/p/17374812.html#1xml_182)
  - [2、内部实体](https://www.cnblogs.com/Hardworking666/p/17374812.html#2_185)
  - [3、外部实体](https://www.cnblogs.com/Hardworking666/p/17374812.html#3_231)
  - [4、通用实体](https://www.cnblogs.com/Hardworking666/p/17374812.html#4_253)
  - [5、参数实体](https://www.cnblogs.com/Hardworking666/p/17374812.html#5_266)
  - [6、有回显XXE](https://www.cnblogs.com/Hardworking666/p/17374812.html#6XXE_279)
  - [7、无回显XXE](https://www.cnblogs.com/Hardworking666/p/17374812.html#7XXE_320)
- [七、学习链接](https://www.cnblogs.com/Hardworking666/p/17374812.html#_361)

# 一、Web 访问过程分析

一次Web访问过程分析： **DNS域名解析、TCP连接、HTTP请求、处理请求返回HTTP响应、页面渲染和关闭连接。**

# 二、SQL注入

SQL注入漏洞是指，

攻击者能够利用现有Web应用程序，将 **恶意的数据** 插入 **SQL查询** 中，提交到 **后台数据库引擎** 执行 **非授权操作**。

SQL注入攻击利用的工具是 **SQL语法**。

## 1、SQL注入的危害

1、非法查询、修改或删除 **数据库资源**。

2、执行 **系统命令**。

3、获取承载主机操作系统和网络的 **访问权限**。

## 2、SQL注入思路

1、注入点选择

2、数字型和字符型注入

3、通过Web端对数据库注入或者直接访问数据库注入

## 3、SQL注入的类型

1、报错注⼊

2、bool 型注⼊

3、延时注⼊

4、宽字节注⼊

## 4、SQL注入防护

1、使⽤安全的 API

2、对输⼊的特殊字符进⾏ Escape 转义处理

3、使⽤⽩名单来规范化输⼊验证⽅法

4、对客户端输⼊进⾏控制，不允许输⼊ SQL 注⼊相关的特殊字符

5、服务器端在提交数据库进⾏ SQL 查询之前，对特殊字符进⾏过滤、转义、替换、删除。

6、规范编码, 字符集

# 三、XSS跨站脚本

跨站脚本(Cross Site Script)，简称XSS。

XSS漏洞是指：应用程序没有对接收到的 **不可信数据** 经过适当的 **验证或转义** 就直接发给客户端浏览器。

原理：web浏览器可以执行HTML页面中嵌入的脚本命令，攻击者利用XSS漏洞将恶意脚本代码注入到网页中，当用户浏览该网页时，便会触发执行恶意脚本。

XSS漏洞主要 **危害**

1 非法访问、篡改 **敏感数据**

2 **会话劫持**

3 控制受害机器 **向其他站点发起攻击**

## 1、反射型XSS漏洞原理

1）最普遍的一种类型。

2）服务器直接使用客户端提供的数据而没有对数据进行无害化处理，就会出现此漏洞。

3）特点： **用户单击时触发，而且只执行一次**，因此反射型XSS也称为非持久型XSS。

反射型XSS通常是由

攻击者诱使用户向有漏洞的Web应用程序 **提供危险内容**，然后危险内容会 **反射给用户** 并由浏览器执行。

XSS漏洞潜在影响的一种攻击：可导致攻击者截获一名通过验证的用户的会话。

劫持用户的会话后，攻击者就可以 **访问该用户经授权访问的所有数据和功能**。

实施这种攻击的步骤如图：

## 2、存储型XSS漏洞原理

存储型XSS也称为持久型XSS，它的危害更大。此类XSS **不需要用户单击特定的URL** 就能执行跨站脚本。攻击者事先将恶意脚本代码上传或者存储到存在漏洞的 **服务器端数据库** 中，只要用户浏览包含此恶意脚本的网页便会触发，遭受攻击。

存储型XSS漏洞通常在 **留言板、个人资料、博客日志** 等位置出现，并常被用于编写危害性更大的 **XSS蠕虫**。

## 3、基于DOM的XSS

基于DOM的XSS又称为 **本地XSS**，DOM 型 XSS 漏洞是基于 **文档对象模型** （Document Objeet Model，DOM）的⼀种漏洞。

由于客户端浏览器JavaScript可以访问浏览器的DOM动态地检查和修改页面的内容，当HTML页面采用不安全的方式从document.location、document.URL、document.referrer或其他攻击者可以修改的对象获取数据时，如果数据包含恶意JavaScript脚本，就会触发基于DOM的XSS攻击。

基于DOM的XSS攻击与反射型XSS和存储型XSS不同，基于DOM的XSS攻击来源于客户端处理的脚本中， **无需服务器端的参与**。

**文档对象模型** (DOM) **将 web 页面与到脚本或编程语言连接** 起来。通常是指 JavaScript，但将 HTML、SVG 或 XML 文档建模为对象并不是 JavaScript 语言的一部分。DOM模型用一个逻辑树来表示一个文档，树的每个分支的终点都是一个节点(node)，每个节点都包含着对象(objects)。DOM的方法(methods)让你可以用特定方式操作这个树，用这些方法你可以改变文档的结构、样式或者内容。节点可以关联上事件处理器，一旦某一事件被触发了，那些事件处理器就会被执行。

## 4、XSS未给出具体位置的解决

如果安全应急响应中心（SRC，Security Response Center） 上报了⼀个 XSS 漏洞，payload 已经写⼊页面，但未给出具体位置，如何快速介⼊？

看是什么类型的 XSS，XSS 反射型看提交的地址，指的参数是哪个位置，通过这个⻚⾯进⾏ fuzzing 测试。如果是存储型⻚⾯查找关键字。

**修复方式**：对字符实体进⾏转义、使⽤ HTTP Only 来禁⽌ JavaScript 读取 Cookie 值、输⼊时校验、浏览器与 Web 应⽤端采⽤相同的字符编码。

## 5、DOM 型和 XSS 自动化测试或人工测试

**人工测试** 思路：找到类似 document.write、innerHTML 赋值、outterHTML 赋值、window.location 操作、写 javascript: 后内容、eval、setTimeout 、setInterval 等直接执⾏之类的函数点。找到其变量，回溯变量来源观察是否可控，是否经过安全函数。

**自动化测** 试参看道哥的博客，思路是从输入入手，观察变量传递的过程，最终检查是否有在危险函数输出，中途是否有经过安全函数。但是这样就需要有⼀个 javascript 解析器，否则会漏掉⼀些通过 js 执⾏带⼊的部分内容容。

# 四、CSRF跨站请求伪造

[从0到1完全掌握 CSRF](https://www.freebuf.com/articles/web/333173.html)

## 1、CSRF与XSS区别

跨站请求伪造(Cross-site request forgery)简称CSRF，尽管与跨站脚本漏洞名称相近，但它与跨站脚本漏洞不同。

XSS利用 **站点内的信任用户**，而CSRF则通过 **伪装来自受信任用户的请求** 来利用受信任的网站。

CSRF和反射型XSS的主要区别是：反射型XSS的目的是在 **客户端** 执行脚本，CSRF的目的是在 **Web应用** 中执行操作。

CSRF跨站请求伪造攻击迫使登录用户的浏览器将伪造的HTTP请求，包括该用户的会话Cookie和其他认证信息，发送到一个存在漏洞的Web应用程序，而这些请求会被应用程序认为是用户的合法请求。

## 2、CSRF的危害

篡改⽬标⽹站上的⽤户数据、盗取⽤户隐私数据、传播 CSRF 蠕虫。

## 3、CSRF的防御

CSRF 防御原理：不让黑客那么容易伪造请求

1、cookie 中加⼊随机数，要求请求中带上，⽽攻击者获取不到 cookie中的随机数。

2、验证HTTP Referer 字段, 在请求地址中添加 takon 验证。

## 4、CSRF护网面试

**token 和 referer 做横向对比，谁安全等级高？**

token 安全等级更⾼，因为并不是任何服务器都可以取得 referer，如果从 HTTPS 跳到 HTTP，也不会发送 referer。并且 FLASH ⼀些版本中可以⾃定义 referer。但是 token 的话，要保证其⾜够随机且不可泄露。(不可预测性原则)

**对 referer 的验证，从什么⻆度去做？**

对 header 中的 referer 的验证，⼀个是空 referer，⼀个是 referer 过滤或者检测不完善。为了杜绝这种问题，在验证的白名单中，正则规则应当写完善。

**针对 token，会对 token的哪方面进⾏测试？**

针对token的攻击，⼀是对它本身的攻击，重放测试⼀次性、分析加密规则、校验⽅式是否正确等，⼆是结合信息泄露漏洞对它的获取，结合着发起组合攻击信息泄露有可能是缓存、⽇志、get，也有可能是利⽤跨站很多跳转登录的都依赖token，有⼀个跳转漏洞加反射型跨站就可以组合成登录劫持了。另外也可以结合着其它业务来描述token的安全性及设计不好怎么被绕过⽐如抢红包业务之类的。

# 五、SSRF服务器端请求伪造

## 1、成因

SSRF(Server-Side Request Forgery，服务器端请求伪造)。是一种由攻击者构造形成 **由服务端发起请求** 的一个安全漏洞。

一般情况下，SSRF攻击的目标是 **从外网无法访问的内部系统**。

很多Web应用都提供了 **从其他服务器上获取数据** 的功能。使用用户指定的URL，Web应用可以获取图片，下载文件，读取文件内容等。这个功能如果被恶意使用，可以利用存在缺陷的web应用 **作为代理** 攻击远程和本地服务器。

## 2、危害

可以对外网服务器所在的内网、本地进行端口扫描，获取一些服务的banner信息 。

攻击运行在内网或者本地的应用程序。

对内网web应用进行指纹识别，通过访问默认文件实现 。

攻击内外网的web应用。sql注入、struct2、redis等。

利用file协议读取本地文件等。

## 3、利用

1、可以对外网、内网、本地进行端口扫描，某些情况下端口的Banner会回显出来（比如3306的）；

2、攻击运行在内网或本地的有漏洞程序（比如溢出）；

3、可以对内网Web应用进行指纹识别，原理是通过请求默认的文件得到特定的指纹

4、攻击内网或外网有漏洞的Web应用

5、使用file:///协议读取本地文件

## 4、漏洞挖掘

一. WEB功能上查找

1、分享：通过URL地址分享网页内容

通过URL地址分享网页内容早期应用中 ，为了更好的用户体验，Web应用在分享功能中，通常会获取目标URL地址网页内容中标签或者<meta name=“description”content=“”/>标签中content的文本内容提供更好的用户体验。

2、转码服务：通过URL地址把原地址的网页内容调优使其适合手机屏幕浏览

3、在线翻译：通过 URL地址翻译对应文本的内容

4、图片加载与下载：通过URL地址加载或下载图片

图片加载远程图片地址此功能用到的地方很多，但大多都是比较隐秘，如有些公司中的加载自家图片服务器上的图片用于展示。（开发者为了有更好的用户体验通常对图片做些微小调整例如加水印、压缩等，就必须要把图片下载到服务器的本地，所以就可能造成SSRF问题）。

5、图片、文章收藏功能

6、未公开的api实现以及其他调用URL的功能

7、从URL关键字中寻找

二. 从URL关键字中寻找

Share、wap、url、link、src、source、target、u、3g、display、sourceURL、imageURL、domain

三. 通用的SSRF实例

Weblogic配置不当，天生ssrf漏洞

Discuz x2.5/x3.0/x3.1/x3.2 ssrf漏洞

## 5、防御方法

1、过滤返回信息，验证远程服务器对请求的响应是比较容易的方法。如果web应用是去获取某一种类型的文件。那么在把返回结果展示给用户之前先验证返回的信息是否符合标准。

2、统一错误信息，避免用户可以根据错误信息来判断远端服务器的端口状态。

3、限制请求的端口为http常用的端口，比如，80,443,8080,8090。

4、黑名单内网ip。避免应用被用来获取获取内网数据，攻击内网。

5、禁用不需要的协议。仅仅允许http和https请求。可以防止类似于file:///,gopher://,ftp:// 等引起的问题。

## 6、绕过方法

1、@

http://abc@127.0.0.1

2、添加端口号

http://127.0.0.1:8080

3、短地址

http://dwz.cn/11SMa

4、可以指向任意ip的域名：xip.io

5、ip地址转换成进制来访问

115.239.210.26 ＝ 16373751032

# 六、XXE（XML 外部实体注入）

[XXE 漏洞详解](https://xz.aliyun.com/t/3357#toc-10)

## 1、xml介绍

XML是一种非常流行的标记语言，在解析外部实体的过程中，XML解析器可以根据URL中指定的方案（协议）来查询各种网络协议和服务（DNS，FTP，HTTP，SMB等）。外部实体对于在文档中创建动态引用非常有用，这样对引用资源所做的任何更改都会在文档中自动更新。但是，在处理外部实体时，可以针对应用程序启动许多攻击。这些攻击包括泄露本地系统文件，这些文件可能包含密码和私人用户数据等敏感数据，或利用各种方案的网络访问功能来操纵内部应用程序。通过将这些攻击与其他实现缺陷相结合，这些攻击的范围可以扩展到客户端内存损坏，任意代码执行，甚至服务中断，具体取决于这些攻击的上下文。

## 2、内部实体

XML 文档有自己的一个格式规范，这个格式规范是由一个叫做 **文档类型定义（DTD，Document Type Definition）** 的东西控制的。

```xml

```

上面这个 DTD 就定义了 XML 的根元素是 message，然后跟元素下面有一些子元素，那么 XML 到时候必须像下面这么写

```xml

```

其实除了在 DTD 中定义元素（其实就是对应 XML 中的标签）以外，我们还能在 DTD 中定义实体(对应XML 标签中的内容)，毕竟 XML 中除了能标签以外，还需要有些内容是固定的

```xml

```

这里 定义元素为 ANY 说明接受任何元素，但是定义了一个 xml 的实体（实体其实可以看成一个变量，到时候我们可以在 XML 中通过 & 符号进行引用），那么 XML 就可以写成这样

示例代码：

```xml

```

我们使用 &xxe 对 上面定义的 xxe 实体进行了引用，到时候输出的时候 &xxe 就会被 “test” 替换。

## 3、外部实体

示例代码：

```xml

```

当然，还有一种引用方式是使用 引用公用 DTD 的方法，语法如下：

```xml

```

我们上面已经将实体分成了两个派别（内部实体和外部外部），但是实际上从另一个角度看，实体也可以分成两个派别（通用实体和参数实体）。

## 4、通用实体

用 `&实体名;`在DTD 中定义，在 XML 文档中引用

```xml

```

## 5、参数实体

(1)使用 `% 实体名`(这里面空格不能少) 在 DTD 中定义，并且只能在 DTD 中使用 `%实体名;` 引用(2)只有在 DTD 文件中，参数实体的声明才能引用其他实体 (3)和通用实体一样，参数实体也可以外部引用

示例代码：

```xml

```

抛转：参数实体在我们 Blind XXE 中起到了至关重要的作用

## 6、有回显XXE

这个实验的攻击场景模拟的是在服务能接收并解析 XML 格式的输入并且有回显的时候，我们就能输入我们自定义的 XML 代码，通过引用外部实体的方法，引用服务器上面的文件。

本地服务器上放上解析 XML 的 php 代码：

xml.php

```php

```

其中：LIBXML\_NOENT: 将 XML 中的实体引用 替换 成对应的值 LIBXML\_DTDLOAD: 加载 DOCTYPE 中的 DTD 文件

**触发XXE**

```xml

```

读取本地服务器C盘的flag文件

```xml

```

引用外部实体读取文件

引用方式是使用 引用公用 DTD 的方法读取

## 7、无回显XXE

有回显的情况可以直接在页面中看到Payload的执行结果或现象，无回显的情况又称为`blind xxe`，可以使用外带数据通道提取数据，先使用php://filter获取目标文件的内容，然后将内容以http请求发送到接受数据的服务器。

xml.php

```php

```

test.dtd

```xml

```

payload

```xml

```

我们从 payload 中能看到 连续调用了三个参数实体 `%remote;%int;%send;`，这就是我们的利用顺序，`%remote` 先调用，调用后请求远程服务器上的 test.dtd ，有点类似于将 test.dtd 包含进来，然后 `%int` 调用 test.dtd 中的 %file, %file 就会去获取服务器上面的敏感文件，然后将 `%file` 的结果填入到 `%send` 以后(因为实体的值中不能有 %, 所以将其转成html实体编码 %)，我们再调用 `%send`; 把我们的读取到的数据发送到我们的远程 vps 上，这样就实现了外带数据的效果，完美的解决了 XXE 无回显的问题。

这样，我们就读到了flag文件的内容。

实战靶场

复制链接到电脑端操作：

https://www.hetianlab.com/expc.do?ec=ECID9117-d620-481d-91f8-344e0ac69dea&pk\_campaign=weixin-wemedia#stu

# 七、学习链接

[GitHub上的安全知识框架：](https://github.com/ffffffff0x/1earn)

项目文件一览

Security

安全工具 \- 各类安全工具的使用介绍

安全资源

靶机

HTB

VulnHub

DC Serial - DC 系列靶场,难度简单至中等,可以学习各种提权和CMS漏洞利用,推荐初学者挑战

It’s\_October

Kioptrix Serial - Kioptrix 系列靶场,难度简单至中等,推荐初学者挑战

Mission-Pumpkin - 难度适中,偏向于加解密比较多,漏洞利用内容较少

symfonos Serial - 挺有难度的靶场,内容丰富,难度中等,漏洞利用内容很多,推荐有一定经验者挑战

Wargames

Bandit

BlueTeam

分析 \- 分析工具与分析案例

加固 \- 系统、应用加固的方法和工具资源

监察 \- 有关查杀、监控、蜜罐的资源

取证 \- 内容涉及操作系统的取证、web 的取证、文件的取证

应急 \- 应急资源、溯源案例

笔记 \- 涉及磁盘取证、内存取证、USB取证等内容

实验 \- 涉及流量分析实战、安防设施搭建等内容

Crypto

Crypto - 介绍各种编码和加密算法及相关的工具

CTF

CTF - 收集 CTF 相关的工具和 writeup 资源

writeup - 自己参与的一些比赛记录

ICS

工控协议 \- 总结各类工控协议的知识点

上位机安全 \- 总结上位机安全相关的知识点

PLC攻击 - 总结 PLC 攻击的相关知识点

S7comm相关 - 记录 S7comm 相关错误类型、功能码和相关参数

实验 \- 仿真环境搭建和 PLC 攻击实验

IOT

固件安全

固件安全 \- 记录 IOT 固件分析的知识点,包括固件提取、固件分析、固件解密等

实验 \- 分析固件实验

无线电安全

实验 \- 无线电安全实验

硬件安全

Device-Exploits - 嵌入式设备相关漏洞利用,不太熟悉这一块,内容不多

HID - 和组员制作的 HID 实物记录

MobileSec

Android安全 - 记录一些安卓安全相关的内容,这块掌握较少

RedTeam

安防设备

Bypass技巧 - 记录 waf 绕过手段

SecDevice-Exploits - 常见的安全设备的漏洞利用方法

后渗透

后渗透 \- 后渗透知识点的大纲

权限提升 \- 操作系统和数据库的提权方法

权限维持 \- 权限维持的各种方法和资源

实验

软件服务安全

CS-Exploits - 收集软件、业务应用服务漏洞的渗透手段和 cve 漏洞

DesktopApps-Exploits - 收集桌面软件的渗透手段和 cve 漏洞

协议安全

Protocol-Exploits - 按照协议归类各种漏洞、攻击手段

笔记

信息收集

端口安全 \- 记录端口渗透时的方法和思路

空间测绘 \- 收集搜索引擎语法资源

信息收集 \- 记录信息收集方面各类技术，如漏扫、IP 扫描、端口扫描、DNS 枚举、目录枚举、指纹等

语言安全

语言安全

云安全

云安全 \- 云主机利用工具,渗透案例,相关知识点

OS安全

Linux安全 - 包含 Linux 口令破解，漏洞利用、获取Shell

OS-Exploits - 收集操作系统的 cve 漏洞

Windows安全 - 包含 windows pth、ptt，漏洞利用、提权、远程执行命令

实验

Web 安全

前端攻防 \- 前端解密,绕过访问

BS-Exploits - 全面收集 web 漏洞 POC \| Payload \| exp

IDOR - 整个部分结构大部分基于乌云的几篇密码找回、逻辑漏洞类文章,在其基础上记录和归纳

靶场

Web\_Generic

Web\_Tricks

Reverse

Reverse

实验

FILE

Develop

版本控制

Git学习笔记 - 记录 git 的用法和平时使用 github 遇到的问题

标记语言

HTML

JSON

XML

可视化

gnuplot

正则

regex - 常用正则表达式和相关资源

Web

Speed-Web

HTTP

笔记

Integrated

数据库

Power-SQL

Speed-SQL

笔记

实验

虚拟化

Docker

Linux

God-Linux - 记录 Linux 下的骚操作,收集的较少,后面会慢慢添加

Power-Linux - 配置指南,记录各种服务搭建与配置过程

Secure-Linux - Linux 加固+维护+应急响应参考

Speed-Linux - 命令速查手册,记录各种基本命令操作

笔记

实验 \- 各种 linux 服务的搭建过程和案例

Network

不同厂商 \- 记录不同厂商配置服务命令的区别

方向实验 \- 按方向分类记录配置

速查 \- 速查各类帧、报文格式、掩码等

SDN笔记 - 记录以前比赛时 SDN 的题目和命令

TCP-IP - 记录 TCP/IP 协议栈的协议

VPN-Security - 记录 VPN 领域的协议

Windows

Secure-Win - Windows 加固+维护+应急响应参考

Speed-Win - 记录 windows 下 CMD 常用命令

笔记

实验 -涉及域环境搭建、基础服务搭建

Powershell

Plan

Misc-Plan - 各种小技巧

Team-Plan - 团队协作解决方案

Thinking-Plan - 问题解决方式的记录和学习

VM-Plan - VMWare 常见问题记录

https://blog.csdn.net/Hardworking666
本人主要使用CSDN，地址献上，请多多指教。

标签:
[SQL注入](https://www.cnblogs.com/Hardworking666/tag/SQL%E6%B3%A8%E5%85%A5/), [安全](https://www.cnblogs.com/Hardworking666/tag/%E5%AE%89%E5%85%A8/), [Web漏洞](https://www.cnblogs.com/Hardworking666/tag/Web%E6%BC%8F%E6%B4%9E/), [xss](https://www.cnblogs.com/Hardworking666/tag/xss/), [csrf](https://www.cnblogs.com/Hardworking666/tag/csrf/)

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/2744426/20220206141030.png)](https://home.cnblogs.com/u/Hardworking666/)

[Hardworking666](https://home.cnblogs.com/u/Hardworking666/)

[粉丝 \- 10](https://home.cnblogs.com/u/Hardworking666/followers/) [关注 \- 0](https://home.cnblogs.com/u/Hardworking666/followees/)

+加关注

0

0

[升级成为会员](https://cnblogs.vip/)

[»](https://www.cnblogs.com/Hardworking666/p/17374809.html) 下一篇： [CTFmisc类密码题思路与多种做法（CyberChef、Ciphey）](https://www.cnblogs.com/Hardworking666/p/17374809.html "发布于 2022-01-02 00:39")

posted @
2021-12-21 20:03 [Hardworking666](https://www.cnblogs.com/Hardworking666)
阅读(310)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2FHardworking666%2Fp%2F17374812.html&targetId=17374812&targetType=0)来源

[刷新页面](https://www.cnblogs.com/Hardworking666/p/17374812.html#) [返回顶部](https://www.cnblogs.com/Hardworking666/p/17374812.html#top)

登录后才能查看或发表评论，立即 登录 或者
[逛逛](https://www.cnblogs.com/) 博客园首页

[【推荐】智能无限 \| 协作无间，TRAE SOLO 中国版正式上线，全面免费](https://www.trae.com.cn/?utm_source=advertising&utm_medium=cnblogs_ug_cpa&utm_term=hw_trae_cnblogs)

[【推荐】科研领域的连接者艾思科蓝，一站式科研学术服务数字化平台](https://ais.cn/u/QjqYJr)

[【推荐】飞算 JavaAI 修复器：无限 tokens 加持，Bug 修复快到飞起](https://www.cnblogs.com/cmt/p/19669319)

[![](https://img2024.cnblogs.com/blog/35695/202604/35695-20260423213336272-1914399152.webp)](https://www.volcengine.com/activity/codingplan?utm_campaign=hw&utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&utm_term=cnblogs)

- [从 305 GB 到 7.4 GB：大模型 KVCache 架构演进全景](https://www.cnblogs.com/cswuyg/p/19981922)
- [DeepSeek V4模型的Agent能力实测](https://www.cnblogs.com/zhayujie/p/19935607/deepseek-v4-eval)
- [C# 15 类型系统改进：Union Types](https://www.cnblogs.com/hez2010/p/19891530/union-types-in-csharp-15)
- [你能被装进一个文件里吗？——7 万人把同事“蒸馏”成了 AI](https://www.cnblogs.com/wmyskxz/p/19854791)
- [别再吹牛了，100% Vibe Coding 存在无法自洽的逻辑漏洞！](https://www.cnblogs.com/mengxiang2/p/19796426)

### 公告

昵称：
[Hardworking666](https://home.cnblogs.com/u/Hardworking666/)

园龄：
[4年3个月](https://home.cnblogs.com/u/Hardworking666/ "入园时间：2022-02-06")

粉丝：
[10](https://home.cnblogs.com/u/Hardworking666/followers/)

关注：
[0](https://home.cnblogs.com/u/Hardworking666/followees/)

+加关注

| |     |     |     |
| --- | --- | --- |
| < | 2026年5月 | > | |
| 日 | 一 | 二 | 三 | 四 | 五 | 六 |
| 26 | 27 | 28 | 29 | 30 | 1 | 2 |
| 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| 24 | 25 | 26 | 27 | 28 | 29 | 30 |
| 31 | 1 | 2 | 3 | 4 | 5 | 6 |

### 搜索

### 常用链接

- [我的随笔](https://www.cnblogs.com/Hardworking666/p/ "我的博客的随笔列表")
- [我的评论](https://www.cnblogs.com/Hardworking666/MyComments.html "我的发表过的评论列表")
- [我的参与](https://www.cnblogs.com/Hardworking666/OtherPosts.html "我评论过的随笔列表")
- [最新评论](https://www.cnblogs.com/Hardworking666/comments "我的博客的评论列表")
- [我的标签](https://www.cnblogs.com/Hardworking666/tag/ "我的博客的标签列表")

### [我的标签](https://www.cnblogs.com/Hardworking666/tag/)

- [系统集成项目管理工程师(21)](https://www.cnblogs.com/Hardworking666/tag/%E7%B3%BB%E7%BB%9F%E9%9B%86%E6%88%90%E9%A1%B9%E7%9B%AE%E7%AE%A1%E7%90%86%E5%B7%A5%E7%A8%8B%E5%B8%88/)
- [逆向工程(9)](https://www.cnblogs.com/Hardworking666/tag/%E9%80%86%E5%90%91%E5%B7%A5%E7%A8%8B/)
- [网络(5)](https://www.cnblogs.com/Hardworking666/tag/%E7%BD%91%E7%BB%9C/)
- [运维(4)](https://www.cnblogs.com/Hardworking666/tag/%E8%BF%90%E7%BB%B4/)
- [服务器(3)](https://www.cnblogs.com/Hardworking666/tag/%E6%9C%8D%E5%8A%A1%E5%99%A8/)
- [sql(3)](https://www.cnblogs.com/Hardworking666/tag/sql/)
- [mysql(3)](https://www.cnblogs.com/Hardworking666/tag/mysql/)
- [CTF(3)](https://www.cnblogs.com/Hardworking666/tag/CTF/)
- [网络安全(2)](https://www.cnblogs.com/Hardworking666/tag/%E7%BD%91%E7%BB%9C%E5%AE%89%E5%85%A8/)
- [渗透测试(2)](https://www.cnblogs.com/Hardworking666/tag/%E6%B8%97%E9%80%8F%E6%B5%8B%E8%AF%95/)
- [更多](https://www.cnblogs.com/Hardworking666/tag/)

### [随笔分类](https://www.cnblogs.com/Hardworking666/post-categories)

- [CNVD(1)](https://www.cnblogs.com/Hardworking666/category/2306662.html)
- [CTF(1)](https://www.cnblogs.com/Hardworking666/category/2306661.html)
- [MySQL等数据库(1)](https://www.cnblogs.com/Hardworking666/category/2306657.html)
- [PHP(1)](https://www.cnblogs.com/Hardworking666/category/2306663.html)
- [操作系统基础知识(1)](https://www.cnblogs.com/Hardworking666/category/2306660.html)
- [等保2.0(1)](https://www.cnblogs.com/Hardworking666/category/2306655.html)
- [护网（HW）(1)](https://www.cnblogs.com/Hardworking666/category/2306656.html)
- [逆向工程(1)](https://www.cnblogs.com/Hardworking666/category/2306659.html)
- [软件安全基础(1)](https://www.cnblogs.com/Hardworking666/category/2306654.html)
- [网络安全(1)](https://www.cnblogs.com/Hardworking666/category/2306653.html)
- [系统集成项目管理工程师(1)](https://www.cnblogs.com/Hardworking666/category/2306652.html)
- [职场生涯与法律法规(1)](https://www.cnblogs.com/Hardworking666/category/2306658.html)

### 随笔档案

- [2023年5月(1)](https://www.cnblogs.com/Hardworking666/p/archive/2023/05)
- [2023年4月(19)](https://www.cnblogs.com/Hardworking666/p/archive/2023/04)
- [2023年3月(3)](https://www.cnblogs.com/Hardworking666/p/archive/2023/03)
- [2023年2月(2)](https://www.cnblogs.com/Hardworking666/p/archive/2023/02)
- [2022年10月(1)](https://www.cnblogs.com/Hardworking666/p/archive/2022/10)
- [2022年9月(2)](https://www.cnblogs.com/Hardworking666/p/archive/2022/09)
- [2022年6月(2)](https://www.cnblogs.com/Hardworking666/p/archive/2022/06)
- [2022年5月(2)](https://www.cnblogs.com/Hardworking666/p/archive/2022/05)
- [2022年4月(4)](https://www.cnblogs.com/Hardworking666/p/archive/2022/04)
- [2022年3月(10)](https://www.cnblogs.com/Hardworking666/p/archive/2022/03)
- [2022年2月(68)](https://www.cnblogs.com/Hardworking666/p/archive/2022/02)
- [2022年1月(4)](https://www.cnblogs.com/Hardworking666/p/archive/2022/01)
- [2021年12月(1)](https://www.cnblogs.com/Hardworking666/p/archive/2021/12)

### [阅读排行榜](https://www.cnblogs.com/Hardworking666/most-viewed)

- [1\. CTF misc图片类总结（入门级）(10110)](https://www.cnblogs.com/Hardworking666/p/15866125.html)
- [2\. CTF压缩包隐写类（zip、RAR、zip伪加密）(6615)](https://www.cnblogs.com/Hardworking666/p/15866093.html)
- [3\. CTF密码题思路与多种做法（CyberChef、Ciphey）(5122)](https://www.cnblogs.com/Hardworking666/p/15866092.html)
- [4\. 七种寻址方式(4640)](https://www.cnblogs.com/Hardworking666/p/17374792.html)
- [5\. 请求页式管理中的置换算法（FIFO、LRU、OPT），求缺页率例题(4539)](https://www.cnblogs.com/Hardworking666/p/15866114.html)

### [评论排行榜](https://www.cnblogs.com/Hardworking666/most-commented)

- [1\. CNVD、CNNVD、CICSVD等区别与联系详解(1)](https://www.cnblogs.com/Hardworking666/p/15866081.html)

### [推荐排行榜](https://www.cnblogs.com/Hardworking666/most-liked)

- [1\. WebShell基础详解（特点、原理、分类、工具）(3)](https://www.cnblogs.com/Hardworking666/p/15866085.html)
- [2\. 七种寻址方式(1)](https://www.cnblogs.com/Hardworking666/p/17374792.html)
- [3\. wav文件隐写：Deepsound+TIFF图片PS处理（ AntCTF x D^3CTF 2022 misc BadW3ter）(1)](https://www.cnblogs.com/Hardworking666/p/17374797.html)
- [4\. Python的pickle模块详解（包括优缺点及和JSON的区别）(1)](https://www.cnblogs.com/Hardworking666/p/15866134.html)
- [5\. CTF misc图片类总结（入门级）(1)](https://www.cnblogs.com/Hardworking666/p/15866125.html)

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)