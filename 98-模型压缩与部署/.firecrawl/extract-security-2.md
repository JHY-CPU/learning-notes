# 【面试】网络安全- sql注入、xss、csrf、ssrf-CSDN博客

URL: https://blog.csdn.net/zqf787351070/article/details/126686054

[![](https://img-home.csdnimg.cn/images/20201124032511.png)](https://www.csdn.net/)

- [博客](https://blog.csdn.net/)
- [下载](https://download.csdn.net/)
- [社区](https://devpress.csdn.net/)
- [![](https://img-home.csdnimg.cn/images/20240829093757.png)AtomGit](https://link.csdn.net/?target=https%3A%2F%2Fgitcode.com%3Futm_source%3Dcsdn_toolbar)
- [![](https://i-operation.csdnimg.cn/images/3c66245675ae423e9cc897dc790b8ac9.png)GPU算力\\
![](https://i-operation.csdnimg.cn/images/b4db3100c53e4a7c9fd6a3d647156191.png)](https://ai.csdn.net/)
- 更多


[会议](https://www.bagevent.com/event/9117243 "会议") [学习](https://edu.csdn.net/?utm_source=zhuzhantoolbar "高质量课程·大会云会员") [![](https://i-operation.csdnimg.cn/images/77c4dd7a760a493498bee1d336b064c0.png)InsCode](https://inscode.net/?utm_source=csdn_blog_top_bar "InsCode")


搜索

AI 搜索

登录

登录后您可以：

- 复制代码和一键运行
- 与博主大V深度互动
- 解锁海量精选资源
- 获取前沿技术资讯

立即登录

[会员·新人礼包 ![](https://i-operation.csdnimg.cn/images/105eda9d414f4250a7c3fe45be3cd15f.png)](https://mall.csdn.net/vip?utm_source=vip_toolbarhyzx_hy)

[消息](https://i.csdn.net/#/msg/index)

[创作中心](https://mp.csdn.net/ "创作中心")

[创作](https://mp.csdn.net/edit)

[![](https://i-operation.csdnimg.cn/images/6e41bd372d1f4ec39b3cd36ab95046c4.png)](https://mp.csdn.net/edit)![](https://i-operation.csdnimg.cn/images/43349e98a45341699652b0b6fa4ea541.png)![](https://i-operation.csdnimg.cn/images/0f13ec529b6b4195ad99894f76653e56.png)

# 【面试】网络安全

原创已于 2023-03-13 17:23:08 修改·159 阅读

·![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png)
0


·![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png)
0
·

CC 4.0 BY-SA版权

版权声明：本文为博主原创文章，遵循 [CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/) 版权协议，转载请附上原文出处链接和本声明。


文章标签：

[#面试](https://so.csdn.net/so/search/s.do?q=%E9%9D%A2%E8%AF%95&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#web安全](https://so.csdn.net/so/search/s.do?q=web%E5%AE%89%E5%85%A8&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#java](https://so.csdn.net/so/search/s.do?q=java&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art)

于 2022-09-05 07:50:58 首次发布

[![](https://i-blog.csdnimg.cn/blog_column_migrate/b9346faac8287efffed28c8b2d898da7.png?x-oss-process=image/resize,m_fixed,h_224,w_224)面试八股文专栏收录该内容](https://blog.csdn.net/zqf787351070/category_11996292.html "面试八股文")

11 篇文章

订阅专栏

#### 网络安全

- [1\. XSS 跨站脚本攻击](https://blog.csdn.net/zqf787351070/article/details/126686054#1_XSS__3)
  - [XSS 产生的原因](https://blog.csdn.net/zqf787351070/article/details/126686054#XSS__6)
  - [XSS 分类](https://blog.csdn.net/zqf787351070/article/details/126686054#XSS__9)
  - [XSS 防御](https://blog.csdn.net/zqf787351070/article/details/126686054#XSS__14)
- [2\. CSRF 跨站请求伪造](https://blog.csdn.net/zqf787351070/article/details/126686054#2_CSRF__18)
  - [CSRF 流程](https://blog.csdn.net/zqf787351070/article/details/126686054#CSRF__21)
  - [CSRF 防御](https://blog.csdn.net/zqf787351070/article/details/126686054#CSRF__24)
- [3\. SSRF 服务端请求伪造](https://blog.csdn.net/zqf787351070/article/details/126686054#3_SSRF__29)
  - [SSRF 产生原因](https://blog.csdn.net/zqf787351070/article/details/126686054#SSRF__33)
  - [SSRF 漏洞出现场景](https://blog.csdn.net/zqf787351070/article/details/126686054#SSRF__38)
  - [SSRF 漏洞危害](https://blog.csdn.net/zqf787351070/article/details/126686054#SSRF__45)
  - [SSRF 防御措施](https://blog.csdn.net/zqf787351070/article/details/126686054#SSRF__53)
- [4\. SQL 注入](https://blog.csdn.net/zqf787351070/article/details/126686054#4_SQL__60)
  - [常见防御手段](https://blog.csdn.net/zqf787351070/article/details/126686054#_65)

## 1\. XSS 跨站脚本攻击

XSS(Cross-Site Scripting) 跨站脚本攻击是一种常见的安全漏洞，恶意攻击者在用户提交的数据中加入一些代码，将代码嵌入到了Web页面中，从而可以盗取用户资料，控制用户行为或者破坏页面结构和样式等。

#### XSS 产生的原因

XSS 产生的原因是过于信任客户端的数据，没有做好过滤或者转义等工作。如果客户端上传的数据中插入一些符号以及 javascript 代码，那么这些数据将会成为应用代码中的一部分了，这样就造成了 XSS 攻击。

#### XSS 分类

- 存储型：攻击者将恶意代码存储到了数据库中，在响应浏览器请求的时候返回恶意代码，并且执行。这种攻击常见于带有用户保存数据的网站功能；
- 反射型：将恶意代码放在 URL 中，将参数提交到服务器。服务器解析后响应，在响应结果中存在 XSS 代码，最终通过浏览器解析执行；
- DOM 型：取出和执行恶意代码由浏览器端完成，属于前端 JavaScript 的安全漏洞。

#### XSS 防御

- 对重要的 cookie 设置 httpOnly, 防止客户端通过document.cookie读取 cookie；
- 对输入内容的特定字符进行编码，前端后端都可以对传入的内容进行过滤，去掉带 javascript 等字段的输入

## 2\. CSRF 跨站请求伪造

CSRF(Cross-site request forgery) 跨站请求伪造，也是一种常见的安全漏洞。XSS 相当于是控制了站点内的信任用户，而 CSRF 则通过伪装成受信任用户的请求来利用受信任的网站。

#### CSRF 流程

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5a36a9a78dea58f6af8f8000d3fba20b.png#pic_center)

#### CSRF 防御

- Referer 头验证：在 HTTP 头中有一个字段叫 Referer，它记录了该 HTTP 请求的来源地址。不靠谱，Referer可以被改变；
- Token验证：服务器发送给客户端一个 Token，客户端提交的表单中（或者 URL 上）带着这个 Token。如果这个 Token 不合法，那么服务器拒绝这个请求；
- 双重Cookie验证：利用恶意网站无法获取 cookie 信息，仅可冒用的特点，我们将 cookie 中的参数取出来，加入到请求参数中，服务端进行校验，如果参数中没有附加额外的 cookie 中的参数，那么就拒绝请求。

## 3\. SSRF 服务端请求伪造

SSRF 是一种由攻击者构造请求，利用服务端发起的一种安全漏洞。一般情况下， SSRF 攻击的目标是外网无法访问的内部系统，借助于公网上的服务器来访问了内网系统。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f1702e75db9d601ac9512cd630ac3b96.png#pic_center)

#### SSRF 产生原因

SSRF 形成的原因大都是由于服务端提供了从其他服务器应用获取数据的功能，且没有对目标地址做过滤与限制。

比如指定 URL 地址获取网页文本内容，加载指定地址的图片和文档等。

#### SSRF 漏洞出现场景

- 分享场景，通过URL地址分享网页内容。
- 转码服务，在线翻译场景。
- 地址加载或下载图片。
- 图片、文章收藏功能。
- 未公开的api实现以及其他调用URL的功能等。

#### SSRF 漏洞危害

因为外网借助了服务端来实现了对内网服务器的访问，所以很多操作都可以进行，包括如下的危害：

- 对服务器所在的内网进行端口扫描，获取一些服务的banner信息等。
- 攻击运行在内网或者本地的应用程序。
- 对内网WEB应用进行指纹识别，通过访问默认文件实现。
- 下载内网的一些资源文件等。

#### SSRF 防御措施

- 对错误信息进行统一处理，避免用户可以根据错误信息来判断远端服务器的端口状态。
- 对请求的端口进行限制，限定为 HTTP 常用的端口，比如，80，443 和 8080 等。
- 设定 IP 黑名单。避免应用被用来获取内网数据，攻击内网。
- 禁用不需要的协议。
- 仅仅允许 HTTP 和 HTTPS 请求。对返回信息进行有效过滤等。

## 4\. SQL 注入

SQL 注入是指通过把 SQL 命令插入到 Web 表单提交或输入域名或页面请求的查询字符串，最终达到欺骗服务器，执行恶意的 SQL 命令。

比如说用户在登录的时候，使用了 or 1=1 来完成身份验证和授权。

#### 常见防御手段

- 使用预编译语句，比如 MyBatis 中的 SQL 语句使用 # 号代替 $ 符号。
- 使用安全的存储过程来防止 SQL 注入。
- 对客户端的输入进行数据类型的检查等。

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png)

确定要放弃本次机会？


福利倒计时

_:_ _:_

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png)立减 ¥

普通VIP年卡可用

[立即使用](https://mall.csdn.net/vip)

[![](https://profile-avatar.csdnimg.cn/6970612f99224087a2b6144466825531_zqf787351070.jpg!1)\\
情绪大瓜皮丶](https://blog.csdn.net/zqf787351070)

关注关注

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png)
0

点赞

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png)
踩

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png)
0




收藏







觉得还不错?

一键收藏
![](https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png)

- [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png)\\
0](https://blog.csdn.net/zqf787351070/article/details/126686054#commentBox)
评论

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/share.png)分享




复制链接



分享到 QQ



分享到新浪微博









![](https://csdnimg.cn/release/blogv2/dist/pc/img/share/icon-wechat.png)扫一扫


- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/more.png)


![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报



![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报


专栏目录

[_网络安全_ _面试_ 必问](https://blog.csdn.net/qq_41755666/article/details/123834539)

[赤赤三的博客](https://blog.csdn.net/qq_41755666)

03-29![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[项目经历\\
因为大家写的都是渗透相关，所以编故事也要编的圆润些，题材可以去freebuf看\\
https://search.freebuf.com/search/?search=%E6%8C%96%E6%B4%9E#article\\
这里主要记录如何挖洞的，实际项目也可以百度一下 看看别人故事咋编的\\
2、技术能力\\
大家底子应该都各不相同，在 _面试_ 中可能会问到你们不懂的知识点，不要慌 可以迂回战术 ，大部分问题应该会围绕 应急 溯源 渗透（近一年比较火的漏洞）来问，也可能会问网络基础的一些东...](https://blog.csdn.net/qq_41755666/article/details/123834539)

[_网络安全_ _面试_ 常见问题](https://blog.csdn.net/weixin_43748615/article/details/107678336)

[weixin\_43748615的博客](https://blog.csdn.net/weixin_43748615)

07-29![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
7196


[常见基础 _面试_ 问题1、请描述常见 _Web_ 攻击？Owasp TOP10有哪些？2、重要协议分布层3、请描述arp协议的工作原理4、rip协议是什么？rip的工作原理5、什么是RARP？工作原理6、OSPF协议是什么？并描述OSPF的工作原理。7、TCP与UDP区别是什么？8、什么是三次握手四次挥手？9、请描述tcp三次握手？为什么？10、dns是什么？请描述dns的工作原理11、请描述一次完整的HTTP请求过程12、请描述Cookies和session区别是什么？13、请描述GET 和 POST 的区别是什么](https://blog.csdn.net/weixin_43748615/article/details/107678336)

参与评论您还未登录，请先登录后发表或查看评论

[IT _安全_ 防护: _SQL_ _注入_、文件上传、 _XSS_、 _SSRF_ 与 _CSRF_ 漏洞及其防御策略,-CSDN...](https://blog.csdn.net/zyw268/article/details/138452712)

4-24

[​ _XSS_ 又叫CSS(Cross Site Script)跨站脚本攻击是指恶意攻击者往 _Web_ 页面里插入恶意Script代码,当用户浏览该页之时,嵌入其中 _Web_ 里面的Script代码会被执行,从而达到恶意攻击用户的目的。 ​ _xss_ 漏洞通常是通过php的输出函数将 _java_ script代码输出到html页面中,通过用户本地浏览器执行的,所以 _xss_ 漏洞关](https://blog.csdn.net/zyw268/article/details/138452712)

[常见漏洞简介, _网络安全_( _sql_ _注入_、 _xss_、xxe、 _CSRF_、文件上传等)](https://blog.csdn.net/weixin_50651979/article/details/137049873)

4-29

[1.2 _sql_ _注入_ 类型 从 _注入_ 参数来分:数值型 _注入_、字符型 _注入_、搜索 _注入_ 从 _注入_ 方法来分:布尔型盲注、时间延迟盲注、报错盲注、二次 _注入_、宽字节 _注入_、联合 _注入_、堆叠查询、内联查询 _注入_ 从提交方式来分:GET _注入_、POST _注入_、COOKIE _注入_、http头部 _注入_ 从数据库分:my _sql_ _注入_、ms _sql_ _注入_、oracle _注入_ 等 1.3 _sql_ _注入_ 防御 防御...](https://blog.csdn.net/weixin_50651979/article/details/137049873)

[_Web_ _安全_ 基础](https://blog.csdn.net/weixin_50906078/article/details/124214659)

[weixin\_50906078的博客](https://blog.csdn.net/weixin_50906078)

04-16![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
4034


[一、HTTP协议\\
1、HTTP\\
什么是HTTP？\\
超文本传输协议，HTTP是基于B/S架构进行通信的，而HTTP的服务器端实现程序有httpd、nginx等，其客户端的实现程序主要\\
是 _Web_ 浏览器，例如Firefox、InternetExplorer、Google chrome、Safari、Opera等\\
HTTP是基于客户/服务器模式，且面向连接的。典型的HTTP事务处理有如下的过程\\
（1）客户与服务器建立连接；\\
（2）客户向服务器提出请求；\\
（3）服务器接受请求，并根据请求返回相应的文件作为应答；\\
（4](https://blog.csdn.net/weixin_50906078/article/details/124214659)

[_XSS_ 攻击、 _CSRF_ 攻击、 _SQL_ _注入_ 攻击](https://blog.csdn.net/guaituo0129/article/details/103369568)

[厚积薄发](https://blog.csdn.net/guaituo0129)

12-03![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
681


[_XSS_\\
\\
概念\\
\\
跨站脚本攻击Cross-site scripting ( _XSS_)是一种 _安全_ 漏洞，攻击者可以利用这种漏洞在网站上 _注入_ 恶意的客户端代码。当被攻击者登陆网站时就会自动运行这些恶意代码，从而，攻击者可以突破网站的访问权限，冒充受害者。\\
\\
类型\\
\\
存储型 _XSS_ _注入_ 型脚本永久存储在目标服务器上。当浏览器请求数据时，脚本从服务器上传回并执行。\\
\\
反射型 _XSS_\\
\\
当用户点击一个恶意链接，或者...](https://blog.csdn.net/guaituo0129/article/details/103369568)

[_Web_ _安全_ 工程师 _面试_( _SQL_、 _XSS_、 _CSRF_、 _SSRF_、XXE)\_ _web_ _安全_ 考试 _sql_ 类型-CS...](https://blog.csdn.net/Hardworking666/article/details/122070324)

5-4

[_SQL_ _注入_ 攻击利用的工具是 _SQL_ 语法。 1、 _SQL_ _注入_ 的危害 1、非法查询、修改或删除数据库资源。 2、执行系统命令。 3、获取承载主机操作系统和网络的访问权限。 2、 _SQL_ _注入_ 思路 1、 _注入_ 点选择 2、数字型和字符型 _注入_ 3、通过 _Web_ 端对数据库 _注入_ 或者直接访问数据库 _注入_ 3、 _SQL_ _注入_ 的类型 1、报错注⼊ 2、bool 型注...](https://blog.csdn.net/Hardworking666/article/details/122070324)

[常见 _Web_ _安全_ 漏洞: _SQL_ _注入_、 _XSS_、 _CSRF_ 与XXE详解与防范](https://blog.csdn.net/weixin_63560942/article/details/137113392)

4-10

[_xss_ 防范 _csrf_ 漏洞原理 _csrf_ 漏洞流程图 前言 这里介绍一下常见的漏洞及其知识。 _sql_ _注入_ _sql_ _注入_ 原理 原理:所谓 _SQL_ _注入_,就是通过把 _SQL_ 命令插入到 _Web_ 表单提交或输入域名或页面请求的查询字符串,最终达到欺骗服务器执行恶意的 _SQL_ 命令。具体来说,它是利用现有应用程序,将(恶意)的 _SQL_ 命令 _注入_ 到后台数据库引擎执行的能力,它...](https://blog.csdn.net/weixin_63560942/article/details/137113392)

[_SQL_ _注入_、 _XSS_ 攻击、 _CSRF_ 攻击](https://blog.csdn.net/weixin_33910434/article/details/92661591)

[weixin\_33910434的博客](https://blog.csdn.net/weixin_33910434)

09-22![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1289


[_SQL_ _注入_、 _XSS_ 攻击、 _CSRF_ 攻击\\
\\
_SQL_ _注入_\\
\\
什么是 _SQL_ _注入_ _SQL_ _注入_，顾名思义就是通过 _注入_ _SQL_ 命令来进行攻击，更确切地说攻击者把 _SQL_ 命令插入到 _web_ 表单或请求参数的查询字符串里面提交给服务器，从而让服务器执行编写的恶意的 _SQL_ 命令。\\
对于 _web_ 开发者来说， _SQL_ _注入_ 已然是非常熟悉的，而且 _SQL_ _注入_ 已经生存了 10 多...](https://blog.csdn.net/weixin_33910434/article/details/92661591)

[_网络安全_ _面试_ 题\\
\\
热门推荐](https://blog.csdn.net/stqer/article/details/123612376)

[stqer的博客](https://blog.csdn.net/stqer)

03-20![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2万+


[网络\\
OSI七层模型\\
重要性\\
对网络有个初步整体了解\\
原理\\
TCP三次握手、四次挥手\\
重要性\\
\\
对于网络运维，关键在于故障排查，而TCP三次握手和四次挥手原理，有助于网络排障，分析数据包的异常状态。\\
故障排查，需要了解通信建立的原理，即TCP三次握手，再了解通信终止的原理，即TCP四次挥手，从中找到异常的地方，进行专项分析。\\
\\
原理\\
\*\*三次握手（Three-Way- Handshake）\*\*即建立TCP连接，就是指建立一个TCP连接时，需要客户端和服务端总共发送3个包以确认连接的建立。在 socket编程](https://blog.csdn.net/stqer/article/details/123612376)

[IT _安全_ 防护: _SQL_ _注入_、 _XSS_、 _CSRF_ 等常见 _Web_ 漏洞及其修复](https://blog.csdn.net/m0_71300782/article/details/132823956)

4-20

[1、数据泄露:攻击者可以利用 _SQL_ _注入_ 漏洞获取数据库中的敏感信息,如用户账号、密码、个人信息等,从而导致用户隐私泄露。 2、数据篡改:攻击者可以通过 _注入_ 恶意的 _SQL_ 语句来修改数据库中的数据,包括增加、删除、修改等操作,从而破坏数据的完整性和准确性。 3、服务器攻击:攻击者可以通过 _注入_ 恶意的 _SQL_ 语句来执行任意的操作...](https://blog.csdn.net/m0_71300782/article/details/132823956)

[_SQL_ _注入_ 防御策略与 _XSS_、 _CSRF_、 _SSRF_ 漏洞分析:PHP _安全_ 实践](https://blog.csdn.net/weixin_65633498/article/details/138026075)

5-1

[DOM型 _xss_ 和别的 _xss_ 最大的区别就是它不经过服务器,仅仅是通过网页本身的 _Java_ Script进行渲染触发的 _Csrf_ 不用获取用户的cookie就能冒充用户的身份,来做一些操作 _SSRF_(Server-Side Request Forgery:服务器端请求伪造) 是一种由内部系统访问的,所以可以通过它攻击外网无法访问的内部系统,也就是把目标网站当中间人) ...](https://blog.csdn.net/weixin_65633498/article/details/138026075)

[_网络安全_ 岗位 _面试_ 题](https://blog.csdn.net/WEARE001/article/details/123291265)

[流浪法师的博客](https://blog.csdn.net/WEARE001)

03-05![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2万+


[介绍了 _网络安全_ 岗位常见的 _面试_ 题，仅供参考！](https://blog.csdn.net/WEARE001/article/details/123291265)

[最新 _网络安全_ 岗位 _面试_ 题汇总（附答案解析）\_甲方 _安全_ 运维 _面试_ 问题](https://blog.csdn.net/ewii12567/article/details/139992961)

[ewii12567的博客](https://blog.csdn.net/ewii12567)

06-26![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2287


[_网络安全_ 产业就像一个江湖，各色人等聚集。相对于欧美国家基础扎实（懂加密、会防护、能挖洞、擅工程）的众多名门正派，我国的人才更多的属于旁门左道（很多白帽子可能会不服气），因此在未来的人才培养和建设上，需要调整结构，鼓励更多的人去做“正向”的、结合“业务”与“数据”、“自动化”的“体系、建设”，才能解人才之渴，真正的为社会全面互联网化提供 _安全_ 保障。](https://blog.csdn.net/ewii12567/article/details/139992961)

[...常见漏洞的挖掘利用与防护方法(如 _sql_ _注入_、 _csrf_ 跨站请求伪造、命令...](https://blog.csdn.net/2401_84466225/article/details/154125701)

4-25

[摘要:​​ 本文系统讲解 _Web_ _安全_ 的核心漏洞原理、攻击手法和防护方案,涵盖 _SQL_ _注入_、 _XSS_、 _CSRF_、 _SSRF_ 等常见漏洞,提供完整的代码示例和实战案例,帮助开发者构建 _安全_ 的 _Web_ 应用。 一、 _SQL_ _注入_ 深度解析与防护​ ​1.1 _SQL_ _注入_ 原理与利用​ ​漏洞代码示例:​​ ...](https://blog.csdn.net/2401_84466225/article/details/154125701)

[_Web_ _安全_ 漏洞详解: _SQL_ _注入_、 _XSS_、文件上传及OWASPTop10](https://blog.csdn.net/weixin_64267091/article/details/136147914)

4-6

[先来总结一下我上学期间学习到的一些漏洞,分别有 _sql_ _注入_ 漏洞,文件上传漏洞, _XSS_, _CSRF_,RCE, _SSRF_。然后再来总结一下OWASP top10。 _sql_ _注入_ 原理: 由于网站的开发者在处理 _sql_ 语句不严谨,从而导致攻击者可以插入恶意的 _sql_ 语句,然后使得原有的 _sql_ 语句产生了歧义,从而达到了攻击者目的。](https://blog.csdn.net/weixin_64267091/article/details/136147914)

[50道 _网络安全_ _面试_ 题（附答案及解析）建议收藏！](https://blog.csdn.net/2401_85688943/article/details/149602453)

[2401\_85688943的博客](https://blog.csdn.net/2401_85688943)

07-24![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1407


[_网络安全_ _面试_ 题！ _面试_ 宝典！总结心得！（附答案及解析）会持续更新哦！ （2025.7月最新版）感谢支持，你们的鼓励是我的动力！](https://blog.csdn.net/2401_85688943/article/details/149602453)

[_网络安全_ 基础知识 _面试_ 题库](https://blog.csdn.net/Algo_x/article/details/114359796)

[Algo\_x的博客](https://blog.csdn.net/Algo_x)

03-06![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[1.基于路由器的攻击手段\\
\\
源IP地址欺骗式攻击\\
\\
入侵者从外部传输一个伪装成来自内部主机的数据包（数据包的IP是内网的合法IP）\\
\\
丢弃所有来自路由器外端口，却使用内部源地址的数据包\\
\\
源路由攻击\\
\\
入侵者让数据包循着一个对方不可预料的路径到达目的地，以逃避 _安全_ 系统的审核\\
\\
丢弃所有包含源路由选项的数据包\\
\\
源路由\\
\\
是指在数据包中还要列出所要经过的路由。某些路由器对源路由包的反应是使用其指定的路由，并使用其反向路由来传送应答数据。这就使一个\*\*\*者可以\\
假冒一个主机的名义通过一个特殊的路径来获得某...](https://blog.csdn.net/Algo_x/article/details/114359796)

[_web_ _安全_ 及防护(一、 _XSS_、二、 _CSRF_、三、 _SQL_ _注入_)、\_ _xss_ 、 _csrf_、 _ssrf_...](https://blog.csdn.net/m0_57142556/article/details/118117119)

4-26

[三、 _SQL_ _注入_ 攻击: 原理: _SQL_ _注入_( _SQL_ Injection),应用程序在向后台数据库传递 _SQL_(Structured Query Language,结构化查询语言)时,攻击者将 _SQL_ 命令插入到 _Web_ 表单提交或输入域名或页面请求的查询字符串,最终达到欺骗服务器执行恶意的 _SQL_ 命令。 _SQL_ 防范: ①增加黑名单或者白名单验证 ...](https://blog.csdn.net/m0_57142556/article/details/118117119)

[网安 _面试_ _网络安全_ _面试_ 2023最新校招指南.zip](https://download.csdn.net/download/yhsbzl/91898367)

09-08

[对于2023年的新一轮校招， _网络安全_ _面试_ 指南会更加注重当前 _网络安全_ 领域的最新发展趋势。例如，云计算和物联网(IoT)的 _安全_、大数据环境下的 _安全_ 防护策略、人工智能与机器学习在 _网络安全_ 中的应用等方面。同时，也会...](https://download.csdn.net/download/yhsbzl/91898367)

精选资源 [93道 _网络安全_ _面试_ 题目](https://download.csdn.net/download/qq_32277727/86246191)

07-20

[“93道 _网络安全_ _面试_ 题目” 从给定的文件信息中，我们可以总结出以下知识点： 1\. _SQL_ _注入_ 攻击 \\* 定义：攻击者在 HTTP 请求中 _注入_ 恶意的 _SQL_ 代码，使服务器使用参数构建数据库 _SQL_ 命令时，恶意 _SQL_ 被一起构造，...](https://download.csdn.net/download/qq_32277727/86246191)

精选资源 [93道 _网络安全_ _面试_ 题PDF资料](https://download.csdn.net/download/xiaoli8748/88427906)

10-14

[_网络安全_ _面试_ 题涵盖了多个关键领域的知识，包括 _SQL_ _注入_ 攻击、 _XSS_ 攻击、 _CSRF_ 攻击、文件上传漏洞和DDoS攻击，以及ARP协议的工作原理。下面将详细解释这些知识点： 1. \*\* _SQL_ _注入_ 攻击\*\*：这是一种利用用户输入数据构造...](https://download.csdn.net/download/xiaoli8748/88427906)

[_网络安全_ _面试_ 类题目-详解](https://download.csdn.net/download/qiqi776532/88849128)

02-19

[【搞定网络协议】之 _网络安全_ _面试_ 题.pdf 北京某安 _安全_ 服务 _面试_ 经验分享,pdf 常考渗透测试 _面试_ 题.pdf 吧)护网 _面试_ 题总结+DD _安全_ 工程师笔试问题.pd吧几率大的 _网络安全_ _面试_ 题(含答案).pdf _面试_.pdf _面试_ 经验分享.pdf 某...](https://download.csdn.net/download/qiqi776532/88849128)

精选资源 [各大 _安全_ 厂商 _网络安全_ _面试_ 题汇总（7份）.zip](https://download.csdn.net/download/goodxianping/44142065)

11-18

[各大 _安全_ 厂商 _网络安全_ _面试_ 题汇总，共7份。适合找工作的 _安全_ 同志。 360-几率大的 _网络安全_ _面试_ 题（含答案） 百度-搞定网络协议】之 _网络安全_ _面试_ 题 京东护网 _面试_ 题总结+DD _安全_ 工程师笔试问题 渗透测试初级 _面试_ 题 渗透...](https://download.csdn.net/download/goodxianping/44142065)

[HarmonyOS Next _面试_ 题之异步并发Promise和async/await的核心机制](https://devpress.csdn.net/v1/article/detail/160652044)

[╰つ栺尖篴夢ゞ](https://blog.csdn.net/Forever_wj)

04-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
321


[本文系统介绍了HarmonyOS Next中的异步并发编程机制。主要内容包括：1）异步编程的核心思想和实现手段，区分了异步I/O任务和多线程并发的适用场景；2）Promise对象的状态机特性及其静态方法all/allSettled/race/any的应用场景；3）async/await语法糖的本质优势，以及与传统Promise链式调用的对比；4）实际开发中的注意事项，如async回调问题、生命周期函数异常处理等；5）深入分析了async/await与多线程的本质区别，以及Promise构造函数内部的同步执行](https://devpress.csdn.net/v1/article/detail/160652044)

[大数据凉了？速看4月的就业数据新鲜出炉！AI时代岗位不会原地消失，而是岗位的标准会被逐步抬高\\
\\
最新发布](https://hero78.blog.csdn.net/article/details/160826329)

[涤生大数据](https://blog.csdn.net/qq_26442553)

05-06![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
192


[AI浪潮下职场生存指南：岗位重塑而非取代 当前AI技术迅猛发展引发职场焦虑，但实际并非岗位消失，而是标准提升。未来3年，多数岗位将经历"AI+"转型： 岗位重塑：数据分析、运维等传统职位将借助AI工具提效 就业趋势：头部大模型岗位门槛高（需211硕士+），更多机会在行业应用层 最新案例：20位求职者成功转型，包括： 3年经验双非硕士斩获45W offer 6年普本开发者获60W中大厂职位 大专学历者突破限制拿到25K月薪 核心建议：不必盲目追求底层开发，应聚焦"AI+行业&qu](https://hero78.blog.csdn.net/article/details/160826329)

[_面试_ 复盘之WHERE和HAVING的区别以及MySL的索引](https://blog.csdn.net/pEOly48AM/article/details/160823857)

[pEOly48AM的博客](https://blog.csdn.net/pEOly48AM)

05-06![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
244


[OFA（One For All）是字节跳动提出的多模态预训练模型，支持视觉问答、图像描述、图像编辑等多种任务，其中视觉问答（VQA）是最常用的功能之一——输入一张图片和一个英文问题（该模型仅支持英文），模型就能输出对应的答案（比如输入“瓶子”图片+问题“What is the main subject?ModelScope 加载 OFA 模型时，会自动检查依赖版本，如果发现版本和它硬编码的要求不一致，会直接卸载你的版本并强制安装指定版本——哪怕你已经安装了正确的版本，也会被覆盖，导致之前的努力白费。](https://blog.csdn.net/pEOly48AM/article/details/160823857)

[HarmonyOS Next _面试_ 题之Worker常驻线程如何通过TaskPool进行多任务并发处理](https://blog.csdn.net/Forever_wj/article/details/160823730)

[╰つ栺尖篴夢ゞ](https://blog.csdn.net/Forever_wj)

05-06![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
196


[本文介绍了HarmonyOS Next中Worker与TaskPool结合的高性能后台任务处理方案。核心架构采用Worker作为常驻调度中心，通过主线程下发任务、Worker维护状态、TaskPool执行具体任务的模式，实现了职责分离与高效并发。文章详细展示了图片批量上传的实战案例，包括任务定义、Worker调度中心实现和主线程交互流程。该方案优势在于：Worker管理上下文保持状态，TaskPool提供无状态并发执行，二者结合既保证了任务调度的灵活性，又充分利用了系统资源。](https://blog.csdn.net/Forever_wj/article/details/160823730)

[_Java_ 高频 _面试_ 考点场景题20](https://blog.csdn.net/DreamComeTrue001/article/details/160720175)

[DreamComeTrue001的博客](https://blog.csdn.net/DreamComeTrue001)

05-02![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
828


[《数据库与系统优化实战精要》摘要：针对深分页 _SQL_ 导致的CPU过载问题，核心在于避免全表扫描和回表操作，可通过索引覆盖、ID分页等方案优化。高并发API统计推荐Flink+Kafka实时处理或ELK日志分析方案，前者实时性强后者维护简单。MVCC机制通过事务ID、UndoLog和ReadView实现读写并发控制，但需注意大事务带来的版本链问题。订单超时处理建议采用RedisZSet或RabbitMQ死信队列替代低效的定时任务。秒杀系统要重点防范对象创建过快引发的OOM，需建立从JVM调优到服务降级的立体防御](https://blog.csdn.net/DreamComeTrue001/article/details/160720175)

- [关于我们](https://www.csdn.net/company/index.html#about)
- [招贤纳士](https://www.csdn.net/company/index.html#recruit)
- [商务合作](https://fsc-p05.txscrm.com/T8PN8SFII7W)
- [寻求报道](https://marketing.csdn.net/questions/Q2202181748074189855)
- ![](https://g.csdnimg.cn/common/csdn-footer/images/tel.png)400-660-0108
- ![](https://g.csdnimg.cn/common/csdn-footer/images/email.png)[kefu@csdn.net](mailto:webmaster@csdn.net)
- ![](https://g.csdnimg.cn/common/csdn-footer/images/cs.png)[在线客服](https://csdn.s2.udesk.cn/im_client/?web_plugin_id=29181)
- 工作时间 8:30-22:00


- ![](https://g.csdnimg.cn/common/csdn-footer/images/badge.png)[公安备案号11010502030143](http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=11010502030143)
- [京ICP备19004658号](http://beian.miit.gov.cn/publish/query/indexFirst.action)
- [京网文〔2020〕1039-165号](https://csdnimg.cn/release/live_fe/culture_license.png)
- [经营性网站备案信息](https://csdnimg.cn/cdn/content-toolbar/csdn-ICP.png)
- [北京互联网违法和不良信息举报中心](http://www.bjjubao.org/)
- [家长监护](https://download.csdn.net/tutelage/home)
- [网络110报警服务](https://cyberpolice.mps.gov.cn/)
- [中国互联网举报中心](http://www.12377.cn/)
- [Chrome商店下载](https://chrome.google.com/webstore/detail/csdn%E5%BC%80%E5%8F%91%E8%80%85%E5%8A%A9%E6%89%8B/kfkdboecolemdjodhmhmcibjocfopejo?hl=zh-CN)
- [账号管理规范](https://blog.csdn.net/blogdevteam/article/details/126135357)
- [版权与免责声明](https://www.csdn.net/company/index.html#statement)
- [版权申诉](https://blog.csdn.net/blogdevteam/article/details/90369522)
- [出版物许可证](https://img-home.csdnimg.cn/images/20250103023206.png)
- [营业执照](https://img-home.csdnimg.cn/images/20250103023201.png)
- ©1999-2026北京创新乐知网络技术有限公司

登录后您可以享受以下权益：

- ![](<Base64-Image-Removed>)免费复制代码
- ![](<Base64-Image-Removed>)和博主大V互动
- ![](<Base64-Image-Removed>)下载海量资源
- ![](<Base64-Image-Removed>)发动态/写文章/加入社区

×立即登录

评论![](https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png)

![](https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowLeftWhite.png)被折叠的  条评论
[为什么被折叠?](https://blogdev.blog.csdn.net/article/details/122245662) [![](https://csdnimg.cn/release/blogv2/dist/pc/img/iconPark.png)到【灌水乐园】发言](https://bbs.csdn.net/forums/FreeZone)

查看更多评论![](https://csdnimg.cn/release/blogv2/dist/pc/img/commentArrowDownWhite.png)

添加红包


祝福语

请填写红包祝福语或标题

红包数量

个

红包个数最小为10个

红包总金额

元

红包金额最低5元

余额支付

当前余额3.43元
[前往充值 >](https://i.csdn.net/#/wallet/balance/recharge)

需支付：10.00元


取消确定

实付元

使用余额支付

![](https://csdnimg.cn/release/blogv2/dist/pc/img/pay-time-out.png)点击重新获取

![](https://csdnimg.cn/release/blogv2/dist/pc/img/weixin.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/zhifubao.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/jingdong.png)扫码支付

钱包余额0

![](https://csdnimg.cn/release/blogv2/dist/pc/img/pay-help.png)

抵扣说明：

1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。

2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。

[![](https://csdnimg.cn/release/blogv2/dist/pc/img/recharge.png)余额充值](https://i.csdn.net/#/wallet/balance/recharge)

![](https://blog.csdn.net/zqf787351070/article/details/126686054)

确定取消![](https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png)

举报

![](https://csdnimg.cn/release/blogv2/dist/pc/img/closeBlack.png)

选择你想要举报的内容（必选）

- 内容涉黄
- 政治相关
- 内容抄袭
- 涉嫌广告
- 内容侵权
- 侮辱谩骂
- 样式问题
- 其他

原文链接（必填）

请选择具体原因（必选）

- 包含不实信息
- 涉及个人隐私

请选择具体原因（必选）

- 侮辱谩骂
- 诽谤

请选择具体原因（必选）

- 搬家样式
- 博文样式

补充说明（选填）

取消

确定

[![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/Group.png)点击体验\\
\\
DeepSeekR1满血版](https://ai.csdn.net/chat?utm_source=cknow_pc_blogdetail&spm=1001.2101.3001.10583)![](https://g.csdnimg.cn/side-toolbar/3.6/images/mobile.png)

下载APP

![程序员都在用的中文IT技术交流社区](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_app.png)

程序员都在用的中文IT技术交流社区

公众号

![专业的中文 IT 技术社区，与千万技术人共成长](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_wechat.png)

专业的中文 IT 技术社区，与千万技术人共成长

视频号

![关注【CSDN】视频号，行业资讯、技术分享精彩不断，直播好礼送不停！](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_video.png)

关注【CSDN】视频号，行业资讯、技术分享精彩不断，直播好礼送不停！

![](https://g.csdnimg.cn/side-toolbar/3.6/images/customer.png)客服

新手引导

![](https://g.csdnimg.cn/side-toolbar/3.6/images/totop.png)返回顶部

![](https://i-operation.csdnimg.cn/images/fb287ddc3c984e04a2021d439632f08c.png)

提问

![](https://csdnimg.cn/release/blogv2/dist/pc/img/quoteClose1White.png)

![](https://i-blog.csdnimg.cn/blog_migrate/5a36a9a78dea58f6af8f8000d3fba20b.png#pic_center)

![](https://i-blog.csdnimg.cn/blog_migrate/f1702e75db9d601ac9512cd630ac3b96.png#pic_center)

-100%+1:1还原