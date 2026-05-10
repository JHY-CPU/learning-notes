# 【面试】运维工程师面试题及答案 - 阿里云开发者社区

URL: https://developer.aliyun.com/article/1363607

[开发者社区](https://developer.aliyun.com/) [开发与运维](https://developer.aliyun.com/group/othertech/) [文章](https://developer.aliyun.com/group/othertech/article/) 正文

# 【面试】运维工程师面试题及答案

2023-10-311516发布于天津

版权
举报

版权声明：

本文内容由阿里云实名注册用户自发贡献，版权归原作者所有，阿里云开发者社区不拥有其著作权，亦不承担相应法律责任。具体规则请查看《
[阿里云开发者社区用户服务协议](https://developer.aliyun.com/article/768092)》和
《 [阿里云开发者社区知识产权保护指引](https://developer.aliyun.com/article/768093)》。如果您发现本社区中有涉嫌抄袭的内容，填写
[侵权投诉表单](https://yida.alibaba-inc.com/o/right) 进行举报，一经查实，本社区将立刻删除涉嫌侵权内容。


**简介：**【面试】运维工程师面试题及答案

[双绞线](https://so.csdn.net/so/search?q=%E5%8F%8C%E7%BB%9E%E7%BA%BF&spm=1001.2101.3001.7020) 两种制作标准的线序：

> EIA/TIA 568A：绿白、绿、橙白、蓝、蓝白、橙、棕白、棕
>
>         EIA/TIA 568B：橙白、橙白、绿白、蓝、蓝白、绿、棕白、棕

连接方法有两种：

> 正线（双绞线两边都按照EIAT/TIA 568B 标准连接）
>
>         反线（一边是按照EIAT/TIA 568A 标准连接，另一边按照EIT/TIA 568B 标准连接）

填写以下各设备的连接方法：

> PC-PC:  反线、PC-HUB:  正线、HUB-HUB  反线 HUB-SWITCH: 正线SWITCH-SWITCH:   反线、SWITCH-ROUTER: 正线、ROUTER-ROUTER:  反线C类 [IP](https://so.csdn.net/so/search?q=IP&spm=1001.2101.3001.7020) 地址，但要连接6个子公司，最大的一个子公司有26台计算机，每个某公司申请到一个子公司在一个网段中，则子网掩码应设为255.255.255.224 IP 地址为201.103.136.184，其子网掩码为255.255.255.192，该主机是在已知某一主机的\_\_\_\_\_c\_\_类网络中，该主机所在子网最多允许有\_\_\_\_63\_\_\_\_\_\_台主机。的传输层提供的服务有两大类，即\_\_\_TCP/IP\_\_\_\_\_\_\_的服务和\_\_\_IPX/SPX \_\_\_的服务。
>
> OSI/RMOSI七层的哪一层？数据链路层交换机工作在均无故障时间的英文缩写\_\_\_MTBF()\_\_\_\_。平Mean Time Between Failure TCP/IP协议集中，传输层的\_UDP\_\_\_\_协议是一种面向无连接的协议，它不能提供可靠的数据包传输，没有差错检测功能。IP网络中，从IP [地址](https://so.csdn.net/so/search?q=%E5%9C%B0%E5%9D%80&spm=1001.2101.3001.7020) 映射到物理地址采用\_\_\_ARP（Address Resolution Protocol）是地址解在析协议\_\_\_协议。\_\_代理防火墙\_\_、\_\_双穴主机防火到目前为止，已出现了三种类型的防火墙，即数据包过滤、墙\_\_\_。

PIX配置是否正确，为什么?

下面几条

access-list 101 permit icmp any host web

access-list 101 permit tcp any host X.X.X.X eq www

access-list 101 permit tcp any host X.X.X.X eq 3389

access-list 101 permit tcp any host X.X.X.X eq ftp

access-list 101 deny tcp any any

access-list 101 permit tcp any host Y.Y.Y.Y eq www

access-list 101 permit tcp any host Y.Y.Y.Y eq 3389

答：

\_\_RAM（random access memory）随机存储器\_\_\_\_。

断电后，会使存储的数据丢失的存储器是

SSH 22 、Telnet 23 、SMTP 25 、POP3 110 、

下列服务的默认端口是多少

DNS 53 、远程桌面3389

和NAT有什么区别？

PAT

答：都是网络地址转换，只不过不同的是一个是一对一，一个是一对多可复用地址

转换，PAT是nat的一种NAT中文全称是地址转换,一般指的是内部IP和内部全局IP一一对应

PAT中文全称是端口转换,一般指的是内部IP与内部全局IP一对多对应Raid技术至少需要几块磁盘：

说明下列

Raid 0 2 、Raid 1 2 、Raid 5 3 、Raid 0＋1 4 、Raid 1+0 4

raid 0、1、5的特点和优点。

描述

> 答：RAID 0即Data Stripping（数据分条技术）。整个逻辑盘的数据是被分条（stripped）分
>
> 布在多个物理磁盘上，可以并行读/写，提供最快的速度，但没有冗余能力。要求至少两个
>
> 磁盘。我们通过RAID 0可以获得更大的单个逻辑盘的容量，且通过对多个磁盘的同时读取获
>
> 得更高的存取速度。RAID 0首先考虑的是磁盘的速度和容量，忽略了安全，只要其中一个磁
>
> 盘出了问题，那么整个阵列的数据都会不保了。
>
> 又称镜像方式，也就是数据的冗余。在整个镜像过程中，只有一半的磁盘容量是有效的（另
>
> 一半磁盘容量用来存放同样的数据）。同RAID 0相比，RAID 1首先考虑的是安全性，容量减
>
> 半、速度不变。
>
>         RAID5把数据和相对应的奇偶校验信息存储到组成RAID5的各个磁盘上，并且奇偶校验信息和
>
>         相对应的数据分别存储于不同的磁盘上，其中任意N-1块磁盘上都存储完整的数据，也就是说有相当于一块磁盘容量的空间用于存储奇偶校验信息。因此当RAID5的一个磁盘发生损坏后，不会影响数据的完整性，从而保证了数据安全。当损坏的磁盘被替换后，RAID还会自动利用剩下奇偶校验信息去重建此磁盘上的数据，来保持RAID5的高可靠性。
>
> DNS集成的活动目录，客户端在加入域的时候提示找不到域控制器，其中最有可配置一个与能出现问题的地方在哪里？
>
> 答：客户端没有将DNS指向和活动目录集成的DNS的缘故复制的基本元素包括SQL出版服务器、订阅服务器、分发服务器、出版物、文章复制技术类型快照复制、事务复制、合并复制SQL某台电脑不能通过UNC方式访问文件服务器，提示找不到路径，但其他客户端均可局域网内正常访问，该如何解决？
>
> 答：得先看看那个机器和其他的在不在同一网段下，你先PING下对方的IP就知道了
>
> drw-r--r--，用数值形式表示该权限，则该八进制数为：644，该文件某文件的权限为：属性是目录。（linux）

22、Nagios 监控系统中负责主机和服务监控的配置文件名称是  NRPE、。

23、简述IBM X系列服务器安装步骤：

答：

24、已知10.105.239.197的用户名为administrator，密码cosft#08#&ATS02，请问如何访问

默认共享？

> 答：在CMD中net use \\\10.105.239.197\\ipc$ cosft#08#&ATS02 /user:administrator

25、用sql语句查询teasttable中姓名为“张”的nikname的字段和email字段

答：

26、简单介绍你使用过的服务器监控软件，并陈述它们的特点

> 答：
>
> MRTG或者Cacti
>
> Mrtg的功能稍微单调些
>
> 而Cacti的功能稍微强大些

27、windows群集（cluster）和网络负载均衡（NLB）各自的特点及用处是什么？

> 答：使用11ns 可以对加入到负载均衡的机器设置权重。系统自动根据权重比例来分析
>
> 访问比例，对于流量比较大的网站，可以通过负载均衡功能把流量分配到几台不同的服务器
>
> 上，以提高网站的运行速度

28、服务器需要监控哪些项目？凭借这些项目如何判断服务器的瓶颈？

> 答：一般监控服务器的CPU，内存，磁盘空间，接口流量，IIS访问数、流量等。。。
>
> 数据库连接数

文章标签：

[负载均衡](https://developer.aliyun.com/label/article_de-product-3-slb)

[网络协议](https://developer.aliyun.com/label/article_de-3-100112)

[监控](https://developer.aliyun.com/label/article_de-3-100072)

[运维](https://developer.aliyun.com/label/article_de-3-100073)

[存储](https://developer.aliyun.com/label/article_de-3-100262)

[SQL](https://developer.aliyun.com/label/article_de-3-100064)

关键词：

[运维工程师](https://www.aliyun.com/sswb/835893.html)

[运维面试题](https://www.aliyun.com/sswb/553775.html)

[面试运维工程师面试题](https://www.aliyun.com/sswb/1343866.html)

[面试运维工程师](https://www.aliyun.com/sswb/656926.html)

[面试运维](https://www.aliyun.com/sswb/1281148.html)

相关实践学习

每个IT人都想学的“Web应用上云经典架构”实战

本实验从Web应用上云这个最基本的、最普遍的需求出发，帮助IT从业者们通过“阿里云Web应用上云解决方案”，了解一个企业级Web应用上云的常见架构，了解如何构建一个高可用、可扩展的企业级应用架构。

[![](https://ucc.alicdn.com/avatar/0fc429d454bd4b198588fb8bf43c7ce6.jpg?x-oss-process=image/resize,h_150,m_lfit)](https://developer.aliyun.com/profile/6h2qt5rxnv6ci)

![](https://ucc.alicdn.com/pic/ucc-admin/88c34b916d704521b87d41daa9a77107.png?x-oss-process=image%2Fresize%2Ch_80%2Cm_lfit%2Fformat%2Cwebp)

[征服Bug](https://developer.aliyun.com/profile/6h2qt5rxnv6ci)

+关注

[210文章](https://developer.aliyun.com/profile/6h2qt5rxnv6ci/article_1) [2问答](https://developer.aliyun.com/profile/6h2qt5rxnv6ci/ask_1)

目录

0

0

0

47

分享

相关文章

[大侠之运维](https://developer.aliyun.com/profile/wvv6ake3fmjoy)

\|

[Prometheus](https://developer.aliyun.com/label/sc/de-3-100271) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [Cloud Native](https://developer.aliyun.com/label/sc/de-3-100024)

[55k star,推荐一份关于devops、SRE、运维的手册，简直就算是一份面试大纲了](https://developer.aliyun.com/article/1582327)

【8月更文挑战第10天】

[大侠之运维](https://developer.aliyun.com/profile/wvv6ake3fmjoy)

63711

[游客g6r3tt6lma5xg](https://developer.aliyun.com/profile/g6r3tt6lma5xg)

\|

[弹性计算](https://developer.aliyun.com/label/sc/de-3-100063) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [安全](https://developer.aliyun.com/label/sc/de-3-100244)

[为了提升运维工程师及开发者](https://developer.aliyun.com/article/1648103)

为了提升运维工程师及开发者

[游客g6r3tt6lma5xg](https://developer.aliyun.com/profile/g6r3tt6lma5xg)

3206666

[阿里云安全-小安](https://developer.aliyun.com/profile/aaahn7m4yotok)

\|

[云安全](https://developer.aliyun.com/label/sc/de-3-100097) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [安全](https://developer.aliyun.com/label/sc/de-3-100244)

[AK泄漏28小时：运维工程师的极限自救](https://developer.aliyun.com/article/1655604)

随着比特币等加密货币的价格持续上涨，挖矿活动成为了黑客们眼中的一块肥肉。尤其是在2024年至2025年间，比特币价格突破历史高位，吸引了大量投资者和投机者的目光。与此同时，这也引发了新一轮的黑客攻击浪潮，目标直指那些拥有强大计算资源的企业和个人用户。

[阿里云安全-小安](https://developer.aliyun.com/profile/aaahn7m4yotok)

12201212

[码农阿豪](https://developer.aliyun.com/profile/gwgnf4umqhevg)

\|

[算法](https://developer.aliyun.com/label/sc/de-3-100255) [Java](https://developer.aliyun.com/label/sc/de-3-100001) [调度](https://developer.aliyun.com/label/sc/de-3-100259)

[《面试专题-----经典高频面试题收集四》解锁 Java 面试的关键：深度解析并发编程进阶篇高频经典面试题（第四篇）](https://developer.aliyun.com/article/1552857)

《面试专题-----经典高频面试题收集四》解锁 Java 面试的关键：深度解析并发编程进阶篇高频经典面试题（第四篇）

[码农阿豪](https://developer.aliyun.com/profile/gwgnf4umqhevg)

27500

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [监控](https://developer.aliyun.com/label/sc/de-3-100072) [网络协议](https://developer.aliyun.com/label/sc/de-3-100112)

[Linux运维工程师必知：如何在 Linux 中使用网络命令netstat？](https://developer.aliyun.com/article/1631925)

【10月更文挑战第21天】

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

95412

[![Linux运维工程师必知：如何在 Linux 中使用网络命令netstat？](https://ucc.alicdn.com/z3pojg2spmpe4/developer-article1631925/20241031/4f6f37a69cd24504b8714d7324791eb4.png?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/1631925)

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

\|

[存储](https://developer.aliyun.com/label/sc/de-3-100262) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [搜索推荐](https://developer.aliyun.com/label/sc/de-3-100046)

[Linux vim 操作大集合，Linux运维工程师收藏！](https://developer.aliyun.com/article/1621276)

【10月更文挑战第2天】

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

28101

[![Linux vim 操作大集合，Linux运维工程师收藏！](https://ucc.alicdn.com/z3pojg2spmpe4/developer-article1621276/20241012/516d78039b134f829ceee329d4f26f6e.png?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/1621276)

[土木林森](https://developer.aliyun.com/profile/mfonu6kasfx3y)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [监控](https://developer.aliyun.com/label/sc/de-3-100072) [网络协议](https://developer.aliyun.com/label/sc/de-3-100112)

[运维工程师日常工作中最常用的20个Linux命令，涵盖文件操作、目录管理、权限设置、系统监控等方面](https://developer.aliyun.com/article/1640642)

本文介绍了运维工程师日常工作中最常用的20个Linux命令，涵盖文件操作、目录管理、权限设置、系统监控等方面，旨在帮助读者提高工作效率。从基本的文件查看与编辑，到高级的网络配置与安全管理，这些命令是运维工作中的必备工具。

[土木林森](https://developer.aliyun.com/profile/mfonu6kasfx3y)

119434

[阿里云云原生](https://developer.aliyun.com/profile/pawmkwdq37c7s)

\|

[人工智能](https://developer.aliyun.com/label/sc/de-3-100052) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [自然语言处理](https://developer.aliyun.com/label/sc/de-3-100040)

[今晚围观—>安全运维工程师现场直播用通义灵码发现和修复代码漏洞](https://developer.aliyun.com/article/1646622)

12 月 18 日晚 19:30 分，阿里云中小企业直播间「AI 编码助手一年养成记：从“打酱油”到企业开发“真正助手”」见。

[阿里云云原生](https://developer.aliyun.com/profile/pawmkwdq37c7s)

40400

[k2otpl7gep5j4](https://developer.aliyun.com/profile/k2otpl7gep5j4)

\|

[弹性计算](https://developer.aliyun.com/label/sc/de-3-100063) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [Linux](https://developer.aliyun.com/label/sc/de-3-100077)

[运维工程师必备的摸鱼神器：阿里云智能助手OS Copilot](https://developer.aliyun.com/article/1563654)

OS Copilot 概述与体验评测摘要

阿里云的OS Copilot是一款基于大模型的智能操作系统助手，作为高级运维工程师，体验者发现它在系统诊断和性能优化上尤其有用，简化了如重置ECS密码和安全组配置等任务，提升了工作效率。
OS Copilot的易用性和安全性得到肯定，操作手册详细且交互性强，减少了用户在不同页面间切换的需要。在辅助编程方面，它能帮助非专业开发者编写和理解代码，对运维工作中的开发技能补充有很大帮助。与GitHub Copilot等产品相比，OS Copilot的独特之处在于结合了Linux操作的支持。

[k2otpl7gep5j4](https://developer.aliyun.com/profile/k2otpl7gep5j4)

77833

[![运维工程师必备的摸鱼神器：阿里云智能助手OS Copilot](https://ucc.alicdn.com/pic/developer-ecology/k2otpl7gep5j4_a9ea930d95904acc8af89a7966994e70.png?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/1563654)

[小Lee](https://developer.aliyun.com/profile/beyuwtntijacc)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [关系型数据库](https://developer.aliyun.com/label/sc/de-3-100067)

[云计算运维工程师面试技巧](https://developer.aliyun.com/article/1578673)

【8月更文挑战第6天】

[小Lee](https://developer.aliyun.com/profile/beyuwtntijacc)

152013

## 热门文章

## 最新文章

[1\\
\\
全网最全95道MongoDB面试题1万字详细解析\\
\\
20](https://developer.aliyun.com/article/764824)
[2\\
\\
Java 最常见的面试题：两个对象的 hashCode()相同，则 equals()也一定为 true，对吗？\\
\\
6](https://developer.aliyun.com/article/1169781)
[3\\
\\
两个对象的 hashCode()相同，则值一定相同吗？面试篇（第三天）\\
\\
8](https://developer.aliyun.com/article/905379)
[4\\
\\
网络协议报文理解刨析篇二(再谈Http和Https), 加上TCP/UDP/IP协议分析(理解着学习), 面试官都惊讶你对网络的见解（1）\\
\\
19](https://developer.aliyun.com/article/1135107)
[5\\
\\
几种有关排序的常见面试问题\\
\\
2](https://developer.aliyun.com/article/316393)
[6\\
\\
百度面试题的java实现\\
\\
586](https://developer.aliyun.com/article/273858)
[7\\
\\
二叉树面试题\\
\\
4](https://developer.aliyun.com/article/556017)
[8\\
\\
CSDN社区分享面试经历活动作品9——找工作之路\\
\\
669](https://developer.aliyun.com/article/639516)
[9\\
\\
【面试虐菜】—— Oracle知识整理《收获，不止Oracle》\\
\\
651](https://developer.aliyun.com/article/327187)
[10\\
\\
经典面试---spring IOC容器的核心实现原理\\
\\
6](https://developer.aliyun.com/article/1612732)

[1\\
\\
AI运维不再是玄学：教你用AI提前预测系统故障，少熬几次夜！\\
\\
764](https://developer.aliyun.com/article/1686299)
[2\\
\\
别再熬夜救火了！自动化运维正在重塑企业IT管理的底层逻辑\\
\\
283](https://developer.aliyun.com/article/1686261)
[3\\
\\
别再靠“拍脑袋”修系统了——聊聊大数据如何让运维更聪明\\
\\
675](https://developer.aliyun.com/article/1686208)
[4\\
\\
别再靠“救火”过日子了：智能运维，正在重塑IT服务的未来\\
\\
1014](https://developer.aliyun.com/article/1686092)
[5\\
\\
AI来了，运维不慌：教你用人工智能把团队管理提速三倍！\\
\\
848](https://developer.aliyun.com/article/1685932)
[6\\
\\
别让运维只会“救火”——用数据点燃业务增长的引擎\\
\\
271](https://developer.aliyun.com/article/1685799)
[7\\
\\
别再“亡羊补牢”了！——聊聊如何优化企业的IT运维监控架构\\
\\
295](https://developer.aliyun.com/article/1685679)
[8\\
\\
别再靠脚本“救火”了！让智能数据治理接管你的运维世界\\
\\
359](https://developer.aliyun.com/article/1685424)
[9\\
\\
当AI遇上自动化：运维测试终于不“加班”了\\
\\
894](https://developer.aliyun.com/article/1685318)
[10\\
\\
拔俗AI巡检系统：让设备“会说话”，让隐患“早发现”，打造更安全高效的智能运维\\
\\
959](https://developer.aliyun.com/article/1685269)

相关商品

## 相关课程

[更多](https://edu.aliyun.com/explore/)

[企业运维之弹性计算原理与实践](https://edu.aliyun.com/course/316534)
[企业运维之云上网络原理与实践课程](https://edu.aliyun.com/course/316264)
[企业级运维之云原生与Kubernetes实战课程](https://edu.aliyun.com/course/315907)
[面向运维的 python 脚本速成-1024程序员节创造营公益课](https://edu.aliyun.com/course/315821)
[Linux企业运维实战 - 入门及常用命令](https://edu.aliyun.com/course/314053)
[玩转云上智能运维](https://edu.aliyun.com/course/312826)

## 相关电子书

[更多](https://developer.aliyun.com/ebook/)

[企业运维之云原生和Kubernetes 实战](https://developer.aliyun.com/ebook/7660)
[可视化架构运维实践](https://developer.aliyun.com/ebook/7442)
[2021云上架构与运维峰会演讲合集](https://developer.aliyun.com/ebook/1)

## 相关实验场景

[更多](https://developer.aliyun.com/adc/)

[使用操作系统智能助手OS Copilot解锁操作系统运维与编程](https://developer.aliyun.com/adc/scenario/311000500000)

热门文章

最新文章

下一篇

[\[网络安全\] Dirsearch 工具的安装、使用详细教程](https://developer.aliyun.com/article/1395854)

### [大模型](https://www.aliyun.com/product/tongyi)

一站式为企业和开发者提供大模型能力体系，大模型原生应用以及最佳解决方案，助力云上开发者轻松完成 AI 落地。

[**AI 体验馆** \\
免费体验前沿 AI 应用和最新通义系列大模型，开启智能创新](https://www.aliyun.com/exp/)

[**大模型服务平台百炼** \\
为企业打造的大模型服务与应用开发平台](https://www.aliyun.com/product/bailian)

#### 大模型

[**Qwen3.6-Plus** \\
原生视觉语言模型，Vibe Coding体验显著提升](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/qwen3.6-plus) [**Qwen3.5-Plus** \\
原生视觉语言模型，代码生成和智能体双强](https://bailian.console.aliyun.com/cn-beijing/?tab=demohouse#/experience/llm) [**Qwen3.6-Max-preview** \\
全能旗舰预览版，编程与知识能力全面进阶](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/qwen3.6-max-preview?serviceSite=asia-pacific-china)

[**HappyHorse-1.0-T2V _NEW_** \\
文生视频，精准理解语义，细节丰富画质流畅](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/happyhorse-1.0-t2v?serviceSite=asia-pacific-china) [**HappyHorse-1.0-I2V _NEW_** \\
图生视频，流畅自然，细节丰富](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/happyhorse-1.0-i2v?serviceSite=asia-pacific-china) [**Qwen3-VL-Plus** \\
视觉 Coding、空间感知、多模态思考等全面升级](https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/vision?currentTab=imageComprehend&modelId=qwen3-vl-plus)

[**Wan2.7-Image** \\
全新图像生成与编辑模型](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/wan2.7-image) [**Wan2.7-Video** \\
全新视频生成模型，超强编辑能力](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/wan2.7-videoedit) [**Fun-ASR** \\
支持中英文自由切换，具备更强的噪声鲁棒性](https://bailian.console.aliyun.com/?tab=model#/efm/model_experience_center/voice?currentTab=voiceAsr&modelId=fun-asr-realtime)

[**Deepseek-v4-pro** \\
旗舰 MoE 大模型，百万上下文与顶尖推理能力](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/deepseek-v4-pro?serviceSite=asia-pacific-china) [**Kimi-k2.6** \\
全能进阶，多项权威基准测试行业领先](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/kimi%2Fkimi-k2.6?serviceSite=asia-pacific-china) [**MiniMax-M2.7** \\
自主构建复杂 Agent 架构，驾驭高难度生产力任务](https://bailian.console.aliyun.com/cn-beijing/?tab=model#/model-market/detail/MiniMax%2FMiniMax-M2.7?serviceSite=asia-pacific-china)

#### [大模型服务](https://bailian.console.aliyun.com/?tab=model\#/model-market)

[**大模型服务平台百炼-Token Plan** \\
面向企业和开发者打造的多模态 AI 订阅服务](https://bailian.console.aliyun.com/cn-beijing/?tab=plan#/efm/subscription/overview) [**大模型服务平台百炼-应用模版** \\
丰富多元化的应用模版和解决方案](https://bailian.console.aliyun.com/?tab=app#/app-market/newTemplate) [**大模型服务平台百炼-微调与部署** \\
一站式大模型服务平台，支持界面化微调与部署](https://bailian.console.aliyun.com/?tab=model#/efm/model_manager)

#### [AI 应用构建](https://bailian.console.aliyun.com/?tab=app\#/app-center)

[**大模型服务平台百炼 \- 模型体验** \\
在线体验全尺寸、多种模态的模型效果](https://bailian.console.aliyun.com/?tab=demohouse#/experience/llm) [**大模型服务平台百炼-智能体** \\
灵活可视化地构建企业级 Agent](https://bailian.console.aliyun.com/?tab=app#/app-center) [**人工智能平台 PAI** \\
AI Native 的算法工程平台，一站式完成建模、训练、推理服务部署](https://www.aliyun.com/product/bigdata/learn)

#### 大模型原生应用

[**Qoder _HOT_** \\
面向真实软件的智能体编程平台](https://www.aliyun.com/product/qoder) [**通义听悟** \\
智能会议助手，实时转写会议记录，支持搜索定位](https://tongyi.aliyun.com/tingwu) [**通义晓蜜** \\
智能客服平台，对话机器人、对话分析、智能外呼](https://tongyi.aliyun.com/xiaomi)

[**通义灵码** \\
智能编码助手，支持企业专属部署](https://www.aliyun.com/product/lingma) [**大模型服务平台百炼 \- 全妙** \\
多模态内容创作工具，已接入 DeepSeek](https://bailian.aliyun.com/quanmiao?from=bailian#/home) [**大模型服务平台百炼 \- 法睿** \\
法律智能助手，支持合同审查、法律咨询与检索、智能阅卷等](https://tongyi.aliyun.com/farui/home)

#### 大模型解决方案

[**快速部署 Dify，高效搭建 AI 应用** \\
依托云原生高可用架构,实现Dify私有化部署](https://www.aliyun.com/solution/tech-solution/rapidly-deploy-dify-to-accelerate-ai-application-development) [**10 分钟在聊天系统中增加一个 AI 助手** \\
在企业官网、通讯软件中为客户提供 AI 客服](https://www.aliyun.com/solution/tech-solution/build-a-chatbot-for-your-website-or-chat-system)

[**10分钟微调：让0.6B模型媲美235B模型** \\
用1%尺寸在特定领域达到大模型90%以上效果](https://www.aliyun.com/solution/tech-solution/qwen3-distill) [**即刻拥有 DeepSeek-R1 满血版** \\
多种方案随心选，轻松解锁专属 DeepSeek](https://www.aliyun.com/solution/tech-solution/deepseek-r1-for-platforms)

[**多模态数据信息提取** \\
从文本、图片、视频中提取结构化的属性信息](https://www.aliyun.com/solution/tech-solution/information-extraction) [**超强辅助，Bolt.diy 一步搞定创意建站** \\
通过自然语言交互简化开发流程,全栈开发支持](https://www.aliyun.com/solution/tech-solution/bolt-diy)

[**与 AI 智能体进行实时音视频通话** \\
构建支持视频理解的 AI 音视频实时通话应用](https://www.aliyun.com/solution/tech-solution/real-time-interaction) [**构建大模型应用的安全防护体系** \\
通过阿里云安全产品对 AI 应用进行安全防护](https://www.aliyun.com/solution/tech-solution/build-large-model-application-security-system)

### [产品](https://www.aliyun.com/product/list)

精选产品 [人工智能与机器学习](https://ai.aliyun.com/) [计算](https://www.aliyun.com/product/list/ecs) [容器](https://www.aliyun.com/product/aliware/containerservice) [存储](https://www.aliyun.com/storage/storage?spm=5176.19720258.J_2686872250.30.3f0e4ff6AwBQKs) [网络与CDN](https://www.aliyun.com/product/network/network) [安全](https://www.aliyun.com/product/list/security) [中间件](https://www.aliyun.com/product/list/aliware) [数据库](https://www.aliyun.com/product/outline/index?spm=5176.19720258.J_2686872250.45.3f0e4ff6AwBQKs) [大数据计算](https://www.aliyun.com/product/bigdata/apsarabigdata) 媒体服务 [企业服务与云通信](https://www.aliyun.com/product/list/ent-cmc) 域名与网站终端用户计算Serverless [开发工具](https://www.aliyun.com/product/list/developertools) [迁移与运维管理](https://www.aliyun.com/product/list/operation-mainenance) [专有云](https://apsara-stack.aliyun.com/)

#### 精选产品

[**大模型服务平台百炼 _大模型_** \\
大模型服务与应用平台](https://www.aliyun.com/product/bailian) [**云服务器 ECS** \\
安全可靠、弹性可伸缩的云计算服务](https://www.aliyun.com/product/ecs) [**云数据库 RDS** \\
全托管，含MySQL、PostgreSQL、SQL Server、MariaDB多引擎](https://www.aliyun.com/product/rds) [**人工智能平台 PAI _大模型_** \\
一站式AI开发、训练和推理服务](https://www.aliyun.com/product/bigdata/learn) [**大数据开发治理平台 DataWorks** \\
Data Agent 驱动的一站式 Data+AI 开发治理平台](https://www.aliyun.com/product/dide) [**云原生数据库 PolarDB** \\
100%兼容MySQL、PostgreSQL，兼容Oracle，支持集中和分布式](https://www.aliyun.com/product/polardb) [**智能商业分析 Quick BI** \\
AI时代的超级数据分析Agent](https://www.aliyun.com/product/quick-bi)

[**域名与网站** \\
提供智能易用的域名与建站服务](https://wanwang.aliyun.com/) [**千问大模型 _大模型_** \\
多元化、高性能、安全可靠的大模型服务](https://www.aliyun.com/product/tongyi) [**数字证书管理服务（原SSL证书）** \\
实现全站 HTTPS，呈现可信的 Web 访问](https://www.aliyun.com/product/cas) [**Qoder** \\
面向真实软件的智能体编程平台](https://www.aliyun.com/product/qoder) [**云解析 DNS** \\
覆盖公网/内网、递归/权威、移动APP等全场景解析服务](https://www.aliyun.com/product/dns) [**容器服务 Kubernetes 版 ACK** \\
提供一站式管理容器应用的 K8s 服务](https://www.aliyun.com/product/kubernetes)

[**轻量应用服务器** \\
快速构建应用程序和网站，即刻迈出上云第一步](https://www.aliyun.com/product/swas) [**对象存储 OSS** \\
稳定、安全、高性价比、高性能的云存储服务](https://www.aliyun.com/product/oss) [**无影云电脑** \\
随时随地安全接入的云上超级电脑](https://www.aliyun.com/product/ecs/gws) [**短信服务** \\
国内短信简单易用，安全可靠，秒级触达，全球覆盖200+国家和地区。](https://www.aliyun.com/product/sms) [**云原生大数据计算服务 MaxCompute** \\
面向分析的企业级SaaS模式云数据仓库](https://www.aliyun.com/product/maxcompute) [**函数计算 FC** \\
事件驱动的Serverless计算服务](https://www.aliyun.com/product/fc)

#### [产品动态](https://www.aliyun.com/product/news/)

[![云安全中心安全告警处置能力优化](https://img.alicdn.com/imgextra/i1/O1CN01XiavUz1DoV8MwPYh9_!!6000000000263-55-tps-36-36.svg)云安全中心安全告警处置能力优化](https://www.aliyun.com/product/news/28945) [![AI 安全护栏语音审核增强版新增语音审核大模型服务](https://img.alicdn.com/imgextra/i2/O1CN018RWjv71wAL79aPwaS_!!6000000006267-55-tps-36-36.svg)AI 安全护栏语音审核增强版新增语音审核大模型服务](https://www.aliyun.com/product/news/28944) [![负载均衡 ALB 扩展版开服上海/迪拜/北京地域](https://img.alicdn.com/imgextra/i4/O1CN01tqdTzW241C8DpMThk_!!6000000007330-55-tps-36-36.svg)负载均衡 ALB 扩展版开服上海/迪拜/北京地域](https://www.aliyun.com/product/news/28909) [![EMR Serverless StarRocks 支持 Compaction Service](https://img.alicdn.com/imgextra/i2/O1CN018RWjv71wAL79aPwaS_!!6000000006267-55-tps-36-36.svg)EMR Serverless StarRocks 支持 Compaction Service](https://www.aliyun.com/product/news/28920) [![百炼 Qwen3.5-Omni 实时 API 支持零样本音色克隆](https://img.alicdn.com/imgextra/i2/O1CN018RWjv71wAL79aPwaS_!!6000000006267-55-tps-36-36.svg)百炼 Qwen3.5-Omni 实时 API 支持零样本音色克隆](https://www.aliyun.com/product/news/28875) [![万镜一刻 HappyHorse1.0 同发登陆](https://img.alicdn.com/imgextra/i2/O1CN018RWjv71wAL79aPwaS_!!6000000006267-55-tps-36-36.svg)万镜一刻 HappyHorse1.0 同发登陆](https://www.aliyun.com/product/news/28925) [![百炼新增 Qwen-Image-2.0-Pro-2026-04-22 模型](https://img.alicdn.com/imgextra/i2/O1CN018RWjv71wAL79aPwaS_!!6000000006267-55-tps-36-36.svg)百炼新增 Qwen-Image-2.0-Pro-2026-04-22 模型](https://www.aliyun.com/product/news/28893)

### [解决方案](https://www.aliyun.com/solution/tech-solution/)

精选解决方案 [AI](https://www.aliyun.com/solution/tech-solution/ai) 互联网应用开发大数据现代化应用安全网络可观测上云与迁云 [企业出海](https://www.aliyun.com/goglobal) 政企业务

#### 精选解决方案

[**Hermes Agent，打造自进化智能体 _NEW_** \\
自主进化，持久记忆，越用越聪明](https://www.aliyun.com/solution/tech-solution/hermes-agent) [**极速搭建专属 SBTI 测评网站 _NEW_** \\
快来测试全新SBTI人格画像](https://www.aliyun.com/solution/tech-solution/sbti) [**深度研究：生成你的独家洞察报告** \\
智能生成洞察、分析等深度研究与决策报告](https://www.aliyun.com/solution/tech-solution/deep-research) [**10 分钟在聊天系统中增加一个 AI 助手** \\
在企业官网、通讯软件中为客户提供 AI 客服](https://www.aliyun.com/solution/tech-solution/build-a-chatbot-for-your-website-or-chat-system)

[**5 分钟轻松部署专属 QwenPaw _HOT_** \\
从聊天伙伴进化为能主动干活的本地数字员工](https://www.aliyun.com/solution/tech-solution/copaw) [**JVS Claw 免费获取，开启私享 AI 助理** \\
免除繁琐配置，快速拥有专属 JVS Claw](https://www.aliyun.com/solution/tech-solution/jvsclaw) [**AI 解题 + 批改：推动课程教学智变** \\
部署可拍照解题和作业批改的 AI 辅学应用](https://www.aliyun.com/solution/tech-solution/ai-homework-helper) [**高效搭建 AI 智能体与工作流应用** \\
通过阿里云百炼高效搭建AI应用,助力高效开发](https://www.aliyun.com/solution/tech-solution/build-ai-applications-based-on-alibaba-cloud-model-studio)

[**Claude Code + GStack 打造工程团队** \\
安装技能 GStack，拥有专属 AI 工程团队](https://www.aliyun.com/solution/tech-solution/gstack) [**效率翻倍，一句话生成专业 PPT** \\
输入一句话想法, 轻松生成专业的 PPT](https://www.aliyun.com/solution/tech-solution/vibe-ppt) [**快速拥有专属 OpenClaw _HOT_** \\
让AI从“聊天伙伴”进化为能干活的“数字员工”](https://www.aliyun.com/solution/tech-solution/clawdbot) [**低代码高效构建企业门户网站** \\
以可视化方式快速构建移动和 PC 门户网站](https://www.aliyun.com/solution/tech-solution/build-a-website)

[**基于 Spark 的分布式 AI 大模型智训方案** \\
Serverless 分布式 AI 训练平台,突破算力运维与成本瓶颈](https://www.aliyun.com/solution/tech-solution/spark-ai-train) [**自然语言编程：人人都能做 Web 开发** \\
自然语言驱动的沉浸式在线编程应用](https://www.aliyun.com/solution/tech-solution/web-vibe-coding) [**漫剧工坊：一站式动画创作平台** \\
快速生产连贯的高质量长漫剧](https://www.aliyun.com/solution/tech-solution/use-bailian-to-intelligently-create-comics) [**10 分钟搭建微信、支付宝小程序** \\
高效部署网站，快速应用到小程序](https://www.aliyun.com/solution/tech-solution/develop-your-wechat-mini-program-in-10-minutes)

### [权益](https://www.aliyun.com/benefit)

上云优选，普惠好价，为开发者和企业提供多款超值优选上云必备产品；超 140 款免费试用产品；初创企业最高可得 100 万抵扣金。

#### 普惠上云

[**普惠上云 官方力荐** \\
云服务器38元/年起，超值低价云产品抢先购](https://www.aliyun.com/benefit/select/cloud-discount) [**一键部署 OpenClaw _NEW_** \\
三步轻松构建 AI 助理，低至9.9元起](https://www.aliyun.com/benefit/scene/moltbot) [**官方推荐返现计划** \\
推荐新用户得奖励，单订单最高返9万](https://dashi.aliyun.com/)

#### 免费试用

[**解决方案免费试用 新老同享** \\
最高领取价值200元试用点，立即开启云上创新](https://www.aliyun.com/solution/free) [**AI 产品 免费试用** \\
7000+万大模型 tokens 和30+款产品免费体验](https://free.aliyun.com/product/ai) [**140+云产品 免费试用** \\
产品新客免费试用，最长12个月](https://free.aliyun.com/)

#### AI 特惠

[**智启 AI 普惠权益** \\
至高享 7000 万免费 tokens，加速 Al 应用落地](https://www.aliyun.com/benefit/ai/discount) [**阿里云百炼 Token Plan _HOT_** \\
多模型灵活切换，兼容主流 AI 工具，多档套餐选择](https://www.aliyun.com/benefit/scene/tokenplan) [**高性价比的 AI 算力, 快速部署大模型** \\
丰富多样的 GPU 算力和 PAI，构建 AI 应用](https://www.aliyun.com/benefit/scene/ainew)

#### AI 活动

[**飞天发布时刻** \\
所见，即是所愿](https://summit.aliyun.com/apsaramoment) [**AI 实训营** \\
从基础到进阶，Agent 创客手把手教你](https://www.aliyun.com/benefit/aihands-on/mainpage) [**有模有样：AI场景实践者说** \\
聚焦真实AI场景，对话模型最佳实践](https://www.aliyun.com/activity/ai-seminar/home)

#### AI 场景体验

[**AI 电商营销** \\
从图文生成到视频创作，一键激活电商全链路生产力](https://www.aliyun.com/benefit/aiuse/e-commerce) [**AI 广告创作** \\
图文、视频一站生成，高效打造优质广告素材](https://www.aliyun.com/benefit/aiuse/ad) [**AI 短剧/漫剧** \\
AI助力短剧漫剧创作，剧本、分镜、视频高效生成](https://www.aliyun.com/benefit/scene/playlet) [**AI Coding _HOT_** \\
智能编程，一键开启高性价比 AI 编程新体验](https://www.aliyun.com/benefit/scene/coding) [**AI 办公** \\
AI智能应用，一键激活高效办公新体验](https://www.aliyun.com/benefit/aiuse/office) [**智能客服** \\
自动承接线索、识别商机，让客服更高效、服务更出色。](https://www.aliyun.com/benefit/scene/callcenter)

#### 企业成长

[**企业上云第一站** \\
数字化转型从这里起步](https://www.aliyun.com/benefit/enterprise/start) [**大模型ACA认证体验** \\
助力企业全员 AI 认知与能力提升](https://edu.aliyun.com/learning/topic/llm-free-trial)

#### 创新加速

[**上云场景组合购** \\
覆盖90%+业务场景，专享组合折扣价](https://www.aliyun.com/benefit/client/package) [**老友焕新 权益中心** \\
100+款云产品超值低价](https://www.aliyun.com/benefit/client/index)

### [定价](https://www.aliyun.com/price)

提供灵活的计费方式和清晰的计费规则，满足不同的业务场景需求；支持自助估算价格、高效采购；专业的成本管理工具，持续优化云上成本。

[**产品定价** \\
了解云产品的定价详情](https://www.aliyun.com/price/detail) [**云上成本管理** \\
管理和优化成本](https://www.aliyun.com/price/cost-management)

[**价格计算器** \\
自助选配和估算价格](https://www.aliyun.com/price/product) [**价格优势** \\
推动算力普惠，释放技术红利](https://www.aliyun.com/price/advantage)

[**配置报价器** \\
一站式生成采购清单，支持单品或批量购买](https://www.aliyun.com/price/cpq/list)

[![5亿算力补贴](https://img.alicdn.com/imgextra/i2/O1CN01rarnNB1diFke7Rby7_!!6000000003769-0-tps-400-500.jpg)**5亿算力补贴** \\
\\
新迁上云，5亿补贴享不停](https://www.aliyun.com/benefit/scene/subsidy)

### [云市场](https://market.aliyun.com/)

提供与阿里云能力融合和互补的优质伙伴产品和服务，满足企业上云和各类业务应用开发需求。

#### [精选商城](https://market.aliyun.com/xinxuan)

[网站建设](https://market.aliyun.com/xinxuan/webdesign) [多端小程序](https://market.aliyun.com/xinxuan/application/miniapps) [Salesforce 国际版订阅](https://market.aliyun.com/products/56790007/cmfw00037956.html?innerSource=search_salesforce#sku=yuncode3195600001) [友盟天域](https://market.aliyun.com/products/56842011/cmfw00040027.html) [观测云](https://market.aliyun.com/products/56838014/cmgj00053362.html) [Tuya 物联网平台阿里云版](https://www.aliyun.com/research/tuya) [蓝凌 OA](https://market.aliyun.com/xinxuan/lanling-oa) [电子合同](https://market.aliyun.com/xinxuan/wyy-2023) [畅捷通](https://market.aliyun.com/products/56764034/cmgj00042861.html) [Tableau 订阅](https://market.aliyun.com/products/56024006/cmfw00062543.html) [AI空中课堂在线直播课堂（旗舰版）](https://market.aliyun.com/products/201204006/cmgj00070018.html)

#### [生态解决方案](https://market.aliyun.com/industry)

[行业生态解决方案](https://market.aliyun.com/industry) [开发者生态解决方案](https://market.aliyun.com/developer/shouye) [AI 开发和 AI 应用解决方案](https://market.aliyun.com/developer/AIGC)

#### [数据与 API](https://market.aliyun.com/data)

[数据集](https://market.aliyun.com/dataexchange) [手机三要素](https://market.aliyun.com/products?k=%E6%89%8B%E6%9C%BA%E4%B8%89%E8%A6%81%E7%B4%A0&scene=market) [身份实名认证](https://market.aliyun.com/products?k=%E8%BA%AB%E4%BB%BD%E5%AE%9E%E5%90%8D%E8%AE%A4%E8%AF%81&scene=market) [短信](https://market.aliyun.com/products?k=%E7%9F%AD%E4%BF%A1&scene=market) [OCR 文字识别](https://market.aliyun.com/products?k=OCR%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB&scene=market) [发票查验](https://market.aliyun.com/products?k=%E5%8F%91%E7%A5%A8%E6%9F%A5%E9%AA%8C&scene=market) [天气预报查询](https://market.aliyun.com/products?k=%E5%A4%A9%E6%B0%94%E9%A2%84%E6%8A%A5%E6%9F%A5%E8%AF%A2&scene=market) [快递物流查询](https://market.aliyun.com/products?k=%E5%BF%AB%E9%80%92%E7%89%A9%E6%B5%81%E6%9F%A5%E8%AF%A2&scene=market)

#### [企业应用](https://market.aliyun.com/enterprise)

[ERP](https://market.aliyun.com/products?k=ERP&scene=market) [CRM](https://market.aliyun.com/products?k=CRM&scene=market) [OA 办公系统](https://market.aliyun.com/products?k=OA%E5%8A%9E%E5%85%AC%E7%B3%BB%E7%BB%9F&scene=market) [财税管理](https://market.aliyun.com/products/56764034?page=1&scene=market) [400电话](https://market.aliyun.com/products?k=400%E7%94%B5%E8%AF%9D&scene=market) [广告营销](https://market.aliyun.com/products/56842011?page=1&scene=market)

#### [基础软件](https://market.aliyun.com/software)

[Windows](https://market.aliyun.com/products?k=Windows&scene=market) [宝塔 Linux](https://market.aliyun.com/products?k=%E5%AE%9D%E5%A1%94+Linux&scene=market) [CentOS](https://market.aliyun.com/products?k=CentOS&scene=market) [Docker](https://market.aliyun.com/products?k=Docker&scene=market) [JAVA](https://market.aliyun.com/products?k=JAVA&scene=market) [全能环境](https://market.aliyun.com/products?k=%E5%85%A8%E8%83%BD%E7%8E%AF%E5%A2%83&scene=market) [操作系统](https://market.aliyun.com/products?k=%E6%93%8D%E4%BD%9C%E7%B3%BB%E7%BB%9F&scene=market) [WordPress](https://market.aliyun.com/products?k=WordPress&scene=market) [Ubuntu](https://market.aliyun.com/products?k=Ubuntu&scene=market) [Red Hat](https://market.aliyun.com/products?k=Red+Hat&scene=market) [SUSE](https://market.aliyun.com/products?k=SUSE&scene=market)

#### [建站小程序](https://market.aliyun.com/jianzhan)

[模板建站](https://market.aliyun.com/products/56598032?page=1&scene=market) [定制建站](https://market.aliyun.com/products/52738005?page=1&scene=market) [模板小程序](https://market.aliyun.com/products/205798005?page=1&scene=market) [定制小程序](https://market.aliyun.com/products/52752001?page=1&scene=market) [APP 开发](https://market.aliyun.com/products/55514022?page=1&scene=market) [建站系统](https://market.aliyun.com/products/57342011?page=1&scene=market)

#### [专业服务](https://market.aliyun.com/service)

[域名](https://market.aliyun.com/products?k=%E5%9F%9F%E5%90%8D&scene=market) [商标](https://market.aliyun.com/products?k=%E5%95%86%E6%A0%87&scene=market) [备案](https://market.aliyun.com/products?k=%E5%A4%87%E6%A1%88&scene=market) [公司注册](https://market.aliyun.com/products?k=%E5%85%AC%E5%8F%B8%E6%B3%A8%E5%86%8C&scene=market) [上云迁移](https://market.aliyun.com/products/52738004?page=1&scene=market) [代维服务](https://market.aliyun.com/products/52732002?page=1&scene=market)

#### [安全](https://market.aliyun.com/security)

[VPN](https://market.aliyun.com/products?k=VPN&scene=market) [SSL 证书](https://market.aliyun.com/products?k=SSL%E8%AF%81%E4%B9%A6&scene=market) [堡垒机](https://market.aliyun.com/products?k=%E5%A0%A1%E5%9E%92%E6%9C%BA&scene=market) [防火墙](https://market.aliyun.com/products?k=%E9%98%B2%E7%81%AB%E5%A2%99&scene=market) [主机安全](https://market.aliyun.com/products?k=%E4%B8%BB%E6%9C%BA%E5%AE%89%E5%85%A8&scene=market)

#### [AI 应用及服务市场](https://market.aliyun.com/common/ai)

[AI 应用](https://market.aliyun.com/products?k=AI%E5%BA%94%E7%94%A8&scene=market) [大模型](https://market.aliyun.com/products?k=%E5%A4%A7%E6%A8%A1%E5%9E%8B&scene=market) [自然语言处理](https://market.aliyun.com/products?k=%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86&scene=market) [数据标注](https://market.aliyun.com/products/201198004?page=1&scene=market) [机器学习](https://market.aliyun.com/products?k=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0&scene=market)

### [伙伴](https://partner.aliyun.com/management/v2)

坚持伙伴优先，为伙伴提供产品、销售和服务的商业合作模式；与伙伴紧密合作，共同为客户提供更完备的产品、更完善的服务。

#### 成为销售伙伴

[分销伙伴](https://partner.aliyun.com/programs/reseller_P) [通用与行业 ISV 伙伴](https://partner.aliyun.com/program/ISV_partner) [咨询伙伴](https://partner.aliyun.com/program/consult_partner)

#### 销售伙伴合作计划

[无影生态合作计划](https://partner.aliyun.com/management/epp_wuying) [Salesforce On Alibaba Cloud Consulting Partner 合作计划](https://partner.aliyun.com/program/Salesforce_program) [AI 大模型销售与服务生态合作计划](https://partner.aliyun.com/program/aimsp)

#### 成为产品伙伴

[产品生态集成认证中心](https://partner.aliyun.com/program/PEIC) [产品生态伙伴](https://chanpinshengtai.aliyun.com/partner/partner) [产品生态伙伴工作台](https://aps.aliyun.com/#/)

#### 产品伙伴合作计划

[阿里云 AI 伙伴计划（繁花）](https://chanpinshengtai.aliyun.com/partner/ai) [弹性计算合作计划](https://partner.aliyun.com/program/txjs_program) [云存储合作计划](https://partner.aliyun.com/program/cunchu) [数据库合作计划](https://partner.aliyun.com/program/sjk_program) [云网络合作计划](https://chanpinshengtai.aliyun.com/chanpinpartner/network) [Salesforce On Alibaba Cloud ISV 合作计划](https://partner.aliyun.com/program/Salesforce-ISV)

#### [成为服务伙伴](https://gts.aliyun.com/)

[服务生态伙伴](https://gts.aliyun.com/)

#### 服务伙伴合作计划

[伙伴信用分合作计划](https://www.aliyun.com/gts/msp/Creditscoresystem)

#### 更多支持

[合作伙伴培训与认证](https://edu.aliyun.com/certification/partner) [查询合作伙伴](https://partner.aliyun.com/management/query#/) [登录合作伙伴管理后台](https://account.aliyun.com/login/qr_login.htm?oauth_callback=https://partner.aliyun.com/management/v2)

### [服务](https://www.aliyun.com/service)

提供多样化的支持计划和专家服务，满足上云咨询、迁移上云、云上运维等场景的全链路服务需求。

#### 售前咨询

[在线服务](https://smartservice.console.aliyun.com/pre-sale/chat?entrance=201&referrer=https%3A%2F%2Fwww.aliyun.com%2F%3Fspm%3Da1z2e.12184483.navigationzhcn.dnavigationzhcn6.469d3247dm8YgK)

#### 售后服务

[自助服务](https://smartservice.console.aliyun.com/tourist-self/self-service-center?from=webnav) [在线服务](https://smartservice.console.aliyun.com/service/robot-chat) [工单服务](https://smartservice.console.aliyun.com/service/create-ticket) [短信专区](https://smartservice.console.aliyun.com/tourist-self/self-service-center/topic?topicCode=dysms&from=webnav)

#### 企业增值服务

[企业支持计划](https://www.aliyun.com/service/supportplans) [专家技术服务](https://www.aliyun.com/service/list) [企业增值服务台](https://custservice.console.aliyun.com/value-added/home)

#### 企业成长

[服务实践](https://www.aliyun.com/service/customer-case) [创新中心](https://chuangke.aliyun.com/)

#### [阿里云认证](https://edu.aliyun.com/)

[大模型认证](https://edu.aliyun.com/certification/llm) [全部认证](https://edu.aliyun.com/certification/) [训练营](https://edu.aliyun.com/trainingcamp)

#### 信息公告

[官网公告](https://www.aliyun.com/notice/) [健康状态](https://status.aliyun.com/)

#### [开发者社区](https://developer.aliyun.com/)

[博文](https://developer.aliyun.com/indexFeed/) [问答](https://developer.aliyun.com/ask/) [电子书](https://developer.aliyun.com/ebook/) [镜像站](https://developer.aliyun.com/mirror/)

#### 我要反馈

[我要建议](https://www.aliyun.com/connect/home) [我要投诉](https://www.aliyun.com/complaint)

### [了解阿里云](https://www.aliyun.com/about)

作为全球领先的全栈人工智能服务商，阿里云坚持让计算成为公共服务，助力全球客户加速价值创新。

#### [为什么选择阿里云](https://www.aliyun.com/why-us)

[什么是云计算](https://www.aliyun.com/about/what-is-cloud-computing) [技术领先](https://www.aliyun.com/why-us/leading-technology) [稳定可靠](https://www.aliyun.com/why-us/reliability) [安全合规](https://www.aliyun.com/why-us/security-compliance) [分析师报告](https://www.aliyun.com/analyst-reports) [研究报告与白皮书](https://www.aliyun.com/reports)

#### [天池大赛](https://tianchi.aliyun.com/)

[AI 算法大赛](https://tianchi.aliyun.com/competition/algorithmList/) [云开发大赛](https://tianchi.aliyun.com/competition/programList) [入门学习赛](https://tianchi.aliyun.com/competition/coupleList)

#### 最佳实践

[云上春晚](https://www.aliyun.com/about/gala) [云上奥运之旅](https://www.aliyun.com/about/games) [云栖战略参考](https://www.aliyun.com/about/magazines) [云上的中国](https://www.aliyun.com/about/ysdzg) [看见新力量](https://startup.aliyun.com/special/seenewpower) [金融模力时刻](https://summit.aliyun.com/market/financial-agent) [客户案例](https://www.aliyun.com/customer-stories/customer-case-index)

#### 市场活动

[2026 云上安全健康体检](https://summit.aliyun.com/health-check) [阿里云中企出海大会](https://summit.aliyun.com/go-global) [云栖大会](https://yunqi.aliyun.com/) [活动全景](https://www.aliyun.com/about/events)

#### 魔搭 ModelScope

[魔搭 ModelScope](https://modelscope.cn/home)

#### 高校合作

[云工开物](https://university.aliyun.com/) [科研合作](https://university.aliyun.com/activity/air)

#### [加入我们](https://careers.aliyun.com/)

[Careers](https://careers.aliyun.com/en/home) [社会招聘](https://careers.aliyun.com/off-campus/home) [校园招聘](https://careers.aliyun.com/campus/home)

### 中国站 \| aliyun.com

[简体中文](https://www.aliyun.com/)

### 国际站 \| alibabacloud.com

[English](https://www.alibabacloud.com/en?_p_lc=1) [简体中文](https://www.alibabacloud.com/zh?_p_lc=1) [繁體中文](https://www.alibabacloud.com/tc?_p_lc=2) [日本語](https://www.alibabacloud.com/ja?_p_lc=1) [한국어](https://www.alibabacloud.com/ko?_p_lc=1) [Deutsch](https://www.alibabacloud.com/de?_p_lc=1) [Français](https://www.alibabacloud.com/fr?_p_lc=1) [Bahasa Indonesia](https://www.alibabacloud.com/id?_p_lc=9) [ไทย](https://www.alibabacloud.com/th?_p_lc=14) [Español](https://www.alibabacloud.com/es?_p_lc=17)

### 联系我们

4008013260 [售前咨询](https://smartservice.console.aliyun.com/pre-sale/chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363607) [售后在线](https://smartservice.console.aliyun.com/service/robot-chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363607)

### 其他服务

[我要建议](https://www.aliyun.com/connect/home) [我要投诉](https://www.aliyun.com/complaint)

![登录插画](https://img.alicdn.com/imgextra/i2/O1CN015QIT9m1FmmyUntYlQ_!!6000000000530-2-tps-320-200.png)

登录以查看您的控制台资源

管理云资源

状态一览

快捷访问

[快捷注册](https://account.aliyun.com/register/qr_register.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363607) [登录阿里云](https://account.aliyun.com/login/login.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363607)