# Linux运维工程师50个常见面试题原创 - CSDN博客

URL: https://blog.csdn.net/yy17111342926/article/details/149415196

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

# Linux运维工程师 50个常见面试题

最新推荐文章于 2025-09-19 11:56:37 发布

原创于 2025-07-17 10:08:28 发布·689 阅读

·![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png)
26


·![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png)
13
·

CC 4.0 BY-SA版权

版权声明：本文为博主原创文章，遵循 [CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/) 版权协议，转载请附上原文出处链接和本声明。


文章标签：

[#运维](https://so.csdn.net/so/search/s.do?q=%E8%BF%90%E7%BB%B4&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#linux](https://so.csdn.net/so/search/s.do?q=linux&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#java](https://so.csdn.net/so/search/s.do?q=java&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#信息安全](https://so.csdn.net/so/search/s.do?q=%E4%BF%A1%E6%81%AF%E5%AE%89%E5%85%A8&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#网络安全](https://so.csdn.net/so/search/s.do?q=%E7%BD%91%E7%BB%9C%E5%AE%89%E5%85%A8&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#计算机](https://so.csdn.net/so/search/s.do?q=%E8%AE%A1%E7%AE%97%E6%9C%BA&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art)

## Kubernetes 生产环境最佳实践

众所周知，Kubernetes很难! 以下是在生产中使用它应遵循的一些最佳实践。遵循这些步骤能够确保更高的安全性和生产效率。

毫无疑问，DevOps已经走过了一段很长的路! 借助于Kubernetes编排平台使得公司比以往更快地发布软件。随着容器用于构建和发布软件的使用量不断增加，Kubernetes已经成为事实上的容器编排工具标准，在软件企业中非常受欢迎。

Kubernetes具有优秀的特性，比如：支持可扩展、零停机部署、服务发现、自动重启和回滚功能等。要大规模管理容器部署，Kubernetes是必须的。它支持灵活地分配资源和工作负载。毫无疑问，生产环境中的Kubernetes是一个很好的解决方案，但需要花费一些时间来设置和熟悉这个工具。由于现在许多公司都希望在生产中使用Kubernetes，因此有必要考虑一些最佳实践。在本文中，我们将讨论一些Kubernetes的最佳实践。

### 生产环境中的Kubernetes

Kubernetes是一个复杂并且学习曲线陡峭的编排工具，但它具有丰富的功能。生产操作应尽可能小心谨慎处理。如果您面临内部人才短缺的问题，您可以将其外包给PaaS供应商，为您提供所有最佳实践。但假设您在生产中独自管理Kubernetes。在这种情况下，关注最佳实践是非常重要的，特别是关于可观察性、日志记录、集群监控和安全配置。

我们很多人都知道，在生产环境中运行容器不是一件容易的事情。它需要大量的工作和计算资源等等。市场上有许多编排平台，但Kubernetes已经获得了巨大的吸引力和大多数云提供商的支持。

总之——Kubernetes、集装箱化和微服务都是美好的基础设施，但同时带来了安全挑战。Kubernetes
Pod
可以在所有基础设施类之间快速切换，从而导致Pod之间的内部流量增加，引发安全隐患。此外，Kubernetes的攻击面通常更大。您必须考虑到Kubernetes的高度动态且全新的环境无法与旧版安全工具完美融合的问题。

Gartner预测，到2022年，超过75%的全球组织将在生产中运行集装箱应用程序，而目前这一比例还不到30%。到2025年，超过85%的全球组织将在生产中推动集装箱应用，较2019年的不到35%有显著增长。本地云应用程序需要高度的基础设施自动化、DevOps和专门的操作技能，这些在普通IT组织中很难找到这些技能。

所以必须使用Kubernetes的一些
策略
，在安全性、监控、网络、治理、存储、容器生命周期管理和平台选择方面应用最佳实践。下面让我们来看看Kubernetes的一些生产最佳实践。

在生产中运行Kubernetes并不容易; 有以下几个方面需要注意。

### 是否使用存活探针和就绪探针进行健康检查？

管理大型分布式系统可能会很复杂，特别是当出现问题时，我们无法及时得到通知。为了确保应用实例正常工作，设置Kubernetes健康检查至关重要。

通过创建自定义运行健康检查，可以有效避免分布式系统中僵尸服务运行，具体可以根据环境和需要对其进行调整。

![img](https://i-blog.csdnimg.cn/img_convert/242d593d6392230dcb473e527933fbba.png)

**Readiness-就绪探针**

就绪探针的目的是让Kubernetes知道该应用是否已经准备好为流量服务。Kubernetes将始终确保准备就绪探针通过之后开始分配服务，将流量发送到Pod。

#### Liveness-存活探针

你怎么知道你的应用程序是活的还是死的?存活探针可以让你做到这一点。如果你的应用死了，Kubernetes会移除旧的Pod并用新Pod替换它。

### Resource Management- 资源管理

为单个容器指定资源请求和限制是一个很好的实践。

![img](https://i-blog.csdnimg.cn/img_convert/6a73170cb77cb039029f06b3e791ba38.png)

另一个好的实践是将Kubernetes环境划分为不同团队、部门、应用程序和客户机的独立名称空间。

#### Kubernetes资源使用情况

Kubernetes资源使用指的是容器/pod在生产中所使用的资源数量。

因此，密切关注pods的资源使用情况是非常重要的。一个明显的原因是成本，因为越高的资源利用证明越少的资源浪费。

#### Resource utilization资源利用率

Ops团队通常希望优化和最大化pods消耗的资源百分比。资源使用情况是Kubernetes环境实际优化程度的指标之一。

您可以认为优化后的Kubernetes环境中运行的容器的平均CPU等资源利用率是最优的。

### 启用RBAC

RBAC代表基于角色的访问控制。它是一种用于限制系统/网络上的用户和应用程序的访问和准入的方法。

![img](https://i-blog.csdnimg.cn/img_convert/a9a52979b48d1f990c68968cff117d27.png)

他们从Kubernetes 1.8版本引入了RBAC。使用rbac.authorization.k8s RBAC用于创建授权策略。

在Kubernetes中，RBAC用于授权，使用RBAC，您将能够授予用户、帐户、添加/删除权限、设置规则等权限。因此，它基本上为Kubernetes集群添加了额外的安全层。RBAC限制谁可以访问您的生产环境和集群。

### 集群置备和  负载均衡

生产级Kubernetes基础设施通常需要考虑某些关键方面，例如高可用性、多主机、多etcd Kubernetes集群等。此类集群的配置通常涉及到Terraform或Ansible等工具。

![img](https://i-blog.csdnimg.cn/img_convert/f0dd316ead1ed18742db8e7f70e377bb.png)

一旦集群都设置好了，并且为运行应用程序创建了pods，这些pods就配备了负载平衡器;这些负载均衡器将流量路由到服务。开源的Kubernetes项目并不是默认的负载平衡器;因此，它需要与NGINX
Ingress  controller与HAProxy或ELB等工具集成，或任何其他工具，扩大Kubernetes的Ingress插件，以提供负载均衡能力。

### 给Kubernetes对象添加标签

![img](https://i-blog.csdnimg.cn/img_convert/3eef66ab97296a861ad855f2dbb3b850.png)

标签就像附加到对象上的键/值对，比如pods。标签是用来标识对象的属性的，这些属性对用户来说是重要的和有意义的。在生产中使用Kubernetes时，不能忽视的一个重要问题是标签;标签允许批量查询和操作Kubernetes对象。标签的特殊之处在于，它们还可以用于识别Kubernetes对象并将其组织成组。这样做的最佳用例之一是根据pod所属的应用程序对它们进行分组。在这里，团队可以构建并拥有任意数量的标签约定。

### 配置网络策略

使用Kubernetes时，设置网络策略至关重要。

![img](https://i-blog.csdnimg.cn/img_convert/e47de1aa144c7adc7fcd26f5c4f29b3c.png)

网络策略只不过是一个对象，它使你能够明确地声明和决定哪些流量是允许的，哪些是不允许的。这样，Kubernetes将能够阻止所有其他不想要的和不符合规则的流量。在我们的集群中定义和限制网络流量是强烈推荐的基本且必要的安全措施之一。

Kubernetes中的每个网络策略都定义了一个如上所述的授权连接列表。无论何时创建任何网络策略，它所引用的所有pod都有资格建立或接受列出的连接。简单地说，网络策略基本上就是授权和允许连接的白名单——一个连接，无论它是`到`还是`从`pod，只有在应用于pod的至少一个网络策略允许的情况下才被允许。

### 集群监控和日志记录

在使用Kubernetes时，监控部署是至关重要的。确保配置、性能和流量保持安全更是重要。如果不进行日志记录和监控，就不可能诊断出发生的问题。为了确保合规性，监视和日志记录变得非常重要。

![img](https://i-blog.csdnimg.cn/img_convert/a4d887fc04e74e19eee255dbbc5b8e8e.png)

在进行监视时，有必要在体系结构的每一层上设置日志记录功能。生成的日志将帮助我们启用安全工具、审计功能和分析性能。

### 从无状态应用程序开始

运行无状态应用要比运行有状态应用简单得多，但随着Kubernetes运营商的不断增长，这种想法正在改变。对于刚接触Kubernetes的团队来说，建议首先使用无状态应用程序。

建议使用无状态后端，这样开发团队就可以确保不存在长时间运行的连接，从而增加了扩展的难度。使用无状态，开发人员还可以更有效地、零停机部署应用程序。

人们普遍认为，无状态应用程序可以方便地根据业务需要进行迁移和扩展。

### 启动自动扩缩容

Kubernetes有三种用于部署的自动伸缩功能:水平pod自动伸缩(HPA)、垂直pod自动伸缩(VPA)和集群自动伸缩。

水平pod autoscaler根据感知到的CPU利用率自动扩展deployment、replicationcontroller, replicaset, statefulset的数量。

Vertical pod autoscaling为CPU和内存请求和限制推荐合适的值，它可以自动更新这些值。

Cluster Autoscaler扩展和缩小工作节点池的大小。它根据当前的利用率调整Kubernetes集群的大小。

### 控制  镜像 拉取来源

控制在集群中运行所有容器的镜像源。如果您允许您的Pod从公共资源中拉取镜像，您就不知道其中真正运行的是什么。

如果从受信任的注册表中提取它们，则可以在注册表上应用策略以提取安全和经过认证的镜像。

### 持续学习

不断评估应用程序的状态和设置，以学习和改进。例如，回顾容器的历史内存使用情况可以得出这样的结论:我们可以分配更少的内存，在长期内节省成本。

### 保护重要服务

使用Pod优先级，您可以决定设置不同服务运行的重要性。例如，为了更好的稳定性，你需要确保RabbitMQ pod比你的应用pod更重要。或者你的入口控制器pods比数据处理pods更重要，以保持服务对用户可用。

### 零停机时间

通过在HA中运行所有服务，支持集群和服务的零停机升级。这也将保证您的客户获得更高的可用性。

使用pod反亲和性来确保在不同的节点上调度一个pod的多个副本，从而通过计划中的和计划外的集群节点停机来确保服务可用性。

使用pod Disruptions策略，不惜一切代价确保您有最低的Pod副本数量!

### 计划失败

硬件最终会失败，软件最终会运行。–（迈克尔·哈顿）

### 结论

众所周知，Kubernetes实际上已经成为DevOps领域的编排平台标准。Kubernetes从可用性、可伸缩性、安全性、弹性、资源管理和监控的角度来应对生产环境产生的风暴。由于许多公司都在生产中使用Kubernetes，因此必须遵循上面提到的最佳实践，以顺利和可靠地扩展应用程序。

_来源：https://my.oschina.net/u/1787735/blog/4870582_

```
- END -分享一个k8s大礼包，包含：k8s思维导图+k8s面试题合集+k8s学习文档笔记+k8s实战案例+k8s中文文档+k8s实战指南等等，想学习和提升k8s的同学，这份资料千万不可错过！大礼包部分内容如下：01k8s中文文档
文档主要内容：Kubernetes是什么?互动教程Mi nikube介绍设计文档概念Kubect1 CLIKubectl命令表安装设置API使用集群管理TASKS
02k8s思维导图
03k8s实战案例
04k8s高频面试题
128道k8s高频面试题：
简述Kubernetes kubelet的作用?简述Kubernetes kubelet监控Worker节点资源是使用什么组件来实现的?简述Kubernetes如何保证集群的安全性?简述Kubernetes准入机制?简述Kubernetes RBAC及其特点（优势）?44、简述Kubernetes Secret作用?简述Kubernetes Secret有哪些使用方式?简述Kubernetes PodSecurityPolicy机制?简述Kubernetes PodSecurityPolicy机制能实现哪些安全策略?简述Kubernetes网络模型?简述Kubernetes CNI模型?简述Kubernetes网络策略?简述Kubernetes网络策略原理简述Kubernetes数据持久化的方式有哪些?56、简述Kubernetes PV和PVC?简述Kubernetes PV生命周期内的阶段?58、简述Kubernetes所支持的存储供应模式?......05阿里云k8s实战案例
文档主要内容：这么理解集群控制器，能行!集群网络详解.集群伸缩原理认证与调度集群服务的三个要点和一种实现镜像拉取这件小事读懂这一篇，集群节点不下线节点下线姊妹篇我们为什么会删除不了集群的命名空间?阿里云ACK产品安全组配置管理二分之一活的微服务半夜两点Ca证书过期问题处理惨况总结
所有学习资料已打包
网络安全学习资源分享:
-----------------------------------------------------------------------------------------------------

**给大家分享一份全套的网络安全学习资料，给那些想学习 网络安全的小伙伴们一点帮助！**

对于从来没有接触过网络安全的同学，我们帮你准备了详细的学习成长路线图。可以说是最科学最系统的学习路线，大家跟着这个大的方向学习准没问题。

因篇幅有限，仅展示部分资料，朋友们如果有需要<mark>全套《**网络安全入门+进阶学习资源包**》</mark>，需要<mark>点击下方链接</mark>即可前往获取

<code>**读者福利 |** </code> **[CSDN大礼包：《网络安全入门&进阶学习资源包》免费分享 ](https://mp.weixin.qq.com/s/QWVo1i9TGDnzoli_KM4Jnw?poc_token=HCZiU2ijBF_dn7QHBnpE_v59shmKv4EF160li9CS)**  <code>**（安全链接，放心点击）**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ca94d7aeb73e4a47842fa8af60cef20c.jpeg#pic_center)

### 👉1.成长路线图&学习规划👈

要学习一门新的技术，作为新手一定要**先学习成长路线图，方向不对，努力白费。**

对于从来没有接触过网络安全的同学，我们帮你准备了详细的学习成长路线图&学习规划。可以说是最科学最系统的学习路线，大家跟着这个大的方向学习准没问题。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8b31b4bbd304070affa9c726d2e20f8.png#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c6e4885811d84247a93d6bcf240fd2c0.webp)

### 👉2.网安入门到进阶视频教程👈

很多朋友都不喜欢**晦涩的文字**，我也为大家准备了视频教程，其中一共有**21个章节**，每个章节都是**当前板块的精华浓缩。**<font color="red">**（全套教程文末领取哈）**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/356c31e97b9448ac9a030276befeb8ce.webp)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f86928c4051649c380cab368ad9f54df.webp)

### 👉3.SRC&黑客文档👈

大家最喜欢也是最关心的**SRC技术文籍&黑客技术**也有收录

**SRC技术文籍：**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/24339152d1874ccfb395f44bdecc2d83.webp)

**黑客资料由于是敏感资源，这里不能直接展示哦！**<font color="red">**（全套教程文末领取哈）**
### 👉4.护网行动资料👈
其中关于**HW护网行动，也准备了对应的资料，这些内容可相当于比赛的金手指！**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9124e6befe844cf39ade6c832e5cea63.webp)

### 👉5.黑客必读书单👈

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c8c8c5c516344f2d8177b503d2433c94.webp)

### 👉6.网络安全岗面试题合集👈

当你自学到这里，你就要开始**思考找工作**的事情了，而工作绕不开的就是**真题和面试题。**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c951f8196954fff816621fdb7557dd8.webp)
**所有资料共282G，朋友们如果有需要全套《网络安全入门+进阶学习资源包》，可以扫描下方二维码或链接免费领取~**

<code>**读者福利 |** </code> **[CSDN大礼包：《网络安全入门&进阶学习资源包》免费分享 ](https://mp.weixin.qq.com/s/QWVo1i9TGDnzoli_KM4Jnw?poc_token=HCZiU2ijBF_dn7QHBnpE_v59shmKv4EF160li9CS)**  <code>**（安全链接，放心点击）**

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/42d7664fd3b64d5aa1fc36145f0a6ee6.jpeg#pic_center)

AI编程工具

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
```

![](https://img-blog.csdnimg.cn/21ca5f10f03f440d918555abd508b5e7.png)2024最新网络安全技术资料领取

![](https://g.csdnimg.cn/extension-box/2.0.2/image/weixin.png)微信名片

![](https://g.csdnimg.cn/extension-box/2.0.2/image/ic_move.png)

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png)

确定要放弃本次机会？


福利倒计时

_:_ _:_

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png)立减 ¥

普通VIP年卡可用

[立即使用](https://mall.csdn.net/vip)

[![](https://profile-avatar.csdnimg.cn/fd1d678a361d4ca1a68e1e37902d3e63_yy17111342926.jpg!1)\\
程序员柚柚](https://blog.csdn.net/yy17111342926)

关注关注

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png)
26

点赞

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png)
踩

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png)
13




收藏







觉得还不错?

一键收藏
![](https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png)

- [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png)\\
0](https://blog.csdn.net/yy17111342926/article/details/149415196#commentBox)
评论

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/share.png)分享




复制链接



分享到 QQ



分享到新浪微博









![](https://csdnimg.cn/release/blogv2/dist/pc/img/share/icon-wechat.png)扫一扫


- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/more.png)


![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报



![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报


[Kubernetes生产环境最佳实践17条](https://blog.csdn.net/m0_37723088/article/details/130324953)

[力哥讲技术](https://blog.csdn.net/m0_37723088)

04-23![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
760


[Kubernetes是一种流行的容器编排工具，但在生产环境中部署和管理Kubernetes可能会面临一些挑战。为了确保生产环境的高可用性、可伸缩性和安全性，需要遵循一些最佳实践。这些实践包括使用合适的硬件和网络基础设施、使用可靠的存储和备份策略、进行监控和日志记录、使用适当的安全措施以及实施自动化和持续集成/持续交付流程。遵循这些最佳实践将有助于确保Kubernetes在生产环境中的稳定性和可靠性。](https://blog.csdn.net/m0_37723088/article/details/130324953)

[1.3kubernetes核心架构\_kubernetes生产化实践之路](https://blog.csdn.net/u012271526/article/details/121194250)

[草办的学习记录](https://blog.csdn.net/u012271526)

12-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
701


[1.3kubernetes核心架构\_kubernetes生产化实践之路](https://blog.csdn.net/u012271526/article/details/121194250)

参与评论您还未登录，请先登录后发表或查看评论

[Kubernetes生产环境最佳实践](https://aillm.blog.csdn.net/article/details/112001207)

[u012516914的专栏](https://blog.csdn.net/u012516914)

12-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
527


[众所周知，Kubernetes很难! 以下是在生产中使用它应遵循的一些最佳实践。遵循这些步骤能够确保更高的安全性和生产效率。毫无疑问，DevOps已经走过了一段很长的路! 借助于Kube...](https://aillm.blog.csdn.net/article/details/112001207)

[Kubernetes生产化实践之路学习笔记](https://blog.csdn.net/niwoxiangyu/article/details/142847292)

[niwoxiangyu的博客](https://blog.csdn.net/niwoxiangyu)

10-16![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
425


[5.1.2 Kubernetes 多租户有限支持。6.3.1 理解 _Linux_ 网络协议栈工作机制。2.6.5 网络性能第3章 构建高可用集群。5.1.3 Kubernetes 租户扩展。6.5 Kubernetes 中的服务发布。3.2 Kubernetes 高可用层级。1.2 Kubernetes 模型设计。1.3 Kubernetes 核心架构。5.2.1 Kubernetes 认证。5.3.1 Kubernetes 授权。5.5.1 Kubernetes 配额。6.5.4 Service 高级特性。](https://blog.csdn.net/niwoxiangyu/article/details/142847292)

[1.1云计算的变革\_kubernetes生产化实践之路](https://blog.csdn.net/u012271526/article/details/120953593)

[草办的学习记录](https://blog.csdn.net/u012271526)

10-26![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
565


[云原生：\\
将应用程序构建为微服务，并将其运行在完全动态地利用云计算模型优势的容器编排平台上的方法。云原生主要关注的是如何创建和部署程序，而不是运行在哪里。\\
设计云原生应用可以从如下考虑：\\
\\
松耦合的微服务：按照业务功能设计成微服务，之间通过轻量级协议通信（例如HTTP）\\
无状态且可规模化部署\\
故障的容忍性和弹性：跨数据中心、跨云厂商部署时，单个服务部分实例异常不影响整体服务的质量\\
\\
1.1、云计算的变革\\
物理机–虚拟机–容器化\\
\\
1.1.2、虚拟化时代\\
\\
1.基础架构即服务（Infrastructure a](https://blog.csdn.net/u012271526/article/details/120953593)

[_Linux_ _运维_ _工程师_ _面试题_ 全面汇总（2023）\_ _linux_ _运维_ 面试](https://blog.csdn.net/2401_89872656/article/details/145085959)

[2401\_89872656的博客](https://blog.csdn.net/2401_89872656)

01-12![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1628


[①PV - 物理卷：物理卷在逻辑卷管理中处于最底层，它可以是实际物理硬盘上的分区，也可以是整个物理硬盘，也可以是raid设备。②VG - 卷组：卷组建立在物理卷之上，一个卷组中至少要包括一个物理卷，在卷组建立之后可动态添加物理卷到卷组中。一个逻辑卷管理系统工程中可以只有一个卷组，也可以拥有多个卷组。③LV - 逻辑卷：逻辑卷建立在卷组之上，卷组中的未分配空间可以用于建立新的逻辑卷，逻辑卷建立后可以动态地扩展和缩小空间。系统中的多个逻辑卷可以属于同一个卷组，也可以属于不同的多个卷组。①添加磁盘。](https://blog.csdn.net/2401_89872656/article/details/145085959)

[最新 _Linux_ 系统 _运维_ _面试题_（共四十页附答案）](https://download.csdn.net/download/ydk888888/10715285)

10-12

[\#### 十、 _运维_ _工程师_ 的角色与职责 \*\*知识点12： _运维_ _工程师_ 的工作内容\\*\\* \- \\*\\*职责\\*\\*: \- 系统部署与维护。 \- 故障排查与解决。 \- 性能优化。 \- 安全保障。 \- \\*\\*核心能力\\*\\*: \- 技术能力: 精通各种操作系统和网络...](https://download.csdn.net/download/ydk888888/10715285)

[_运维_ _工程师_，总结40个 _面试题_](https://aaaedu.blog.csdn.net/article/details/138185715)

[张晨光老师的播客](https://blog.csdn.net/zhangchen124)

04-25![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2308


[下面是一名 _运维_ 人员求职数十家公司总结的 _Linux_ _运维_ _面试题_，给大家参考下~1、什么是 _运维_？什么是游戏 _运维_？1） _运维_ 是指大型组织已经建立好的网络软硬件的维护，就是要保证业务的上线与运作的正常，在他运转的过程中，对他进行维护，他集合了网络、系统、数据库、开发、安全、监控于一身的技术 _运维_ 又包括很多种，有DBA _运维_、网站 _运维_、虚拟化 _运维_、监控 _运维_、游戏 _运维_ 等等2）游戏 _运维_ 又有分工，分为开发 _运维_、应用 _运维_（业务 _运维_）和系统 _运维_ 开发 _运维_：是给应用 _运维_ 开发 _运维_ 工具和 _运维_ 平台的。](https://aaaedu.blog.csdn.net/article/details/138185715)

[_运维_ _工程师_ _面试题_](https://blog.csdn.net/qq_41946216/article/details/143307505)

[qq\_41946216的博客](https://blog.csdn.net/qq_41946216)

10-28![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2072


[36、已知 apache 服务的访问日志按天记录在服务器本地目录/app/logs下，由于磁盘空间紧张现在要求只能保留最近7天的访问日志!请给出解决办法或配置或处理命令?22、使用tcpdump监听主机为192.168.1.1，tcp端口为8 0的数据，同时将输出结果保存输出到tcpdump.log。35、写一个脚本，实现判断192.168.1.0/24网络里，当前在线的IP有哪些，能ping通则认为在线?144.简述 Kubernetes 中，如何使用 EFK 实现日志的统一管理？](https://blog.csdn.net/qq_41946216/article/details/143307505)

[中级 _运维_ _工程师_ _面试题_ 汇总(含答案)](https://blog.csdn.net/qaz9821/article/details/145342620)

[qaz9821的专栏](https://blog.csdn.net/qaz9821)

01-24![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1652


[这时 RS 的包通过网关（LVS）中转，LVS 会做源地址转换（SNAT），将包的源地址改为 VIP，这样，这个包对客户端看起来就仿佛是 LVS 直接返回给它的。Django有模板，表单，路由，认证，基本的数据库管理等等内建功能。Flask 比 Django 更灵活 用Flask来构建应用之前，选择组件的时候会给开发者带来更多的灵活性 ，可能有的应用场景不适合使用一个标准的ORM(Object-Relational Mapping 对象关联映射)，或者需要与不同的工作流和模板系统交互。](https://blog.csdn.net/qaz9821/article/details/145342620)

[Kubernetes学习指南，轻松拥抱云原生](https://blog.csdn.net/broadview2006/article/details/110385621)

[博文视点（北京）官方博客](https://blog.csdn.net/broadview2006)

11-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1274


[你的书架，由我承包\\
\\
Kubernetes书单来啦！\\
\\
作为云原生环境下非常热门的开源技术，K8s能够帮助我们更好地拥抱云原生，加速创新！\\
\\
马上进入12月了，趁着最后一个月，跟着博文菌再冲刺一波！学起来！\\
\\
1《阿里云数字新基建系列：云原生操作系统Kubernetes》\\
\\
2《Kubernetes权威指南：从Docker到Kubernetes实践全接触（第4版）》\\
\\
3《Kubernetes生产化实践之路》\\
\\
4《Kubernetes源码剖析》\\
\\
5《Kube...](https://blog.csdn.net/broadview2006/article/details/110385621)

[Kubernetes 全流程技术实践教程（从集群搭建到生产 _运维_）\\
\\
最新发布](https://huc0day.blog.csdn.net/article/details/151861820)

[fearhacker的专栏](https://blog.csdn.net/fearhacker)

09-19![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1162


[本教程系统性地介绍了Kubernetes（K8s）容器编排平台的全生命周期管理，从基础概念到生产环境实践。内容涵盖：1）K8s核心架构与组件；2）集群部署方案（包括高可用架构）；3）基础配置与调优；4） _运维_ 管理（监控、日志、备份等）；5）安全配置（认证授权、网络策略等）；6）测试实践方法。教程特别强调生产环境注意事项，提供版本选择建议、性能优化参数和安全加固方案，并附有典型应用部署示例和 _常见_ 问题排查指南。通过本教程，读者可全面掌握K8s集群的规划、部署、 _运维_ 和优化能力，适用于容器化应用管理和云原生技术实践。](https://huc0day.blog.csdn.net/article/details/151861820)

[最新 _运维_ 面试2396道题 都掌握吊打面试官 offer到手](https://devpress.csdn.net/v1/article/detail/117459176)

[辅导自学转行](https://blog.csdn.net/qq_39418469)

06-02![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3万+


[244\. 在/home/work目录下，有10000个文件，你如何统计出里面含有yicha.cn的文件，并将包含有yicha.cn的文件发送到yunwei@yicha.cn。88. 将/home/stud1/wang目录做归档压缩，压缩后生成wang.tar.gz文件，并将此文件保存到/home目录下。247. 一台 _linux_ 服务器，上面只运行apche，突然访问变的很慢，你是如何分析问题所在的？60. 在一个系统中，cpu正常，内存使用非常大，系统各项服务都很慢，你会怎么判断和解决。](https://devpress.csdn.net/v1/article/detail/117459176)

[中高级 _运维_ _工程师_ 高频 _面试题_（附万能回答公式）](https://aaaedu.blog.csdn.net/article/details/138185752)

[张晨光老师的播客](https://blog.csdn.net/zhangchen124)

04-25![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1000


[分级发布的一个核心点是，必须要做大量的检查，这样就会给上线效率造成很大的影响，因为可能是多人ci，一个人上线，那么这个上线的同学可以不清楚别人业务的指标是否正常，是否检查完整性会有大打折扣，另外，检查必然会有大量的时间浪费，那么是否可以有一个工具来实现所有指标的自动化和智能化检查呢？如果是推荐系统，比如抖音和快手，那么时长就是一个非常核心的指标。智能在哪里呢，异常指标的判断，一个上线可能有几百上千的指标，不可能去定义每个指标的检查算法，那么这个智能检查就会集成一些默认算法指标，以及上下游的服务。](https://aaaedu.blog.csdn.net/article/details/138185752)

[_运维_ 经典 _面试题_ 总结100道（含答案）---2026 _常见_ _面试题_\\
\\
热门推荐](https://blog.csdn.net/weixin_50014016/article/details/139865528)

[流水\_鱼鱼的博客](https://blog.csdn.net/weixin_50014016)

06-21![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
6万+


[_运维_ _工程师_ 面试总结(含答案)\_ _运维_ 面试](https://blog.csdn.net/weixin_50014016/article/details/139865528)

[最新整理的 _运维_ _工程师_ 面试真的太给力了，整整 _50_ 道，速度收藏！](https://wljslmz.blog.csdn.net/article/details/132236358)

[网络技术联盟站](https://blog.csdn.net/weixin_43025343)

08-11![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
9801


[K8s是kubernetes的简称，其本质是一个开源的容器编排系统，主要用于管理容器化的应用，其目标是让部署容器化的应用简单并且高效（powerful）,Kubernetes提供了应用部署，规划，更新，维护的一种机制。说简单点：k8s就是一个编排容器的系统，一个可以管理容器应用全生命周期的工具，从创建应用，应用的部署，应用提供服务，扩容缩容应用，应用更新，都非常的方便，而且还可以做到故障自愈，所以，k8s是一个非常强大的容器编排系统。](https://wljslmz.blog.csdn.net/article/details/132236358)

[_运维_ _工程师_ 面试总结(含答案)](https://ncayu.blog.csdn.net/article/details/130911409)

[qq\_44534541的博客](https://blog.csdn.net/qq_44534541)

05-28![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[原文链接：https://www.cuiliangblog.cn/detail/article/2。](https://ncayu.blog.csdn.net/article/details/130911409)

[Kubernetes 生产化集群管理 操作系统选择](https://blog.csdn.net/qq_34556414/article/details/126358573)

[小楼一夜听春雨，深巷明朝卖杏花](https://blog.csdn.net/qq_34556414)

08-16![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1476


[不可变架构就是希望主机的操作系统是不可变的，然后运行在主机上面的容器镜像也是不可变的，那么就让主机的大部分文件目录都是只读的，我只做必要的变更，然后再通过某些机制从流程上面管控这些变更的可能性，那么就使得整个基础架构是不可变的。过去很多的生产化 _运维_，它是比较随意的，比如说操作系统有各种各样的工具，然后所有的文件目录都是可以修改配置的，那么这种架构下面，管理员或者用户习惯性的去登入这些节点上面，去做一些配置的更改，这样改来改去，积年累月，那么可能一个事件的迁移就会搞出大问题。......](https://blog.csdn.net/qq_34556414/article/details/126358573)

[Kubernetes容器平台建设中，F5解决方案好不好？](https://blog.csdn.net/hanniuniu11/article/details/98618491)

[hanniuniu11的博客](https://blog.csdn.net/hanniuniu11)

08-06![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3397


[近年来，随着OpenStack、Kubernetes等云技术的兴起，应用系统的微服务化、快速迭代对资源的弹性伸缩能力提出了更高的要求。基于多年在负载均衡领域的经验，Kubernetes容器平台建设中，F5解决方案好不好？\\
\\
F5推出了Kubernetes容器服务解决方案\\
前不久，民生银行在Kubernetes容器平台建设中，探索使用了一种灵活的软件F5解决方案，在利用F5传统优势的同时，也满足...](https://blog.csdn.net/hanniuniu11/article/details/98618491)

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

![](https://blog.csdn.net/yy17111342926/article/details/149415196)

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

微信名片![](https://g.csdnimg.cn/extension-box/2.0.2/image/ic_close.png)

![](https://i-blog.csdnimg.cn/direct/31c249bc5f38405bad9b22847e5738bf.png) 微信 ID：2023最新python资料领取微信扫码添加好友或搜索 ID

复制微信 ID

![](https://csdnimg.cn/release/blogv2/dist/pc/img/quoteClose1White.png)

![](https://i-blog.csdnimg.cn/img_convert/242d593d6392230dcb473e527933fbba.png)

![](https://i-blog.csdnimg.cn/img_convert/6a73170cb77cb039029f06b3e791ba38.png)

![](https://i-blog.csdnimg.cn/img_convert/a9a52979b48d1f990c68968cff117d27.png)

![](https://i-blog.csdnimg.cn/img_convert/f0dd316ead1ed18742db8e7f70e377bb.png)

![](https://i-blog.csdnimg.cn/img_convert/3eef66ab97296a861ad855f2dbb3b850.png)

![](https://i-blog.csdnimg.cn/img_convert/e47de1aa144c7adc7fcd26f5c4f29b3c.png)

![](https://i-blog.csdnimg.cn/img_convert/a4d887fc04e74e19eee255dbbc5b8e8e.png)

-100%+1:1还原