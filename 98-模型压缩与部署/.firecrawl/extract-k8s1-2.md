# 「高频必考」Docker&K8S面试题和答案

URL: https://developer.aliyun.com/article/1394001

[开发者社区](https://developer.aliyun.com/) [云原生](https://developer.aliyun.com/group/cloudnative/) [文章](https://developer.aliyun.com/group/cloudnative/article/) 正文

# 「高频必考」Docker&K8S面试题和答案

2023-12-121152发布于吉林

版权
举报

版权声明：

本文内容由阿里云实名注册用户自发贡献，版权归原作者所有，阿里云开发者社区不拥有其著作权，亦不承担相应法律责任。具体规则请查看《
[阿里云开发者社区用户服务协议](https://developer.aliyun.com/article/768092)》和
《 [阿里云开发者社区知识产权保护指引](https://developer.aliyun.com/article/768093)》。如果您发现本社区中有涉嫌抄袭的内容，填写
[侵权投诉表单](https://yida.alibaba-inc.com/o/right) 进行举报，一经查实，本社区将立刻删除涉嫌侵权内容。


**简介：**「高频必考」Docker&K8S面试题和答案

## Docker

#### 如何在Docker容器内部访问主机上的服务？

可以通过设置主机网络模式，使用--net=host参数来访问主机上的服务。这样，容器和主机将共享一个网络命名空间，容器将可以直接访问主机上的服务。

#### 如何在Docker容器中运行多个进程？

Docker推荐每个容器只运行一个进程。如果需要在容器中运行多个进程，可以使用supervisord等进程管理工具来管理多个进程。

#### 如何在Docker容器中使用环境变量？

可以通过在Dockerfile中使用ENV指令定义环境变量，或者使用docker run命令的-e选项来设置环境变量。在容器内部，可以使用$ENV\_NAME的方式来引用环境变量。

#### 如何在Docker容器中共享数据？

可以使用Docker数据卷来共享数据。数据卷是一个可以被容器内外访问的特殊目录，可以在docker run命令中使用-v选项来创建和挂载数据卷。

#### 如何在Docker容器中安装和使用ssh服务？

可以在Dockerfile中安装openssh-server，然后启动sshd服务。同时，需要在docker run命令中使用-p选项将容器内部的22端口映射到主机上的一个端口，以便可以通过ssh连接到容器。

#### 如何在Docker容器中使用多个镜像？

可以使用Docker Compose来组合多个镜像。在Docker Compose配置文件中，可以定义多个服务，每个服务对应一个镜像，然后使用docker-compose up命令来启动多个容器。

#### 如何在Docker容器中运行GUI应用？

可以在Dockerfile中安装图形界面相关的软件包，然后使用docker run命令的--env DISPLAY选项来设置显示环境变量，再使用--volume选项将主机上的X11套接字文件挂载到容器内部。这样，就可以在容器中运行GUI应用了。

#### 如何在Docker容器中限制CPU和内存使用？

可以使用docker run命令的--cpu-shares和--memory选项来限制CPU和内存使用。--cpu-shares选项可以设置CPU时间片的权重，--memory选项可以设置容器能够使用的内存大小。

#### 如何在Docker容器中设置时区？

可以在Dockerfile中安装tzdata软件包，然后使用docker run命令的-e选项设置TZ环境变量为所需时区，或者使用--volume选项将主机上的/etc/localtime文件挂载到容器内部的相应位置。

#### 如何在Docker容器中实现容器间通信？

可以使用Docker网络来实现容器间通信。Docker提供了多种网络模式，如默认的bridge网络、host网络、overlay网络等。可以使用docker network命令来创建和管理网络，并使用--network选项指定容器所属的网络。在同一个网络中的容器可以直接通过容器名或IP地址相互访问。

## kubernetes

#### 什么是Kubernetes？

Kubernetes是一个用于自动部署、扩展和管理容器化应用程序的开源平台。它提供了一个可扩展的、高可用的集群，并包含了自动化部署、负载平衡、存储管理、自我修复、自动扩容等功能。

#### Kubernetes中的Pod是什么？

Pod是Kubernetes中最小的部署单元，它包含一个或多个紧密耦合的容器和共享的存储/网络资源。Pod提供了一种抽象层，使得容器可以在同一个主机上共享文件系统、网络命名空间等资源。

#### Kubernetes中的ReplicaSet是什么？

ReplicaSet是一个用于管理Pod副本数量的控制器。它可以根据用户定义的副本数量，自动调整Pod的数量，以保证应用程序的高可用性。

#### Kubernetes中的Service是什么？

Service是Kubernetes中一个抽象的逻辑概念，用于暴露Pod的网络服务。Service可以将Pod的IP地址和端口映射到一个虚拟的IP地址和端口上，从而实现了Pod的负载均衡和服务发现功能。

#### Kubernetes中的Deployment是什么？

Deployment是一个用于管理Pod部署的控制器。它可以自动创建和更新Pod，以便保持应用程序的可用性和可伸缩性。

#### Kubernetes中的ConfigMap和Secret是什么？

ConfigMap是一个用于存储应用程序配置信息的对象，可以通过环境变量、命令行参数等方式使用。Secret是一个用于存储敏感信息（如密码、密钥等）的对象，可以加密存储。

#### Kubernetes中的DaemonSet是什么？

DaemonSet是一个用于在每个节点上运行一个Pod的控制器。它可以用于运行一些需要在每个节点上运行的系统级别的服务，如日志收集、监控等。

#### Kubernetes中的StatefulSet是什么？

StatefulSet是一个用于管理有状态应用程序的控制器。它可以保证Pod的唯一性和有序性，从而实现有状态应用程序的可靠性。

#### Kubernetes中的HorizontalPodAutoscaler是什么？

HorizontalPodAutoscaler是一个用于自动扩展Pod数量的控制器。它可以根据应用程序的负载情况，自动调整Pod数量，以保证应用程序的性能和可用性。

#### Kubernetes中的CSI是什么？

CSI（Container Storage Interface）是一个用于存储管理的标准接口，它可以让存储提供商为Kubernetes提供自定义的存储解决方案。CSI可以让Kubernetes与不同的存储提供商进行集成，从而实现高度可定制化的存储管理。

文章标签：

[容器服务Kubernetes版](https://developer.aliyun.com/label/article_de-product-3-csk)

[容器](https://developer.aliyun.com/label/article_de-3-100018)

[Docker](https://developer.aliyun.com/label/article_de-3-100208)

[Kubernetes](https://developer.aliyun.com/label/article_de-3-100209)

[Perl](https://developer.aliyun.com/label/article_de-3-100007)

[存储](https://developer.aliyun.com/label/article_de-3-100262)

关键词：

[Docker k8s](https://www.aliyun.com/sswb/361017.html)

[容器服务Kubernetes版docker](https://www.aliyun.com/sswb/1303469.html)

[docker容器服务Kubernetes版](https://www.aliyun.com/sswb/816800.html)

[Docker面试](https://www.aliyun.com/sswb/577878.html)

[Docker面试题](https://www.aliyun.com/sswb/715667.html)

相关实践学习

深入解析Docker容器化技术

Docker是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的Linux机器上，也可以实现虚拟化，容器是完全使用沙箱机制，相互之间不会有任何接口。Docker是世界领先的软件容器平台。开发人员利用Docker可以消除协作编码时“在我的机器上可正常工作”的问题。运维人员利用Docker可以在隔离容器中并行运行和管理应用，获得更好的计算密度。企业利用Docker可以构建敏捷的软件交付管道，以更快的速度、更高的安全性和可靠的信誉为Linux和Windows Server应用发布新功能。 在本套课程中，我们将全面的讲解Docker技术栈，从环境安装到容器、镜像操作以及生产环境如何部署开发的微服务应用。本课程由黑马程序员提供。 &nbsp; &nbsp; 相关的阿里云产品：容器服务 ACK 容器服务 Kubernetes 版（简称 ACK）提供高性能可伸缩的容器应用管理能力，支持企业级容器化应用的全生命周期管理。整合阿里云虚拟化、存储、网络和安全能力，打造云端最佳容器化应用运行环境。 了解产品详情: https://www.aliyun.com/product/kubernetes

[![](https://ucc.alicdn.com/avatar/avatar3.jpg?x-oss-process=image/resize,h_150,m_lfit)](https://developer.aliyun.com/profile/6k2lvb6jt2j4s)

![](https://ucc.alicdn.com/pic/ucc-admin/88c34b916d704521b87d41daa9a77107.png?x-oss-process=image%2Fresize%2Ch_80%2Cm_lfit%2Fformat%2Cwebp)

[antony王中阳](https://developer.aliyun.com/profile/6k2lvb6jt2j4s)

+关注

[231文章](https://developer.aliyun.com/profile/6k2lvb6jt2j4s/article_1)

目录

1

0

0

11

分享

相关文章

[蓝易云](https://developer.aliyun.com/profile/3c4vysbj27yje)

\|

[Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [Docker](https://developer.aliyun.com/label/sc/de-3-100208) [容器](https://developer.aliyun.com/label/sc/de-3-100018)

[Kubernetes与Docker参数对照：理解Pod中的command、args与Dockerfile中的CMD、ENTRYPOINT。](https://developer.aliyun.com/article/1658986)

需要明确的是，理解这些都需要对Docker和Kubernetes有一定深度的理解，才能把握二者的区别和联系。虽然它们都是容器技术的二个重要组成部分，但各有其特性和适用场景，理解它们的本质和工作方式，才能更好的使用这些工具，将各自的优点整合到生产环境中，实现软件的快速开发和部署。

[蓝易云](https://developer.aliyun.com/profile/3c4vysbj27yje)

5332525

[刘大猫.](https://developer.aliyun.com/profile/urngpoxx4ky3i)

\|

[监控](https://developer.aliyun.com/label/sc/de-3-100072) [NoSQL](https://developer.aliyun.com/label/sc/de-3-100068) [时序数据库](https://developer.aliyun.com/label/sc/de-3-100069)

[《docker高级篇（大厂进阶）：7.Docker容器监控之CAdvisor+InfluxDB+Granfana》包括：原生命令、是什么、compose容器编排，一套带走](https://developer.aliyun.com/article/1644678)

《docker高级篇（大厂进阶）：7.Docker容器监控之CAdvisor+InfluxDB+Granfana》包括：原生命令、是什么、compose容器编排，一套带走

[刘大猫.](https://developer.aliyun.com/profile/urngpoxx4ky3i)

7367878

[栈江湖](https://developer.aliyun.com/profile/qcazreabeg5vk)

\|

[存储](https://developer.aliyun.com/label/sc/de-3-100262) [Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [开发者](https://developer.aliyun.com/label/sc/de-3-100267)

[容器化时代的领航者：Docker 和 Kubernetes 云原生时代的黄金搭档](https://developer.aliyun.com/article/1646388)

Docker 是一种开源的应用容器引擎，允许开发者将应用程序及其依赖打包成可移植的镜像，并在任何支持 Docker 的平台上运行。其核心概念包括镜像、容器和仓库。镜像是只读的文件系统，容器是镜像的运行实例，仓库用于存储和分发镜像。Kubernetes（k8s）则是容器集群管理系统，提供自动化部署、扩展和维护等功能，支持服务发现、负载均衡、自动伸缩等特性。两者结合使用，可以实现高效的容器化应用管理和运维。Docker 主要用于单主机上的容器管理，而 Kubernetes 则专注于跨多主机的容器编排与调度。尽管 k8s 逐渐减少了对 Docker 作为容器运行时的支持，但 Doc

[栈江湖](https://developer.aliyun.com/profile/qcazreabeg5vk)

76855

[![容器化时代的领航者：Docker 和 Kubernetes 云原生时代的黄金搭档](https://ucc.alicdn.com/qcazreabeg5vk/developer-article1646388/20241223/d7be1288782d447c9154f1114fb757ee.webp?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/1646388)

[蓝易云](https://developer.aliyun.com/profile/3c4vysbj27yje)

\|

[存储](https://developer.aliyun.com/label/sc/de-3-100262) [Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [Docker](https://developer.aliyun.com/label/sc/de-3-100208)

[Kubernetes（k8s）和Docker Compose本质区别](https://developer.aliyun.com/article/1646902)

理解它们的区别和各自的优势，有助于选择合适的工具来满足特定的项目需求。

[蓝易云](https://developer.aliyun.com/profile/3c4vysbj27yje)

17571920

[RossyYan](https://developer.aliyun.com/profile/xr6p7tt2s7v4u)

\|

[Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [Linux](https://developer.aliyun.com/label/sc/de-3-100077) [虚拟化](https://developer.aliyun.com/label/sc/de-3-100136)

[入门级容器技术解析：Docker和K8s的区别与关系](https://developer.aliyun.com/article/1648449)

本文介绍了容器技术的发展历程及其重要组成部分Docker和Kubernetes。从传统物理机到虚拟机，再到容器化，每一步都旨在更高效地利用服务器资源并简化应用部署。容器技术通过隔离环境、减少依赖冲突和提高可移植性，解决了传统部署方式中的诸多问题。Docker作为容器化平台，专注于创建和管理容器；而Kubernetes则是一个强大的容器编排系统，用于自动化部署、扩展和管理容器化应用。两者相辅相成，共同推动了现代云原生应用的快速发展。

[RossyYan](https://developer.aliyun.com/profile/xr6p7tt2s7v4u)

39831111

[IT行业工程师](https://developer.aliyun.com/profile/leha2ktgjdiu4)

\|

[Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [开发者](https://developer.aliyun.com/label/sc/de-3-100267) [Docker](https://developer.aliyun.com/label/sc/de-3-100208)

[Docker与Kubernetes的协同工作](https://developer.aliyun.com/article/1642639)

Docker与Kubernetes的协同工作

[IT行业工程师](https://developer.aliyun.com/profile/leha2ktgjdiu4)

36200

[南瓜佬](https://developer.aliyun.com/profile/56dndxy5qvfgg)

\|

8月前

\|

[Kubernetes](https://developer.aliyun.com/label/sc/de-3-100209) [Devops](https://developer.aliyun.com/label/sc/de-3-100023) [Docker](https://developer.aliyun.com/label/sc/de-3-100208)

[Kubernetes 和 Docker Swarm：现代 DevOps 的理想容器编排工具](https://developer.aliyun.com/article/1680826)

本指南深入解析 Kubernetes 与 Docker Swarm 两大主流容器编排工具，涵盖安装、架构、网络、监控等核心维度，助您根据团队能力与业务需求精准选型，把握云原生时代的技术主动权。

[南瓜佬](https://developer.aliyun.com/profile/56dndxy5qvfgg)

723115115

## 热门文章

## 最新文章

[1\\
\\
寻找 K8s 1.14 Release 里的“蚌中之珠”\\
\\
8869](https://developer.aliyun.com/article/695664)
[2\\
\\
【云栖大会】Docker与阿里云达成战略合作 为企业级客户提供容器服务\\
\\
8379](https://developer.aliyun.com/article/61595)
[3\\
\\
在K8S中，什么是CNI？平时K8s集群常用什么网络插件？\\
\\
15](https://developer.aliyun.com/article/1594616)
[4\\
\\
使用Kubernetes进行CI/CD的最佳实践\\
\\
10](https://developer.aliyun.com/article/1214330)
[5\\
\\
在Kubernetes（k8s）中部署 jenkins\\
\\
5](https://developer.aliyun.com/article/974810)
[6\\
\\
K8s集群v1.26.1版本的简单部署实践\\
\\
5](https://developer.aliyun.com/article/1430569)
[7\\
\\
快速搭建kubernetes与kubeSphere环境（亲测有效）\\
\\
9](https://developer.aliyun.com/article/1132285)
[8\\
\\
基于ACK One注册集群实现IDC中K8s集群以Serverless方式使用云上CPU/GPU资源\\
\\
5](https://developer.aliyun.com/article/1290956)
[9\\
\\
Kubernetes 上 Java 应用的最佳实践\\
\\
8](https://developer.aliyun.com/article/1204040)
[10\\
\\
阿里云容器服务 ACK 产品技术动态（202208）\\
\\
6](https://developer.aliyun.com/article/1024725)

[1\\
\\
在虚拟机Docker环境下部署Nginx的步骤。\\
\\
346](https://developer.aliyun.com/article/1686215)
[2\\
\\
手把手教你使用 Docker 部署 Nginx 教程\\
\\
3182](https://developer.aliyun.com/article/1686131)
[3\\
\\
n8n Docker 部署手册\\
\\
3413](https://developer.aliyun.com/article/1684784)
[4\\
\\
【Docker】（3）学习Docker中 镜像与容器数据卷、映射关系！手把手带你安装 MySql主从同步 和 Redis三主三从集群！并且进行主从切换与扩容操作，还有分析 哈希分区 等知识点！\\
\\
788](https://developer.aliyun.com/article/1684460)
[5\\
\\
【Docker】（2）还在浏览网页寻找Docker命令？本文全面列举与使用Docker里的各个命令！想要什么命令直接从本文拿！\\
\\
818](https://developer.aliyun.com/article/1684459)
[6\\
\\
【Docker】（1）Docker的概述与架构，手把手带你安装Docker，云原生路上不可缺少的一门技术！\\
\\
626](https://developer.aliyun.com/article/1684458)
[7\\
\\
【赵渝强老师】Docker容器的资源管理机制\\
\\
696](https://developer.aliyun.com/article/1684297)
[8\\
\\
手把手教你用 Docker 部署 Redis\\
\\
2243](https://developer.aliyun.com/article/1684285)
[9\\
\\
如何在Kubernetes环境下使用第三方监控系统监控Docker容器性能？\\
\\
863](https://developer.aliyun.com/article/1684226)
[10\\
\\
如何使用第三方监控系统监控Docker容器性能？\\
\\
727](https://developer.aliyun.com/article/1684225)

相关商品

## 相关课程

[更多](https://edu.aliyun.com/explore/)

[深入解析Docker容器化技术](https://edu.aliyun.com/course/314620)
[基于Docker与Jenkins实现自动化部署](https://edu.aliyun.com/course/314498)
[Docker 快速入门](https://edu.aliyun.com/course/314379)
[Docker完全自学手册图文教程](https://edu.aliyun.com/course/313684)
[AI开发者的Docker实践](https://tianchi.aliyun.com/course/351)
[Docker 入门](https://edu.aliyun.com/course/312466)

## 相关电子书

[更多](https://developer.aliyun.com/ebook/)

[ACK集群类型选择最佳实践](https://developer.aliyun.com/ebook/8094)
[容器服务重磅升级，打造高效安全、智能无界新平台](https://developer.aliyun.com/ebook/212)
[《拥抱容器存储，使用阿里云容器服务 ACK +文件存储 NAS 构建现代化企业应用》](https://developer.aliyun.com/ebook/478)

## 推荐镜像

[更多](https://developer.aliyun.com/mirror/)

[docker-ce](https://developer.aliyun.com/mirror/docker-ce)
[docker-toolbox](https://developer.aliyun.com/mirror/docker-toolbox)

热门文章

最新文章

下一篇

[\[网络安全\] Dirsearch 工具的安装、使用详细教程](https://developer.aliyun.com/article/1395854)

目录

- [Docker](https://developer.aliyun.com/article/1394001#slide-0)
- [如何在Docker容器内部访问主机上的服务？](https://developer.aliyun.com/article/1394001#slide-1)
- [如何在Docker容器中运行多个进程？](https://developer.aliyun.com/article/1394001#slide-2)
- [如何在Docker容器中使用环境变量？](https://developer.aliyun.com/article/1394001#slide-3)
- [如何在Docker容器中共享数据？](https://developer.aliyun.com/article/1394001#slide-4)
- [如何在Docker容器中安装和使用ssh服务？](https://developer.aliyun.com/article/1394001#slide-5)
- [如何在Docker容器中使用多个镜像？](https://developer.aliyun.com/article/1394001#slide-6)
- [如何在Docker容器中运行GUI应用？](https://developer.aliyun.com/article/1394001#slide-7)
- [如何在Docker容器中限制CPU和内存使用？](https://developer.aliyun.com/article/1394001#slide-8)
- [如何在Docker容器中设置时区？](https://developer.aliyun.com/article/1394001#slide-9)
- [如何在Docker容器中实现容器间通信？](https://developer.aliyun.com/article/1394001#slide-10)
- [kubernetes](https://developer.aliyun.com/article/1394001#slide-11)
- [什么是Kubernetes？](https://developer.aliyun.com/article/1394001#slide-12)
- [Kubernetes中的Pod是什么？](https://developer.aliyun.com/article/1394001#slide-13)
- [Kubernetes中的ReplicaSet是什么？](https://developer.aliyun.com/article/1394001#slide-14)
- [Kubernetes中的Service是什么？](https://developer.aliyun.com/article/1394001#slide-15)
- [Kubernetes中的Deployment是什么？](https://developer.aliyun.com/article/1394001#slide-16)
- [Kubernetes中的ConfigMap和Secret是什么？](https://developer.aliyun.com/article/1394001#slide-17)
- [Kubernetes中的DaemonSet是什么？](https://developer.aliyun.com/article/1394001#slide-18)
- [Kubernetes中的StatefulSet是什么？](https://developer.aliyun.com/article/1394001#slide-19)
- [Kubernetes中的HorizontalPodAutoscaler是什么？](https://developer.aliyun.com/article/1394001#slide-20)
- [Kubernetes中的CSI是什么？](https://developer.aliyun.com/article/1394001#slide-21)

目录

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

4008013260 [售前咨询](https://smartservice.console.aliyun.com/pre-sale/chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1394001) [售后在线](https://smartservice.console.aliyun.com/service/robot-chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1394001)

### 其他服务

[我要建议](https://www.aliyun.com/connect/home) [我要投诉](https://www.aliyun.com/complaint)

![登录插画](https://img.alicdn.com/imgextra/i2/O1CN015QIT9m1FmmyUntYlQ_!!6000000000530-2-tps-320-200.png)

登录以查看您的控制台资源

管理云资源

状态一览

快捷访问

[快捷注册](https://account.aliyun.com/register/qr_register.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1394001) [登录阿里云](https://account.aliyun.com/login/login.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1394001)