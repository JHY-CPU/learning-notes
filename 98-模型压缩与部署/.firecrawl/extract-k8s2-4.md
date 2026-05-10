# Docker+Kubernetes精选面试题汇总 - CSDN博客

URL: https://blog.csdn.net/weixin_41737291/article/details/148538005

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

# Docker+Kubernetes精选面试题汇总

最新推荐文章于 2025-06-24 00:54:29 发布

原创 [![](https://csdnimg.cn/release/blogv2/dist/pc/img/identityVipNew.png)](https://mall.csdn.net/vip) 于 2025-06-09 17:06:34 发布·1k 阅读

·![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png)
26


·![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png)
16


文章标签：

[#docker](https://so.csdn.net/so/search/s.do?q=docker&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#kubernetes](https://so.csdn.net/so/search/s.do?q=kubernetes&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#容器](https://so.csdn.net/so/search/s.do?q=%E5%AE%B9%E5%99%A8&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#运维](https://so.csdn.net/so/search/s.do?q=%E8%BF%90%E7%BB%B4&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#云原生](https://so.csdn.net/so/search/s.do?q=%E4%BA%91%E5%8E%9F%E7%94%9F&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art)

## Docker+Kubernetes精选面试题汇总

在数字化转型的浪潮中，云计算已成为企业技术架构的核心底座，而容器化与容器编排技术则是支撑云原生应用的关键引擎。Docker 凭借其轻量级容器化能力，彻底改变了应用部署与运行的方式，让 “一次构建，到处运行” 成为现实；Kubernetes 则以强大的集群管理与自动化调度能力，将容器技术推向大规模生产环境的舞台中央，成为云原生时代的事实标准。

本文精心整理的 105 道核心面试题，全面覆盖 Docker 与 Kubernetes 的核心概念、架构设计与实践应用，旨在为云计算从业者提供一份系统的知识梳理与面试备考指南。从 Docker 的镜像构建、容器生命周期管理到网络通信模式，从 Kubernetes 的核心组件原理、资源调度策略到复杂的存储与监控体系，每一道题目都凝聚着对技术本质的深入解析，既包含基础概念的精准阐释，也涵盖生产环境中的实践要点。

对于初入行业的技术新人，本书可作为快速掌握云原生核心技术的入门手册，通过清晰的概念对比（如 Docker 容器与传统虚拟机的差异）、详尽的流程解析（如 Dockerfile 的指令详解），逐步建立起对容器技术栈的完整认知；对于经验丰富的开发者与运维工程师，书中深入的架构分析（如 Kubernetes 的调度器原理、存储卷机制）与实战场景解答（如蓝绿部署、日志收集方案），可帮助进一步深化技术理解，梳理知识体系，从容应对高阶技术面试。

#### 文章目录

- [Docker+Kubernetes精选面试题汇总](https://blog.csdn.net/weixin_41737291/article/details/148538005#DockerKubernetes_0)
  - [1.什么是Docker？](https://blog.csdn.net/weixin_41737291/article/details/148538005#1Docker_8)
  - [2.Docker容器与传统虚拟机有何区别？](https://blog.csdn.net/weixin_41737291/article/details/148538005#2Docker_16)
  - [3.Docker镜像（Image）是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#3DockerImage_66)
  - [4.Docker容器（Container）是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#4DockerContainer_74)
  - [5.Docker容器的生命周期管理包括哪些操作？](https://blog.csdn.net/weixin_41737291/article/details/148538005#5Docker_82)
  - [6.Docker中的“端口映射”是什么意思？](https://blog.csdn.net/weixin_41737291/article/details/148538005#6Docker_88)
  - [7.Dockerfile是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#7Dockerfile_96)
  - [8.Docker的优点都有哪些](https://blog.csdn.net/weixin_41737291/article/details/148538005#8Docker_104)
  - [9.Dockerfile常用指令都有哪些？](https://blog.csdn.net/weixin_41737291/article/details/148538005#9Dockerfile_120)
  - [10.COPY和ADD命令在Dockerfile中有什么区别?](https://blog.csdn.net/weixin_41737291/article/details/148538005#10COPYADDDockerfile_148)
  - [11.什么是Docker容器编排？](https://blog.csdn.net/weixin_41737291/article/details/148538005#11Docker_160)
  - [12.Docker Compose是什么？它的主要用途是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#12Docker_Compose_170)
  - [13.Docker Compose的基本命令有哪些？](https://blog.csdn.net/weixin_41737291/article/details/148538005#13Docker_Compose_178)
  - [14.Docker私有仓库的作用是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#14Docker_196)
  - [15.Docker 的网络通信模式。](https://blog.csdn.net/weixin_41737291/article/details/148538005#15Docker__202)
  - [16.Docker Swarm是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#16Docker_Swarm_224)
  - [17.Swarm中的角色有哪些？](https://blog.csdn.net/weixin_41737291/article/details/148538005#17Swarm_232)
  - [18.Swarm与Kubernetes相比有哪些特点？](https://blog.csdn.net/weixin_41737291/article/details/148538005#18SwarmKubernetes_240)
  - [19.如何扩缩Swarm服务的副本数量？](https://blog.csdn.net/weixin_41737291/article/details/148538005#19Swarm_246)
  - [20.如何加入一个Swarm集群作为工作节点？](https://blog.csdn.net/weixin_41737291/article/details/148538005#20Swarm_252)
  - [21.Docker中如何查看容器的日志？](https://blog.csdn.net/weixin_41737291/article/details/148538005#21Docker_258)
  - [22.Logstash的作用是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#22Logstash_268)
  - [23.Filebeat如何与Docker集成收集日志？](https://blog.csdn.net/weixin_41737291/article/details/148538005#23FilebeatDocker_278)
  - [24.如何在Docker容器中安装和运行Filebeat？](https://blog.csdn.net/weixin_41737291/article/details/148538005#24DockerFilebeat_286)
  - [25.Docker日志如何实现持久化存储？](https://blog.csdn.net/weixin_41737291/article/details/148538005#25Docker_296)
  - [26.什么是Containerd？](https://blog.csdn.net/weixin_41737291/article/details/148538005#26Containerd_304)
  - [27.Containerd和Docker的关系是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#27ContainerdDocker_312)
  - [28.Containerd的架构包含哪些主要组件？](https://blog.csdn.net/weixin_41737291/article/details/148538005#28Containerd_324)
  - [29.Containerd的性能优势是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#29Containerd_332)
  - [30.Containerd如何管理容器网络？](https://blog.csdn.net/weixin_41737291/article/details/148538005#30Containerd_338)
  - [31.简述Kubernetes是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#31Kubernetes_346)
  - [32.Kubernetes的核心组件有哪些？](https://blog.csdn.net/weixin_41737291/article/details/148538005#32Kubernetes_354)
  - [33.有了Docker为什么还用Kubernetes？](https://blog.csdn.net/weixin_41737291/article/details/148538005#33DockerKubernetes_366)
  - [34.在kubernetes中，etcd有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#34kubernetesetcd_382)
  - [35.ETCD在部署时需要多少个节点](https://blog.csdn.net/weixin_41737291/article/details/148538005#35ETCD_390)
  - [36.在kubernetes中API Server组件有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#36kubernetesAPI_Server_396)
  - [37.在kubernetes中Scheduler组件有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#37kubernetesScheduler_404)
  - [38.在kubernetes中Controller Manager组件有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#38kubernetesController_Manager_412)
  - [39.在kubernetes中kubelet组件有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#39kuberneteskubelet_420)
  - [40.在kubernetes中kube-proxy代理组件有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#40kuberneteskubeproxy_428)
  - [41.什么是Pod](https://blog.csdn.net/weixin_41737291/article/details/148538005#41Pod_446)
  - [42.Pod探针有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#42Pod_456)
  - [43.Pod探针实现方式都有哪些](https://blog.csdn.net/weixin_41737291/article/details/148538005#43Pod_462)
  - [44.简述存活探针（Liveness Probe）](https://blog.csdn.net/weixin_41737291/article/details/148538005#44Liveness_Probe_478)
  - [45.简述就绪探针（Readiness Probe）](https://blog.csdn.net/weixin_41737291/article/details/148538005#45Readiness_Probe_488)
  - [46.简述启动探针（Startup Probe）](https://blog.csdn.net/weixin_41737291/article/details/148538005#46Startup_Probe_498)
  - [47.在kubernetes中镜像拉取策略都有哪些，默认是什么方式](https://blog.csdn.net/weixin_41737291/article/details/148538005#47kubernetes_508)
  - [48.Pod进行部署或者运行时的重启策略都有哪些，默认策略是什么](https://blog.csdn.net/weixin_41737291/article/details/148538005#48Pod_520)
  - [49.什么是静态pod，并简述它的特性](https://blog.csdn.net/weixin_41737291/article/details/148538005#49pod_532)
  - [50.简述什么是Replication Controller（复制控制器，RC）](https://blog.csdn.net/weixin_41737291/article/details/148538005#50Replication_ControllerRC_542)
  - [51.简述什么是ReplicaSet（复制集，RS），它和RC的区别是什么](https://blog.csdn.net/weixin_41737291/article/details/148538005#51ReplicaSetRSRC_552)
  - [52.什么是DaemonSet，他有什么特性](https://blog.csdn.net/weixin_41737291/article/details/148538005#52DaemonSet_560)
  - [53.简述什么是无状态服务，请列举相应的例子](https://blog.csdn.net/weixin_41737291/article/details/148538005#53_570)
  - [54.简述什么是有状态服务，请列举相应的例子](https://blog.csdn.net/weixin_41737291/article/details/148538005#54_576)
  - [55.在Kubernetes使用Deployment部署Pod时，都提供了哪些高级功能](https://blog.csdn.net/weixin_41737291/article/details/148538005#55KubernetesDeploymentPod_585)
  - [56.在Kubernetes中创建的Service有什么作用](https://blog.csdn.net/weixin_41737291/article/details/148538005#56KubernetesService_590)
  - [57.解释Kubernetes中EndPoint的定义](https://blog.csdn.net/weixin_41737291/article/details/148538005#57KubernetesEndPoint_597)
  - [58.在Kubernetes中，目前Service支持的类型都有哪些](https://blog.csdn.net/weixin_41737291/article/details/148538005#58KubernetesService_606)
  - [59.在Kubernetes的Service类型中，简述ExternalName类型的运行原理](https://blog.csdn.net/weixin_41737291/article/details/148538005#59KubernetesServiceExternalName_621)
  - [60.简述Kubernetes中Headless Service的特点](https://blog.csdn.net/weixin_41737291/article/details/148538005#60KubernetesHeadless_Service_628)
  - [61.什么是configmap](https://blog.csdn.net/weixin_41737291/article/details/148538005#61configmap_633)
  - [62.ConfigMap有什么缺陷](https://blog.csdn.net/weixin_41737291/article/details/148538005#62ConfigMap_640)
  - [63.在Kubernetes中，使用ConfigMap都有哪些限制](https://blog.csdn.net/weixin_41737291/article/details/148538005#63KubernetesConfigMap_647)
  - [64.pod使用configmap的方式有哪些](https://blog.csdn.net/weixin_41737291/article/details/148538005#64podconfigmap_662)
  - [65.在Kubernetes中，valueFrom字段和envFrom字段的区别是什么](https://blog.csdn.net/weixin_41737291/article/details/148538005#65KubernetesvalueFromenvFrom_671)
  - [66.在kubernetes的卷中，emptDir的卷类型有什么特征](https://blog.csdn.net/weixin_41737291/article/details/148538005#66kubernetesemptDir_686)
  - [67.简述kubernetes中HostPath卷的特点](https://blog.csdn.net/weixin_41737291/article/details/148538005#67kubernetesHostPath_693)
  - [68.谈一谈你对PV和PVC的理解](https://blog.csdn.net/weixin_41737291/article/details/148538005#68PVPVC_700)
  - [69.简述PV的回收策略](https://blog.csdn.net/weixin_41737291/article/details/148538005#69PV_707)
  - [70.简述PV的访问策略](https://blog.csdn.net/weixin_41737291/article/details/148538005#70PV_718)
  - [71.Init容器与普通的容器的不同之处是什么](https://blog.csdn.net/weixin_41737291/article/details/148538005#71Init_731)
  - [72.什么是Pause容器？为什么Kubernetes中每个Pod都有一个Pause容器？](https://blog.csdn.net/weixin_41737291/article/details/148538005#72PauseKubernetesPodPause_744)
  - [73.Pause容器的具体作用是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#73Pause_753)
  - [74.简述什么kubernetes的HPA](https://blog.csdn.net/weixin_41737291/article/details/148538005#74kubernetesHPA_770)
  - [75.HPA如何决定何时扩展或收缩Pod的数量？](https://blog.csdn.net/weixin_41737291/article/details/148538005#75HPAPod_777)
  - [76.HPA在执行扩展操作时有哪些限制和注意事项？](https://blog.csdn.net/weixin_41737291/article/details/148538005#76HPA_782)
  - [77.简述污点和容忍的概念](https://blog.csdn.net/weixin_41737291/article/details/148538005#77_797)
  - [78.在设置污点时，effect用来描述污点的作用，他有三个选项可以使用，是哪三个选项，并解释这三个选项](https://blog.csdn.net/weixin_41737291/article/details/148538005#78effect_810)
  - [79.亲和性调度可以分成软策略和硬策略两种方式，请解释软策略和硬策略](https://blog.csdn.net/weixin_41737291/article/details/148538005#79_825)
  - [80.简述亲和性的配置规则](https://blog.csdn.net/weixin_41737291/article/details/148538005#80_834)
  - [81.Kubernetes中有哪些基本的资源对象？](https://blog.csdn.net/weixin_41737291/article/details/148538005#81Kubernetes_845)
  - [82.如何在Kubernetes中定义资源的CPU和内存限制？](https://blog.csdn.net/weixin_41737291/article/details/148538005#82KubernetesCPU_852)
  - [83.如何诊断和解决Kubernetes资源调度失败的问题？](https://blog.csdn.net/weixin_41737291/article/details/148538005#83Kubernetes_859)
  - [84.什么是ResourceQuota？如何使用它来管理集群资源？](https://blog.csdn.net/weixin_41737291/article/details/148538005#84ResourceQuota_878)
  - [85.解释kubernetes中QoS分类及其含义。](https://blog.csdn.net/weixin_41737291/article/details/148538005#85kubernetesQoS_885)
  - [86.什么是Rook，以及它如何与Ceph关联？](https://blog.csdn.net/weixin_41737291/article/details/148538005#86RookCeph_900)
  - [87.Rook如何利用Ceph提供存储服务？](https://blog.csdn.net/weixin_41737291/article/details/148538005#87RookCeph_909)
  - [88.Rook部署Ceph的步骤有哪些？](https://blog.csdn.net/weixin_41737291/article/details/148538005#88RookCeph_918)
  - [89.Rook如何确保Ceph集群的高可用性？](https://blog.csdn.net/weixin_41737291/article/details/148538005#89RookCeph_939)
  - [90.Rook管理Ceph的存储资源时，如何处理数据持久化？](https://blog.csdn.net/weixin_41737291/article/details/148538005#90RookCeph_948)
  - [91.在Kubernetes中，Pod的日志是如何存储的，如何查看pod的日志？](https://blog.csdn.net/weixin_41737291/article/details/148538005#91KubernetesPodpod_959)
  - [92.Kubernetes日志收集面临的挑战有哪些？](https://blog.csdn.net/weixin_41737291/article/details/148538005#92Kubernetes_970)
  - [93.什么是Sidecar容器模式？它如何应用于日志收集？](https://blog.csdn.net/weixin_41737291/article/details/148538005#93Sidecar_985)
  - [94.什么是DaemonSet？它如何用于日志收集？](https://blog.csdn.net/weixin_41737291/article/details/148538005#94DaemonSet_994)
  - [95.如何在Kubernetes中实现日志的实时监控？](https://blog.csdn.net/weixin_41737291/article/details/148538005#95Kubernetes_1001)
  - [96.Prometheus是什么？它的主要特点是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#96Prometheus_1006)
  - [97.Prometheus如何与Kubernetes集成？](https://blog.csdn.net/weixin_41737291/article/details/148538005#97PrometheusKubernetes_1013)
  - [98.如何利用Prometheus监控Kubernetes集群的资源使用情况？](https://blog.csdn.net/weixin_41737291/article/details/148538005#98PrometheusKubernetes_1020)
  - [99.如何使用Grafana与Prometheus集成以实现数据可视化？](https://blog.csdn.net/weixin_41737291/article/details/148538005#99GrafanaPrometheus_1031)
  - [100.Prometheus 的监控系统主要包括哪些核心组件](https://blog.csdn.net/weixin_41737291/article/details/148538005#100Prometheus__1040)
  - [101.什么是Kubernetes Ingress？](https://blog.csdn.net/weixin_41737291/article/details/148538005#101Kubernetes_Ingress_1057)
  - [102.Ingress与Service的区别是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#102IngressService_1064)
  - [103.Ingress Controller的作用是什么？](https://blog.csdn.net/weixin_41737291/article/details/148538005#103Ingress_Controller_1073)
  - [104.如何使用Ingress实现蓝绿部署或金丝雀发布？](https://blog.csdn.net/weixin_41737291/article/details/148538005#104Ingress_1080)
  - [105.Ingress的Class是什么？有什么作用？](https://blog.csdn.net/weixin_41737291/article/details/148538005#105IngressClass_1087)

#### 1.什么是Docker？

答案：Docker是一个开源的容器化平台，它允许开发者将应用及其依赖打包到一个轻量级、可移植的容器中，确保应用在任何支持Docker的环境中都能一致地运行。

#### 2.Docker容器与传统虚拟机有何区别？

答案：

Docker容器和传统虚拟机（Virtual Machines, VMs）都是实现应用程序隔离和封装的技术，但它们在实现方式、资源利用率、启动速度等方面存在显著差异。以下是两者的主要区别：

（1）资源使用

Docker容器：容器共享宿主机的操作系统内核，不需要为每个应用运行独立的操作系统实例。因此，容器在内存和CPU资源的使用上更为高效。

传统虚拟机：每个VM都有自己的完整操作系统副本，包括内核和用户空间服务。这使得VM占用更多的磁盘空间和内存。

（2）启动速度

Docker容器：容器可以在几秒钟内启动，因为它们只需要加载应用程序及其依赖项，而无需加载整个操作系统。

传统虚拟机：启动VM可能需要几分钟的时间，因为它们需要完全启动一个完整的操作系统。

（3）隔离性

Docker容器：容器之间的隔离主要是通过命名空间（namespaces）和控制组（cgroups）来实现的，这些机制允许在单一操作系统实例中隔离进程和资源。

传统虚拟机：每个VM都有自己的硬件资源（如CPU、内存等），并且运行着独立的操作系统，提供更高级别的隔离。

（4）可移植性

Docker容器：容器将应用及其依赖打包在一起，可以在任何支持Docker的环境中运行，提高了可移植性。

传统虚拟机：虽然VM也提供了良好的可移植性，但由于包含了完整的操作系统，其迁移通常会更加耗时且占用更多存储空间。

（5）应用场景

Docker容器：适用于微服务架构、持续集成/持续部署（CI/CD）流程、开发环境快速搭建等场景。

传统虚拟机：更适合需要更高安全性和资源隔离级别的场景，例如生产环境中的关键业务应用或托管敏感数据的应用程序。

（6）管理复杂度

Docker容器：由于容器轻量级的特性，管理和扩展相对简单。

传统虚拟机：管理多个VM可能会更加复杂，尤其是在配置和维护操作系统层面的问题时。

总的来说，Docker容器提供了一种轻量级的隔离方法，特别适合于现代的云原生和微服务架构，而传统虚拟机则适用于需要更强隔离性和安全性保证的场合。

#### 3.Docker镜像（Image）是什么？

答案：Docker镜像是创建容器的一个只读模版，包含应用程序及其运行所需的全部文件和依赖，是只读的，可通过构建过程从Dockerfile创建。

#### 4.Docker容器（Container）是什么？

答案：Docker容器是镜像的一个运行实例，它是启动时从镜像创建的可写的运行时环境，包含应用程序及其运行时环境。

#### 5.Docker容器的生命周期管理包括哪些操作？

答案：包括创建（create）、启动（start）、停止（stop）、重启（restart）、删除（rm）和查看状态（ps）等操作。

#### 6.Docker中的“端口映射”是什么意思？

答案：端口映射是将宿主机的一个端口映射到容器内的端口，使得外界可以通过宿主机的地址和端口访问到容器内的服务。使用-p或–publish选项实现，例如docker run -p 8080:80 nginx会将宿主机的8080端口映射到容器的80端口。

#### 7.Dockerfile是什么？

答案：Dockerfile是一个文本文件，包含了用户可以调用的所有命令，用于自动化构建Docker镜像的过程。它定义了从基础镜像开始，到最终形成所需运行环境的每一步操作。

#### 8.Docker的优点都有哪些

启动速度快（秒级）

计算机能力几乎没有损耗

性能接近原生

单台主机能支持上千个容器的运行

通过dockerfile可以自动化部署容器，方便、灵活

#### 9.Dockerfile常用指令都有哪些？

答：

FROM 指定基础镜像（必须为第一个指令，因为需要指定使用哪个基础镜像来构建镜像）；

MAINTAINER 设置镜像作者相关信息，如作者名字，日期，邮件，联系方式等；

COPY 复制文件到镜像；

ADD 复制文件到镜像（ADD与COPY的区别在于，ADD会自动解压tar、zip、tgz、xz等归档文件）；

ENV 设置环境变量；

EXPOSE 暴露容器进程的端口，仅仅是提示别人容器使用的哪个端口，没有过多作用；

VOLUME 数据卷持久化，挂载一个目录；

WORKDIR 设置工作目录，如果目录不在，则会自动创建目录；

RUN 在容器中运行命令，RUN指令会创建新的镜像层，RUN指令经常被用于安装软件包；

CMD 指定容器启动时默认运行哪些命令，如果有多个CMD，则只有最后一个生效，另外，CMD指令可以被docker run之后的参数替换；

ENTRYOINT 指定容器启动时运行哪些命令，如果有多个ENTRYOINT，则只有最后一个生效

#### 10.COPY和ADD命令在Dockerfile中有什么区别?

答:

一般来说，虽然COPY的AD和COPY功能相似，但COPY是首选。那是因为它比ADD更容易理解。COPY只支持将本地文件复制到容器中，而ADD具有一些不明显的功能(如局部TAR提取和远程URL支持)。因此，ADD的最佳用途是自动将本地TAR文件提取到镜像中

#### 11.什么是Docker容器编排？

答案：Docker容器编排是一种管理多个容器及其复杂部署、网络配置、存储卷以及服务发现的自动化技术。它确保容器能够在集群环境中高效、可靠地运行。常见的容器编排工具包括Docker Compose、Kubernetes和Swarm。

#### 12.Docker Compose是什么？它的主要用途是什么？

答案：Docker Compose是一个用于定义和运行多容器Docker应用的工具。通过一个YAML文件来配置服务、网络和卷，用户可以一次性启动或停止整个应用的所有服务，简化了开发和部署流程

#### 13.Docker Compose的基本命令有哪些？

答案：基本命令包括：

docker-compose up：启动或重建（如果已存在）所有服务。

docker-compose down：停止并移除服务、网络、卷和镜像（默认保留匿名卷）。

docker-compose ps：列出所有服务的状态。

docker-compose build：构建服务镜像。

docker-compose run 服务名：在新容器中运行指定服务的一次性命令。

#### 14.Docker私有仓库的作用是什么？

答案：Docker私有仓库允许组织在内部网络中存储和分发Docker镜像，提高了安全性、减少了对外部网络的依赖，并且可以控制访问权限，加速镜像的拉取速度。

#### 15.Docker 的网络通信模式。

答∶

（1）host 模式，使用 --net=host 指定。

和宿主机共用一个 Network Namespace。容器将不会虚拟出自己的网卡，配置自己的 IP 等，而是使用宿主机的 IP 和端口。

（2）container 模式，使用 --net=container∶NAMEorlD 指定。

指定新创建的容器和已经存在的一个容器共享一个 Network Namespace，而不是和宿主机共享。

（3）none 模式，使用 --net=none 指定。

告诉 docker 将新容器放到自己的网络堆栈中，但是不要配置它的网络。

（4）bridge 模式，使用 --net=bridge 指定，默认设置。

bridge 模式是 Docker 默认的网络设置，此模式会为每一个容器分配 Network Namespace、设置IP等，并将一个主机上的Docker容器连接到一个虚拟网桥上。

#### 16.Docker Swarm是什么？

答案：Docker Swarm是Docker的原生集群管理系统，用于将多个Docker主机组织成一个单一的虚拟系统，从而可以轻松地在多台机器上部署和管理容器化应用。

#### 17.Swarm中的角色有哪些？

答案：Swarm中有两种角色：管理节点（Manager Node）负责集群的管理任务，如服务部署、维护集群状态；工作节点（Worker Node）则执行实际的容器运行任务。

#### 18.Swarm与Kubernetes相比有哪些特点？

答案：Swarm是Docker的原生解决方案，与Docker集成紧密，操作相对简单。相比之下，Kubernetes功能更丰富，支持更复杂的部署场景，社区更活跃，但学难度较大。

#### 19.如何扩缩Swarm服务的副本数量？

答案：使用docker service scale命令，如docker service scale my\_web=5将服务my\_web的副本数增加到5个。

#### 20.如何加入一个Swarm集群作为工作节点？

答案：使用docker swarm join命令，并提供由管理节点生成的令牌和管理节点的地址，即可使Docker主机加入Swarm作为工作节点。

#### 21.Docker中如何查看容器的日志？

答案：使用`docker logs [容器ID或名称]`命令查看容器的日志输出。

#### 22.Logstash的作用是什么？

答案：Logstash是一个数据处理管道，负责接收、转换和转发日志数据到Elasticsearch或其它存储系统。

#### 23.Filebeat如何与Docker集成收集日志？

答案：Filebeat通常作为sidecar容器部署，配置为监听Docker容器的日志文件或使用Docker API收集日志，然后转发给Logstash或直接到Elasticsearch。

#### 24.如何在Docker容器中安装和运行Filebeat？

答案：通过编写Dockerfile或使用Docker-compose，将Filebeat配置文件和容器日志路径映射进容器，并启动Filebeat服务。

#### 25.Docker日志如何实现持久化存储？

答案：通过配置日志驱动如使用syslog或通过volume挂载日志文件到宿主机的持久存储卷上。

#### 26.什么是Containerd？

答案：Containerd是一个行业标准的容器运行时，主要负责容器的创建、启动、停止、删除等生命周期管理，以及容器镜像的拉取、存储和分发。它为更高层的工具（如Kubernetes）提供低级别的容器操作接口。它专注于提供在宿主机上运行容器的基础功能，如容器生命周期管理、镜像管理等。它是CNCF托管的项目，广泛应用于Kubernetes等容器编排平台。

#### 27.Containerd和Docker的关系是什么？

答案：从Docker 1.11开始，Docker引入了Containerd作为其核心容器运行时组件，负责容器和镜像的管理。现在，Containerd可以独立于Docker运行，为其他容器管理工具提供服务。

#### 28.Containerd的架构包含哪些主要组件？

答案：Containerd的核心架构包括一个守护进程（containerd-shim-runc-v2），用于管理容器和执行runc；一个API服务器，用于接收外部请求；以及一个镜像服务（containerd’s image service），负责镜像的存储和分发。

#### 29.Containerd的性能优势是什么？

答案：Containerd作为一个轻量级的容器运行时，专为大规模部署优化，提供了高效的容器启动速度、低资源消耗和稳定的性能表现，特别适合在大规模容器编排平台中使用。

#### 30.Containerd如何管理容器网络？

答案：Containerd本身不直接管理网络，而是通过与CNI（Container Network Interface）兼容的网络插件（如Flannel、Calico）集成来实现容器网络的配置和管理。

#### 31.简述Kubernetes是什么？

答案：Kubernetes是一个开源的容器编排系统，用于自动化部署、扩展和管理容器化应用程序。它能够跨多个主机节点上运行容器，确保它们高

![](https://csdnimg.cn/release/blogv2/dist/pc/img/lock.png)最低0.47元/天 解锁文章![](https://img-home.csdnimg.cn/images/20240516053626.png)

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png)

确定要放弃本次机会？


福利倒计时

_:_ _:_

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png)立减 ¥

普通VIP年卡可用

[立即使用](https://mall.csdn.net/vip)

[![](https://i-avatar.csdnimg.cn/e43b25a5601f4f7fa909be0613d8c944_weixin_41737291.jpg!1)\\
郑州课工场-王俊卿](https://blog.csdn.net/weixin_41737291)

关注关注

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png)
26

点赞

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png)
踩

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png)
16




收藏







觉得还不错?

一键收藏
![](https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png)

- [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png)\\
0](https://blog.csdn.net/weixin_41737291/article/details/148538005#commentBox)
评论

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/share.png)分享




复制链接



分享到 QQ



分享到新浪微博









![](https://csdnimg.cn/release/blogv2/dist/pc/img/share/icon-wechat.png)扫一扫


- ![打赏](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/reward.png)打赏
打赏

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/more.png)


![打赏](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/reward.png)打赏![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报



![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报


[【k8s面试】超详细 _kubernetes_ _面试题_ 总结，面试必问!（附200道K8s _Docker_ 面试真题 _+_ 答案详解(1)](https://blog.csdn.net/2401_83620927/article/details/138735528)

[2401\_83620927的博客](https://blog.csdn.net/2401_83620927)

05-12![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1187


[最近很多小伙伴找我要Linux学习资料，于是我翻箱倒柜，整理了一些优质资源，涵盖视频、电子书、PPT等共享给大家！](https://blog.csdn.net/2401_83620927/article/details/138735528)

[【k8s面试】超详细 _kubernetes_ _面试题_ 总结，面试必问!（附200道K8s _Docker_ 面试真题 _+_ 答案详解](https://blog.csdn.net/2401_83620959/article/details/138735559)

[2401\_83620959的博客](https://blog.csdn.net/2401_83620959)

05-12![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1344


[最近很多小伙伴找我要Linux学习资料，于是我翻箱倒柜，整理了一些优质资源，涵盖视频、电子书、PPT等共享给大家！](https://blog.csdn.net/2401_83620959/article/details/138735559)

参与评论您还未登录，请先登录后发表或查看评论

[_Kubernetes_ 与 _Docker_ 相关 _面试题_ _汇总_\_ _docker_ _容器_ 化技术以及 _kubernetes_ 容...](https://blog.csdn.net/m0_57836225/article/details/142427859)

4-24

[_Kubernetes_ 可以集成各种日志收集工具,如 Fluentd、Logstash 等,将 _容器_ 的日志收集到集中的日志存储中。 可以使用 Prometheus、Grafana 等监控工具来监控 _Kubernetes_ 集群的性能和资源使用情况。 故障恢复和自我修复: _Kubernetes_ 中的控制器会自动监测 Pod 的状态,如果 Pod 出现故障,会自动重新创建新的 Pod。 使用存活探针...](https://blog.csdn.net/m0_57836225/article/details/142427859)

[_运维_ 笔记 \-\- _docker_、 _kubernetes_\_大神wushirenfei](https://blog.csdn.net/duxiangwushirenfei/article/details/82843318)

4-18

[感觉现在多数互联网公司就在朝着 _容器_ 化的路上前行,大势所趋啊。之前陆续有过 _docker_ 相关记录,管理方式也是 _docker_-compose形式,其实对于一般小量级的 _docker_-compose管理足以应付。但是如果追求更高些, _kubernetes_ 是绕不过去的。倘若 _docker_ 不是太熟悉,可以参见之前的 _docker_ 笔记一, _docker_ 笔记二。](https://blog.csdn.net/duxiangwushirenfei/article/details/82843318)

[_Docker_/K8s 常见 _面试题_ 整理\\
\\
最新发布](https://blog.csdn.net/weixin_43795588/article/details/148857626)

[weixin\_43795588的博客](https://blog.csdn.net/weixin_43795588)

06-24![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2142


[以上问题覆盖了 _Docker_/K8s 面试的核心考点，建议结合原理理解和实战经验（如亲手部署集群、调试 Pod）加深记忆。面试中可能会结合具体场景追问，例如 “如何优化 _Docker_ 镜像构建速度？” 或 “K8s 节点故障时 Pod 如何迁移？”，需灵活运用知识回答。](https://blog.csdn.net/weixin_43795588/article/details/148857626)

[_docker_ _+_ k8s相关 _面试题_](https://devpress.csdn.net/v1/article/detail/136807234)

[qq\_51537858的博客](https://blog.csdn.net/qq_51537858)

03-18![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
4244


[_Kubernetes_（通常简称为K8s）是一个开源的 _容器_ 编排平台，用于自动化应用程序的部署、扩展和管理。它最初由Google开发，并于2014年发布为开源项目，现在由Cloud Native Computing Foundation（CNCF）进行维护。 _Kubernetes_ 旨在解决在分布式系统中部署和管理 _容器_ 化应用程序时遇到的挑战。下面是一些关键概念和组件，帮助你更好地理解 _Kubernetes_：节点（Nodes）：节点是运行 _Kubernetes_ 的物理或虚拟机器。每个节点都可以托管一个或多个 _容器_。](https://devpress.csdn.net/v1/article/detail/136807234)

[_Docker_ 与 _Kubernetes_ 实战](https://blog.csdn.net/w405722907/article/details/80943467)

4-2

[_docker_ \[CMD\] \[options\] 基本命令与释义 attach进入到正在运行的 _容器_ build由 _Docker_ file构建镜像 commit由 _容器_ 的改变创建一个新的镜像 cp在 _容器_ 中复制文件或文件夹到本地文件或文件夹中. logs获取 _容器_ 日志 network管理 _Docker_ 网络 node管理 _Docker_ 集群节点 pause暂停一个或多个 _容器_ 内的所有进程 ...](https://blog.csdn.net/w405722907/article/details/80943467)

[_Docker_ 和K8s _面试题_](https://blog.csdn.net/wy990880/article/details/147578389)

4-12

[_docker_ attach是attach到 _容器_ 启动命令的终端。 _docker_ exec是在 _容器_ 内启动一个新的TTY终端。 🌟🌟🌟K8s _面试题_ 😊😊😊😊 10.什么是k8s?说出你的理解 K8s是 _kubernetes_ 的简称、其本质是一个开源的 _容器_ 编排系统、主要用于管理 _容器_ 化的应用、其目标是让部署 _容器_ 化的应用简单并且高效(powerful)。 _Kubernetes_ 提供了应...](https://blog.csdn.net/wy990880/article/details/147578389)

[_docker_ 和k8s _面试题_ 总结（未完待续）](https://blog.csdn.net/weixin_56752399/article/details/120903554)

[临江仙我亦是行人](https://blog.csdn.net/weixin_56752399)

10-22![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[_docker_ 有四种网络模式none桥接（默认模式）Hostcontainer _容器_ 之间通过桥接模式进行通信跨主机的 _容器_ 之间通过静态路由进行通信，A主机的 _容器_ 的下一跳指向B主机，这样B主机接收到请求解包后转发给本机的 _容器_ _容器_ 其实就是 Namespace _+_ CgroupsNamespace：帮助 _容器_ 实现各种资源的隔离Cgroups：主要对 _容器_ 使用某种资源量的多少做一个限制 _docker_ 使用宿主机的内核，启动速度快，隔离性差， _docker_ 确切来说是一个 _容器_ 引擎虚拟机使用自己的内核，启动速度慢，因为有自己的内核](https://blog.csdn.net/weixin_56752399/article/details/120903554)

[_Docker_ 和K8s _面试题_](https://devpress.csdn.net/v1/article/detail/147578389)

[wy990880的博客](https://blog.csdn.net/wy990880)

04-28![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1639


[K8s是 _kubernetes_ 的简称、其本质是一个开源的 _容器_ 编排系统、主要用于管理 _容器_ 化的应用、其目标是让部署 _容器_ 化的应用简单并且高效（powerful）。 _Kubernetes_ 提供了应用部署、规划、更新、维护的一种机制。说简单点：K8s就是一个编排 _容器_ 的系统、一个它可以管理 _容器_ 应用全生命周期的工具、从创建应用、应用的部署、应用提供服务、扩容缩容应用、应用更新、都非常的方便、而且还可以做到故障自愈所以K8s是一个非常强大的 _容器_ 编排系统。](https://devpress.csdn.net/v1/article/detail/147578389)

[_docker_ 与k8s _面试题_ 基础\_vmware、 _docker_、 _kubernetes_ 的 _面试题_](https://blog.csdn.net/kali_yao/article/details/120605132)

5-2

[12. _docker_ 自定义镜像,如果写了多条CMD,那么默认执行时会执行哪一条命令呢? 13. _docker_ 如何做持久化存储 14. _容器_ 的六大命名空间 15. _docker_ fifile中ADD和Copy区别 16. _docker_ 命令 17. _docker_ 怎样实现 _容器_ 件的独立 18.什么是ELK 19.微服务: 20.K8S架构:](https://blog.csdn.net/kali_yao/article/details/120605132)

[【 _Kubernetes_】常见 _面试题_ _汇总_(三十八)](https://blog.csdn.net/Songyaxuan075118/article/details/142513857)

3-24

[题目1-68 属于【 _Kubernetes_】的常规概念题,即“ _汇总_(一)~(二十二)”。 题目69-113 属于【 _Kubernetes_】的生产应用题。 91. _Docker_ 的网络通信模式。 _Docker_ 的 4 种网络通信模式: 1、host 模式: - host 模式,使用 --net=host 指定。 - 和宿主机共用一个 NetworkNamespace。](https://blog.csdn.net/Songyaxuan075118/article/details/142513857)

[_Docker_&k8s常见 _面试题_](https://blog.csdn.net/qq_57207718/article/details/127597153)

[qq\_57207718的博客](https://blog.csdn.net/qq_57207718)

10-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3693


[_docker_ & k8s常见 _面试题_](https://blog.csdn.net/qq_57207718/article/details/127597153)

[_Kubernetes_ 招聘 _面试题_ _汇总_](https://download.csdn.net/download/beck_li/89741326)

09-12

[简述 _Kubernetes_ 和 _Docker_ 的关系？ 3 简述 _Kubernetes_ 中什么是Minikube、Kubectl、Kubelet？ 3 简述 _Kubernetes_ 常见的部署方式？ 3 简述 _Kubernetes_ 如何实现集群管理？ 3 简述 _Kubernetes_ 的优势、适应场景及其特点？ 3 ...](https://download.csdn.net/download/beck_li/89741326)

[【k8s面试】超详细 _kubernetes_ _面试题_ 总结,面试必问!(附200道K8s _Docker_...](https://blog.csdn.net/2401_83620959/article/details/138735546)

4-11

[(附200道K8s _Docker_ 面试真题 _+_ 答案详解(2) 先自我介绍一下,小编浙江大学毕业,去过华为、字节跳动等大厂,目前在阿里 深知大多数程序员,想要提升技能,往往是自己摸索成长,但自己不成体系的自学效果低效又漫长,而且极易碰到天花板技术停滞不前! 因此收集整理了一份《2024年最新Linux _运维_ 全套学习资料》,初衷也很简单,就...](https://blog.csdn.net/2401_83620959/article/details/138735546)

精选资源 [_kubernetes_（2021-12月整理k8s经典面试常问题目） _面试题_ _汇总_.pdf](https://download.csdn.net/download/qq_39418469/61159944)

12-11

[_Kubernetes_（简称k8s）是Google开源的 _容器_ 编排...以上是 _Kubernetes_ 面试中可能遇到的一些核心问题和解答，涵盖了基础概念、架构、管理和监控等多个方面。理解并掌握这些知识点对于成为高级 _运维_ 或 _运维_ 开发者至关重要。](https://download.csdn.net/download/qq_39418469/61159944)

[_Kubernetes_ 自测题](https://blog.csdn.net/flynetcn/article/details/120047736)

[flynetcn的专栏](https://blog.csdn.net/flynetcn)

09-01![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
546


[k8s 100问](https://blog.csdn.net/flynetcn/article/details/120047736)

[《两地书》-- _Kubernetes_(K8s)基础知识( _docker_ _容器_ 技术)](https://blog.csdn.net/weixin_34055910/article/details/86261329)

[weixin\_34055910的博客](https://blog.csdn.net/weixin_34055910)

05-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2414


[大家都知道历史上有段佳话叫“司马相如和卓文君”。“皑如山上雪，皎若云间月”。卓文君这么美，却也抵不过多情女儿薄情郎。\\
司马相如因一首《子虚赋》得汉武帝赏识，飞黄腾达之后便要与卓文君“故来相决绝”，寄来给家乡留守的妻子一封《两地书》，上面只有一行数字：“一二三四五六七八九十百千万。”意义是：无亿，我已经无意于你啦。\\
卓文君看了这封信也不示弱，回了一首《怨郎诗》，司马相如看了发现虽然我...](https://blog.csdn.net/weixin_34055910/article/details/86261329)

[_docker_ _面试题_ 和解答(一)](https://devpress.csdn.net/v1/article/detail/102361251)

[weixin\_30606461的博客](https://blog.csdn.net/weixin_30606461)

08-21![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
4036


[什么 _Docker_ _Docker_ 是一个 _容器_ 化平台，它以 _容器_ 的形式将您的应用程序及其所有依赖项打包在一起，以确保您的应用程序在任何环境中无缝运行。\\
_Docker_ 与虚拟机有何不同\\
_Docker_ 不是虚拟化方法。它依赖于实际实现基于 _容器_ 的虚拟化或操作系统级虚拟化的其他工具。为此， _Docker_ 最初使用LXC驱动程序，然后移动到libcontainer现在重命名为runc。 _Docker_ 主要专注于在应...](https://devpress.csdn.net/v1/article/detail/102361251)

[40道常见的 K8S _面试题_ 总结\\
\\
热门推荐](https://mingongge.blog.csdn.net/article/details/100613465)

[民工哥的博客](https://blog.csdn.net/mingongge)

09-07![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
10万+


[点击上方“民工哥技术之路”选择“星标”每天10点为你分享不一样的干货读者福利！多达 2048G 各种资源免费赠送作者：fiisio译文：https://zhuanlan...](https://mingongge.blog.csdn.net/article/details/100613465)

[_Kubernetes_ 面试 11 问：面试官想考察什么？](https://blog.csdn.net/M2l0ZgSsVc7r69eFdTj/article/details/89049292)

[Docker的专栏](https://blog.csdn.net/M2l0ZgSsVc7r69eFdTj)

04-05![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
703


[文章来源：K8sMeetup社区ID： _Kubernetes_ china2近几年， _Kubernetes_ 相关工作岗位在互联网圈中越来越火爆。很多开发人员都因其发展前景与薪资而...](https://blog.csdn.net/M2l0ZgSsVc7r69eFdTj/article/details/89049292)

[【虚拟化】 _Kubernetes_ 面试100题](https://xinzhe.blog.csdn.net/article/details/98476793)

[小哲的博客](https://blog.csdn.net/ARPOSPF)

08-05![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3813


[_Kubernetes_ 面试100题\\
\\
_Kubernetes_ 是什么？ _kubernetes_，简称K8s，是用8代替8个字符“ubernete”而成的缩写。是一个开源的，用于管理云平台中多个主机上的 _容器_ 化的应用， _Kubernetes_ 的目标是让部署 _容器_ 化的应用简单并且高效（powerful）, _Kubernetes_ 提供了应用部署，规划，更新，维护的一种机制。\\
\\
Q： _Kubernetes_ 是什么？\\
\\
A：Kube...](https://xinzhe.blog.csdn.net/article/details/98476793)

[_kubernetes_ 和 _Docker_ —— _kubernetes_ 实用随笔（一）](https://blog.csdn.net/whdxjbw/article/details/80739529)

[little prince，blue coder](https://blog.csdn.net/whdxjbw)

06-25![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3万+


[系列文章： \\
\\
_kubernetes_ 和 _Docker_ —— _kubernetes_ 实用随笔（一）\\
\\
最常用的kubectl命令(附带场景和截图) —— _kubernetes_ 实用随笔（二）\\
\\
_kubernetes_ 核心对象 —— _kubernetes_ 实用随笔（三）\\
\\
最近项目用到 _kubernetes_（以下简称k8s，k和s之间有8个字母），虽然之前也有简单使用过，但最近发现k8s概念较多，命...](https://blog.csdn.net/whdxjbw/article/details/80739529)

[_kubernetes_ _面试题_ _汇总_](https://blog.csdn.net/kingzdd/article/details/106076969)

[zhangdd的博客](https://blog.csdn.net/kingzdd)

05-12![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1855


[注：以下所有问题，均为自己总结，若有错误之处，还请指出。\\
1、 k8s是什么？请说出你的了解？\\
\\
答：Kubenetes是一个针对 _容器_ 应用，进行自动部署，弹性伸缩和管理的开源系统。主要功能是生产环境中的 _容器_ 编排。\\
K8S是Google公司推出的，它来源于由Google公司内部使用了15年的Borg系统，集结了Borg的精华。\\
2、 K8s架构的组成是什么？\\
\\
答：和大多数分布式系统一样，K8S集群至少需要一个主节点（Master）和多个计算节点（Node）。\\
\\
主节点主要用于暴露API，调度部署和节..](https://blog.csdn.net/kingzdd/article/details/106076969)

[60道重要的 _Kubernetes_ _面试题_](https://blog.csdn.net/M2l0ZgSsVc7r69eFdTj/article/details/117267374)

[Docker的专栏](https://blog.csdn.net/M2l0ZgSsVc7r69eFdTj)

05-25![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1769


[简述etcd及其特点etcd是CoreOS团队发起的开源项目，是一个管理配置信息和服务发现（service discovery）的项目，它的目标是构建一个高可用的分布式键值（key-val...](https://blog.csdn.net/M2l0ZgSsVc7r69eFdTj/article/details/117267374)

[_docker_ 及k8s _容器_ 面试精华 _汇总_（一），祝大家顺利通过企业面试！](https://devpress.csdn.net/v1/article/detail/108117341)

[栗神的博客](https://blog.csdn.net/bj_licq)

08-20![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2483


[_docker_ 及k8s _容器_ 面试精华 _汇总_，希望可以加深大家对 _容器_ 的理解，祝大家顺利通过企业面试。\\
一、如何在 _Kubernetes_ 中实现负载均衡？\\
node中有kube-proxy，他可以提供负载均衡。每个 Node 都会运行 kube-proxy 服务，它负责将访问 service 的 TCP/UPD 数据流转发到后端的 _容器_。如果有多个副本，kube-proxy 会实现负载均衡。\\
二.在生产中，你如何实现 _Kubernetes_ 自动化？\\
1.日志：严重依赖日志，与任何分布式系统一样，日志的精准定位会提供非常](https://devpress.csdn.net/v1/article/detail/108117341)

[面试之—K8S、 _Docker_ _面试题_ 整理](https://blog.csdn.net/Mr_XiMu/article/details/125628672)

[Mr\_XiMu的博客](https://blog.csdn.net/Mr_XiMu)

07-05![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2996


[面试之—K8S、 _Docker_ _面试题_ 整理](https://blog.csdn.net/Mr_XiMu/article/details/125628672)

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

打赏作者![](https://csdnimg.cn/release/blogv2/dist/pc/img/closeBt.png)

[![](https://i-avatar.csdnimg.cn/e43b25a5601f4f7fa909be0613d8c944_weixin_41737291.jpg!1)](https://blog.csdn.net/weixin_41737291)

郑州课工场-王俊卿

你的鼓励将是我创作的最大动力

¥1¥2¥4¥6¥10¥20

扫码支付：¥1

![](https://csdnimg.cn/release/blogv2/dist/pc/img/pay-time-out.png)获取中

![](https://csdnimg.cn/release/blogv2/dist/pc/img/newWeiXin.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newZhiFuBao.png)扫码支付

您的余额不足，请更换扫码支付或 [充值](https://i.csdn.net/#/wallet/balance/recharge?utm_source=RewardVip)

打赏作者

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

![](https://blog.csdn.net/weixin_41737291/article/details/148538005)

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