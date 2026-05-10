# 云原生k8s高频面试题 - 博客园

URL: https://www.cnblogs.com/cat1/p/18812637

🖖\|

[收藏](https://wz.cnblogs.com/) [闪存](https://ing.cnblogs.com/) [小组](https://group.cnblogs.com/) [博问](https://q.cnblogs.com/)

[没有猫的猫奴](https://www.cnblogs.com/cat1) \+ 关注

园龄：5年9个月 [粉丝：1](https://home.cnblogs.com/u/cat1/followers) [关注：3](https://home.cnblogs.com/u/cat1/followees)

# [云原生k8s高频面试题](https://www.cnblogs.com/cat1/p/18812637 "发布于 2025-04-07 14:10")

**1\. k8s service有哪些类型？**

```yaml

通过创建service，可以为一组具有相同功能的容器应用提供一个统一的入口地址，并且将请求负载分发到后端的各个容器应用上。其主要类型有：
ClusterIP：虚拟的服务IP地址，该地址用于k8s集群内部的pod访问，在Node上kube-proxy通过设置的iptables规则进行转发；
NodePort: 使用宿主机的端口，是能够访问各Node的外部客户端通过Node的IP地址和端口就能访问服务；
LoadBalancer：使用外部负载均衡骑完成到服务的负载分发，需要在spec.status.loadBalancer字段指定外部负载均衡器的IP地址，通常用于公有云。
```

**2\. K8S标签与标签选择器的作用是什么？**

```yaml
在k8s中，标签（Label）和标签选择器（Label Selector）适用于表示和筛选对象（如pod、service）的一种机制。它们在k8s管理和组织资源方面起着重要作用。
标签（Labels）：
标签是附件在杜伊向上的键值对，可以用于标识和组织对象。标签没有固定的意义，它们可以根据用户的需要任意设定。常见的标签用途包括：
	- 环境：区分不同的环境，如`env=prd`或者`env=dev`
	- 应用：标识应用程序，如`app=nginx`
	- 版本：标识版本，如`version=1.0`
	- 角色：标识角色，如`role=frontend`或`role=backend`
```

```yaml
标签选择器（Label Selectors）:
标签选择器用于U型安泽一组具有特定标签的对象。它们在k8s中广泛用于各种控制器和服务，以便根据标签筛选和操作对象。标签选择器主要有两种类型：等式选择器和集合选择器。

等式选择器：
等式选择器通过键和值的匹配来选择对象。
 - 等于（= 或 ==）
 - 不等于（！=）

集合选择器：
集合选择器用于选择一组匹配特定条件的标签
 - 包含（in）
 - 不包含（notin）
 - 存在（exists）
```

```yaml
作用与优势
（1）灵活管理：通过标签，用户可以灵活地对资源进行分类和管理。
（2）动态分组：标签选择器允许根据标签动态选择一组对象，便于自动化和扩展操作。
（3）简化配置：通过标签和选择器，可以简化复杂环境的配置管理，避免编码具体的对象名称。
总之，标签与标签选择器是k8s中强大的机制，帮助用户邮箱的组织和管理集群中的资源。
```

**3\. KUBERNETES 如何实现集群管理？**

```yaml
k8s实现集群管理主要依靠以下几个核心组件和概念：

（1）kubernetes Master Node（主节点）：
主节点负责管理整个集群，处理集群的所有控制请求。它包含以下关键组件：
 - API Server：k8s API的前端，所有的操作（如创建、更新、删除资源）都通过API Server进行。
 - Etcd：一个分布式键值存储，用于保存集群的所有配置信息和状态数据。
 - Controller Manager：负责管理控制器的组件，控制器用于确保集群处于期望的状态。
 - Scheduler：负责将新建的pod分配到合适的节点上。

（2）kubernetes Node（工作节点）
工作节点运行pod，并由以下关键组件组成：
 - kubelet：主节点与工作节点之间的代理，负责接收来自API Server的pod规范并管理容器的生命周期。
 - kube-proxy：维护网络规则，允许pod之间的网络通信和pod与服务之间的网络通信。
 - Container Runtime：运行容器的基础组件，如Docker、containerd等。

（3）Pod
pod是kubernetes中最小的单元，通常包含一个或多个容器。pod共享网络和存储资源，并共同调度。

（4）ReplicaSet
ReplicaSet 确保指定数量的pod副本在集群中运行。如果某个pod出现故障，ReplicaSet会创建新的pod来代替它。

（5）Deployment
Deployment是一种更高级的资源，用于管理应用的声明式更新和版本控制。它能够创建和管理ReplicaSet，实现滚动更新和回滚。

（6）Service
Service 提供了一种稳定的方式来访问一组pod。它为一组pod提供一个单一的访问入口，并通过标签选择器来确定后端pod。

（7）ConfigMap 和 Secret
ConfigMap 和 Secret 用于在pod中注入配置信息和敏感数据（如密码、秘钥等），从而实现配置与代码的分离。

（8）Namespace
Namespace 用于在同一个物理集群中创建多个虚拟集群，以实现资源的隔离和管理。不同的团队或项目可以使用不同的Namespace来避免资源冲突。

（9）Ingress
Ingress管理外部集群访问到集群内部的路由规则，通常用于HTTP和HTTPS流量。
```

**4\. 如何解释k8s架构组件之间的不同？**

```yaml
k8s由两层组成：控制平面和数据平面
控制平面是容器编排层包括：
（1）控制集群的k8s对象。
（2）有关集群状态和配置的数据。
数据平面是处理数据请求的层，由控制平面管理。
```

**5\. kube-proxy的作用？**

```yaml
kube-proxy 运行在所有节点上，它监听apiserver和service和endpoint的变化情况，创建路由规则以提供服务IP和负载均衡功能。简单理解此进程是Service的透明代理兼负载均衡器，其主要功能是将某个service的访问请求转发到后端的多个pod实例上。
```

**6\. 为什么需要k8s，它能做什么？**

```yaml
k8s是一个开源的容器编排平台，用于自动化容器化应用的部署、扩展和管理，它解决了容器化应用在生产环境中面临的一些主要挑战，提供了一套完整的解决方案。以下是k8s的一些关键功能和优势：

为什么需要k8s？

（1）自动化操作
	- 自动化部署和恢复：k8s可以自动部署应用并在节点或容器失败时自动恢复。
	- 滚动更新和回滚：k8s支持以滚动方式更新应用，同时能够在出现问题时回滚到以前的版本。
（2）弹性和扩展性
	- 自动扩展：根据流量和负载自动调整应用的副本数量，确保资源的高效利用。
	- 负载均衡：自动分配流量到多个实例，实现高可用性。
（3）资源管理
	- 资源分配和调度：根据资源需求和策略将应用调度到最合适的节点上，优化资源利用。
	- 命名空间隔离：通过命名空间对不同团队和项目进行资源隔离和管理。
（4）易于维护和扩展
	- 声明式配置：使用声明式配置文件管理基础设施和应用，使得配置管理更加简单和一直。
	- 可扩展的架构：通过插件和自定义资源定义（CRD），可以扩展k8s的功能，满足不同的需求。
（5）一致的环境
	- 跨环境一致性：无论是在开发、测试环境还是生产环境中，k8s都能够提供一致的运行环境，减少环境差异带来的问题。

k8s能做什么？

（1）容器编排
	- pod管理：k8s将容器打包成pod进行管理，一个pod通常包含一个或多个紧密相关的容器。
	- 自动重启和恢复：k8s会自动监控和重启失败的容器，确保应用的高可用性。
（2）服务发现和负载均衡
	- 服务发现：k8s自动为每个pod分配IP地址，并提供DNS名称，使服务发现变得简单。
	- 负载均衡：k8s会将流量分配到不同的pod，实现流量的负载分布。
（3）存储编排
	- 持久化存储：通过pv和pvc，k8s提供持久化存储，确保不会因为pod的重启或迁移而丢失。
	- 存储集成：支持多种存储后端，如本地存储、NFS、云存储等。
（4）配置管理和秘钥管理
	- configMap和secret：k8s提供configmap和secret，用于管理应用的配置数据和敏感信息，并将其注入到pod中。
（5）安全性和访问控制
	- 身份认证和授权：支持多种身份认证方式，并通过角色和权限控制（RBAC）管理用户访问权限。
	- 网络策略：通过网络策略控制pod之间的网络访问，实现更细粒度的安全控制。
（6）监控和日志记录
	- 内置监控和日志：k8s提供对集群和应用的监控和日志记录功能，集成了prometheus、grafana等流行的工具。
	- 健康检查和自我修复：通过探针监控应用的健康状态，自动重启异常的容器。

总结：k8s提供了一套强大的工具集，帮助开发和运维团队高效管理和扩展容器化应用。它通过自动化操作、资源管理和一致性运行环境，提高了应用的可靠性和可扩展性，同时减少了手动操作和运维复杂性。
```

**7\. 什么是容器编排？**

```yaml
容器编排是与运行容器相关的组件和流程的自动化，它包括诸如配置和调度容器、容器的可用性、容器之间的资源分配以及保护容器之间的交互等内容。
```

**8\. daemonset、deployment、replication之间有什么区别？**

```yaml
（1）daemonset：确保每一个节点都运行pod的一个副本，适用于守护进程。
（2）deployment：用于无状态应用的部署和管理，支持滚动更新和回滚。
（3）ReplicaSet：用于确保指定数量的pod副本运行，通常由Deployment管理。
（4）statefulSet：用于有状态应用的部署和管理，确保pod的稳定性、持久性和顺序。
```

**9\. k8s-中镜像的下载策略是什么**

```yaml
k8s的镜像下载策略有以下三种：Always、Never、ifNotPresent
 - Always：镜像标签为latest时，总是去指定的仓库获取镜像。
 - Never：禁止从仓库中下载镜像，也就是说只能使用本地镜像。
 - ifNotPresent：仅当本地没有对应的镜像时，才从目标仓库中下载。
```

**10\. 删除一个pod会发生什么事情？**

```yaml
kube-apiserver 会接收到用户的删除指令，默认有30秒时间等待优雅退出，超过30s会被标记为死亡状态，此时的pod处于Terminating，kubelet看到pod被标记为Terminating就开始了关闭pod的工作

关闭流程：
	- pod从service的endpoint列表中移除
	- 如果该pod定义了一个停止的钩子，其会在pod内部被调用，停止钩子一般定义了如何优雅的结束进程
	- 进程被发送TERM信号（kill -14）
	- 超过优雅退出的时间后，pod中所有进程都会被发送sigkill信号（kill -9）
```

**11\. k8s scheduler的作用是什么？**

```yaml
scheduler 的主要作用是将新创建的pod分配到合适的节点上。scheduler决定每个pod应该运行到哪个节点上，以满足资源需求、策略约束和其它调度的要求。它的主要任务包括：
 - 资源分配：根据pod的资源需求（如CPU和内存），选择满足条件的节点。
 - 负载均衡：将pod分布都不同的节点上，避免资源过度集中，保持集群的负载均衡。
 - 策略约束：遵循各种调度策略和约束条件，如节点亲和性/反亲和性、pod亲和性/反亲和性、污点和容忍度等。
 - 性能优化：选择最优的节点以提升集群性能和资源利用率。

k8s scheduler 的实现原理

scheduler的实现原理主要包括以下几个步骤：
（1）过滤节点
	- 首先，scheduler会筛选出一组候选节点，这些节点能够满足pod的基本要求。
	- 过滤条件节点的资源情况（如cpu和内存）、节点标签、污点和容忍度、pod亲和性和反亲和性等。
（2）打分节点
	- 对通过过滤的候选节点进行打分，评估每个节点的优势。
	- 打分根据多个调度策略进行计算，这些策略可以包括资源利用率、节点间均衡、节点亲和性等。
（3）选择节点
	- 根据打分结果选择得分最高的节点，将pod调度到该节点上。
	- 如果多个节点得分相同，scheduler会随机选择一个节点。
（4）绑定节点
	- 最后，scheduler将pod绑定到选定的节点上。
	- 通过向API Server发出绑定请求来实现这一操作。
```

**12\. 简述k8s的探针**

```yaml
kubernetes探针（Probes）是一种用于检查健康状况的机制。探针能够帮助kubernetes自动检测和处理容器运行的状态，从而提高应用的可用性和稳定性。kubernetes提供了三种类型的探针：“liveness Probe”、“Readiness Probe”、“Startup Probe”。每种探针都有不同的用途和配置方式。

（1）liveness probe（存活探针）
用途：
	检查容器是否处于健康状态，如果liveness probe检测失败，kubernetes会重启该容器。
典型场景：
	用于检测容器是否进入了无法恢复的错误状态。
（2）readliness probe（就绪探针）
用途：检查容器是否已准备好接收流量，如果readliness probe检测失败，kubernetes会将该容器从service的端点中移除，停止向其发送流量。
典型场景：
	用于确保只有在容器完全初始化并能够处理请求后才会接收流量。
（3）Startup Probe
用途：检查容器应用是否已成功启动。如果startup probe配置成功，kubernetes会停止对liveness probe的检查。如果startup probe检测失败，kubernetes会重启容器。
典型场景：用于需要较长时间启动的应用，确保在应用启动完成前不会被误判为失败。

探针的配置选项

探针可以通过以下几种方式进行配置：
（1）HTTP探针
	- 通过发送HTTP GET请求来检测服务健康。
	- 配置参数：“httpGet”（包含“path”和“port”）
（2）TCP探针
	- 通过尝试打开与容器的TCP连接来检测服务健康。
	- 配置参数：“tcpSocket”（包含“port”）。
（3）命令探针：
	- 通过在容器中运行命令来检测服务健康。
	- 配置参数：“exec”（包含“command”）。

探针的配置字段
	- “initiaDelaySeconds”：容器启动后第一次探测的延迟时间（秒）。
	- “periodSeconds”：探测的时间间隔（秒）。
	- “timeoutSeconds”：探测的超时时间（秒）。
	- “successThreshold”：探测成功的最少次数，连续成功多少次才算成功。
	- “failureThreshold”：探测失败的最少次数，连续失败多少次才算失败。
```

**13\. 什么是-sidecar-容器使用它做什么？**

```yaml
什么是sidecar容器？
sidecar容器是一种运行在同一个pod中的辅助容器，它与主应用容器协同工作，共享同一个网络命名空间和存储卷。sidecar容器通常用于提供辅助功能，例如日志收集、监控、代理服务、配置管理等，而不需要对主应用进行修改。
```

**14\. kubernetes-中如何隔离资源？**

```yaml
(1) 命名空间（Namespace）
用途：命名空间用于将集群中的资源划分为多个逻辑组，从而实现不同团队或应用之间的资源隔离。

（2）资源配额（Resource Quota）
用途：限制命名空间中资源的使用量，例如cpu、内存、pod数量等。它可以防止单个命名空间的资源使用过多，影响其它命名空间中的应用。

（3）限制范围（Limit Range）
用途：限制范围定义了pod和容器的资源请求和限制的默认值和最大值。通过设置LimitRange，可以控制单个pod或容器的资源使用，确保资源使用符合预期。

（4）节点选择器和节点亲和性（Node Selector and Node Affinity）
用途：通过节点选择器和节点亲和性，可以将pod调度到特定节点上，从而实现资源隔离。

（5）污点和容忍度（Taints and Tolerations）
用途：通过给节点打污点，可以防止特定pod调度到这些节点上。

（6）网络策略（Network Policy）
用途：网络策略用于控制pod之间以及pod与外部之间的网络流量，提供基于命名空间和标签的网络隔离。
```

**15\. 容器和主机部署应用的区别是什么？**

```yaml
容器的中心思想就是秒级启动；一次封装，到处运行；这是主机部署应用无法达到的效果，但同时也更注意容器的数据持久化问题。
另外，容器部署可以将各个服务进行隔离，互不影响，这也是容器的另一个核心概念。
```

**16\. 描述一下deployment的升级过程**

```yaml
在deployment的定义中，可以通过spec.strategy指定pod的更新的策略，目前支持两种策略：Recreate（重建）和RollingUpdate（滚动更新），默认值是RollingUpdate。
- Recreate：设置spec.strategy.type=Recreate，表示Deployment在更新pod时，会先杀掉所有正在运行的pod，然后创建新的pod。
- RollingUpdate：设置spec.strategy.type=RollingUpdate，表示Deployment会以滚动更新的方式来逐个更新pod。同时，可以通过设置spec.strategy.rollingUpdate下的两个参数（maxUnavaliable和maxSurge）来控制滚动更新的过程。
```

**17\. metric-service-有什么作用**

```yaml
Metrics Server 是一个集群范围内的资源数据集和工具，同样的，metrics-server也只是显示数据，并不提供数据存储服务，主要关注的是资源度量API的实现，如CPU、文件描述符、内存、请求延时等指标，metric-server收集数据给k8s集群内使用，如：kubectl，hpa，secheduler等

metrics-server从kubelet收集资源指标，并通过Metrics API在apiserver中公开它们，以提供Horizontal Pod Autoscaler（HPA）和vertical pod Autoscaler（VPA）使用。Metricas API也可以通过访问kubectl top，从而更容易调试自动缩放管道。

Metrics server不适用于非自动缩放目的。例如，不要使用它来将指标转发给监控方案，或作为监控解放方来指标的来源。在这种情况下，请直接从kubelet/metrics/resource端点收集指标。

Metrics Server提供：
	- 适用于大多数集群的单一部署；
	- 快速自动缩放，每15秒收集一次指标；
	- 资源效率，集群中每个节点使用1 mili的cpu和2MB的内存
	- 可扩展支持多达5000个节点集群
```

**18\. kubernetes中RBAC是什么？有什么优势？**

```yaml
RBAC 是基于角色的访问控制，是一种基于个人用户的角色来管理对计算机或网络资源的访问的方法。
相对于其它授权模式，RBAC具有如下的优势：
- 对集群中的资源和非资源权限均有完整的覆盖；
- 整个RBAC完全由几个API对象完成，同其它API对象一样，可以用kubectl或API进行操作；
- 可以在运行时进行调整，无须重新启动API Server。
```

**19\. 是否手动创建Pod，如果想要创建同一个容器的多份拷贝，需要一个个分别创建出来么，能否将Pods划到逻辑组里？**

```yaml
Replication Controller确保任意时间都有指定数量的pod副本在运行，如果为某个pod创建了Replication COntroller并且指定3个副本，它会创建3个pod，并且持续监视它们。如果某个pod不响应，那么Replication Controller会替换它，但保持总数为3
```

**20\. 如何监视部署在Docker容器上的应用程序？**

```yaml
kubernetes可以通过设定liveness Probe属性来为一个pod做健康检查。
```

**21\. Docker+kubernetes 只能在linux环境下运行吗？**

```yaml
docker可以再windows环境下运行，并且，从1.5版本开始，kubernetes加入了对windows server容器的支持，控制器仍然还跑在linux上，然后kubelet和kube-proxy则可以在windows上运行。
```

**22\. kubernetes和openstack发展方向是怎样的？它们之间存在很多分歧吗？**

```yaml
kubernetes和openstack是两个完全不同的东西，没有关系。
```

**23\. 应用和运行时平台是怎么解耦的？**

```yaml
容器是设计成自包含的。因此可以创建一个包含了系统的所有内容，让它拥有完备的移植性。我们也应该明白一点，应用程序不可能完全和运行是平台解耦。
```

参考原文链接： [https://blog.csdn.net/qq\_46654855/article/details/125612617](https://blog.csdn.net/qq_46654855/article/details/125612617)

[上一篇记不同服务接入ldap配置](https://www.cnblogs.com/cat1/p/16255632.html)

[下一篇node镜像](https://www.cnblogs.com/cat1/p/18812643)

本文作者：没有猫的猫奴

本文链接：https://www.cnblogs.com/cat1/p/18812637

版权声明：本作品采用知识共享署名-非商业性使用-禁止演绎 2.5 中国大陆 [许可协议](https://www.cnblogs.com/cat1/p/18812637) 进行许可。

分类:
[云原生学习记录](https://www.cnblogs.com/cat1/category/2106796.html)

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/2096608/20210705101300.png)](https://home.cnblogs.com/u/cat1/)

[没有猫的猫奴](https://home.cnblogs.com/u/cat1/)

[粉丝 \- 1](https://home.cnblogs.com/u/cat1/followers/) [关注 \- 3](https://home.cnblogs.com/u/cat1/followees/)

+加关注

0

0

[«](https://www.cnblogs.com/cat1/p/16255632.html) 上一篇： [记不同服务接入ldap配置](https://www.cnblogs.com/cat1/p/16255632.html "发布于 2022-05-10 22:05")

[»](https://www.cnblogs.com/cat1/p/18812643) 下一篇： [node镜像](https://www.cnblogs.com/cat1/p/18812643 "发布于 2025-04-07 14:14")

posted @
2025-04-07 14:10 [没有猫的猫奴](https://www.cnblogs.com/cat1)
阅读(118)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2Fcat1%2Fp%2F18812637&targetId=18812637&targetType=0)

[刷新页面](https://www.cnblogs.com/cat1/p/18812637#) [返回顶部](https://www.cnblogs.com/cat1/p/18812637#top)

登录后才能查看或发表评论，立即 登录 或者
[逛逛](https://www.cnblogs.com/) 博客园首页

[【推荐】智能无限 \| 协作无间，TRAE SOLO 中国版正式上线，全面免费](https://www.trae.com.cn/?utm_source=advertising&utm_medium=cnblogs_ug_cpa&utm_term=hw_trae_cnblogs)

[【推荐】科研领域的连接者艾思科蓝，一站式科研学术服务数字化平台](https://ais.cn/u/QjqYJr)

[【推荐】飞算 JavaAI 修复器：无限 tokens 加持，Bug 修复快到飞起](https://www.cnblogs.com/cmt/p/19669319)

[![](https://img2024.cnblogs.com/blog/35695/202512/35695-20251205182619157-1150461542.webp)](https://ais.cn/u/3Qf22e)

- [从 305 GB 到 7.4 GB：大模型 KVCache 架构演进全景](https://www.cnblogs.com/cswuyg/p/19981922)
- [DeepSeek V4模型的Agent能力实测](https://www.cnblogs.com/zhayujie/p/19935607/deepseek-v4-eval)
- [C# 15 类型系统改进：Union Types](https://www.cnblogs.com/hez2010/p/19891530/union-types-in-csharp-15)
- [你能被装进一个文件里吗？——7 万人把同事“蒸馏”成了 AI](https://www.cnblogs.com/wmyskxz/p/19854791)
- [别再吹牛了，100% Vibe Coding 存在无法自洽的逻辑漏洞！](https://www.cnblogs.com/mengxiang2/p/19796426)

[博客园](https://www.cnblogs.com/) [首页](https://www.cnblogs.com/cat1) [新随笔](https://i.cnblogs.com/posts/edit) [草稿箱](https://i.cnblogs.com/posts) [联系](https://msg.cnblogs.com/send/%E6%B2%A1%E6%9C%89%E7%8C%AB%E7%9A%84%E7%8C%AB%E5%A5%B4)订阅
[管理](https://i.cnblogs.com/)

- 随笔：44
- 文章：0
- 评论：0
- 阅读：5983

### 公告

昵称：
[没有猫的猫奴](https://home.cnblogs.com/u/cat1/)

园龄：
[5年9个月](https://home.cnblogs.com/u/cat1/ "入园时间：2020-07-13")

粉丝：
[1](https://home.cnblogs.com/u/cat1/followers/)

关注：
[3](https://home.cnblogs.com/u/cat1/followees/)

+加关注

![](https://images.cnblogs.com/cnblogs_com/guangzan/1894231/o_230626114104_summer.png)

Summer Wonderland

08 May, 2026

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

### 常用链接

- [我的随笔](https://www.cnblogs.com/cat1/p/ "我的博客的随笔列表")
- [我的评论](https://www.cnblogs.com/cat1/MyComments.html "我的发表过的评论列表")
- [我的参与](https://www.cnblogs.com/cat1/OtherPosts.html "我评论过的随笔列表")
- [最新评论](https://www.cnblogs.com/cat1/comments "我的博客的评论列表")
- [我的标签](https://www.cnblogs.com/cat1/tag/ "我的博客的标签列表")

### 最新随笔

- [1\. Linux服务器安全管理规范](https://www.cnblogs.com/cat1/p/18812663)
- [2\. vsftpd部署文档(centos7.9)](https://www.cnblogs.com/cat1/p/18812662)
- [3\. kubeadm搭建k8s集群，更新证书](https://www.cnblogs.com/cat1/p/18812653)
- [4\. centos7搭建OpenVpn](https://www.cnblogs.com/cat1/p/18812650)
- [5\. k8s部署kubepi](https://www.cnblogs.com/cat1/p/18812648)
- [6\. node镜像](https://www.cnblogs.com/cat1/p/18812643)
- [7\. 云原生k8s高频面试题](https://www.cnblogs.com/cat1/p/18812637)
- [8\. 记不同服务接入ldap配置](https://www.cnblogs.com/cat1/p/16255632.html)
- [9\. docekr 单独资源限制](https://www.cnblogs.com/cat1/p/16131318.html)
- [10\. harbor](https://www.cnblogs.com/cat1/p/16131301.html)

### 积分与排名

- 积分 \-
5980

- 排名 \-
168871


### [随笔分类](https://www.cnblogs.com/cat1/post-categories)  (44)

- [docker(2)](https://www.cnblogs.com/cat1/category/2142433.html)
- [jenkins(3)](https://www.cnblogs.com/cat1/category/2012388.html)
- [zabbix(1)](https://www.cnblogs.com/cat1/category/2116456.html)
- [探索运维(21)](https://www.cnblogs.com/cat1/category/1994804.html)
- [云原生学习记录(17)](https://www.cnblogs.com/cat1/category/2106796.html)

### 随笔档案  (44)

- [2025年4月(7)](https://www.cnblogs.com/cat1/p/archive/2025/04)
- [2022年5月(1)](https://www.cnblogs.com/cat1/p/archive/2022/05)
- [2022年4月(3)](https://www.cnblogs.com/cat1/p/archive/2022/04)
- [2022年3月(5)](https://www.cnblogs.com/cat1/p/archive/2022/03)
- [2022年2月(8)](https://www.cnblogs.com/cat1/p/archive/2022/02)
- [2021年12月(2)](https://www.cnblogs.com/cat1/p/archive/2021/12)
- [2021年11月(1)](https://www.cnblogs.com/cat1/p/archive/2021/11)
- [2021年10月(1)](https://www.cnblogs.com/cat1/p/archive/2021/10)
- [2021年9月(2)](https://www.cnblogs.com/cat1/p/archive/2021/09)
- [2021年8月(5)](https://www.cnblogs.com/cat1/p/archive/2021/08)
- [2021年7月(9)](https://www.cnblogs.com/cat1/p/archive/2021/07)

### [阅读排行榜](https://www.cnblogs.com/cat1/most-viewed)

- [1\. Linux关闭防火墙命令(948)](https://www.cnblogs.com/cat1/p/15001967.html)
- [2\. shell脚本中执行curl遇到异常继续执行(808)](https://www.cnblogs.com/cat1/p/15561706.html)
- [3\. Linux 数据盘在线扩容(382)](https://www.cnblogs.com/cat1/p/15206370.html)
- [4\. 一键安装docker+docker-compose(338)](https://www.cnblogs.com/cat1/p/15272093.html)
- [5\. 记搭建smb共享服务器(290)](https://www.cnblogs.com/cat1/p/15012093.html)

01. 1好想爱这个世界啊华晨宇
02. 2Lemon米津玄師
03. 3打上花火Daoko / 米津玄師
04. 4グランドエスケープ (Movie edit)三浦透子 / RADWIMPS
05. 5愛にできることはまだあるかいRADWIMPS
06. 6アイロニmajiko
07. 7过膝袜之歌多多poi
08. 8Illusionary DaytimeShirfine
09. 9Move Up (Lost Gravity)Mr. Polska
10. 10喜帖街谢安琪
11. 11国王与乞丐华晨宇 / 杨宗纬
12. 12嚣张en（王翊恩）
13. 13玫瑰花的葬礼许嵩
14. 14海阔天空Beyond
15. 15遇见孙燕姿
16. 16美丽的神话Ⅰ成龙 / 金喜善

Lemon \- 米津玄師

00:00 / 00:00

An audio error has occurred, player will skip forward in 2 seconds.

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)

评论

收藏

关注

推荐

深色

回顶

展开

收起

1. 1404 not foundREOL

404 not found \- REOL

00:00 / 00:00

An audio error has occurred.

作曲 : Reol

作词 : Reol

fade away...do over again...

fade away...do over again...

歌い始めの一文字目 いつも迷ってる

歌い始めの一文字目 いつも迷ってる

どうせとりとめのないことだけど

伝わらなきゃもっと意味がない

どうしたってこんなに複雑なのに

どうしたってこんなに複雑なのに

噛み砕いてやらなきゃ伝わらない

ほら結局歌詞なんかどうだっていい

僕の音楽なんかこの世になくたっていいんだよ

Everybody don't know why.

Everybody don't know why.

Everybody don't know much.

僕は気にしない 君は気付かない

何処にももういないいない

Everybody don't know why.

Everybody don't know why.

Everybody don't know much.

忘れていく 忘れられていく

We don't know,We don't know.

目の前 広がる現実世界がまた歪んだ

目の前 広がる現実世界がまた歪んだ

何度リセットしても

僕は僕以外の誰かには生まれ変われない

「そんなの知ってるよ」

気になるあの子の噂話も

シニカル標的は次の速報

麻痺しちゃってるこっからエスケープ

麻痺しちゃってるこっからエスケープ

遠く遠くまで行けるよ

安定なんてない 不安定な世界

安定なんてない 不安定な世界

安定なんてない きっと明日には忘れるよ

fade away...do over again...

fade away...do over again...

そうだ世界はどこかがいつも嘘くさい

そうだ世界はどこかがいつも嘘くさい

綺麗事だけじゃ大事な人たちすら守れない

くだらない 僕らみんなどこか狂ってるみたい

本当のことなんか全部神様も知らない

Everybody don't know why.

Everybody don't know why.

Everybody don't know much.

僕は気にしない 君は気付かない

何処にももういないいない

Everybody don't know why.

Everybody don't know why.

Everybody don't know much.

忘れていく 忘れられていく

We don't know,We don't know.