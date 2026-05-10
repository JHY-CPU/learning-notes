# 13.k8s(kubernetes)常见运维面试题- 马俊南- 博客园

URL: https://www.cnblogs.com/junnan/p/18830991

[![订阅](https://www.cnblogs.com/skins/sea/images/xml.gif)](https://www.cnblogs.com/junnan/rss/)

随笔 -
194
文章 -
0
评论 -
20
阅读 \-

21728

# [13.k8s(kubernetes)常见运维面试题](https://www.cnblogs.com/junnan/p/18830991 "发布于 2025-04-18 17:34")

[视频课程](https://www.bilibili.com/video/BV1ZTDEYmEbL/?spm_id_from=333.337.search-card.all.click)  <\-\-\-\-\-\- 点击

#### 1 k8s中常见类型的资源介绍和区别？

##### 1.1 常用资源介绍

Pod、最早的控制器RS(ReplicaSet)、deployment(RS的升级)、daemonset、statefuset、service、Ingress、Ingress-controller、configmap、secret、serviceAccount(服务账号)

##### 1.2 作用

Pod：运行服务部署中的最小工作单元，pod里运行服务的容器，可运行一个或者多个容器，一般运行一个容器。（花生壳与花生）

RS控制器：即为：ReplicaSet，是pod的控制器，维护pod的副本数，更新pod的时候，手动删除pod才能更新（已被Deployment取代）。

Deployment 控制器：和RS作用一样，也是pod的控制器，维护pod的副本数，但是比RS更好用，更新pod的时候，能自动更新升级，不用手动进行删除，在yaml文件里修改pod数量，自动会删除旧的pod，升级成新的pod，它是RS的升级版本（对RS的又一层封装，可以不停机更新）。

Daemonset 控制器：作用是保证k8s中所有节点都运行同一个pod，主要用于部署监控pod或日志收集的pod（例如zabbix\_agentd、node-exporter等）。

Statefuset 控制器：管理有状态pod的控制器，主要部署需要有持久化存储数据的pod（如：mysql、redis等数据库的pod）或pod中多个容器之间有严格的启动顺序（如：mysql主从等、持久化存储：MySql、Redis、MongDB）

Service 资源：主要通过匹配pod的label标签（Pod经就绪探针检测为就绪后，Service再把新就绪的Pod加入负载中），从而对pod实现负载均衡

Ingress/Ingress-controller 资源：主要是把部署好的服务，通过域名暴露出去，能够实现对外访问的作用。

Ingress，我们可以理解为：通过编写ingress的yaml文件，前端匹配域名，后端匹配相应服务的service名，达到修改nginx服务配置文件的目的。

Ingress-controller 我们可以理解为：就只是一个nginx的代理服务，但没有修改过nginx的配置，只是保持了默认的nginx配置，并没有修改成需要的nginx配置文件。

Configmap资源：主要起到修改服务配置文件解藕的作用。想要修改服务的配置文件，直接进行修改pod的配置文件很不方便，尤其是多副本时候，而且pod可能随时会重建。可以通过configmap间接修改，将配置文件的内容创建成configmap资源，pod挂载该configmap，修改configmap的内容，pod的配置文件内容也会跟着更新，这样就修改好了pod的配置文件。

Secret资源：创建加密内容的时候使用的资源。如：创建相应的账号和密码时。

ServiceAccount资源：服务账号，用来登录k8s验证登录时使用，不同的服务账号绑定不同的角色，访问不同的k8s资源等。

#### 2.k8s中pod服务健康检查方式有哪几种？

三种：startupProbe（启动检查）、livenessProbe（存活检查）、readnessProbe（就绪检查）

startupProbe启动检查：最先进行启动检查，到启动检查成功为止，再执行其他检查，成功后将不再进行探测。

livenessProbe存活检查：当检查出pod状态不正常或服务不健康时，会重建pod，重启容器。

readnessProbe就绪检查：当检查出pod状态不正常或服务不健康时，不会重建pod，而是将其设为不可用。

#### 3.k8s认证方式有哪几种？

两种登录方式： x509证书 +role/rolebinding认证 和 服务账号 +role/rolebinding认证

#### 4.k8s中的证书和私钥种类有哪些？

证书一共有三类：etcd数据库集群内部的证书（etcd集群内通信）、APIServer到etcd数据库的证书、其他到APIServer的证书（大多是双向认证证书）。

#### 5.k8s中各个节点上组件有哪些？各自作用是什么？

Master节点（控制节点）：

Kube-apiserver：是访问整个k8s集群的入口。

Kube-controller-manager：整个k8s集群的核心控制管理组件，主要负责管理和维护集群的状态。

Kube-scheduler：是k8s中调度服务组件，主要做分析资源，通过预选和优选机制，调度服务pod部署在哪个节点。

Node节点（工作节点）：

Kubelet服务：用来创建、管理、删除pod的组件，启动的时候向api-server中注册，然后定时的向api-server汇报本节点信息状态，资源使用状态等。

Kube-proxy服务：它运行在集群中的每个节点上，负责实现服务发现和负载均衡功能。

比如：多个pod副本，可以通过service，进行label标签匹配，通过service作为统一的访问入口，负载均衡访问后面的多个pod副本服务，实现多个pod的负载均衡，就会用到kube-proxy服务。

公共组件：ETCD数据库集群、flannel网络或calico网络（都属于overlay网络，给pod或容器分配IP的网络插件）、CoreDNS提供集群内部DNS服务，用于服务发现或服务之间的访问调用。

#### 6.k8s集群中有没有高可用？你们公司的高可用架构是什么样的？

高可用集群是：3个master、50台node节点机器

k8s的高可用：主要是针对组件apiserver进行高可用，它是集群的入口，都是通过访问APIServer来访问k8s集群的。

k8s的高可用可以用两种方式进行部署：

方式1：二进制方式进行安装

3个master上都安装相同的组件，controller-manager、scheduler、apiserver，3个节点都再增加部署一个haproxy+keepalived，实现用haproxy代理apiserver，提供出一个haproxy的代理访问地址（即：vip访问地址：haproxy服务端口），其他node节点的kubelet服务和kube-proxy服务配置连接代理APIServer服务的haproxy地址，通过haproxy代理apiserver，实现APIServer的高可用。

方式2：kubeadm初始化安装

3个master节点，都先安装keepalived+haproxy，让haproxy代理3个master的apiserver

第1台master节点初始化集群，初始化的最后会生成两种加入集群信息的提示：一种是以主的身份加入集群，一种是以从的身份加入，自带了加入地址：vip:haproxy的端口

后面2台master节点复制给的提示信息，以主的身份加入k8s集群既可。

其他的node节点复制给的提示信息，以从的身份加入k8s集群。

#### 7.k8s中镜像下载策略有哪几种？

1.Always（默认）：无论本地是否存在该镜像，总是尝试从远程仓库拉取最新的镜像。这对于使用 latest 标签或者希望总是获取最新镜像内容的场景非常有用。

2.Nerver：始终只依赖本地存在的镜像，绝不会尝试连接到任何外部仓库进行拉取操作。这种策略适用于不希望自动升级镜像版本且确保始终使用固定镜像的情况。需要注意的是，当使用 Never 策略时，必须确保目标机器确实拥有对应的 Docker 镜像文件，否则 Pod 将无法成功调度执行。

3.IfNotPresent：如果本地节点上已经存在该镜像，则不会尝试从镜像仓库拉取镜像；仅当本地不存在该镜像时，kubelet 才会去远程仓库拉取镜像。这是默认的策略，适用于具有明确版本标签（如 v1.0）的镜像，以避免不必要的镜像下载。

#### 8.k8s中Pod故障重启策略有哪几种？

1.Always：无论容器因任何原因终止（无论是正常退出还是异常退出），kubelet 都会自动重启该容器。

2.Nerver：无论容器因任何原因终止，都不会重启该容器。

3.OnFailure（默认）：仅当容器以非零退出码（异常退出）终止时，kubelet 才会重启该容器。如果容器正常退出（退出码为 0），则不会重启。

#### 9.k8s中PV有几种访问模式？

PV提供持久化存储的资源

4种：只让一个节点读写ReadWriteOnce (RWO)（基本的访问模式）、只让一个节点读写，即使这个 Pod 被调度到不同的节点上ReadWriteOncePod (RWOP)(Kubernetes 1.22 版本引入的新特性)、同时多个节点读写ReadWriteMany (RWX)、同时多个节点读ReadOnlyMany (ROX)。

#### 10.k8s中pv和pvc的作用是什么？pv和pvc和底层存储的关联顺序是什么？

pv和pvc都是用来做持久化存储数据的。

pv是持久化存储卷，定义的持久化存储在宿主机上的目录或其他共享目录，nfs等其他，是具体的实现方式。

pvc只是对某种持久化存储的描述，不提供具体的实现。

关联顺序：应用pod关联（通过pvc的名字）pvc，pvc关联（通过容量和访问模式）pv，pv关联底层存储设备（如：host宿主机、nfs共享存储、ceph或glusterfs分布式存储等）

pv和pvc需要绑定才能使用。

#### 11.客户端访问k8s资源需要经过几关？分别是什么？

验证通过 ------> 权限授权通过 ------\> 资源限制

Authentication（认证）：

这是验证用户身份的过程，确保请求是由合法的用户发出的。Kubernetes 支持多种认证方法，如客户端证书、静态token文件、Bootstrap tokens、OpenID Connect tokens、Authenticating Proxy等。

Authorization（授权）：

在认证之后，接下来是授权阶段，确定已认证的用户是否有权执行所请求的操作。Kubernetes 使用 Role-Based Access Control (RBAC) 系统来管理权限。通过定义角色（Role）和角色绑定（RoleBinding），可以控制用户对集群资源的访问级别。

Admission Control（准入控制）：

在请求被处理之前，它会经过一系列的准入控制器（Admission Controllers）。这些控制器可以修改请求的内容或者根据某些策略决定是否允许请求继续进行。例如，NamespaceLifecycle 准入控制器可以阻止在已标记为终止的命名空间中创建新对象。

Network Policies（网络策略）：

如果请求涉及到跨 Pod 的通信，还需要考虑网络策略。Kubernetes 允许定义网络策略来限制哪些 Pods 可以相互通信。这提供了额外的安全层，防止未经授权的访问。

Service Account Tokens and Permissions（服务账号令牌及权限）：

当应用内部的组件或服务需要与 Kubernetes API 交互时，它们通常使用服务账号（Service Account）。每个服务账号都有一个关联的令牌（Token），该令牌用于向 API 服务器证明其身份，并且与特定的角色绑定来控制其权限。

API Server Endpoint Security（API服务器端点安全性）：

最后，所有的请求都必须通过 API 服务器的安全端点。API 服务器本身应该配置有适当的 TLS 加密来保护传输中的数据，并且只能接受来自可信来源的连接。

#### 12.k8s集群中的数据是存储在哪个位置？

k8s集群中的数据都存储在etcd数据库中

#### 13.什么是Headless（无头）Service?

无头service，类似普通服务，没有集群IP，可以直接访问Pod，不需要通过代理访问它。

#### 14.docker怎么用Dockerfile文件构建镜像？具体命令是什么？

进入Dockerfile文件所在目录执行

\# docker build -t 镜像名:tag . \[--build-arg Dockerfile 中的变量名=值\]

#### 15.docker怎么用镜像运行一个容器？如何设置在后台运行？

运行容器命令：-d是后台运行

\# docker run -d --name 容器名 --net=host -v 宿主机目录:容器目录 镜像名:版本号(默认laster)

#### 16.docker-habor是怎么安装的？有哪几种安装方式？

在线安装(添加远程仓库)或离线包安装(tar压缩包)都可以，也可以通过docker-compose(镜像容器)拉起安装。

#### 17.Dockerfile中都有哪些关键字？各自的作用是什么？

FROM：定义一个基础镜像，以这个基础镜像进行扩展构建成其他服务镜像

WORKDIR：登录容器后，切换到容器时的落脚点路径定义

ARG：指定用于定义构件时的参数，可以在构建镜像时通过 --build-arg 选项传递参数值，build时传递参数值给下面的变量，下面的变量可以接收到传进来的值，构建镜像时使用

ENV：指定用于设置环境变量，这些环境变量在容器运行时可用

EXPOSE：暴露服务的端口

RUN：用于构建镜像时执行的命令，用于在构建镜像时使用。

CMD：容器运行启动后执行的命令。

CMD特点：

1）.设置的命令能被docker run ... 后面的命令行参数替换。

2）.存在多个CMD时，仅仅最后的一个生效，最后一个会覆盖掉前面所有的。

ENTRYPOINT：容器启动时执行的命令，设置的命令不会被docker run ... 后面的命令行参数替换，里面设置的命令肯定能被执行，可以和CMD配合使用。

#### 18.如何使用docker快速运行相关服务？比如docker安装nginx、mysql、redis、tomcat等？

准备好相应服务的镜像，可以运行时在线拉取 pull，也可提前导入 load，然后运行命令：

\# docker run -d --name 容器名 --net=host（或-p端口映射） -v 映射数据目录或配置文件 镜像名:版本号(默认laster)

#### 19.使用过docker-compose吗？比docker来讲，有什么优点？相关常用命令是？

使用过docker-compose，docker只是管理运行单个容器，docker-compose可以将要运行的很多容器编写在yaml文件中，可以一键拉起很多容器，部署效率会提高很多。常用命令：

\# docker-compose build   # 构建所有镜像

\# docker-compose up -d  # 后台拉起所有服务

\# docker-compose stop   # 停止

\# docker-compose start   # 启动

\# docker-compose down  # 停止并移除由 docker-compose up 启动的所有容器、网络以及可能存在的默认外部卷

\# docker-compose rm      # 移除已停止的容器

#### 20.会编写docker-compose的yaml文件吗？如何编写yaml文件，使用docker-compose拉起一连串的服务？

会编写，拉取一连串的多个容器服务，是docker-compose的基本功能，在yaml文件中，先定义拉取一个服务，相关任务配置好后，包括：build镜像，需不需要和宿主机映射、服务端口映射等，先调试好一个服务，其他服务直接进行复制修改，所有服务修改好后，直接执行docker-compose命令一键拉起所有服务。

#### 21.docker有几种网络模式？分别是哪些？各自作用是什么？

host：以宿主机网络映射出去提供服务

container：容器共享IP网络模式，多个容器同一个IP

None：无网络模式，进行内部测试使用

Bridge(nat)：默认的nat桥接网络模式

#### 22.你们用的k8s的版本是什么？

版本是：k8s1.20.4

#### 23.Dockerfile中RUN、CMD、ENTRYPOINT的区别？

RUN：用于构建镜像时执行的命令，用于在构建镜像时使用。

CMD：容器运行启动后执行的命令。

CMD特点：

1).设置的命令能被 docker run --- 后面的命令行参数替换。

2).存在多个CMD时，仅仅最后的一个生效，最后一个会覆盖掉前面所有的命令。

ENTRYPOINT：容器启动时执行的命令，设置的命令不会被docker run ... 后面的命令行参数替换，里面设置的命令肯定能被执行，可以和CMD配合使用。

#### 24.Dockerfile中ADD和COPY的区别？

COPY：能拷贝本地文件或目录到容器中（拷贝目录时，容器里也要带上该目录名）

ADD：

1).拷贝本地文件或目录到容器中，该功能和上面的功能一样，一般此功能用上面的COPY既可

2).拷贝本地压缩文件到容器中时可以自动进行解压

3).还可以拷贝网络文件到容器中，并且可以自动解压，如：ADD http://xxxx/a.tar.gz /root

#### 25.docker中镜像分层是怎么样的？

1).镜像是分层创建的。

2).每一条指令都会新建一层。

3).当前层不会影响上一层。

#### 26.k8s中pod是如何实现代理和负载均衡？

通过创建service资源，label的标签和pod的标签保持一致，通过label标签匹配到后端的pod的标签，代理pod，实现pod的负载均衡（service底层依赖kube-proxy服务，才能实现负载均衡）。

#### 27.k8s中创建pod的过程或流程是怎么样的？

1).kubectl命令或点击图像化界面向APIServer发送一个创建pod的请求。

2).k8s中APIServer接收到请求后，先生成一个创建pod信息的yaml文件，将yaml文件写入到etcd数据库中，添加一条记录。

3).Scheduler调度服务会查看etcd数据库信息（类似有通知机制会通知给Scheduler调度服务组件），判断这条信息是不是新来的，若是新来的信息需要创建，则会进行调度计算（预选、优选），找到节点中“最闲”的，资源使用率最低的Node节点，让其调度到节点上面，将这个调度信息更新到etcd数据库中。

4).各Node节点上的kubelet服务，也是不停的监控etcd数据库，发现有新的数据进来，也会自行判断，根据分配的调度信息和自己对号入座，如果发现任务不是分配给自己的就不操作，如果发现任务是分配给自己的，那么就会调用自己节点的容器API（如：docker api、containerd api）来创建出pod容器。

#### 28.k8s中如何批量删除pod?

批量过滤出需要删除的pod，使用命令或脚本进行删除。

如：过滤出状态为Error的pod进行批量删除

\# kubectl get pod -n 命名空间\|grep Error\|awk  '{print $1}'\|xargs kubelet delete pod -n 命名空间

#### 29.kubeadm初始化的k8s集群，token过期后，集群中如何增加一个新的node节点？

以k8s集群中加入一个新的node节点为例：

##### 1).在k8s-master上重新生成新的token---在k8s-master节点操作

\# kubeadm token create # 创建新的token

\# kubeadm token list       # 查看新创建的token值

##### 2).在k8s-master上根据ca证书获取hash值---在k8s-master节点操作

\# openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt \| openssl rsa -pubin -outform der 2>/dev/null \| openssl dgst -sha256 -hex \| sed 's/^.\* //'      # 获取hash值

##### 3)在新的node节点上根据 token 和 hash 值加入k8s集群---在新节点上操作

新节点也需要安装基础环境，kubeadm，kubelet，和kubectl

\# yum install -y kubelet-1.20.4 kubeadm-1.20.4 kubectl-1.20.4

节点根据token值和hash值加入集群：

\# kubeadm join <想加入节点的IP>:<服务端口> --token <1中token值> --discovery-token-ca-cert-hash sha256:<根据1token生成2的hash值>

#### 30.k8s运维过程中遇到过那些问题，如何解决的？

##### 1).kubeadm init 初始化 k8s 集群时，初始化失败

查看原因，是之前初始化用过的机器，之前机器没有删除干净，有部分遗留文件，重新清空原来环境，删除原来文件后，重新初始化后成功，问题解决。

##### 2).k8s部署服务时，pod一直处于pending状态，无法部署

以下几种原因都遇到过：

a).node节点资源不足，yaml文件中资源限制中分配的内存，cpu资源太大，node宿主机资源没有那么大，导致无法调度部署，临时解决：yaml文件中资源限制处资源分配小一点，永久解决：扩展node节点，增加宿主机资源。

b).部署pod的yaml文件中有标签选择器节点亲和性 nodeAffinity 的名称配置，并且采用的硬策略(RequirdDuringSchedulingIgnoreDuringExection)，但是所有node节点上没有相对应的节点标签，导致无法部署，解决：把yaml文件中节点亲和性配置去掉，或者在相应node节点上打上相对应的标签。

c).node节点上有污点，而部署pod的yaml文件中没有配置污点容忍度，导致无法调度部署。当时同事为了分组，不让不相关的服务部署到相应节点。解决：在yaml文件中配置污点容忍度或者将node节点上打的污点去掉。

#### 31.执行kubectl get node命令后看不到某些节点的原因？

若节点没有部署kubelet服务，就看不到（一般二进制方式部署的master节点未部署，就看不到）

#### 32.执行kubelet get cs 查看集群状态不正常，显示unhealthy，如何解决？

这是因为k8s组件kubelet 服务配置文件里没有配置本地监听端口或设置为了0，可以在本地kubelet服务配置文件中，将端口改成kubelet服务对应的端口，或者将里面的port=0 注释，然后重启kubelet服务。

#### 33.kubectl 命令中 create 和 apply 创建资源的区别？

create：一般用来初次创建，当创建资源后，修改了yaml文件，再create会报错，显示已经创建了，create仅支持第一次创建，不支持后续更新。

apply：不仅支持创建，还支持后续更新，应用较多。

#### 34.pod资源共享机制如何实现？即：如何实现pod中两个容器共享同一个存储数据资源？

首先在pod的yaml文件里定义一个emptyDir空目录，挂载宿主机上的临时目录（会随pod生存而生存），然后定义个名字，pod中的两个容器，同时挂载同一个名字，将资源挂载到自己容器的相应路径，此后两个容器就可以共享该数据资源了。

#### 35.容器之间时通过什么进行隔离的？

通过namespace命名空间进行隔离。

#### 36.pod常用的状态？

Pending：pod未调度或者已经调度正在拉取镜像。

Running：pod已经运行，正常状态。

Failed：pod内的容器停止运行。

Success：pod内容器运行成功结束。（是任务型的pod）

Unknown：master节点和node节点失联，pod状态无法获取到。

#### 37.节点选择器都有什么？各自的区别是什么？

有3种节点选择器：

分别是：nodeSelector、nodeAffinity、nodeName（一般不经过调度器，很少用）

nodeSelector节点选择器，通过给node节点打标签，yaml文件中需要配置匹配到相应node标签上，从而将pod调度到相应node节点上，若没有匹配到相应的node标签，则无法调度。

nodeAffinity节点亲和性：与nodeSelector作用一样，但是比较灵活，调度策略分为：软策略和硬策略

硬策略：是必须满足条件，才能给调度，

软策略：是尝试满足，尽量满足，但不能保证。

硬策略举例：在yaml文件中指定了打在具体标签的node节点上，就必须只能在该标签的node节点上创建资源，如果所有节点都没有符合的标签，就不会给调度创建。

软策略举例：在yaml文件中指定了打在具体标签的node节点上，尽可能在该标签的node节点上创建资源，如果所有节点中都没有符合的该标签，则再按其他调度算法创建在相应的节点上。

#### 38.污点和污点容忍是什么？两者是如何配合使用的？

污点（Taint）：主要作用是：避免pod调度到特定node节点上，主要用于分组管理

污点容忍（Tolerations）：允许pod调度到持有Taint污点的node节点上，需要和Taint结合使用。

#### 39.service的4种类型？

ClusterIP：

默认值，k8s系统给service自动分配的虚拟IP，只能在集群内部访问，集群内部使用IP（默认类型，只能集群内部访问）。

NodePort：

对外暴露应用，通过访问node节点和ip和端口可以访问到对应应用（集群外访问），将Service通过指定的Node上的端口暴露给外部，访问任意一个NodeIP:nodePort都将路由到ClusterIP。

Loadbalancer：（升级版的nodePort）

对外暴露应用（适用于公有云），在NodePort的基础上，借助cloud provider 创建一个外部的负载均衡器，并将请求转发到<NodeIP>:NodePort，此模式只能在云服务器上使用。

ExternalName：

ExternalName类型的Service，就是将该Service名跟集群外部服务地址做一个映射，使之访问Service名称就是访问外部服务。

这里使用两个案例来介绍，一个绑定外部域名，也可以绑定其他命名空间的域名。

1.绑定集群外部域名

把外部的服务绑定到集群内部的ExternalName类型的Service上，这样访问该Service名就能访问到外部服务，先来看一个小例子

1.先创建一个带有curl命令的busybox的pod

2.再创建一个ExternalName类型的Service，绑定的是外部域名www.k8sec.com

3.查看Service

4.访问

这时我们就能通过访问名为 k8sec.default.svc的Service访问到www.k8sec.com

#### 40.service两种代理模式？

两种代理模式：iptables（默认）和ipvs模式。

iptables：通过iptables规则进行转发代理。

ipvs：使用了类似lvs负载均衡的技术，使用rr轮询模式进行转发代理。

iptables模式和ipvs模式可以互相转换。

#### 41.k8s提供了哪几种对外暴露访问方式？

NodePort方式：通过node节点的ip和暴露端口，提供外部访问

Loadbalance模式：适用于云产品厂商进行暴露访问。

Ingress模式：通过提供域名，代理分流不同域名进行各自访问。

#### 42.k8s的监控（Prometheus）常用监控组件有哪些？各自的作用是什么？

##### 1)alertmanager报警组件安装：

可以在宿主机直接安装，也可以部署在k8s集群当中，服务的配置文件中可以定义收发邮件人，进行邮件的报警通知，也可以配置钉钉告警（或微信告警），首先先部署一个dingtalk服务，部署好后，暴露出一个访问地址，包括ip:端口，alertmanager的配置中可以配置dingtalk服务的ip和端口，调用digntalk服务，将报警信息通过dingtalk服务发送到指定的钉钉群。

##### 2)安装部署node\_exporter服务或其他各类exporter服务。

以node\_exporter服务为例：node\_exporter主要是收集k8s集群中宿主机节点上的资源信息，包括cpu、内存、磁盘空间等，通过这些信息对宿主机节点进行监控，可以手动部署到各个宿主机上，也可以daemonset方式部署在k8s集群中，最终暴露出一个访问地址既可，暴露出的访问地址供Prometheus连接，其他各类exporter服务也是类似。

##### 3).安装Prometheus服务：（自带cadvisor服务采集pod容器的数据信息）

Prometheus服务以daemonset方式或deployment方式部署在k8s集群中（有官方的yaml文件，可自定义修改配置），主要是采集pod容器的数据信息，对容器进行监控。

Prometheus的配置文件中：定义了连接alertmanager的地址，进行告警通知，连接了node\_exporter等各类exporter的地址，收集采集各exporter的数据，对宿主机和其他mysql、redis等进行监控，定义了各种报警规则和触发条件，用于实现监控触发报警机制。

##### 4).安装metric服务，收集k8s资源组件的数据信息，对k8s内部资源进行监控metric服务以daemonset方式部署在k8s集群中。

##### 5).安装grafana服务，进行监控大屏展示

可以以deployment方式部署在k8s集群中，访问grafana的访问地址，web界面上添加数据源为Prometheus，并添加Prometheus的地址，从Prometheus收集过来的监控数据，进行大屏展示。

#### 43.pod常见的状态有哪些？

Runing：正常状态

Pending：正在调度或者无法调度，可能是node节点资源不够，可能nodeName和nodeSelector不匹配，也可能node节点有污点。

ContainerCreating：后面node节点故障不能工作，或者网络i已被注册，也可能无法访问镜像，可能参数secret未配置，无法访问镜像。

Crashloopbackoff：pod里的容器退出，多次重启或准备删除。

Error：可能依赖的configmap、secret、pv等不存在，资源超过limit限制，无法访问集群内资源。

Termination：容器里的应用挂了

Unkown：master节点和node节点失联，pod状态无法获取到，需要查看node节点上kubelet服务是否正常，实在不行可以先剔除node节点，重置后重新加入。

#### 44.node节点不能工作的处理？

1).先将该节点正在运行的pod驱逐到其他node节点

2).# kubeadm reset 重置node节点 （相当于所有东西清空）

3).停止kubelet服务

4).停止docker、containerd服务

5).删除清空相关旧的配置 # rm -rf /var/lib/kubelet/\*  # rm -rf /etc/cni/

6).网卡向down 掉 # ifconfig cni0 down # ifconfig flannel.1 down

7).# iplink delete cni0 # iplink delete flannel.1

8).启动docker

9).启动kubelet服务

10).重新加入集群， # kubeadm join ...

#### 45.k8s常见健康检查的探针有几种？

1).httpGet：状态码检测，相应2xx，则认为成功

2).tcpSocker：建立tcp连接检测，探针与容器指定端口建立tcp连接，连接成功则探测成功

3).exec：在容器内执行任意命令，并检查命令退出状态码，状态码为0，则认为成功

#### 46.k8s中网络通信类型有几种？

1).同一个pod中的多个容器间的通信

2).pod之间的通信，从一个pod到另一个pod之间的通信

3).pod与service通信，pod的ip到cluserIP的通信

4).和外部通信，集群和外部客户端的通信，可通过nodePort或Ingress暴露出访问地址

#### 47.k8s中网络插件有哪些？各自特点是什么？

1).flannel网络：能提供ip，但不能配置网络策略

2).calico网络：既能提供ip，也能配置网络策略（可以限制pod允许从那个网段访问）

3).cannel网络：既能提供ip，也能配置网络策略（是flannel网络和calico网络的结合）

那个性能好？从性能上讲，calico网络和flannel网络差不多

flannel网络有多种模式：vxlan、host-gw、udp模式（性能最差，已弃用）

vxlan模式：叠加网络模式就是增加一个flannel.1网卡，连接docker网桥和宿主机网卡，进行通信。

host-gw模式：以宿主机充当网关。

#### 48.pod网络连接超时的几种情况？

1).pod和pod之间的连接超时（不分跨不跨宿主机）

解决排查：查看calico或clannel网络是否是running，查看calico或flannel网络组件的日志，提取重要信息，查看pod网段和宿主机网段是否重合（不能让其重合）

2).pod和虚拟主机（宿主机）的服务器连接超时

解决排查：检查pod网络，能否ping通同网段pod的ip

3).pod和外网连接超时

解决排查：检查物理网络，在容器内ping外网域名或其他pod的ip，如：www.baidu.com，不通时可抓包测试

#### 49.访问pod的ip:端口或service的ip显示超时的处理？

宿主机上检查是否开启ipv4的转发

\# vim /etc/sysctl.conf

net.ipv4.ip\_forward=1

\# sysctl -p

#### 50.pod的生命周期阶段？

Pending：开始创建

Running：正常运行

Terminating：容器终止

#### 51.pod处于Running，但应用不正常的几种情况？

1).端口配置错误

如果应用实际需要监听的端口与pod定义中配置的端口不相符，可能会导致虽然pod运行起来了，但外部无法通过正确的端口访问到应用，从而表现为应用不正常。

2).应用自身代码错误或故障

应用的代码中可能存在逻辑漏洞、未处理的异常情况等，导致即使进程启动了，也不能正常执行功能。

3).依赖项缺失或异常

应用可能依赖其他的服务、数据库或组件，如果这些依赖项不可用、配置错误或版本不兼容等，应用也不能正常工作。

4).网络配置问题

虽然pod与网络连接看似正常，但可能存在网络策略限制、路由错误、DNS解析问题等，导致应用看似Running，但是内部的网络通信不能按预期进行。

5).环境变量配置错误

应用可能依赖特定的环境变量来正确运行，如果这些环境变量的值不正确或缺失，就会导致pod看似Running，但是访问不正常。

6).容器内存OOM（内存泄漏），已经僵死，虽然进程还在，但无响应

#### 52.当遇到coreDns经常重启和报错，这种故障如何排查？

1).检查基本系统状况

a).检查系统资源监控，使用系统监控工具如：top持续观察cpu、磁盘I/O等资源的使用情况，看是否存在资源耗尽或过度使用的情况，因为这可能导致coreDns不稳定。

b).检查系统日志：仔细查看/var/log/messages等，看是否有与coreDns相关的系统层面的提示或错误。

2).针对coreDns本身检查

a).coredns服务的日志，查看报错信息，根据日志报错进行分析服务本身

b).检查coreDns的配置文件，确保配置参数设置正确，没有语法错误或不合理的配置项，检查域名解析规则等

c).coreDns版本检查

确定当前使用的coreDns版本是否存在已知问题或是否有更新版本可用，有时版本更新可能会解决一些潜在的故障。

3).网络相关检查

a).网络连接测试：使用网络测试工具，如：ping、traceroute等，检查coreDns与其他相关服务器或网络节点的连续性，看是否存在网络延迟、丢包等问题。

b).DNS流量分析：可以使用网络流量分析工具查看coreDns处理的DNS请求和响应的流量模式，发现可能的异常流量或请求类型。

4).依赖服务检查

coreDns可能依赖于其他服务或组件，如：数据库、缓存等，确保这些依赖项都正常工作且配置正确。

5).安全因素排除

检查是否存在安全策略或防火墙规则等限制了coreDns的正常运行

6).考虑与其他应用兼容性问题

排查近期是否安装或更新的其他应用程序，是否与coreDns存在兼容性问题，可能会导致冲突。

#### 53.k8s集群节点状态为not ready的都有哪些情况？

1).刚安装好k8s集群的node节点为not ready：

原因是：没有部署网络插件calico或flannel网络，需安装网络插件

2).k8s运行一段时间后，node节点为not ready：

原因可能是：

a).可能宿主机资源不足（磁盘、cpu、内存资源不足）

b).可能是node节点上kubelet服务不正常，需要查看kubelet服务是否正常，必要时重启kubelet

c).通过describe查看node节点的详细信息

#### 54.k8s的几种调度方式？

1).scheduler的预选和优选，选择合适node节点调度

预选：通过资源不足问题，过滤掉一些不符合要求的node节点，如：资源request不符。

优选：调度考虑整体的优化，如：多个副本尽量分布到不同的主机节点上，负载均衡。

2).通过定义nodeName或nodeSelector标签选择器进行调度

3).可以根据节点亲和性nodeAffinity进行调度

节点亲和性：nodeAffinity，作用和nodeSelecor一样，但更灵活，有软策略和硬策略

#### 55.k8s中一个node节点突然断电，修复后上面的pod无法启动，故障如何排查？

当node节点宕机，k8s会自动为这个节点上不可调度的污点，有可能开机后，污点没有自动消失，导致pod无法调度。

解决：

1).查看node节点是否有污点存在，如果有污点，可删除污点。

2).检查node节点主机名是否修改，主机名更改后连接不到k8s集群，也会添加污点，无法调度。可以将主机名的名称修改正确，重启kubelet服务恢复。

3).检查node节点kubelet服务是否正常，为not ready状态，也会被打上污点，可以重启kubelet服务，恢复正常解决。

#### 56.pod超过节点资源限制的故障情况有哪些？

1).pod数量太多，超过物理节点数量限制：

现象：大量pod批量重启后，pod一直处于pending状态，无法调度

处理思路：

a).查看相应的node节点上pod数量是否超过了默认的110个，node节点上默认能运行的最大pod的数量是110个，如果node节点有资源，可以修改node节点kubelet服务的配置文件，将pod运行数量调大。

\# vim /var/lib/kubelet/config.yaml 修改参数：maxPods为指定的值，重启kubelet服务既可。如果node节点上资源不够，可扩展几个node节点，将部分pod分配到新node节点上。

b).通过describe命令查看pod的详细信息

c).通过describe命令查看node节点详细信息

2).pod的资源配置超过了宿主机节点的资源限制：

例如：pod的yaml文件中，有配置cpu、内存的资源限制，宿主机上所有运行的pod，yaml文件中配置的cpu、内存等的总和，超过了该宿主机上最大的cpu，也会导致pod处于pending状态，无法调度到该节点上。

处理思路：

a).查看pod的详细信息或日志，看看有无污点，有无OOM内存溢出，cpu、内存、磁盘等资源。

b).查看pod的yaml文件中limit限制的配置，降低limit的值，在pod能运行的许可范围内，给pod分配较小的资源。

c).扩展宿主机的资源或增加node节点等扩展资源。

#### 57.pod的自动扩容和缩容的方法有哪些？

1).HPA方式

利用监控指标（cpu使用率、磁盘、内存）等自动扩容或缩容pod的数量

2).KPA方式

基于请求数对pod自动扩缩容（但不支持cpu的自动扩缩容）

3).基于pod资源使用情况，自动为集群设置资源占用限制和要求比例，保持最初的比例值。

如：request和limitd的比值

#### 58.k8s中service访问异常的常见问题有哪些？如何处理？

1).service没有正确匹配到后端pod的label标签

可以排查service的yaml文件里定义的标签和pod的标签是否匹配上，修改匹配一致既可。

2).kube-proxy服务故障，导致service无法提供服务

暴露service访问的时候，kube-proxy会在iptables中追加一些规则（SNAT或DNAT转发），或通过ipvs负载均衡模式，为我们实现路由和负载均衡功能，使得我们通过service能访问到后端的pod服务，如果kube-proxy服务有问题，则会导致访问不正常，可以查看kube-proxy服务是否正常，查看kube-proxy相关日志，有无资源报错、cpu内存啥的错误信息。如果是资源不够问题，可以增加资源或node节点，同时限制kube-proxy的yaml文件，limit限制调整小点资源。

#### 59.k8s中pod删除失败，有哪些情况？和如何解决？

pod删除失败的情况：

1).pod被其他资源引用，无法删除

pod可能被其他资源（如：deployment、ReplicaSet等）引用，导致无法直接删除。

解决：先删除引用该pod的资源，在删除pod

2).pod处于异常状态：如：持续崩溃重启CrashLoopBackOff、镜像拉取错误ImagePullError等，会影响删除操作，可能会导致删除失败。

解决：先解决pod的异常状态，如：修复镜像问题，调整应用配置等，让pod恢复到正常状态，然后再尝试删除。

3).k8s版本问题：

某些旧版本的k8s可能存在特定情况下与pod删除相关的bug问题或兼容性问题，可能导致pod删除失败。

解决：升级到较新的k8s稳定版本。

4).node节点故障，可能会导致pod删除失败：

如果节点出现故障，如：网络问题、存储故障等，可能导致pod删除异常。

解决：对节点进行诊断和修复，修复节点故障，恢复都正常运行状态，然后再尝试删除pod。

5).网络问题，可能会导致pod删除失败

网络问题，如：网络不稳定或存在连接问题时，可能会导致pod删除命令无法正确传递到k8s集群，从而导致删除失败。

解决：检查网络连接，确保能够正常访问k8s集群。

6).强制删除pod

如果以上访问都无法解决问题，可以尝试使用强制删除命令，如：

\# kubectl delete pod pod名称 --force--grace-period=0

\# --grace-period=0 立即删除pod而不等待默认的优雅停机时长

#### 60.pause容器的概念和作用？

每个pod里运行着一个特殊的跟容器pause容器，也叫初始化容器，其他容器为业务容器。

pod中pause容器是第一个被创建，先提前运行，提前准备好相应的运行环境，如：网络ip和volume存储数据卷资源，其他业务容器通过join加入的方式，和pause容器联在一起。

Pause容器启动前，会为容器创建虚拟一对ethernet接口，一个保留在宿主机网卡vethxx（插在网桥上），一个保留在容器网络命名空间内，并重新命名为eth0，两虚拟接口，一端进，一端出。

#### 61.同一个节点多个pod之间通信示意图？

#### 62.跨主机之间的多个pod之间通信示意图？

flannel默认是vxlan方式的如下图：

解释：flannel会为所在主机从集群网络地址中，获取一个小的网段子网subnet，本主机内所有容器ip地址从中分配。

pause容器创建一对ethernet：一个宿主机，插在网桥上（veth1），一个在pod空间中eth0.

#### 63.k8s中同一个命名空间下服务间是怎么调用的？

在同一个namespace命名空间中，服务间调用，是通过service name（服务名）访问的。

#### 64.k8s中不同命名空间下服务是怎么调用的？

对于跨命名空间的服务访问，service提供了2种访问方式：

1).直接访问：访问service名称，需要带命名空间，带集群域名的全路径访问

可以直接使用{SERVICE\_NAME}.{NAMESPACE\_NAME}.svc\_cluster.local的格式进行跨命名空间访问。

2).创建一个ExternalName类型的Service：

创建一个ExternalName类型的service，

然后用{SERVICE\_NAME}.{NAMESPACE\_NAME}.svc.cluster.local的格式访问目标命名空间下的服务。

解释：

{SERVICE\_NAME}：是服务的名称

{NAMESPACE\_NAME}：是服务所在的命名空间的名称

svc：是固定的后缀，表示这是一个服务

cluster.local：是集群的域名

在k8s中，每个服务都有一个唯一的域名，通过这个域名可以访问到服务。这种域名格式使得服务可以在集群内被轻松访问，并且可以在不同的命名空间之间进行隔离和访问控制。

ExternalName类型的Service：

就是将该Service名跟集群外部服务地址（或集群内其他命名空间的服务）做一个映射，使之访问Service名称就是访问外部服务。

#### 65.ExternalName类型的Service的概念深入理解？

理解1：ExternalName类型的Service：就是该Service名跟集群外部服务地址做一个映射，使之访问Service名称就是访问外部服务。

理解2：ExternalName类型的Service：用于引入集群外部的服务，它通过externalName属性指定外部服务的地址，可以是一个域名，如：www.baidu.com，也可以是其他命名空间的服务地址（带命名空间，带集群名的全路径的service名称）

ExternalName类型的Service的应用案例：

1).绑定集群外部域名：

#### 66.docker镜像的优化方法有哪些？

1).选择最精简的基础镜像

2).减少镜像的层数，将所有运行命令放到一层，如：RUN 命令1 命令2

3).清理镜像构建过程中的中间产物，如：RUN命令1 命令2 && rm -rf 不需要的中间产物

4).多阶段构建镜像，一个Dockefile中，有两个FROM基础镜像，上面基础镜像运行过程中的产物镜像，被下面的步骤接着引用

5).去除非必要的文件

#### 67.k8s中针对标准化输出方式的日志，如何进行收集？

标准化输出方式的日志，都会存放在各自宿主机上的/var/lib/docker/containers/各个容器id的目录中，所以只需采集宿主机上的该目录下所有容器的id的日志既可，以daemonset方式在每一个node节点部署一个filebeat的pod，该pod中的容器配置挂载宿主机上相应目录，将宿主机撒花姑娘相应目录的日志路径挂载到filebeat的pod容器中的相应路径，filebeat的配置文件和pod容器中挂载的日志路径保持一致，进而就可以搜集到pod挂载的相应宿主机上容器的日志。

#### 68.k8s中针对容器内部的日志，如何进行收集？

对于应用容器内部的日志，一般采用，一个pod中跑两个容器，一个容器是应用容器，一个容器是收集日志的filebeat容器，两个容器配置共享存储日志的资源进行收集。

具体是：pod的yaml文件中，定义一个空目录资源emptyDir（挂载宿主机上的临时目录，随pod的消失而消失），定义一个挂载卷名称，应用容器和filebeat容器同时挂载该空目录的挂载卷名称，挂载到各自容器相应的目录中，实现filebeat容器和应用容器共享日志资源，因filebeat的容器已经将日志挂载到了自己容器的相应路径下，可以修改filebeat的配置文件，收集日志的路径和挂载日志到自己容器的路径保持一致，就可以搜集相对应的应用容器日志。

#### 69.k8s中阿里云开源软件log-pilot如何收集标准化输出的日志？

1).使用log-pilot.yaml里的镜像部署一个log-poilt组件服务，里面配置了连接es的地址

2).创建要收集日志的k8s资源pod，并将容器内日志路径安log-pilot搜索使用的变量格式定义全局变量并映射到宿主机

上面详细解释：

1).log-Pilot支持声名式日志配置，可依据容器的Label或者ENV两种方法来动态地收集标准化输出或容器内日志

下面两种方法都可收集标准化输出日志或容器内日志

（1）.容器标签方式（label）,aliyun。logsSname=Spath

\# log-poilt可以动态地监听容器的时间变化，然后根据容器的标签进行分析，生成采集配置文件，收集容器日志（包括自身容器和其他容器）

（2）.环境变量方式（ENV）,aliyun\_logs\_$name=$path

# log-poilt 可以依据环境变量，动态地生成日志采集配置文件，收集容器日志（包括自身容器和其他容器）变量$name和$path的含义

$name：是我们自定义的一个字符串，它在不同的场景下指代不同的含义，在本场景中，将日志采集到ElasticSearch的时候，这个$name表示的是index.

$path

#### 71.k8s中pod的亲和性和反亲和性的概念？如何配置亲和性和反亲和性？

1).pod亲和性和反亲和性的概念：

pod亲和性：希望将带有某些标签的多个pod部署在同一个域或同一个node节点上。

pod亲和性：主要是想把pod和某个依赖的pod放在一起

例如：一个pod在一个节点上了，那么我这个pod也得在这个节点上（注：两个pod的标签不一定要相同，只是根据匹配规则决定）

注意：是以pod的标签进行标记的，如：pod1的标签是app=love，部署在node1节点，pod2的标签是app=love2，那么配置pod亲和性，通过匹配规则限制，当pod2匹配上pod1标签的时候，让pod2也部署在node1，和pod1放到一起。

pod反亲和性：某个或某些pod部署时，希望不要和带有某些标签的pod部署在同一个域或同一个node节点上。

pod反亲和性：主要相把pod和某个pod分开

例如：你这个pod在节点上了，那么我就不想和你待在同一个节点上（注：两个pod的标签不一定要相同，只是根据匹配规则决定）。

注意：是以pod的标签进行标记的，如：pod1的标签是app=love，部署在node1节点，pod2的标签是app=love2，那么配置pod反亲和性，不让pod2部署在node1，不要和pod1放到一起。

2).对于亲和性和反亲和性都有这两种规则可以设置，软策略和硬策略

软策略：preferredDuringSchedulingIgnoreDuringExecution

硬策略：requiredDuringSchedulingIgnoredDuringExecution

亲合度可以分成软策略和硬策略两种方式

软策略：尽量满足条件，满足条件最好，不满足也无所谓，若没有满足调度要求的节点的话，pod会忽略这条规则，继续完成调度过程

硬策略：必须满足条件，比较强硬，如果没有满足条件的节点的话，就不断重试直到满足条件为止。

对于亲和性和反亲和性都有这两种规则可以设置。

3).pod亲和性和反亲和性案例

pod亲和性和反亲和性案例：

（1）.先部署一个标签为app=love的pod1的nginx（应用nginx版本nginx:1.19.5）

（2）.再使用pod亲和性部署另一个pod2（标签也是app=love,nginx版本为nginx:1.20.0），和pod1部署到一起

（3）.再使用pod反亲和性部署另一个pod2（标签也是app=love，nginx版本为nginx:1.20.）不和pod1部署到一起

#### 72.简述kubernetes中PV生命周期内的阶段有哪些？

Provisioning：配置阶段

a).静态配置：管理员手动创建 PV

b).动态配置：根据PV请求，自动创建PV

Binding：绑定阶段

kubernetes：将符合要求的PV绑定到PVC上

Using：使用阶段

PV被挂载到Pod，Pod可以通过PVC访问和使用存储资源。

Reclaiming：回收阶段

Retain（保留）：数据保留，PV标记为Released，管理员手动处理。

Delete（删除）：PV和存储资源被删除。

Available：可用阶段

回收后，还未与某个PVC绑定，PV可以重新绑定到新的PVC

Released：释放阶段

PVC删除后，PV进入Released状态，等待回收处理。

Failed：失败阶段

回收失败或其他错误，PV进入Failed状态，需要管理员干预

#### 73.k8s中Etcd的特点？

一致性：提供强一致性的数据模型，确保在分布式环境中数据的一致性。

高可用性：支持多节点部署，通过Faft算法实现高可用性，即使在节点故障时也能保持系统可用。

分布式：可在多个节点上运行，适应大规模数据和高并发读写请求。

支持事务：提供原子事务支持，确保一组操作要么全部成功，要么全部失败。

监控和通知：具备监控功能，支持Watcher机制，可订阅数据变更通知。

安全性：支持TLS安全通信，提供身份验证机制和访问控制，确保数据传输的安全性。

轻量级：是一个轻量级系统，易于部署和管理，适用于简单的分布式存储需求。

#### 74.简述ETCD适应的场景？

1).适用于需要数据高一致性的场景，确保分布式环境中的数据是一致的

2).适用于服务高可用时的场景

3).适用于多节点数据分布式存储的场景

4).适用于服务之间协调和交互使用的场景

#### 75.etcd集群节点之间是怎么同步数据的？

在etcd集群中，节点之间通过Raft一致性算法实现数据同步，

Raft保证了数据的高可用性和一致性，确保在集群中的节点保持相同的数据状态。

———————————————————————————————————————————————————————————————————————————

                                                                                                                         无敌小马爱学习

分类:
[K8s云原生](https://www.cnblogs.com/junnan/category/2382299.html)

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/2469482/20210719112251.png)](https://home.cnblogs.com/u/junnan/)

[马俊南](https://home.cnblogs.com/u/junnan/)

[粉丝 \- 12](https://home.cnblogs.com/u/junnan/followers/) [关注 \- 5](https://home.cnblogs.com/u/junnan/followees/)

+加关注

0

0

[升级成为会员](https://cnblogs.vip/)

[«](https://www.cnblogs.com/junnan/p/18825195) 上一篇： [12.Alertmanager告警配置文件和告警规则详解](https://www.cnblogs.com/junnan/p/18825195 "发布于 2025-04-14 16:54")

[»](https://www.cnblogs.com/junnan/p/18843658) 下一篇： [14.k8s专题 你点到为止 我一醉方](https://www.cnblogs.com/junnan/p/18843658 "发布于 2025-04-23 22:27")

posted on
2025-04-18 17:34 [马俊南](https://www.cnblogs.com/junnan)
阅读(533)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2Fjunnan%2Fp%2F18830991&targetId=18830991&targetType=0)

[刷新页面](https://www.cnblogs.com/junnan/p/18830991#) [返回顶部](https://www.cnblogs.com/junnan/p/18830991#top)

登录后才能查看或发表评论，立即 登录 或者
[逛逛](https://www.cnblogs.com/) 博客园首页

[![](https://img2024.cnblogs.com/blog/35695/202604/35695-20260423213336272-1914399152.webp)](https://www.volcengine.com/activity/codingplan?utm_campaign=hw&utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&utm_term=cnblogs)

- [从 305 GB 到 7.4 GB：大模型 KVCache 架构演进全景](https://www.cnblogs.com/cswuyg/p/19981922)
- [DeepSeek V4模型的Agent能力实测](https://www.cnblogs.com/zhayujie/p/19935607/deepseek-v4-eval)
- [C# 15 类型系统改进：Union Types](https://www.cnblogs.com/hez2010/p/19891530/union-types-in-csharp-15)
- [你能被装进一个文件里吗？——7 万人把同事“蒸馏”成了 AI](https://www.cnblogs.com/wmyskxz/p/19854791)
- [别再吹牛了，100% Vibe Coding 存在无法自洽的逻辑漏洞！](https://www.cnblogs.com/mengxiang2/p/19796426)

昵称：
[马俊南](https://home.cnblogs.com/u/junnan/)

园龄：
[4年9个月](https://home.cnblogs.com/u/junnan/ "入园时间：2021-07-19")

粉丝：
[12](https://home.cnblogs.com/u/junnan/followers/)

关注：
[5](https://home.cnblogs.com/u/junnan/followees/)

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

- [我的随笔](https://www.cnblogs.com/junnan/p/ "我的博客的随笔列表")
- [我的评论](https://www.cnblogs.com/junnan/MyComments.html "我的发表过的评论列表")
- [我的参与](https://www.cnblogs.com/junnan/OtherPosts.html "我评论过的随笔列表")
- [最新评论](https://www.cnblogs.com/junnan/comments "我的博客的评论列表")
- [我的标签](https://www.cnblogs.com/junnan/tag/ "我的博客的标签列表")

### [随笔分类](https://www.cnblogs.com/junnan/post-categories)

- [00.闲杂(5)](https://www.cnblogs.com/junnan/category/2493325.html)
- [01.Linux基础(5)](https://www.cnblogs.com/junnan/category/2424851.html)
- [02.Linux提升篇(1)](https://www.cnblogs.com/junnan/category/2476831.html)
- [03.Nginx(34)](https://www.cnblogs.com/junnan/category/2492857.html)
- [04.MySQL数据库《黑马》(2)](https://www.cnblogs.com/junnan/category/2238502.html)
- [Docker容器(6)](https://www.cnblogs.com/junnan/category/2247973.html)
- [Java(42)](https://www.cnblogs.com/junnan/category/2479860.html)
- [K8s云原生(40)](https://www.cnblogs.com/junnan/category/2382299.html)
- [Python(59)](https://www.cnblogs.com/junnan/category/2468411.html)
- [shell+CI+CD(3)](https://www.cnblogs.com/junnan/category/2420861.html)
- [报错！(2)](https://www.cnblogs.com/junnan/category/2422653.html)
- [震惊！！！作为运维工程师，你还不知道这些事？(10)](https://www.cnblogs.com/junnan/category/2393219.html)

### 随笔档案

- [2026年4月(9)](https://www.cnblogs.com/junnan/p/archive/2026/04)
- [2026年3月(28)](https://www.cnblogs.com/junnan/p/archive/2026/03)
- [2026年2月(3)](https://www.cnblogs.com/junnan/p/archive/2026/02)
- [2026年1月(4)](https://www.cnblogs.com/junnan/p/archive/2026/01)
- [2025年12月(20)](https://www.cnblogs.com/junnan/p/archive/2025/12)
- [2025年11月(33)](https://www.cnblogs.com/junnan/p/archive/2025/11)
- [2025年10月(17)](https://www.cnblogs.com/junnan/p/archive/2025/10)
- [2025年9月(14)](https://www.cnblogs.com/junnan/p/archive/2025/09)
- [2025年8月(8)](https://www.cnblogs.com/junnan/p/archive/2025/08)
- [2025年7月(4)](https://www.cnblogs.com/junnan/p/archive/2025/07)
- [2025年6月(4)](https://www.cnblogs.com/junnan/p/archive/2025/06)
- [2025年5月(7)](https://www.cnblogs.com/junnan/p/archive/2025/05)
- [2025年4月(7)](https://www.cnblogs.com/junnan/p/archive/2025/04)
- [2025年3月(5)](https://www.cnblogs.com/junnan/p/archive/2025/03)
- [2025年2月(2)](https://www.cnblogs.com/junnan/p/archive/2025/02)
- [2025年1月(1)](https://www.cnblogs.com/junnan/p/archive/2025/01)
- [2024年12月(7)](https://www.cnblogs.com/junnan/p/archive/2024/12)
- [2024年11月(4)](https://www.cnblogs.com/junnan/p/archive/2024/11)
- [2024年10月(4)](https://www.cnblogs.com/junnan/p/archive/2024/10)
- [2024年9月(6)](https://www.cnblogs.com/junnan/p/archive/2024/09)
- [2024年5月(1)](https://www.cnblogs.com/junnan/p/archive/2024/05)
- [2024年4月(3)](https://www.cnblogs.com/junnan/p/archive/2024/04)
- [2022年11月(2)](https://www.cnblogs.com/junnan/p/archive/2022/11)

### [阅读排行榜](https://www.cnblogs.com/junnan/most-viewed)

- [1\. 12.Alertmanager告警配置文件和告警规则详解(1855)](https://www.cnblogs.com/junnan/p/18825195)
- [2\. sftp 报错 Connection closed(1396)](https://www.cnblogs.com/junnan/p/18426952)
- [3\. SecureCRT命令行快捷键（功能归类版）(1319)](https://www.cnblogs.com/junnan/p/18145993)
- [4\. Docker与containerd的关系与区别(1073)](https://www.cnblogs.com/junnan/p/18589605)
- [5\. SecureCRT命令行快捷键（字母顺序版）(784)](https://www.cnblogs.com/junnan/p/18163352)

### [评论排行榜](https://www.cnblogs.com/junnan/most-commented)

- [1\. 25.B站薪享宏福笔记——第十章（1）k8s HELM(8)](https://www.cnblogs.com/junnan/p/18911293)
- [2\. 27.B站薪享宏福笔记——第十一章（1）镜像仓库Harbor && 日志收集Loki(5)](https://www.cnblogs.com/junnan/p/18974073)
- [3\. 37.B站薪享宏福笔记——第十三章（2）CRD 自定义资源(3)](https://www.cnblogs.com/junnan/p/19028466)
- [4\. 23.B站薪享宏福笔记——第八章 k8s 调度器(3)](https://www.cnblogs.com/junnan/p/18911288)
- [5\. 9.K8s集群常见报错3(1)](https://www.cnblogs.com/junnan/p/18816135)

### [推荐排行榜](https://www.cnblogs.com/junnan/most-liked)

- [1\. 24.B站薪享宏福笔记——第九章 k8s 集群安全机制(1)](https://www.cnblogs.com/junnan/p/18911291)
- [2\. 16.B站薪享宏福笔记——第一章 k8s介绍说明(1)](https://www.cnblogs.com/junnan/p/18866042)

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)