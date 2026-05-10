# k8s面试题大全（持续更新中）_ks8 面试题【转】 - 博客园

URL: https://www.cnblogs.com/paul8339/p/18844452

# [k8s面试题大全（持续更新中）\_ks8 面试题【转】](https://www.cnblogs.com/paul8339/p/18844452 "发布于 2025-04-24 13:48")

**阅读目录**

- [kubelet的功能、作用是什么？（重点，经常会问）](https://www.cnblogs.com/paul8339/p/18844452#_label0)
- [kube-api-server的端口是多少？各个pod是如何访问kube-api-server的？](https://www.cnblogs.com/paul8339/p/18844452#_label1)
- [pod是什么？](https://www.cnblogs.com/paul8339/p/18844452#_label2)
- [pause容器作用是什么？（经常问）](https://www.cnblogs.com/paul8339/p/18844452#_label3)
- [pod的重启策略有哪些？（经常问）](https://www.cnblogs.com/paul8339/p/18844452#_label4)
- [pod的镜像拉取策略有哪几种？（经常问）](https://www.cnblogs.com/paul8339/p/18844452#_label5)
- [pod的存活探针有哪几种？（必须记住3种探测方式，重点，经常问）](https://www.cnblogs.com/paul8339/p/18844452#_label6)
- [pod的就绪探针有哪几种？（必须记住3种探测方式，重点，经常问）](https://www.cnblogs.com/paul8339/p/18844452#_label7)

前言

本篇模拟面试官提问的各种docker，k8s问题，意在提高面试通过率，欢迎在评论区探讨，同步进步。

docker的工作原理是什么，讲一下

docker是一个Client-Server结构的系统，docker守护进程运行在宿主机上，守护进程从客户端接受命令并管理运行在主机上的容器，容器是一个运行时环境，这就是我们说的集装箱。

docker的组成包含哪几大部分

一个完整的docker有以下几个部分组成：

1、docker client，客户端，为用户提供一系列可执行命令，用户用这些命令实现跟 docker daemon 交互；

2、docker daemon，守护进程，一般在宿主主机后台运行，等待接收来自客户端的请求消息；

3、docker image，镜像，镜像run之后就生成为docker容器；

4、docker container，容器，一个系统级别的服务，拥有自己的ip和系统目录结构；运行容器前需要本地存在对应的镜像，如果本地不存在该镜像则就去镜像仓库下载。

docker 使用客户端-服务器 (C/S) 架构模式，使用远程api来管理和创建docker容器。docker 容器通过 docker 镜像来创建。容器与镜像的关系类似于面向对象编程中的对象与类。

docker与传统虚拟机的区别什么？

1、传统虚拟机是需要安装整个操作系统的，然后再在上面安装业务应用，启动应用，通常需要几分钟去启动应用，而docker是直接使用镜像来运行业务容器的，其容器启动属于秒级别；

2、Docker需要的资源更少，Docker在操作系统级别进行虚拟化，Docker容器和内核交互，几乎没有性能损耗，而虚拟机运行着整个操作系统，占用物理机的资源就比较多;

3、Docker更轻量，Docker的架构可以共用一个内核与共享应用程序库，所占内存极小;同样的硬件环境，Docker运行的镜像数远多于虚拟机数量，对系统的利用率非常高;

4、与虚拟机相比，Docker隔离性更弱，Docker属于进程之间的隔离，虚拟机可实现系统级别隔离;

5、Docker的安全性也更弱，Docker的租户root和宿主机root相同，一旦容器内的用户从普通用户权限提升为root权限，它就直接具备了宿主机的root权限，进而可进行无限制的操作。虚拟机租户root权限和宿主机的root虚拟机权限是分离的，并且虚拟机利用如Intel的VT-d和VT-x的ring-1硬件隔离技术，这种技术可以防止虚拟机突破和彼此交互，而容器至今还没有任何形式的硬件隔离;

6、Docker的集中化管理工具还不算成熟，各种虚拟化技术都有成熟的管理工具，比如：VMware vCenter提供完备的虚拟机管理能力;

7、Docker对业务的高可用支持是通过快速重新部署实现的，虚拟化具备负载均衡，高可用、容错、迁移和数据保护等经过生产实践检验的成熟保障机制，Vmware可承诺虚拟机99.999%高可用，保证业务连续性;

8、虚拟化创建是分钟级别的，Docker容器创建是秒级别的，Docker的快速迭代性，决定了无论是开发、测试、部署都可以节省大量时间;

9、虚拟机可以通过镜像实现环境交付的一致性，但镜像分发无法体系化，Docker在Dockerfile中记录了容器构建过程，可在集群中实现快速分发和快速部署。

docker技术的三大核心概念是什么？

镜像：镜像是一种轻量级、可执行的独立软件包，它包含运行某个软件所需的所有内容，我们把应用程序和配置依赖打包好形成一个可交付的运行环境(包括代码、运行时需要的库、环境变量和配置文件等)，这个打包好的运行环境就是image镜像文件。

容器：容器是基于镜像创建的，是镜像运行起来之后的一个实例，容器才是真正运行业务程序的地方。如果把镜像比作程序里面的类，那么容器就是对象。

镜像仓库：存放镜像的地方，研发工程师打包好镜像之后需要把镜像上传到镜像仓库中去，然后就可以运行有仓库权限的人拉取镜像来运行容器了。

centos镜像几个G，但是docker centos镜像才几百兆，这是为什么？

一个完整的Linux操作系统包含Linux内核和rootfs根文件系统，即我们熟悉的/dev、/proc/、/bin等目录。我们平时看到的centOS除了rootfs，还会选装很多软件，服务，图形桌面等，所以centOS镜像有好几个G也不足为奇。

而对于容器镜像而言，所有容器都是共享宿主机的Linux 内核的，而对于docker镜像而言，docker镜像只需要提供一个很小的rootfs即可，只需要包含最基本的命令，工具，程序库即可，所有docker镜像才会这么小。

讲一下镜像的分层结构以及为什么要使用镜像的分层结构？

一个新的镜像其实是从 base 镜像一层一层叠加生成的。每安装一个软件，dockerfile中使用RUN命令，就会在现有镜像的基础上增加一层，这样一层一层的叠加最后构成整个镜像。所以我们docker pull拉取一个镜像的时候会看到docker是一层层拉取的。

分层机构最大的一个好处就是 ： 共享资源。比如：有多个镜像都从相同的 base 镜像构建而来，那么 Docker Host 只需在磁盘上保存一份 base 镜像；同时内存中也只需加载一份 base 镜像，就可以为所有容器服务了。而且镜像的每一层都可以被共享。

讲一下容器的copy-on-write特性，修改容器里面的内容会修改镜像吗？

我们知道，镜像是分层的，镜像的每一层都可以被共享，同时，镜像是只读的。当一个容器启动时，一个新的可写层被加载到镜像的顶部，这一层通常被称作“容器层”，“容器层”之下的都叫“镜像层”。

所有对容器的改动 \- 无论添加、删除、还是修改文件，都只会发生在容器层中，因为只有容器层是可写的，容器层下面的所有镜像层都是只读的。镜像层数量可能会很多，所有镜像层会联合在一起组成一个统一的文件系统。如果不同层中有一个相同路径的文件，比如 /a，上层的 /a 会覆盖下层的 /a，也就是说用户只能访问到上层中的文件 /a。在容器层中，用户看到的是一个叠加之后的文件系统。

添加文件时：在容器中创建文件时，新文件被添加到容器层中。

读取文件：在容器中读取某个文件时，Docker 会从上往下依次在各镜像层中查找此文件。一旦找到，立即将其复制到容器层，然后打开并读入内存。

修改文件：在容器中修改已存在的文件时，Docker 会从上往下依次在各镜像层中查找此文件。一旦找到，立即将其复制到容器层，然后修改之。

删除文件：在容器中删除文件时，Docker 也是从上往下依次在镜像层中查找此文件。找到后，会在容器层中记录下此删除操作。

只有当需要修改时才复制一份数据，这种特性被称作 Copy-on-Write。可见，容器层保存的是镜像变化的部分，不会对镜像本身进行任何修改。

简单描述一下Dockerfile的整个构建镜像过程

好的。

1、首先，创建一个目录用于存放应用程序以及构建过程中使用到的各个文件等；

2、然后，在这个目录下创建一个Dockerfile文件，一般建议Dockerfile的文件名就是Dockerfile；

3、编写Dockerfile文件，编写指令，如，使用FROM 指令指定基础镜像，COPY指令复制文件，RUN指令指定要运行的命令，ENV设置环境变量，EXPOSE指定容器要暴露的端口，WORKDIR设置当前工作目录，CMD容器启动时运行命令，等等指令构建镜像；

4、Dockerfile编写完成就可以构建镜像了，使用docker build -t 镜像名:tag . 命令来构建镜像，最后一个点是表示当前目录，docker会默认寻找当前目录下的Dockerfile文件来构建镜像，如果不使用默认，可以使用-f参数来指定dockerfile文件，如：docker build -t 镜像名:tag -f /xx/xxx/Dockerfile ；

5、使用docker build命令构建之后，docker就会将当前目录下所有的文件发送给docker daemon，顺序执行Dockerfile文件里的指令，在这过程中会生成临时容器，在临时容器里面安装RUN指定的命令，安装成功后，docker底层会使用类似于docker commit命令来将容器保存为镜像，然后删除临时容器，以此类推，一层层的构建镜像，运行临时容器安装软件，直到最后的镜像构建成功。

Dockerfile构建镜像出现异常，如何排查？

首先，Dockerfile是一层一层的构建镜像，期间会产生一个或多个临时容器，构建过程中其实就是在临时容器里面安装应用，如果因为临时容器安装应用出现异常导致镜像构建失败，这时容器虽然被清理掉了，但是期间构建的中间镜像还在，那么我们可以根据异常时上一层已经构建好的临时镜像，将临时镜像运行为容器，然后在容器里面运行安装命令来定位具体的异常。

Dockerfile的基本指令有哪些？

FROM 指定基础镜像（必须为第一个指令，因为需要指定使用哪个基础镜像来构建镜像）；

MAINTAINER 设置镜像作者相关信息，如作者名字，日期，邮件，联系方式等；

COPY 复制文件到镜像；

ADD 复制文件到镜像（ADD与COPY的区别在于，ADD会自动解压tar、zip、tgz、xz等归档文件，而COPY不会，同时ADD指令还可以接一个url下载文件地址，一般建议使用COPY复制文件即可，文件在宿主机上是什么样子复制到镜像里面就是什么样子这样比较好）；

ENV 设置环境变量；

EXPOSE 暴露容器进程的端口，仅仅是提示别人容器使用的哪个端口，没有过多作用；

VOLUME 数据卷持久化，挂载一个目录；

WORKDIR 设置工作目录，如果目录不在，则会自动创建目录；

RUN 在容器中运行命令，RUN指令会创建新的镜像层，RUN指令经常被用于安装软件包；

CMD 指定容器启动时默认运行哪些命令，如果有多个CMD，则只有最后一个生效，另外，CMD指令可以被docker run之后的参数替换；

ENTRYOINT 指定容器启动时运行哪些命令，如果有多个ENTRYOINT，则只有最后一个生效，另外，如果Dockerfile中同时存在CMD和ENTRYOINT，那么CMD或docker run之后的参数将被当做参数传递给ENTRYOINT；

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

如何加固docker镜像的安全？

不懂，未完。

如何进入容器？使用哪个命令

进入容器有两种方法：docker attach、docker exec；

docker attach命令是attach到容器启动命令的终端，docker exec 是另外在容器里面启动一个TTY终端。

docker run -d centos /bin/bash -c "while true;do sleep 2;echo I\_am\_a\_container;done"

3274412d88ca4f1d1292f6d28d46f39c14c733da5a4085c11c6a854d30d1cde0

docker attach 3274412d88ca4f #attach进入容器

Ctrl + c 退出，Ctrl + c会直接关闭容器终端，这样容器没有进程一直在前台运行就会死掉了

Ctrl + pq 退出（不会关闭容器终端停止容器，仅退出）

docker exec -it 3274412d88ca /bin/bash #exec进入容器

\[root@3274412d88ca /\]# ps -ef #进入到容器了开启了一个bash进程

UID PID PPID C STIME TTY TIME CMD

root 1 0 0 05:31 ? 00:00:01 /bin/bash -c while true;do sleep 2;echo I\_am\_a\_container;done

root 306 0 1 05:41 pts/0 00:00:00 /bin/bash

root 322 1 0 05:41 ? 00:00:00 /usr/bin/coreutils --coreutils-prog-shebang=sleep /usr/bin/sleep 2

root 323 306 0 05:41 pts/0 00:00:00 ps -ef

\[root@3274412d88ca /\]#exit #退出容器，仅退出我们自己的bash窗口

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

小结：attach是直接进入容器启动命令时的终端，不会启动新的进程；exec则是在容器里面打开新的终端，会启动新的进程；一般建议使用exec进入容器。

什么是k8s？说出你的理解

K8s是kubernetes的简称，其本质是一个开源的容器编排系统，主要用于管理容器化的应用，其目标是让部署容器化的应用简单并且高效（powerful）,Kubernetes提供了应用部署，规划，更新，维护的一种机制。

说简单点：k8s就是一个编排容器的系统，一个可以管理容器应用全生命周期的工具，从创建应用，应用的部署，应用提供服务，扩容缩容应用，应用更新，都非常的方便，而且还可以做到故障自愈，所以，k8s是一个非常强大的容器编排系统。

k8s的组件有哪些，作用分别是什么？

参考官网：https://kubernetes.io/zh-cn/docs/concepts/overview/components/

k8s主要由master节点和node节点构成。master节点负责管理集群，node节点是容器应用真正运行的地方。

master节点包含的组件有：kube-api-server、kube-controller-manager、kube-scheduler、etcd。

node节点包含的组件有：kubelet、kube-proxy、container-runtime。

kube-api-server：以下简称api-server，api-server是k8s最重要的核心组件之一，它是k8s集群管理的统一访问入口，提供了RESTful API接口, 实现了认证、授权和准入控制等安全功能；api-server还是其他组件之间的数据交互和通信的枢纽，其他组件彼此之间并不会直接通信，其他组件对资源对象的增、删、改、查和监听操作都是交由api-server处理后，api-server再提交给etcd数据库做持久化存储，只有api-server才能直接操作etcd数据库，其他组件都不能直接操作etcd数据库，其他组件都是通过api-server间接的读取，写入数据到etcd。

kube-controller-manager：以下简称controller-manager，controller-manager是k8s中各种控制器的的管理者，是k8s集群内部的管理控制中心，也是k8s自动化功能的核心；controller-manager内部包含deployment控制器、replicaSet控制器、statefulset控制器、daemonset控制器、job控制器、cronjob控制器、node控制器、endpoint控制器等等各种资源对象的控制器，每种控制器都负责一种特定资源的控制流程，而controller-manager正是这些controller的核心管理者。

kube-scheduler：以下简称scheduler，scheduler负责集群资源调度，其作用是将待调度的pod通过一系列复杂的调度算法计算出最合适的node节点，然后将pod绑定到目标节点上。shceduler会根据pod的信息，全部节点信息列表，过滤掉不符合要求的节点，过滤出一批候选节点，然后给候选节点打分，选分最高的就是最优节点，scheduler就会把目标pod安置到该节点。

etcd：etcd是一个分布式的键值对存储数据库，主要是用于保存k8s集群状态数据，比如，pod，service等资源对象的信息；etcd可以是单个也可以有多个，多个就是etcd数据库集群，etcd通常部署奇数个实例，在大规模集群中，etcd有5个或7个节点就足够了；另外说明一点，etcd本质上可以不与master节点部署在一起，只要master节点能通过网络连接etcd数据库即可。

kubelet：每个node节点上都有一个kubelet服务进程，kubelet作为连接master和各node之间的桥梁，负责维护pod和容器的生命周期，当监听到master下发到本节点的任务时，比如创建、更新、终止pod等任务，kubelet 即通过控制docker来创建、更新、销毁容器；kubelet还会定时执行pod中容器定义的探针，然后根据容器重启策略执行对应的操作。每个kubelet进程都会在api-server上注册本节点自身的信息，用于定期向master汇报本节点资源的使用情况。

kube-proxy：kube-proxy运行在node节点上，在Node节点上实现pod网络代理，维护网络规则和四层负载均衡工作，kube-proxy会监听api-server中从而获取service和endpoint的变化情况，创建并维护路由规则以提供服务ip和负载均衡功能。简单理解此进程是Service的透明代理兼负载均衡器，其核心功能是将到某个Service的访问请求转发到后端的多个pod实例上。

container-runtime：容器运行时环境，即运行容器所需要的一系列程序，目前k8s支持的容器运行时有很多，如docker、rkt、containerd或其他，比较受欢迎的是docker，但是新版的k8s已经宣布弃用docker。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## kubelet的功能、作用是什么？（重点，经常会问）

答：kubelet部署在每个node节点上的，它主要有4个功能：

1、节点管理。kubelet启动时会向api-server进行注册，然后会定时的向api-server汇报本节点信息状态，资源使用状态等，这样master就能够知道node节点的资源剩余，节点是否失联等等相关的信息了。master知道了整个集群所有节点的资源情况，这对于 pod 的调度和正常运行至关重要。

2、pod管理。kubelet负责维护node节点上pod的生命周期，当kubelet监听到master的下发到自己节点的任务时，比如要创建、更新、删除一个pod，kubelet 就会通过CRI（容器运行时接口）插件来调用不同的容器运行时来创建、更新、删除容器；常见的容器运行时有docker、containerd、rkt等等这些容器运行时，我们最熟悉的就是docker了，但在新版本的k8s已经弃用docker了，k8s1.24版本中已经使用containerd作为容器运行时了。

3、容器健康检查。pod中可以定义启动探针、存活探针、就绪探针等3种，我们最常用的就是存活探针、就绪探针，kubelet 会定期调用容器中的探针来检测容器是否存活，是否就绪，如果是存活探针，则会根据探测结果对检查失败的容器进行相应的重启策略；

4、Metrics Server资源监控。在node节点上部署Metrics Server用于监控node节点、pod的CPU、内存、文件系统、网络使用等资源使用情况，而kubelet则通过Metrics Server获取所在节点及容器的上的数据。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## kube-api-server的端口是多少？各个pod是如何访问kube-api-server的？

kube-api-server的端口是8080和6443，前者是http的端口，后者是https的端口，（注意：有些8080是k8s低版本的才有的端口，高版本中不开放此端口了）以我本机使用kubeadm安装的k8s为例：

在命名空间的kube-system命名空间里，有一个名称为kube-api-master的pod，这个pod就是运行着kube-api-server进程，它绑定了master主机的ip地址和6443端口，但是在default命名空间下，存在一个叫kubernetes的服务，该服务对外暴露端口为443，目标端口6443，这个服务的ip地址是ClusterIP地址池里面的第一个地址，同时这个服务的yaml定义里面并没有指定标签选择器，也就是说这个kubernetes服务所对应的endpoint是手动创建的，该endpoint也是名称叫做kubernetes，该endpoint的yaml定义里面代理到master节点的6443端口，也就是kube-api-server的ip和端口。这样一来，其他pod访问kube-api-server的整个流程就是：pod创建后嵌入了环境变量，pod获取到了kubernetes这个服务的ip和443端口，请求到kubernetes这个服务其实就是转发到了master节点上的6443端口的kube-api-server这个pod里面。

k8s中命名空间的作用是什么？

namespace是kubernetes系统中的一种非常重要的资源，namespace的主要作用是用来实现多套环境的资源隔离，或者说是多租户的资源隔离。

k8s通过将集群内部的资源分配到不同的namespace中，可以形成逻辑上的隔离，以方便不同的资源进行隔离使用和管理。不同的命名空间可以存在同名的资源，命名空间为资源提供了一个作用域。

可以通过k8s的授权机制，将不同的namespace交给不同的租户进行管理，这样就实现了多租户的资源隔离，还可以结合k8s的资源配额机制，限定不同的租户能占用的资源，例如CPU使用量、内存使用量等等来实现租户可用资源的管理。

k8s提供了大量的REST接口，其中有一个是Kubernetes Proxy API接口，简述一下这个Proxy接口的作用，已经怎么使用。

好的。kubernetes proxy api接口，从名称中可以得知，proxy是代理的意思，其作用就是代理rest请求；Kubernets API server 将接收到的rest请求转发到某个node上的kubelet守护进程的rest接口，由该kubelet进程负责响应。我们可以使用这种Proxy接口来直接访问某个pod，这对于逐一排查pod异常问题很有帮助。

下面是一些简单的例子：

```
http://<kube-api-server>:<api-sever-port>/api/v1/nodes/node名称/proxy/pods #查看指定node的所有pod信息
http://<kube-api-server>:<api-sever-port>/api/v1/nodes/node名称/proxy/stats #查看指定node的物理资源统计信息
http://<kube-api-server>:<api-sever-port>/api/v1/nodes/node名称/proxy/spec #查看指定node的概要信息

http://<kube-api-server>:<api-sever-port>/api/v1/namespace/命名名称/pods/pod名称/pod服务的url/ #访问指定pod的程序页面
http://<kube-api-server>:<api-sever-port>/api/v1/namespace/命名名称/servers/svc名称/url/ #访问指定server的url程序页面
```

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## pod是什么？

在kubernetes的世界中，k8s并不直接处理容器，而是使用多个容器共存的理念，这组容器就叫做pod。pod是k8s中可以创建和管理的最小单元，是资源对象模型中由用户创建或部署的最小资源对象模型，其他的资源对象都是用来支撑pod对象功能的，比如，pod控制器就是用来管理pod对象的，service或者imgress资源对象是用来暴露pod引用对象的，persistentvolume资源是用来为pod提供存储等等，简而言之，k8s不会直接处理容器，而是pod，pod才是k8s中可以创建和管理的最小单元，也是基本单元。

pod的原理是什么？

在微服务的概念里，一般的，一个容器会被设计为运行一个进程，除非进程本身产生子进程，这样，由于不能将多个进程聚集在同一个单独的容器中，所以需要一种更高级的结构将容器绑定在一起，并将它们作为一个单元进行管理，这就是k8s中pod的背后原理。

pod有什么特点？

1、每个pod就像一个独立的逻辑机器，k8s会为每个pod分配一个集群内部唯一的ip地址，所以每个pod都拥有自己的ip地址、主机名、进程等；

2、一个pod可以包含1个或多个容器，1个容器一般被设计成只运行1个进程，1个pod只可能运行在单个节点上，即不可能1个pod跨节点运行，pod的生命

周期是短暂，也就是说pod可能随时被消亡（如节点异常，pod异常等情况）；

2、每一个pod都有一个特殊的被称为"根容器"的pause容器，也称Infra容器，pause容器对应的镜像属于k8s平台的一部分，除了pause容器，每个pod还

包含一个或多个跑业务相关组件的应用容器；

3、一个pod中的容器共享network命名空间；

4、一个pod里的多个容器共享pod ip，这就意味着1个pod里面的多个容器的进程所占用的端口不能相同，否则在这个pod里面就会产生端口冲突；既然每

个pod都有自己的ip和端口空间，那么对不同的两个pod来说就不可能存在端口冲突；

5、应该将应用程序组织到多个pod中，而每个pod只包含紧密相关的组件或进程；

6、pod是k8s中扩容、缩容的基本单位，也就是说k8s中扩容缩容是针对pod而言而非容器。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## pause容器作用是什么？（经常问）

每个pod里运行着一个特殊的被称之为pause的容器，也称根容器，而其他容器则称为业务容器；

1、创建pause容器主要是为了为业务容器提供 Linux命名空间，共享基础：包括 pid、icp、net 等，这些业务容器共享pause容器的网络命名空间和volume挂载卷，当pod被创建时，pod首先会创建pause容器，从而把其他业务容器加入pause容器，从而让所有业务容器都在同一个网络命名空间中，这样就可以实现网络共享。这种网络命名空间共享设计确保了同一个pod内的容器可以直接通过localhost地址互相通信，从而实现高效的内部通信。

2、pod还可以共享存储，在pod级别引入数据卷volume，业务容器都可以挂载这个数据卷从而实现持久化存储。

3、pause容器还负责处理僵尸进程。在传统的Unix系统中，当一个子进程结束而其父进程尚未读取其退出状态时，子进程会成为僵尸进程，占用系统资源。pause容器通过持续监听并清理这些僵尸进程，优化了系统的资源管理。

4、由于pause容器始终保持运行状态，它还承担了维护pod ip地址的角色。pod的ip地址通常是动态分配的，但只要pause容器在运行，就可以维持这个ip地址不变，即便pod内的其他容器重新启动也不会影响ip地址。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## pod的重启策略有哪些？（经常问）

pod的重启策略（RestartPolicy）决定了当容器异常退出或健康检查失败时，kubelet将如何响应。（注意是kubelet重启容器，因为是kubelet负责容器的健康检测）

需要注意的是，虽然名为pod的重启策略（更规范的说法应该是pod中容器重启策略），但实际上是作用于pod内的所有容器。所有容器都将遵守这个策略，而不是单独的某个容器。

可以通过pod.spec.restartPolicy字段配置重启容器的策略，重启策略如下3种配置：

Always: 当容器终止退出后，总是重启容器，默认策略就是Always。

OnFailure: 当容器异常退出，退出状态码非0时，才重启容器。

Never: 当容器终止退出，不管退出状态码是什么，从不重启容器。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## pod的镜像拉取策略有哪几种？（经常问）

pod镜像拉取策略可以通过imagePullPolicy字段配置镜像拉取策略，主要有3种镜像拉取策略，如下：

Always: 默认值，总是重新拉取，即每次创建pod都会重新从镜像仓库拉取一次镜像。

IfNotPresent: 镜像在node节点宿主机上不存在时才拉取。

Never: 永远不会主动拉取镜像，仅使用本地镜像，需要你手动拉取镜像到node节点，如果本地节点不存在镜像则pod启动失败。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## pod的存活探针有哪几种？（必须记住3种探测方式，重点，经常问）

kubernetes可以通过存活探针检查容器是否还在运行，可以为pod中的每个容器单独定义存活探针，kubelet将定期执行探针，如果探测失败，将杀死容器，并根据restartPolicy策略来决定是否重启容器，kubernetes提供了3种探测容器的存活探针，如下：

```
httpGet：通过容器的ip、端口、路径发送http 请求，返回200-400范围内的状态码表示成功。
exec：在容器内执行shell命令，根据命令退出状态码是否为0进行判断，0表示健康，非0表示不健康。
TCPSocket：与容器的ip、端口建立TCP Socket链接，能建立则说明探测成功，不能建立则说明探测失败。
```

存活探针的属性参数有哪几个？

存活探针的附加属性参数有以下几个：

initialDelaySeconds：表示在容器启动后延时多久秒才开始探测；

periodSeconds：表示执行探测的频率，即间隔多少秒探测一次，默认间隔周期是10秒，最小1秒；

timeoutSeconds：表示探测超时时间，默认1秒，最小1秒，表示容器必须在超时时间范围内做出响应，否则视为本次探测失败；

successThreshold：表示最少连续探测成功多少次才被认定为成功，默认是1，对于liveness必须是1，最小值是1；

failureThreshold：表示连续探测失败多少次才被认定为失败，默认是3，连续3次失败，k8s 将根据pod重启策略对容器做出决定；

注意：定义存活探针时，一定要设置initialDelaySeconds属性，该属性为初始延时，如果不设置，默认容器启动时探针就开始探测了，这样可能会存在

应用程序还未启动就绪，就会导致探针检测失败，k8s就会根据pod重启策略杀掉容器然后再重新创建容器的莫名其妙的问题。

在生产环境中，一定要定义一个存活探针。

[回到顶部](https://www.cnblogs.com/paul8339/p/18844452#_labelTop)

## pod的就绪探针有哪几种？（必须记住3种探测方式，重点，经常问）

我们知道，当一个pod启动后，就会立即加入service的endpoint ip列表中，并开始接收到客户端的链接请求，假若此时pod中的容器的业务进程还没有初始化完毕，那么这些客户端链接请求就会失败，为了解决这个问题，kubernetes提供了就绪探针来解决这个问题的。

在pod中的容器定义一个就绪探针，就绪探针周期性检查容器，如果就绪探针检查失败了，说明该pod还未准备就绪，不能接受客户端链接，则该pod将从endpoint列表中移除，被剔除了service就不会把请求分发给该pod，然后就绪探针继续检查，如果随后容器就绪，则再重新把pod加回endpoint列表。k8s提供了3种就绪探针，如下：

httpGet：通过容器的ip、容器的端口以及路径来发送http get请求，返回200-400范围内的状态码表示请求成功。

exec：在容器内执行shell命令，它根据shell命令退出状态码是否为0进行判断，0表示健康，非0表示不健康。

TCPSocket：通过容器的ip、端口建立TCP Socket链接，能正常建立链接，则说明探针成功，不能正常建立链接，则探针失败。

就绪探针的属性参数有哪些

就绪探针的附加属性参数有以下几个：

initialDelaySeconds：延时秒数，即容器启动多少秒后才开始探测，不写默认容器启动就探测；

periodSeconds ：执行探测的频率（秒），默认为10秒，最低值为1；

timeoutSeconds ：超时时间，表示探测时在超时时间内必须得到响应，负责视为本次探测失败，默认为1秒，最小值为1；

failureThreshold ：连续探测失败的次数，视为本次探测失败，默认为3次，最小值为1次；

successThreshold ：连续探测成功的次数，视为本次探测成功，默认为1次，最小值为1次；

1

2

3

4

5

就绪探针与存活探针区别是什么？

两者作用不一样。

存活探针，是检测容器是否存活，如果检测失败，kubelet将调用容器运行时（如docker）将检查失败的容器杀死，创建新的启动容器来保持pod正常工作；

就绪探针，是检测容器是否可以正常接收流量，当就绪探针检查失败，并不重启容器，而是将pod移出endpoint列表，就绪探针确保了service中的pod都是可用的，确保客户端只与正常的pod交互并且客户端永远不会知道系统存在问题。

简单讲一下 pod创建过程（经常问，必须牢记）

情况一、如果面试官问的是使用kubectl run命令创建的pod，可以这样说：

#注意：kubectl run 在旧版本中创建的是deployment，但在新的版本中创建的是pod则其创建过程不涉及deployment

如果是单独的创建一个pod，则其创建过程是这样的：

1、首先，用户通过kubectl或其他api客户端工具提交需要创建的pod信息给api-server；

2、api-server验证客户端的用户权限信息，验证通过开始处理创建请求生成pod对象信息，并将信息存入etcd，然后返回确认信息给客户端；

3、api-server开始反馈etcd中pod对象的变化，其他组件使用watch机制跟踪api-server上的变动；

4、scheduler发现有新的pod对象要创建，开始调用内部算法机制为pod分配最佳的主机，并将结果信息更新至api-server；

5、node节点上的kubelet通过watch机制跟踪api-server发现有pod调度到本节点，通过CRI容器运行时接口调用底层的docker启动容器，如果pod定义了pv,此时还会调用CSI容器存储接口分配存储，然后还会调用CNI容器网络接口给pod 分配IP，这样等等一系列的创建pod过程，并将创建成功的结果反馈api-server，然后kubelet就会根据pod定义的探针持续的对容器进行健康检查探测。

6、api-server将收到的pod状态信息存入etcd中。

至此，整个pod创建完毕。

情况二、如果面试官说的是使用deployment来创建pod，则可以这样回答：

1、首先，用户使用kubectl create命令或者kubectl apply命令提交了要创建一个deployment资源请求；

2、api-server收到创建资源的请求后，会对客户端操作进行身份认证，在客户端的~/.kube文件夹下，已经设置好了相关的用户认证信息，这样api-server会知道是哪个用户请求，并对此用户进行鉴权，当api-server确定客户端的请求合法后，就会接受本次操作，并把相关的信息保存到etcd中，然后返回确认信息给客户端。（仅返回创建的信息并不是返回是否成功创建的结果）

3、api-server开始反馈etcd中过程创建的对象的变化，其他组件使用watch机制跟踪api-server上的变动。

4、controller-manager组件会监听api-server的信息，controller-manager是有多个类型的，比如Deployment Controller, 它的作用就是负责监听Deployment，此时Deployment Controller发现有新的deployment要创建，那么它就会去创建一个ReplicaSet，一个ReplicaSet的产生，又被另一个叫做ReplicaSet Controller监听到了，紧接着它就会去分析ReplicaSet的语义，它了解到是要依照ReplicaSet的template去创建pod, 它一看这个pod并不存在，那么就新建此pod，当pod刚被创建时，它的nodeName属性值为空，代表着此pod未被调度。

5、接着调度器Scheduler组件开始介入工作，Scheduler也是通过watch机制跟踪api-server上的变动，发现有未调度的pod，则根据内部算法、节点资源情况，pod定义的亲和性反亲和性等等，调度器会综合的选出一批候选节点，在候选节点中选择一个最优的节点，然后将pod绑定到该节点，将信息反馈给api-server。

6、kubelet组件布署于Node之上，它也是通过watch机制跟踪api-server上的变动，监听到有一个pod应该要被调度到自身所在Node上来，kubelet首先判断本地是否在此pod，如果不存在，则会进入创建pod流程，创建pod有分为几种情况，第一种是容器不需要挂载外部存储，则相当于直接docker run把容器启动，但不会直接挂载docker网络，而是通过CNI调用网络插件配置容器网络，比如分配pod IP等，如果需要挂载外部存储，则还要调用CSI来挂载存储。kubelet创建完pod，将信息反馈给api-server，api-servier将pod信息写入etcd。

7、pod建立成功后，ReplicaSet Controller会对其持续进行关注，如果pod因意外或被我们手动退出，ReplicaSet Controller会知道，并创建新的pod，以保持replicas数量期望值。

以上即是pod的调度过程。

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

简单描述一下pod的终止过程（记住，经常问）

1、用户向api-server发送删除pod对象的命令；

2、api-server中的pod对象信息会随着时间的推移而更新，在宽限期内（默认30s），pod被视为dead；

3、将pod标记为terminating状态；

4、kubectl通过watch机制监听api-server，监控到pod对象为terminating状态了就会启动pod关闭过程；

5、kube-proxy 更新转发规则，endpoint控制器监控到pod对象的关闭行为时将其从所有匹配到此endpoint的server资源endpoint列表中删除；

6、如果当前pod对象定义了preStop钩子处理器，则在其被标记为terminating后会以同步的方式启动执行；

7、pod对象中的容器进程收到停止信息；

8、宽限期结束后，若pod中还存在运行的进程，那么pod对象会收到立即终止的信息；

9、kubelet请求api-server将此pod资源的宽限期设置为0从而完成删除操作，此时pod对用户已不可见。

1

2

3

4

5

6

7

8

9

描述一下pod的终止流程（记住，经常问）

以下为容器在 Kubernetes 环境中的pod终止流程：

1、Pod 被删除，此时 Pod 里有 DeletionTimestamp，且状态置为 Terminating。此时调整 CLB 到该 Pod 的权重为 0。

2、kube-proxy 更新转发规则，将 Pod 从 service 的 endpoint 列表中摘除掉，新的流量不再转发到该 Pod。

3、如果 Pod 配置了 preStop Hook ，将会执行。

4、kubelet 将对 Pod 中各个 container 发送 SIGTERM 信号，以通知容器进程开始优雅停止。

5、等待容器进程完全停止，如果在 terminationGracePeriodSeconds 内 (默认30s) 还未完全停止，将发送 SIGKILL 信号强制停止进程。

6、所有容器进程终止，清理 Pod 资源。

1

2

3

4

5

6

7

pod的生命周期有哪几种？（记住，经常问）

pod生命周期有的5种状态（也称5种相位），如下：

Pending（挂起）：API server已经创建pod，但是该pod还有一个或多个容器的镜像没有创建，包括正在下载镜像的过程；

Running（运行中）：pod内所有的容器已经创建，且至少有一个容器处于运行状态、正在启动括正在重启状态；

Succeed（成功）：pod内所有容器均已退出，且不会再重启；

Failed（失败）：pod内所有容器均已退出，且至少有一个容器为退出失败状态

Unknown（未知）：某于某种原因api-server无法获取该pod的状态，可能由于网络通行问题导致；

1

2

3

4

5

pod状态一般有哪些？

pod的状态一般会有以下这些：

ContainerCreating（容器正在创建）：容器正在创建中

Pending（挂起）：API server已经创建pod，但是该pod还有一个或多个容器的镜像没有创建，包括正在下载镜像的过程；

Running（运行中）：pod内所有的容器已经创建，且至少有一个容器处于运行状态、正在启动括正在重启状态；

MatchNodeSelector （匹配节点选择器）：pod正在等待被调度到匹配其nodeSelector的节点上，当一个pod定义有节点选择器但没有任何节点存在指定的标签时，pod将处于“MatchNodeSelector”状态。

ErrImagePull（镜像拉取异常）: 这个错误表示Kubernetes无法从指定的镜像仓库拉取镜像。可能的原因有很多，比如网络问题、镜像名称或标签错误、或者没有权限访问这个镜像仓库等。

ImagePullBackOff（镜像拉取异常）: 这个错误表示Kubernetes尝试拉取镜像，但是失败了，然后它回滚了之前的操作。这通常是因为镜像仓库的问题，比如网络问题、镜像不存在、或者没有权限访问这个镜像仓库等。

Error（pod异常）：可能是容器运行时异常

CrashLoopBackOff（崩溃重启） ：pod正在经历一个无限循环的崩溃和重启过程。

Succeed（成功）：pod内所有容器均已退出，且不会再重启；

Failed（失败）：pod内所有容器均已退出，且至少有一个容器为退出失败状态

Unknown（未知）：某于某种原因api-server无法获取该pod的状态，可能由于网络通行问题导致；

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

pod一直处于pending状态一般有哪些情况，怎么排查？（重点，持续更新）

（这个问题被问到的概率非常大）

答：一个pod一开始创建的时候，它本身就是会处于pending状态，这时可能是正在拉取镜像，正在创建容器的过程。

如果等了一会发现pod还一直处于pending状态，那么我们可以使用kubectl describe命令查看一下pod的Events详细信息。一般可能会有这么几种情况导致pod一直处于pending状态：

1、调度器调度失败。Scheduer调度器无法为pod分配一个合适的node节点。而这又会有很多种情况，比如，node节点处在cpu、内存压力，导致无节点可

调度；pod定义了资源请求，没有node节点满足资源请求；node节点上有污点而pod没有定义容忍；pod中定义了亲和性或反亲和性而没有节点满足这些亲

和性或反亲和性；以上是调度器调度失败的几种情况。

2、pvc、pv无法动态创建。如果因为pvc或pv无法动态创建，那么pod也会一直处于pending状态，比如要使用StatefulSet创建redis集群，因为粗心大

意，定义的storageClassName名称写错了，那么会造成无法创建pvc，这种情况pod也会一直处于pending状态，或者，即

使pvc是正常创建了，但是由于某些异常原因导致动态供应存储无法正常创建pv，那么这种情况pod也会一直处于pending状态。

1

2

3

4

5

6

pod的钩子函数有哪几种，作用是什么？

pod的钩子函数有PostStart和PreStop两种，它们在容器的生命周期中特定时刻被调用并执行指定的操作。

postStart构子，作用是在容器创建后立即执行，但并不能保证该钩子会在容器的ENTRYPOINT之前运行。它主要用于资源部署、环境准备等。如果该钩子

执行失败或花费太长时间，容器将无法达到“Running”状态。一个典型的PostStart应用是在容器启动时执行一些配置或准备工作，比如修改配置文件、

更新本地资源等。例如，可以在此钩子中修改Nginx的默认首页。

1

2

3

preStop构子，作用是在容器终止之前立即被调用，是阻塞的，即同步的。它主要用于优雅关闭应用程序、通知其他系统以及完成清理工作。如果钩子执行

期间挂起，pod的状态将停留在“Running”状态并且不会达到“Failed”状态。在其完成之前会阻塞删除容器的操作。

应用场景：一个常见的PreStop应用是在容器关闭前优雅地停止一个服务，如Nginx或MySQL服务。

1

2

3

注意：

postStart构子是异步的，并不能保证该钩子会在容器的ENTRYPOINT之前运行；

preStop构子是同步的，它会阻塞当前容器的结束流程，直到Hook定义操作完成之后才允许容器被结束。

不管是postStart构子还是preStop构子，都可以使用 exec、httpGet、tcpSocket等3种方法来定义。

如何优雅的终止pod或者说如何保证pod不丢失流量（重点，经常问）

在一些重点领域，比如涉及金钱交易的系统，保证pod终止时不丢失流量是很重要的，这就涉及到如何保证pod优雅的退出的问题。

1、在开发层面，程序务必在业务代码里处理 SIGTERM 信号。主要逻辑是不接受新的流量进入，继续处理剩余流量，保证所有连接全部断开程序才退出。

2、在k8s层面，Pod里面可以设置preStop构子，preStop构子的作用是在容器终止之前立即被调用，主要用于优雅关闭应用程序、或者完成一些清理工作等等，这个钩子是同步的，即具有阻塞性的，也就是说它是会阻塞删除容器的操作。举个例子假设是容器运行的是nginx进程，则可以设置preStop钩子为nginx -s quit 让nginx进程优雅退出。

3、在一些程序终止时常较长的场景下，可以适当增加pod终止宽限期，即terminationGracePeriodSeconds参数，默认pod终止宽限期是30s，参数具体设置在deployment.spec.template.spec.terminationGracePeriodSeconds。当终止终止宽限期到达之后，无论Pod是否完成终止，也无论Pod是否正在被preStop阻塞，k8s都会发送强制退出信号给Pod让其终止。

1

2

3

腾讯云的这篇文章讲的不错：https://www.tencentcloud.com/zh/document/product/457/42070

pod的初始化容器是干什么的？

init container，初始化容器用于在启动应用容器之前完成应用容器所需要的前置条件，初始化容器本质上和应用容器是一样的，但是初始化容器是仅运行一次就结束的任务，初始化容器具有两大特征：

1、初始化容器必须运行完成直至成功结束，若某初始化容器运行失败，那么kubernetes需要重启它直到成功完成；

2、初始化容器必须按照定义的顺序执行，当且仅当前一个初始化容器成功之后，后面的一个初始化容器才能运行；

举个例子，我们最常见的es容器里面就有一个初始化容器，这个初始化容器的执行命令就是配置内核参数，因为es对某些内核参数要求设置比较大，所以

直接通过初始化容器修改了内核参数。（容器与宿主机共享内核，所以修改的就是宿主内核）

1

2

3

4

5

pod的资源请求、限制如何定义？

pod的资源请求、资源限制可以直接在pod中定义，主要包括两块内容，limits，限制pod中容器能使用的最大cpu和内存，requests，pod中容器启动时申请的cpu和内存。

resources: #资源配额

limits: #限制最大资源，上限

cpu: 2 #CPU限制，单位是code数

memory: 2G #内存最大限制

requests: #请求资源（最小，下限）

cpu: 1 #CPU请求，单位是code数

memory: 500G #内存最小请求

1

2

3

4

5

6

7

pod的定义中有个command和args参数，这两个参数不会和docker镜像的entrypointc冲突吗？

不会。

在pod中定义的command参数用于指定容器的启动命令列表，如果不指定，则默认使用Dockerfile打包时的启动命令，args参数用于容器的启动命令需要的参数列表；

特别说明：

kubernetes中的command、args其实是实现覆盖dockerfile中的ENTRYPOINT的功能的。当：

1、如果command和args均没有写，那么使用Dockerfile的配置；

2、如果command写了但args没写，那么Dockerfile默认的配置会被忽略，执行pod容器指定的command；

3、如果command没写但args写了，那么Dockerfile中的ENTRYPOINT的会被执行，使用当前args的参数；

4、如果command和args都写了，那么Dockerfile会被忽略，执行pod当前定义的command和args。

1

2

3

4

标签及标签选择器是什么，如何使用？

标签是键值对类型，标签可以附加到任何资源对象上，主要用于管理对象，查询和筛选。标签常被用于标签选择器的匹配度检查，从而完成资源筛选；一个资源可以定义一个或多个标签在其上面。

标签选择器，标签要与标签选择器结合在一起，标签选择器允许我们选择标记有特定标签的资源对象子集，如pod，并对这些特定标签的pod进行查询，删除等操作。

标签和标签选择器最重要的使用之一在于，在deployment中，在pod模板中定义pod的标签，然后在deployment定义标签选择器，这样就通过标签选择器来选择哪些pod是受其控制的，service也是通过标签选择器来关联哪些pod最后其服务后端pod。

service是如何与pod关联的？

答案：通过标签选择器。每一个由deployment创建的pod都带有标签，这样，service就可以定义标签选择器来关联哪些pod是作为其后端了，就是这样，service就与pod关联在一起了。

service的域名解析格式、pod的域名解析格式

service的DNS域名表示格式为<servicename>.<namespace>.svc.<clusterdomain>，servicename是service的名称，namespace是service所处的命名空间，clusterdomain是k8s集群设置的域名后缀，一般默认为 cluster.local，一般的，我们不会去改k8s集群设置的域名后缀，同时，当pod要链接的svc处于pod当前命名空间时，可以省略<namespace>以及后面的.svc不写，这样，就可以有下面三种方式来表示svc的域名：

#查看k8s集群设置的域名后缀

grep -i clusterDomain /opt/kubernetes/config/kubelet-config.yml #二进制安装的k8s集群，可以这样查看

grep -i clusterDomain /etc/kubernetes/kubelet.conf #kubeadm安装的k8s集群,各个节点的kubelet.conf文件中的字段clusterDomain

grep -i clusterDomain /var/lib/kubelet/config.yaml #kubeadm安装的k8s集群,各个节点的config.yaml文件中的字段clusterDomain

kubectl -n kube-system get cm coredns -oyaml #coredns cm里面也可以看到

kubectl exec -it busybox -- cat /etc/resolv.conf #直接看pod里面的resolv.conf文件亦可

1

2

3

4

5

6

svc-nginx.default.svc.cluster.local #完整的写法

svc-nginx.default #带命名空间写法,省略了后面的.svc.<clusterdomain>

svc-nginx #如果pod与svc在同一个命名空间,可以将命名空间省略不写,换句话说链接的是当前命名空间的svc

#于是,svc域名+svc的端口,我们就可以在pod里面访问svc对应的应用了,如下

wget http://svc-deployment-nginx.default.svc.cluster.local:80 #完整的写法

wget http://svc-deployment-nginx.default:80 #带命名空间写法

wget http://svc-deployment-nginx:80 #如果pod与svc在同一个命名空间,可以将命名空间省略不写

1

2

3

4

5

6

7

8

pod的DNS域名格式为：<pod-ip>.<namespace>.pod.<clusterdomain> ，其中，pod-ip需要使用-将ip之间的点替换掉，namespace为pod所在的命名空间，clusterdomain是k8s集群设置的域名后缀，一般默认为 cluster.local ，如果没有改变k8s集群默认的域名后缀，则可以省略该后缀不写。除此之外，其他的均不可省略，这一点与svc域名有所不同。

演示如下：

#进入default命名空间的busybox pod里面，测试下载文件

kubectl -n default exec -it deployment-busybox-567674bd67-lmrgw -- sh

wget 10-244-166-167.helm.pod.cluster.local:80 #可以正常下载,这里下载的是helm命名空间里的ip为10.244.166.167的pod

wget 10-244-166-167.helm.pod:80 #可以正常下载,这里把k8s集群设置的域名后缀默认省略了

wget 10-244-166-143.default.pod:80 #可以正常下载,这里下载的是default命名空间里的ip为10.244.166.143的pod

wget 10-244-166-143.default:80 #报错了,错误写法,说明不能省略pod关键字

wget 10-244-166-143:80 #报错了,错误写法,说明不能省略命名空间和pod关键字

1

2

3

4

5

6

7

对于deployment、daemonsets等创建的无状态的pod，还还可以通过<pod-ip>.<deployment-name>.<namespace>.svc.<clusterdomain> 这样的域名访问。（这点存疑，一直测试失败，不指定是书中写错了还是什么）

对于StatefulSet创建的pod，statefulset.spec.serviceName字段解释如下：

\[root@matser ~\]# kubectl explain statefulset.spec.serviceName

KIND: StatefulSet

VERSION: apps/v1

FIELD: serviceName <string>

DESCRipTION:

serviceName is the name of the service that governs this StatefulSet. This

service must exist before the StatefulSet, and is responsible for the

network identity of the set. pods get DNS/hostnames that follow the

pattern: pod-specific-string.serviceName.default.svc.cluster.local where

"pod-specific-string" is managed by the StatefulSet controller.

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

也就是说StatefulSet创建的pod，其pod的域名为：pod-specific-string.serviceName.default.svc.cluster.local，而pod-specific-string就是pod的名称。

例如：redis-sts-0.redis-svc.default.svc.cluster.local:6379,redis-sts-1.redis-svc.default.svc.cluster.local:6379,redis-sts-2.redis-svc.default.svc.cluster.local:6379,redis-sts-3.redis-svc.default.svc.cluster.local:6379,redis-sts-4.redis-svc.default.svc.cluster.local:6379,redis-sts-5.redis-svc.default.svc.cluster.local:6379，pod里面的后端应用程序就可以拿这串字符串去连接Redis集群了。

service的类型有哪几种

service的类型一般有4种，分别是：

ClusterIP：表示service仅供集群内部使用,默认值就是ClusterIP类型

NodePort：表示service可以对外访问应用,会在每个节点上暴露一个端口,这样外部浏览器访问地址为：任意节点的ip：NodePort就能连上service了

LoadBalancer：表示service对外访问应用,这种类型的service是公有云环境下的service,此模式需要外部云厂商的支持,需要有一个公网ip地址

ExternalName：这种类型的service会把集群外部的服务引入集群内部,这样集群内直接访问service就可以间接的使用集群外部服务了

一般情况下，service都是ClusterIP类型的，通过ingress接入的外部流量。

1

2

3

4

5

6

一个应用pod是如何发现service的，或者说，pod里面的容器用于是如何连接service的？

答：有两种方式，一种是通过环境变量，另一种是通过service的dns域名方式。

1、环境变量：当pod被创建之后，k8s系统会自动为容器注入集群内有效的service名称和端口号等信息为环境变量的形式，这样容器应用直接通过取环境变量值就能访问service了，如，每个pod都会自动注入了api-server的svc：curl http://${KUBERNETES\_SERVICE\_HOST}:{KUBERNETES\_SERVICE\_PORT}

2、DNS方式：使用dns域名解析的前提是k8s集群内有DNS域名解析服务器，默认k8s中会有一个CoreDNS作为k8s集群的默认DNS服务器提供域名解析服务器；service的DNS域名表示格式为<servicename>.<namespace>.svc.<clusterdomain>，servicename是service的名称，namespace是service所处的命名空间，clusterdomain是k8s集群设置的域名后缀，一般默认为 cluster.local ，这样容器应用直接通过service域名就能访问service了，如wget http://nginx-svc.default.svc.cluster.local:80，另外，service的port端口如果定义了名称，那么port也可以通过DNS进行解析，格式为：\_<portname>.\_<protocol>.<servicename>.<namespace>.svc.<clusterdomain>

如何创建一个service代理外部的服务，或者换句话来说，在k8s集群内的应用如何访问外部的服务，如数据库服务，缓存服务等?

答：可以通过创建一个没有标签选择器的service来代理集群外部的服务。

1、创建service时不指定selector标签选择器，但需要指定service的port端口、端口的name、端口协议等，这样创建出来的service因为没有指定标签选择器就不会自动创建endpoint；

2、手动创建一个与service同名的endpoint，endpoint中定义外部服务的ip和端口，endpoint的名称一定要与service的名称一样，端口协议也要一样，端口的name也要与service的端口的name一样，不然endpoint不能与service进行关联。

完成以上两步，k8s会自动将service和同名的endpoint进行关联，这样，k8s集群内的应用服务直接访问这个service就可以相当于访问外部的服务了。

service、endpoint、kube-proxy三种的关系是什么？

service：在kubernetes中，service是一种为一组功能相同的pod提供单一不变的接入点的资源。当service被建立时，service的ip和端口不会改变，这样外部的客户端（也可以是集群内部的客户端）通过service的ip和端口来建立链接，这些链接会被路由到提供该服务的任意一个pod上。通过这样的方式，客户端不需要知道每个单独提供服务的pod地址，这样pod就可以在集群中随时被创建或销毁。

endpoint：service维护一个叫endpoint的资源列表，endpoint资源对象保存着service关联的pod的ip和端口。从表面上看，当pod消失，service会在endpoint列表中剔除pod，当有新的pod加入，service就会将pod ip加入endpoint列表；但是正在底层的逻辑是，endpoint的这种自动剔除、添加、更新pod的地址其实底层是由endpoint controller控制的，endpoint controller负责监听service和对应的pod副本的变化，如果监听到service被删除，则删除和该service同名的endpoint对象，如果监听到新的service被创建或者修改，则根据该service信息获取得相关pod列表，然后创建或更新service对应的endpoint对象，如果监听到pod事件，则更新它所对应的service的endpoint对象。

kube-proxy：kube-proxy运行在node节点上，在Node节点上实现pod网络代理，维护网络规则和四层负载均衡工作，kube-proxy会监听api-server中从而获取service和endpoint的变化情况，创建并维护路由规则以提供服务ip和负载均衡功能。简单理解此进程是Service的透明代理兼负载均衡器，其核心功能是将到某个Service的访问请求转发到后端的多个pod实例上。

kube-proxy有哪几种模式？

kube-proxy有3种模式,分别是userspace（用户空间）、iptables、ipvs，其中user namespace已经被废弃了这种就不在讲解了。

目前用的最多的就是iptables和ipvs这两种模式。

你们生产环境kube-proxy使用哪种模式，为什么？

生产环境中使用ipvs。因为ipvs性能比iptables高。

iptables本质上是Linux的一个高效的防火墙，提供数据包处理和过滤方面的能力，kube-proxy使用iptables这种模式的时候，其连接处理算法复杂度是O(n)，其中的n随集群规模同步增长，换句话说，当k8s集群中的service数量很多，比如集群有1000个service，每个service后端又有多个Pod副本，那每个节点上的iptable规则将非常多，kube-proxy需要动态维护这些庞大的规则将带来严重的节点性能问题。

而ipvs是Linux内核功能中专门用于处理高性能负载均衡的功能，它使用更高效的数据结构，如hash表并支持索引，IPVS有一套优化过的 API，使用优化的查找算法，而不是简单的从列表中查找规则，ipvs还支持多种调度算法，如rr、wrr、lc、wlc，而iptables就只有一种随机平等的选择算法。其次，kube-proxy在IPVS模式下，连接处理算法复杂度是O(1)而不是O(n)。换句话说，多数情况下，连接处理效率是和集群规模大小无关。

iptables和ipvs

ipvs与iptables相比较，其优势为：

（1）ipvs为大型集群提供了更好的可扩展性和性能

（2）ipvs支持比iptables更复杂的负载均衡算法，如rr、wrr、lc、wlc，而iptables 就只有一种随机平等的选择算法。

（3）ipvs支持服务健康检查和链接重试等功能

（4）ipvs可以动态修改ipset集合

无头service和普通的service有什么区别，无头service使用场景是什么？

答：无头service没有cluster ip，在定义service时将 service.spec.ClusterIP：None，就表示创建的是无头service。

普通的service是用于为一组后端pod提供请求连接的负载均衡，让客户端能通过固定的service ip地址来访问pod，这类的pod是没有状态的，同时service还具有负载均衡和服务发现的功能。普通service跟我们平时使用的nginx反向代理很相识。

但是，试想这样一种情况，有6个redis pod ,它们相互之间要通信并要组成一个redis集群，不在需要所谓的service负载均衡，这时无头service就是派上用场了，无头service由于没有cluster ip，kube-proxy就不会处理它也就不会对它生成规则负载均衡，无头service直接绑定的是pod 的ip。无头service仍会有标签选择器，有标签选择器就会有endpoint资源。

使用场景：无头service一般用于有状态的应用场景，如Kaka集群、Redis集群等，这类pod之间需要相互通信相互组成集群，不在需要所谓的service负载均衡。

deployment怎么扩容或缩容？

答：直接修改pod副本数即可，可以通过下面的方式来修改pod副本数：

1、直接修改yaml文件的replicas字段数值，然后kubectl apply -f xxx.yaml来实现更新；

2、使用kubectl edit deployment xxx 修改replicas来实现在线更新；

3、使用kubectl scale --replicas=5 deployment/deployment-nginx命令来扩容缩容。

deployment的更新升级策略有哪些？

答：deployment的升级策略主要有两种。

1、Recreate 重建更新：这种更新策略会杀掉所有正在运行的pod，然后再重新创建的pod；

2、rollingUpdate 滚动更新：这种更新策略，deployment会以滚动更新的方式来逐个更新pod，同时通过设置滚动更新的两个参数maxUnavailable、maxSurge来控制更新的过程。

deployment的滚动更新策略有两个特别主要的参数，解释一下它们是什么意思？

答：deployment的滚动更新策略，rollingUpdate 策略，主要有两个参数，maxUnavailable、maxSurge。

maxUnavailable：最大不可用数，maxUnavailable用于指定deployment在更新的过程中不可用状态的pod的最大数量，maxUnavailable的值可以是

一个整数值，也可以是pod期望副本的百分比，如25%，计算时向下取整。

maxSurge：最大激增数，maxSurge指定deployment在更新的过程中pod的总数量最大能超过pod副本数多少个，maxUnavailable的值可以是一个整数

值，也可以是pod期望副本的百分比，如25%，计算时向上取整。

1

2

3

4

deployment更新的命令有哪些？

答：可以通过三种方式来实现更新deployment。

1、直接修改yaml文件的镜像版本，然后kubectl apply -f xxx.yaml来实现更新；

2、使用kubectl edit deployment xxx 实现在线更新；

3、使用kubectl set image deployment/nginx busybox=busybox nginx=nginx:1.9.1 命令来更新。

简述一下deployment的更新过程 （经常问）

deployment是通过控制replicaset来实现，由replicaset真正创建pod副本，每更新一次deployment，都会创建新的replicaset，下面来举例deployment的更新过程：

假设要升级一个nginx-deployment的版本镜像为nginx:1.9，deployment的定义滚动更新参数如下：

replicas: 3

deployment.spec.strategy.type: RollingUpdate

maxUnavailable：25%

maxSurge：25%

通过计算我们得出，3\*25%=0.75，maxUnavailable是向下取整，则maxUnavailable=0，maxSurge是向上取整，则maxSurge=1，所以我们得出在整个deployment升级镜像过程中，不管旧的pod和新的pod是如何创建消亡的，pod总数最大不能超过3+maxSurge=4个，最大pod不可用数3-maxUnavailable=3个。

现在具体讲一下deployment的更新升级过程：

使用\`kubectl set image deployment/nginx nginx=nginx:1.9 --record\` 命令来更新；

1、deployment创建一个新的replaceset，先新增1个新版本pod，此时pod总数为4个，不能再新增了，再新增就超过pod总数4个了；旧=3，新=1，总=4；

2、减少一个旧版本的pod，此时pod总数为3个，这时不能再减少了，再减少就不满足最大pod不可用数3个了；旧=2，新=1，总=3；

3、再新增一个新版本的pod，此时pod总数为4个，不能再新增了；旧=2，新=2，总=4；

4、减少一个旧版本的pod，此时pod总数为3个，这时不能再减少了；旧=1，新=2，总=3；

5、再新增一个新版本的pod，此时pod总数为4个，不能再新增了；旧=1，新=3，总=4；

6、减少一个旧版本的pod，此时pod总数为3个，更新完成，pod都是新版本了；旧=0，新=3，总=3；

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

1、用户通过kubectl set image命令或kubectl edit命令直接修改镜像将新的配置应用到集群中,api-server对当前用户操作进行鉴权，鉴权成功后

接收操作请求并返回信息给用户；

2、Deployment控制器检测到配置变化后，开始创建一个新的replicaset，replicaset开始创建新的pod，根据deployment定义的更新策略来执行更新

升级。默认情况下，它采用滚动更新（RollingUpdate）策略。

滚动更新策略有两个主要的参数，maxUnavailable和maxSurge 参数。maxUnavailable 参数定义可以有多少个pod不可用（未就绪），maxSurge则指定了可以比期望副本数多创建多少个pod，通过这两个参数，deployment会来逐步创建新版本pod，并同时删除旧版本pod。

3、在每个新pod被创建之后，kubelet会根据定义的livenessProbe和readinessProbe来检查新pod是否已经启动并准备好接收流量。当新pod通过存活探针并标记为"就绪"时，pod的ip和端口将会被endpoint检查并加入端点列表，此时Service会逐渐将流量从旧pod转移到新pod。

4、一旦所有旧pod都被新pod替换并且新版本的所有pod都已准备就绪，则升级过程结束。

1

2

3

4

5

6

7

deployment的回滚使用什么命令

在升级deployment时kubectl set image 命令加上 --record 参数可以记录具体的升级历史信息，使用kubectl rollout history deployment/deployment-nginx 命令来查看指定的deployment升级历史记录，如果需要回滚到某个指定的版本，可以使用kubectl rollout undo deployment/deployment-nginx --to-revision=2 命令来实现。

讲一下都有哪些存储卷，作用分别是什么?

卷 作用 常用场景

emptyDir 用于存储临时数据的简单空目录 一个pod中的多个容器需要共享彼此的数据 ，emptyDir的数据随着容器的消亡也会销毁

hostPath 用于将目录从工作节点的文件系统挂载到pod中 不常用，缺点是，pod的调度不是固定的，也就是当pod消失后deployment重新创建一个pod，而这pod如果不是被调度到之前pod的节点，那么该pod就不能访问之前的数据

configMap 用于将非敏感的数据保存到键值对中，使用时可以使用作为环境变量、命令行参数arg，存储卷被pods挂载使用 将应用程序的不敏感配置文件创建为configmap卷，在pod中挂载configmap卷，可是实现热更新

secret 主要用于存储和管理一些敏感数据，然后通过在 pod 的容器里挂载 Volume 的方式或者环境变量的方式访问到这些 Secret 里保存的信息了，pod会自动解密Secret 的信息 将应用程序的账号密码等敏感信息通过secret卷的形式挂载到pod中使用

downwardApi 主要用于暴露pod元数据，如pod的名字 pod中的应用程序需要指定pod的name等元数据，就可以通过downwardApi 卷的形式挂载给pod使用

projected 这是一种特殊的卷，用于将上面这些卷一次性的挂载给pod使用 将上面这些卷一次性的挂载给pod使用

pvc pvc是存储卷声明 通常会创建pvc表示对存储的申请，然后在pod中使用pvc

网络存储卷 pod挂载网络存储卷，这样就能将数据持久化到后端的存储里 常见的网络存储卷有nfs存储、glusterfs 卷、ceph rbd存储卷

pv的访问模式有哪几种

pv的访问模式有3种，如下：

ReadWriteOnce，简写：RWO 表示，只仅允许单个节点以读写方式挂载；

ReadOnlyMany，简写：ROX 表示，可以被许多节点以只读方式挂载；

ReadWriteMany，简写：RWX 表示，可以被多个节点以读写方式挂载；

1

2

3

pv的回收策略有哪几种

主要有2中回收策略：retain 保留、delete 删除。

Retain：保留，该策略允许手动回收资源，当删除PVC时，PV仍然存在，PV被视为已释放，管理员可以手动回收卷。

Delete：删除，如果Volume插件支持，删除PVC时会同时删除PV，动态卷默认为Delete，目前支持Delete的存储后端包括AWS EBS，GCE PD，Azure Disk，OpenStack Cinder等。

Recycle：回收，如果Volume插件支持，Recycle策略会对卷执行rm -rf清理该PV，并使其可用于下一个新的PVC，但是本策略将来会被弃用，目前只有NFS和HostPath支持该策略。（这种策略已经被废弃，不用记）

在pv的生命周期中，一般有几种状态

pv一共有4中状态，分别是：

创建pv后，pv的的状态有以下4种：Available（可用）、Bound（已绑定）、Released（已释放）、Failed（失败）

Available，表示pv已经创建正常，处于可用状态；

Bound，表示pv已经被某个pvc绑定，注意，一个pv一旦被某个pvc绑定，那么该pvc就独占该pv，其他pvc不能再与该pv绑定；

Released，表示pvc被删除了，pv状态就会变成已释放；

Failed，表示pv的自动回收失败；

1

2

3

4

存储类的资源回收策略:

主要有2中回收策略，delete 删除，默认就是delete策略、retain 保留。

Retain：保留，该策略允许手动回收资源，当删除PVC时，PV仍然存在，PV被视为已释放，管理员可以手动回收卷。

Delete：删除，如果Volume插件支持，删除PVC时会同时删除PV，动态卷默认为Delete，目前支持Delete的存储后端包括AWS EBS，GCE PD，Azure Disk，OpenStack Cinder等。

注意：使用存储类动态创建的pv默认继承存储类的回收策略，当然当pv创建后你也可以手动修改pv的回收策略。

怎么使一个node脱离集群调度，比如要停机维护单又不能影响业务应用

要使一个节点脱离集群调度可以使用kubectl cordon <node-name> 命令使节点不可调度，该命令其实背后原理就是给节点打上node-status.kubernets.io/unschedulable污点，这样新的pod如果没有容忍将不会调度到该节点，但是已经存在于该节点的pod仍然可以继续在该节点上运行不受影响，除非pod消忙了被重新调度了。如果需要恢复节点重新调度，可以使用kubectl uncordon <node-name> 命令恢复节点可调度。

如果节点是要停机维护，则可以对节点上的pod 进行驱逐：使用kubectl drain <node-name>命令用于将节点上的pod驱逐出去，以便对节点进行维护。

kubectl drain 命令的语法如下：

kubectl drain <node-name>

1

--ignore-daemonsets 参数用于忽略由DaemonSet控制器管理的pods，不加该参数会报错；

--delete-local-data 参数用于在节点上删除所有本地数据，包括PersistentVolume和PersistentVolumeClaim等资源；

--force 参数强制删除pod，默认删除的是ReplicationController, ReplicaSet, Job, DaemonSet 或者StatefulSet创建的pod，如果有静态pod，则需要设置强制执行删除的参数--force。

这个命令会将节点上所有的pods驱逐出去，包括由DaemonSet控制器管理的pods。但是需要注意的是， kubectl drain 命令会将节点上所有的pods驱逐出去，包括由DaemonSet控制器管理的pods，由于ds创建的pod会具有容忍，所以又会马上在正在清空的节点上启动新的pod，我们可以使用 --ignore-daemonsets 参数来忽略由DaemonSet控制器管理的pods。

kubectl drain 命令背后原理其实还是首先将指定的节点标记为不可调，从而阻止新 pod 分配到节点上（实质上是 kubectl cordon），然后删除pod。

综上所述，要停机维护：

1、kubectl cordon node01 #设置节点不可调度

2、kubectl drain node01 --ignore-daemonsets --force #驱逐pod

3、kubectl uncordon node01 #恢复节点调度

1

2

3

pv存储空间不足怎么扩容?

一般的，我们会使用动态分配存储资源，在创建storageclass时指定参数 allowVolumeExpansion：true，表示允许用户通过修改pvc申请的存储空间自动完成pv的扩容，当增大pvc的存储空间时，不会重新创建一个pv，而是扩容其绑定的后端pv。这样就能完成扩容了。但是allowVolumeExpansion这个特性只支持扩容空间不支持减少空间。

k8s生产中遇到什么特别影响深刻的问题吗，问题排查解决思路是怎么样的？（重点）

（此问题被问到的概率高达90%，所以可以自己准备几个自己在生产环境中遇到的问题进行讲解）

答：前端的lb负载均衡服务器上的keepalived出现过脑裂现象。

1、当时问题现象是这样的，vip同时出现在主服务器和备服务器上，但业务上又没受到影响；

2、这时首先去查看备服务器上的keepalived日志，发现有日志信息显示凌晨的时候备服务器出现了vrrp协议超时，所以才导致了备服务器接管了vip；查看主服务器上的keepalived日志，没有发现明显的报错信息，继续查看主服务器和备服务器上的keepalived进程状态，都是running状态的；查看主服务器上检测脚本所检测的进程，其进程也是正常的，也就是说主服务器根本没有成功执行检测脚本（成功执行检查脚本是会kill掉keepalived进程，脚本里面其实就是配置了检查nginx进程是否存活，如果检查到nginx不存活则kill掉keepalived，这样来实现备服务器接管vip）；

3、排查服务器上的防火墙、selinux，防火墙状态和selinux状态都是关闭着的；

4、使用tcpdump工具在备服务器上进行抓取数据包分析，分析发现，现在确实是备接管的vip，也确实是备服务器也在对外发送vrrp心跳包，所以现在外部流量应该都是流入备服务器上的vip；

5、怀疑：主服务器上设置的vrrp心跳包时间间隔太长，以及检测脚本设置的检测时间设置不合理导致该问题；

6、修改vrrp协议的心跳包时间间隔，由原来的2秒改成1秒就发送一次心跳包；检测脚本的检测时间也修改短一点，同时还修改检测脚本的检测失败的次数，比如连续检测2次失败才认定为检测失败；

7、重启主备上的keepalived，现在keepalived是正常的，主服务器上有vip，备服务器上没有vip；

8、持续观察：第二天又发现keepalived出现过脑裂现象，vip又同时出现在主服务器和备服务器上，又是凌晨的时候备服务器显示vrrp心跳包超时，所以才导致备服务器接管了vip；

9、同样的时间，都是凌晨，vrrp协议超时；很奇怪，很有理由怀疑是网络问题，询问第三方厂家上层路由器是否禁止了vrrp协议，第三方厂家回复，没有禁止vrrp协议；

10、百度、看官方文档求解；

11、百度、看官网文档得知，keepalived有2种传播模式，一种是组播模式，一种是单播模式，keepalived默认在组播模式下工作，主服务器会往主播地址224.0.0.18发送心跳包，当局域网内有多个keepalived实例的时候，如果都用主播模式，会存在冲突干扰的情况，所以官方建议使用单播模式通信，单播模式就是点对点通行，即主向备服务器一对一的发送心跳包；

12、将keepalived模式改为单播模式，继续观察，无再发生脑裂现象。问题得以解决。

答：测试环境二进制搭建etcd集群，etcd集群出现2个leader的现象。

1、问题现象就是：刚搭建的k8s集群，是测试环境的，搭建完成之后发现，使用kubectl get nodes 显示没有资源，kubectl get namespace 一会能正常显示全部的命名空间，一会又显示不了命名空间，这种奇怪情况。

2、当时经验不是很足，第一点想到的是不是因为网络插件calico没装导致的，但是想想，即使没有安装网络插件，最多是node节点状态是notready，也不可能是没有资源发现呀；

3、然后想到etcd数据库，k8s的资源都是存储在etcd数据库中的；

4、查看etcd进程服务的启动状态，发现etcd服务状态是处于running状态，但是日志有大量的报错信息，日志大概报错信息就是集群节点的id不匹配，存在冲突等等报错信息；

5、使用etcdctl命令查看etcd集群的健康状态，发现集群是health状态，但是居然显示有2个leader，这很奇怪（当初安装etcd的时候其实也只是简单看到了集群是健康状态，然后没注意到有2个leader，也没太关注etcd服务进程的日志报错信息，以为etcd集群状态是health状态就可以了）

6、现在etcd出现了2个leader，肯定是存在问题的；

7、全部检测一遍etcd的各个节点的配置文件，确认配置文件里面各个参数配置都没有问题，重启etcd集群，报错信息仍未解决，仍然存在2个leader；

8、尝试把其中一个leader节点踢出集群，然后再重新添加它进入集群，仍然是报错，仍然显示有2个leader；

9、尝试重新生成etcd的证书，重新颁发etcd的证书，问题仍然存在，仍然显示有2个leader；日志仍是报错集群节点的id不匹配，存在冲突；

10、计算etcd命令的MD5值，确保各个节点的etcd命令是相同的，确保在scp传输的时候没有损耗等等，问题仍未解决；

11、无解，请求同事，架构师介入帮忙排查问题，仍未解决；

12、删除全部etcd相关的文件，重新部署etcd集群，etcd集群正常了，现在只有一个leader，使用命令kubectl get nodes 查看节点，也能正常显示了；

13、最终问题的原因也没有定位出来，只能怀疑是环境问题了，由于是刚部署的k8s测试环境，etcd里面没有数据，所以可以删除重新创建etcd集群，如果是线上环境的etcd集群出现这种问题，就不能随便删除etcd集群了，必须要先进行数据备份才能进行其他方法的处理。

etcd集群节点可以设置为偶数个吗，为什么要设置为奇数个呢？

不能，也不建议这么设置。

etcd采用了Raft一致性算法来确保数据的一致性和高可用性。根据Raft算法的要求，为了确保算法的正确性和容错性，集群一般包含2n+1个节点，所以进行Leader选举和数据复制时，节点数必须是奇数个。

奇数个节点与配对的偶数个节点（如3个节点和4个节点）相比，容错能力相同，但可以少一个节点；其次，偶数个节点的集群在选举过程中由于等额选票的存在，有较大概率触发下一轮选举，从而增加了不可用的风险。因此，综合考虑性能和容错能力，etcd官方文档推荐的etcd集群大小是3, 5, 7。同时需要注意的是，虽然增加节点可以提高读的吞吐和提高集群的可用性，但节点数越多可能会导致写操作的吞吐降低。

etcd官方推荐3、5、7个节点，虽然raft算法也是半数以上投票才能有 leader，但奇数只是推荐，其实偶数也是可以的。如 2、4、8个节点。下面分情况说明：

1 个节点：就是单实例，没有集群概念，不做讨论

2 个节点：是集群，但没人会这么配，这里说点废话：双节点的etcd能启动，启动时也能有主，可以正常提供服务，但是一台挂掉之后，就选不出主了，因为他只能拿到1票，剩下的那台也无法提供服务，也就是双节点无容错能力，不要使用。

3 节点：标准的3 节点etcd 集群只能容忍1台机器宕机，挂掉 1 台此时等于2个节点的情况，如果再挂 1 台，就和 2节点的情形一致了，一直选，一直增加任期，但就是选不出来，服务也就不可用了

4 节点：最大容忍1台服务器宕机

5 节点：最大容忍2台服务器宕机

6 节点：最大容忍2台服务器宕机

7和8个节点，最大容忍3台服务器宕机

以此类推，9和10个节点，最大容忍4台服务器宕机

1

2

3

4

5

6

7

8

总结以上可以得出结论：偶数节点虽然多了一台机器，但是容错能力是一样的，也就是说，虽然可以设置偶数节点，但没增加什么容错能力，还浪费了一台机器。同时etcd 是通过复制数据给所有节点来达到一致性，因此偶数集群多出一台机器既增加不了性能，反而还会拉低写入速度。

你们生产环境etcd节点一般是几个节点？

我们使用的3节点的etcd集群，3节点etcd集群允许存在1台机器宕机，如果此时两台etcd节点宕机，那此时剩余的1台节点由于无法进行选举，所以整个etcd集群服务就不可用了，同理5个节点则可以最大容忍2个节点不可用，7节点可以容忍3个节点不可用。

目前etcd官方推荐etcd集群节点为3节点、5节点、7个节点，3节点可以支撑小规模的k8s集群，5到7节点可以支持中大型规模的k8s集群。

1

2

etcd节点是越多越好吗？

不是，etcd 集群是一个 Raft Group，没有 shared。所以它的极限有两部分，一是单机的容量限制，内存和磁盘；二是网络开销，每次 Raft 操作需要所有节点参与，每一次写操作需要集群中大多数节点将日志落盘成功后，Leader 节点才能修改内部状态机，并将结果返回给客户端。因此节点越多性能越低，并且出错的概率会直线上升，并且是呈现线性的性能下降，所以扩展很多 etcd 节点是没有意义的，其次，如果etcd集群超过7个达到十几个几十个，那么，对运维来说也是一个不小的压力了，并且集群的配置什么的也会更加的复杂，而不是简单易用了。因此，etcd集群的数量一般是 3、5、7， 3 个是最低标准，7个已经是最高了。

etcd集群节点之间是怎么同步数据的？

（待补充）

pod内容器之间通信，同节点pod通信、不同节点的pod的通信是如何实现的，大概的流程是什么

假设k8s采用flannel插件，则同节点的pod通信、不同节点的pod通信是这样子的：

1、pod内容器之间通信

在创建一个pod时，首先会创建一个pause容器，也就是根容器，而根容器负责创建pod内所有容器共享的网络命名空间，而其他业务容器都会加入到这个网络命名空间中去，所以在一个pod里面，容器都是处于同一个命名空间，既然处于同一个命名空间，则容器进程彼此之间的访问直接通过localhost+端口即可。我们可以把pod想象成一个微型的虚拟机，虚拟机对外只有一个ip，而虚拟机内的应用彼此访问直接通过localhost+端口访问即可。

2、通节点pod的通信

在部署flannel插件的时候，k8s默认是采用daemonsets这种资源类型部署的，会在每个宿主机上有一个flanneld进程，这个进程就是用来管理k8s网络的。然后会在每个宿主机上生成一个网桥，这个网桥叫做cni0；

当一个pod被创建之后，cni插件会为pod创建一个虚拟以太网接口（veth pair），这个虚拟以太网接口（veth pair）它是一端接在pod里面，一端接在cni0网桥上，所以，当同一个节点上的pod相互通信，其实就是通过cni0进行通信的。

3、不同节点的pod的通信（待完善）

pod发送数据给到cni0网桥，cni0根据路由规则转发给flannel.1，flannel.1转发给宿主机上的eth0物理网卡，这样数据就发送到对端pod所在的物理机网卡上。

一个用户请求流量是如何进入k8s集群内部的？

以一个内网的k8s集群为例：

1、用户访问域名，域名必要对应一个服务器ip地址，此时电脑开始解析域名，解析的流程可能是先检查浏览器是否缓存有这个域名的解析记录、检查window的host文件、最后是请求windows网络配置的dns服务器；

2、域名被解析到内网LB负载均衡服务器，LB负载均衡服务器上使用nginx定义了一组后端负载服务器ip端口，而这组后端服务器其实就是k8s的节点；

3、这组k8s节点是ingress-nginx-controller的暴露的端口，所以此时流量交给ingress-nginx-controller处理；

4、k8s上定义了一组ingress规则，规则里定义了对应的后端service，当用户流量来到ingress后会进行规则匹配并将交给后端service处理；

5、service会将流量分发给对应的endpoint，而endpoint对应这一组pod ip端口，所以此时，用户流量已经交给了pod中的容器处理了。

上面就是用户请求流量进入k8s集群内部的大致流程。

用户访问我们的k8s集群里面的应用网站，出现500报错，你是如何排查这种问题的？

1、按F12 打开开发者模式，切换到网络栏，按F5刷新一下页面，看哪个请求地址是红色的，红色就是访问有问题的，看地址，响应值、接口url等信息；

2、得到了接口，那么应该很容易知道是哪个后端服务出现问题，此时直接去查看日志即可。如果有elk，直接登录Kibana查看日志即可，如果没有，则直接去服务器上查看应用的日志即可。

k8s后端存储使用的是什么？

后端存储方面，我们不同的项目使用的方案不一样，这可能是早期系统架构决定的，现在也是一直沿用，主要就是包含3种存储方案：

1、第一种存储方案是，hostPath和local-path-provisioner，先说hostPath：pod中直接使用hostPath卷来挂载数据，然后pod中要节点选择器固定调度到指定的节点，我们一般采用这种方式来实现pod直接挂载服务器数据或者pod日志落盘到服务器磁盘；但是hostPath卷属于静态卷，所以我们还使用了local-path-provisioner来动态供给localPath，local-path-provisioner能够让Kubernetes的本地存储支持动态pv，当使用local-path-provisioner的pod被调度时，scheduler调度器和pv控制器会同时进行控制，然后在pod所在节点上创建对应的本地存储目录，当pod被重新调度后，因为pod所对应的pv存在节点选择器，所以pod仍然能够调度到之前的节点上，从而继续使用或读取之前的数据。

2、第二中存储方案是glusterfs：我们使用glusterfs作为后端存储，先在服务器上大磁盘单独划分数据分区，创建目录，然后创建一个glusterfs文件系统（glusterfs没有选举一说，可以任意数量扩展），然后在glusterfs文件系统上创建卷，然后在k8s创建存储类，这样就实现了静态供给pv，如果需要动态供给pv，还需要使用heketi软件。有时候我们也会把glusterfs中的卷直接挂载到服务器上使用。

3、最后一种存储方案是cephfs：ceph官网上推荐在k8s集群中使用rook-ceph，我们使用的就是rook-ceph，rook-ceph我们使用的helm安装的，创建完成ceph集群之后再创建cephfs,然后创建存储类进行动态pv供给。

1

2

3

你们的服务发布怎么做的？或者你们的cicd流程是怎么做的？

我们服务采用cicd流程发布的，具体的说，就是只要用到两个工具: jenkins和argocd。jenkins负责实现ci，argocd负责实现cd。主要是这么几个步骤:

1、开发写好代码将代码提交到gitlab仓库；

2、然后手动在jenkins页面上点击构建任务；

3、jenkins上定义了多个流水线项目，每个pipeline流水线项目都对应一个gitlab仓库的项目代码，流水线的配置都写在jenkinsfile文件里面，而jenkinsfile文件也是存放在gitlab仓库上进行托管的；

4、构建的流程只要是jenkinsfile文件定义的，jenkinsfile内容大致有这么几步：

第一步、先使用git命令克隆代码到工作目录并检出全部的分支然后写入到一个临时文件。

第二步、读取临时文件全部分支，提示用户选择要构建的分支和要发布的环境。

第三步、开始编译源代码，前端代码使用npm命令编译，后端代码使用maven编译。

第四步、代码编译完成就可以得到jar包了，这时开始构建镜像并推送镜像到harbor镜像仓库。

第五步、部署，这里的部署并不是真正意思上的部署，而是镜像的tag写回到gitlab仓库里去，并且使用kustomize命令修改yaml资源清单文件，这步主要是让argocd实现部署。

第六步、发送钉钉通知消息。

以上，第五步的更新gitlab仓库的时候，argocd会监听到gitlab仓库里面的k8s资源清单文件发生了改变，然后就会自动的应用部署，部署方式仍然是滚动更新deployment的镜像，这样就实现了自动化部署。

拿到一台新的服务器如何优化，如何进行加固安全

禁用SELinux

精简开机自启动服务

安装的Linux系统最小化，yum 安装软件也最小化，无用的包不安装。

更改ssh的默认22端口，改成其他端口，ssh禁用root远程登录，禁止空密码登录

配置sudoers文件，控制用户对系统命令的使用权限

设置linux时间同步，可以结合定时任务来同步时间服务器

调整系统文件描述符数量，在/etc/security/limits.conf文件里面调整

服务器内核参数优化

锁定系统关键的文件，使用chattr命令对文件锁定，锁定后所有用户都不能对文件进行修改删了，还可以将chattr命令重命名，防止被黑客识别。

服务器禁止被ping

开启防火墙

设置用户密码复杂度、过期策略，如最少密码长度、密码中必须包含的数字、大写字母、特殊字符等，以及密码的最大使用天数和到期警告天数。

修改PAM（Pluggable Authentication Modules）配置文件来设置账户锁定策略以对抗口令暴力破解

配置ssh登录超时策略。表示用户无操作多少秒超时自动退出 echo 'export TMOUT=300' >> /etc/profile

配置用户登录失败策略，在/etc/pam.d/system-auth中配置，比如密码错误锁定多少分钟

对多余帐户进行删除、锁定或禁止其登录，如：uucp、nuucp、lp、adm、sync、shutdown、halt、news、operator、gopher、shutdown等

限制保留的历史命令，HISTSIZE值，用于控制history命令保留历史记录数量；HISTFILESIZE值，控制.bash\_history文件中存储历史记录数量；echo 'HISTSIZE=30' >>/etc/profile;echo 'HISTFILESIZE=30' >>/etc/profile

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

如何进行k8s的安全加固

————————————————

转自

k8s面试题大全（持续更新中）\_ks8 面试题-CSDN博客

https://blog.csdn.net/MssGuo/article/details/125267817

版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/MssGuo/article/details/125267817

分类:
[K8S](https://www.cnblogs.com/paul8339/category/2047698.html)

标签:
[K8S](https://www.cnblogs.com/paul8339/tag/K8S/)

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/715460/20160603152231.png)](https://home.cnblogs.com/u/paul8339/)

[paul\_hch](https://home.cnblogs.com/u/paul8339/)

[粉丝 \- 174](https://home.cnblogs.com/u/paul8339/followers/) [关注 \- 20](https://home.cnblogs.com/u/paul8339/followees/)

+加关注

0

0

[«](https://www.cnblogs.com/paul8339/p/18831664) 上一篇： [Java进阶：如何反编译jar包获取源码\_jar包反编译成源代码【转】](https://www.cnblogs.com/paul8339/p/18831664 "发布于 2025-04-17 21:17")

[»](https://www.cnblogs.com/paul8339/p/18845061) 下一篇： [Ubuntu学习篇【学习篇】](https://www.cnblogs.com/paul8339/p/18845061 "发布于 2025-04-24 17:49")

posted @
2025-04-24 13:48 [paul\_hch](https://www.cnblogs.com/paul8339)
阅读(842)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2Fpaul8339%2Fp%2F18844452&targetId=18844452&targetType=0)

[刷新页面](https://www.cnblogs.com/paul8339/p/18844452#) [返回顶部](https://www.cnblogs.com/paul8339/p/18844452#top)

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

- 2020-04-24
[系统架构的演进过程【转】](https://www.cnblogs.com/paul8339/p/12769158.html)
- 2017-04-24
[java系统的优化](https://www.cnblogs.com/paul8339/p/6757170.html)
- 2017-04-24
[JBoss6.1.0修改启动jvm内存以及修改日志级别【转】](https://www.cnblogs.com/paul8339/p/6757090.html)
- 2017-04-24
[JAVA\_OPTS讲解【转】](https://www.cnblogs.com/paul8339/p/6755499.html)

### 公告

昵称：
[paul\_hch](https://home.cnblogs.com/u/paul8339/)

园龄：
[11年3个月](https://home.cnblogs.com/u/paul8339/ "入园时间：2015-01-19")

粉丝：
[174](https://home.cnblogs.com/u/paul8339/followers/)

关注：
[20](https://home.cnblogs.com/u/paul8339/followees/)

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

### [我的标签](https://www.cnblogs.com/paul8339/tag/)

- [Linux(362)](https://www.cnblogs.com/paul8339/tag/Linux/)
- [mysql(324)](https://www.cnblogs.com/paul8339/tag/mysql/)
- [nginx(85)](https://www.cnblogs.com/paul8339/tag/nginx/)
- [python(60)](https://www.cnblogs.com/paul8339/tag/python/)
- [shell(48)](https://www.cnblogs.com/paul8339/tag/shell/)
- [K8S(44)](https://www.cnblogs.com/paul8339/tag/K8S/)
- [oracle(36)](https://www.cnblogs.com/paul8339/tag/oracle/)
- [redis(35)](https://www.cnblogs.com/paul8339/tag/redis/)
- [ansible(35)](https://www.cnblogs.com/paul8339/tag/ansible/)
- [Windows(26)](https://www.cnblogs.com/paul8339/tag/Windows/)
- [更多](https://www.cnblogs.com/paul8339/tag/)

### 积分与排名

- 积分 \-
1406511

- 排名 \-
206


### [随笔分类](https://www.cnblogs.com/paul8339/post-categories)  (1100)

- [AI(1)](https://www.cnblogs.com/paul8339/category/2488076.html)
- [K8S(29)](https://www.cnblogs.com/paul8339/category/2047698.html)
- [Linux(291)](https://www.cnblogs.com/paul8339/category/1012254.html)
- [MongoDB(8)](https://www.cnblogs.com/paul8339/category/2350608.html)
- [MySQL(289)](https://www.cnblogs.com/paul8339/category/1012265.html)
- [Oracle(25)](https://www.cnblogs.com/paul8339/category/1012267.html)
- [PostgreSQL(5)](https://www.cnblogs.com/paul8339/category/2363640.html)
- [Python(47)](https://www.cnblogs.com/paul8339/category/1156958.html)
- [redis(6)](https://www.cnblogs.com/paul8339/category/2350607.html)
- [tomcat(2)](https://www.cnblogs.com/paul8339/category/2370128.html)
- [Windows(23)](https://www.cnblogs.com/paul8339/category/1012259.html)
- [编程语言(8)](https://www.cnblogs.com/paul8339/category/1012266.html)
- [高可用(103)](https://www.cnblogs.com/paul8339/category/1016155.html)
- [其他(49)](https://www.cnblogs.com/paul8339/category/1013777.html)
- [容器(34)](https://www.cnblogs.com/paul8339/category/1780249.html)
- [数据库(2)](https://www.cnblogs.com/paul8339/category/2350024.html)
- [中间件(125)](https://www.cnblogs.com/paul8339/category/1012255.html)
- [自动化运维(53)](https://www.cnblogs.com/paul8339/category/1025054.html)
- 更多

### 随笔档案  (1277)

- [2026年4月(5)](https://www.cnblogs.com/paul8339/p/archive/2026/04)
- [2026年3月(8)](https://www.cnblogs.com/paul8339/p/archive/2026/03)
- [2026年1月(1)](https://www.cnblogs.com/paul8339/p/archive/2026/01)
- [2025年12月(11)](https://www.cnblogs.com/paul8339/p/archive/2025/12)
- [2025年11月(9)](https://www.cnblogs.com/paul8339/p/archive/2025/11)
- [2025年10月(2)](https://www.cnblogs.com/paul8339/p/archive/2025/10)
- [2025年9月(1)](https://www.cnblogs.com/paul8339/p/archive/2025/09)
- [2025年8月(3)](https://www.cnblogs.com/paul8339/p/archive/2025/08)
- [2025年7月(4)](https://www.cnblogs.com/paul8339/p/archive/2025/07)
- [2025年6月(4)](https://www.cnblogs.com/paul8339/p/archive/2025/06)
- [2025年5月(3)](https://www.cnblogs.com/paul8339/p/archive/2025/05)
- [2025年4月(5)](https://www.cnblogs.com/paul8339/p/archive/2025/04)
- [2025年3月(3)](https://www.cnblogs.com/paul8339/p/archive/2025/03)
- [2025年2月(4)](https://www.cnblogs.com/paul8339/p/archive/2025/02)
- [2025年1月(14)](https://www.cnblogs.com/paul8339/p/archive/2025/01)
- [2024年12月(8)](https://www.cnblogs.com/paul8339/p/archive/2024/12)
- [2024年11月(7)](https://www.cnblogs.com/paul8339/p/archive/2024/11)
- [2024年10月(12)](https://www.cnblogs.com/paul8339/p/archive/2024/10)
- [2024年9月(3)](https://www.cnblogs.com/paul8339/p/archive/2024/09)
- [2024年8月(4)](https://www.cnblogs.com/paul8339/p/archive/2024/08)
- [2024年7月(3)](https://www.cnblogs.com/paul8339/p/archive/2024/07)
- [2024年5月(3)](https://www.cnblogs.com/paul8339/p/archive/2024/05)
- [2024年4月(14)](https://www.cnblogs.com/paul8339/p/archive/2024/04)
- [2024年3月(7)](https://www.cnblogs.com/paul8339/p/archive/2024/03)
- [2024年2月(4)](https://www.cnblogs.com/paul8339/p/archive/2024/02)
- [2024年1月(3)](https://www.cnblogs.com/paul8339/p/archive/2024/01)
- [2023年12月(9)](https://www.cnblogs.com/paul8339/p/archive/2023/12)
- [2023年11月(2)](https://www.cnblogs.com/paul8339/p/archive/2023/11)
- [2023年10月(14)](https://www.cnblogs.com/paul8339/p/archive/2023/10)
- [2023年9月(10)](https://www.cnblogs.com/paul8339/p/archive/2023/09)
- [2023年8月(6)](https://www.cnblogs.com/paul8339/p/archive/2023/08)
- [2023年7月(1)](https://www.cnblogs.com/paul8339/p/archive/2023/07)
- [2023年6月(5)](https://www.cnblogs.com/paul8339/p/archive/2023/06)
- [2023年5月(4)](https://www.cnblogs.com/paul8339/p/archive/2023/05)
- [2023年4月(6)](https://www.cnblogs.com/paul8339/p/archive/2023/04)
- [2023年3月(15)](https://www.cnblogs.com/paul8339/p/archive/2023/03)
- [2023年2月(14)](https://www.cnblogs.com/paul8339/p/archive/2023/02)
- [2023年1月(12)](https://www.cnblogs.com/paul8339/p/archive/2023/01)
- [2022年12月(13)](https://www.cnblogs.com/paul8339/p/archive/2022/12)
- [2022年11月(9)](https://www.cnblogs.com/paul8339/p/archive/2022/11)
- [2022年10月(4)](https://www.cnblogs.com/paul8339/p/archive/2022/10)
- [2022年9月(5)](https://www.cnblogs.com/paul8339/p/archive/2022/09)
- [2022年8月(10)](https://www.cnblogs.com/paul8339/p/archive/2022/08)
- [2022年7月(6)](https://www.cnblogs.com/paul8339/p/archive/2022/07)
- [2022年6月(3)](https://www.cnblogs.com/paul8339/p/archive/2022/06)
- [2022年4月(1)](https://www.cnblogs.com/paul8339/p/archive/2022/04)
- [2022年3月(5)](https://www.cnblogs.com/paul8339/p/archive/2022/03)
- [2022年2月(5)](https://www.cnblogs.com/paul8339/p/archive/2022/02)
- [2022年1月(13)](https://www.cnblogs.com/paul8339/p/archive/2022/01)
- [2021年12月(11)](https://www.cnblogs.com/paul8339/p/archive/2021/12)
- [2021年11月(11)](https://www.cnblogs.com/paul8339/p/archive/2021/11)
- [2021年10月(7)](https://www.cnblogs.com/paul8339/p/archive/2021/10)
- [2021年9月(7)](https://www.cnblogs.com/paul8339/p/archive/2021/09)
- [2021年8月(3)](https://www.cnblogs.com/paul8339/p/archive/2021/08)
- [2021年7月(8)](https://www.cnblogs.com/paul8339/p/archive/2021/07)
- [2021年6月(6)](https://www.cnblogs.com/paul8339/p/archive/2021/06)
- [2021年5月(8)](https://www.cnblogs.com/paul8339/p/archive/2021/05)
- [2021年4月(18)](https://www.cnblogs.com/paul8339/p/archive/2021/04)
- [2021年3月(7)](https://www.cnblogs.com/paul8339/p/archive/2021/03)
- [2021年2月(12)](https://www.cnblogs.com/paul8339/p/archive/2021/02)
- [2021年1月(5)](https://www.cnblogs.com/paul8339/p/archive/2021/01)
- [2020年12月(8)](https://www.cnblogs.com/paul8339/p/archive/2020/12)
- [2020年11月(17)](https://www.cnblogs.com/paul8339/p/archive/2020/11)
- [2020年10月(9)](https://www.cnblogs.com/paul8339/p/archive/2020/10)
- [2020年9月(22)](https://www.cnblogs.com/paul8339/p/archive/2020/09)
- [2020年8月(13)](https://www.cnblogs.com/paul8339/p/archive/2020/08)
- [2020年7月(4)](https://www.cnblogs.com/paul8339/p/archive/2020/07)
- [2020年6月(2)](https://www.cnblogs.com/paul8339/p/archive/2020/06)
- [2020年5月(6)](https://www.cnblogs.com/paul8339/p/archive/2020/05)
- [2020年4月(3)](https://www.cnblogs.com/paul8339/p/archive/2020/04)
- [2020年3月(5)](https://www.cnblogs.com/paul8339/p/archive/2020/03)
- [2020年2月(6)](https://www.cnblogs.com/paul8339/p/archive/2020/02)
- [2020年1月(9)](https://www.cnblogs.com/paul8339/p/archive/2020/01)
- [2019年12月(9)](https://www.cnblogs.com/paul8339/p/archive/2019/12)
- [2019年11月(4)](https://www.cnblogs.com/paul8339/p/archive/2019/11)
- [2019年10月(7)](https://www.cnblogs.com/paul8339/p/archive/2019/10)
- [2019年9月(11)](https://www.cnblogs.com/paul8339/p/archive/2019/09)
- [2019年8月(18)](https://www.cnblogs.com/paul8339/p/archive/2019/08)
- [2019年7月(2)](https://www.cnblogs.com/paul8339/p/archive/2019/07)
- [2019年6月(4)](https://www.cnblogs.com/paul8339/p/archive/2019/06)
- [2019年5月(10)](https://www.cnblogs.com/paul8339/p/archive/2019/05)
- [2019年4月(7)](https://www.cnblogs.com/paul8339/p/archive/2019/04)
- [2019年3月(16)](https://www.cnblogs.com/paul8339/p/archive/2019/03)
- [2019年2月(2)](https://www.cnblogs.com/paul8339/p/archive/2019/02)
- [2019年1月(7)](https://www.cnblogs.com/paul8339/p/archive/2019/01)
- [2018年12月(16)](https://www.cnblogs.com/paul8339/p/archive/2018/12)
- [2018年11月(20)](https://www.cnblogs.com/paul8339/p/archive/2018/11)
- [2018年10月(16)](https://www.cnblogs.com/paul8339/p/archive/2018/10)
- [2018年9月(47)](https://www.cnblogs.com/paul8339/p/archive/2018/09)
- [2018年8月(2)](https://www.cnblogs.com/paul8339/p/archive/2018/08)
- [2018年7月(17)](https://www.cnblogs.com/paul8339/p/archive/2018/07)
- [2018年6月(23)](https://www.cnblogs.com/paul8339/p/archive/2018/06)
- [2018年5月(18)](https://www.cnblogs.com/paul8339/p/archive/2018/05)
- [2018年4月(15)](https://www.cnblogs.com/paul8339/p/archive/2018/04)
- [2018年3月(6)](https://www.cnblogs.com/paul8339/p/archive/2018/03)
- [2018年2月(10)](https://www.cnblogs.com/paul8339/p/archive/2018/02)
- [2018年1月(24)](https://www.cnblogs.com/paul8339/p/archive/2018/01)
- [2017年12月(23)](https://www.cnblogs.com/paul8339/p/archive/2017/12)
- [2017年11月(23)](https://www.cnblogs.com/paul8339/p/archive/2017/11)
- [2017年10月(27)](https://www.cnblogs.com/paul8339/p/archive/2017/10)
- [2017年9月(28)](https://www.cnblogs.com/paul8339/p/archive/2017/09)
- [2017年8月(39)](https://www.cnblogs.com/paul8339/p/archive/2017/08)
- [2017年7月(26)](https://www.cnblogs.com/paul8339/p/archive/2017/07)
- [2017年6月(34)](https://www.cnblogs.com/paul8339/p/archive/2017/06)
- [2017年5月(41)](https://www.cnblogs.com/paul8339/p/archive/2017/05)
- [2017年4月(34)](https://www.cnblogs.com/paul8339/p/archive/2017/04)
- [2017年3月(16)](https://www.cnblogs.com/paul8339/p/archive/2017/03)
- [2017年2月(11)](https://www.cnblogs.com/paul8339/p/archive/2017/02)
- [2017年1月(15)](https://www.cnblogs.com/paul8339/p/archive/2017/01)
- [2016年12月(29)](https://www.cnblogs.com/paul8339/p/archive/2016/12)
- [2016年11月(4)](https://www.cnblogs.com/paul8339/p/archive/2016/11)
- [2016年10月(7)](https://www.cnblogs.com/paul8339/p/archive/2016/10)
- [2016年9月(12)](https://www.cnblogs.com/paul8339/p/archive/2016/09)
- [2016年8月(25)](https://www.cnblogs.com/paul8339/p/archive/2016/08)
- [2016年7月(15)](https://www.cnblogs.com/paul8339/p/archive/2016/07)
- [2016年6月(4)](https://www.cnblogs.com/paul8339/p/archive/2016/06)
- [2016年5月(1)](https://www.cnblogs.com/paul8339/p/archive/2016/05)
- [2016年4月(8)](https://www.cnblogs.com/paul8339/p/archive/2016/04)
- [2016年3月(6)](https://www.cnblogs.com/paul8339/p/archive/2016/03)
- [2016年2月(3)](https://www.cnblogs.com/paul8339/p/archive/2016/02)
- [2016年1月(2)](https://www.cnblogs.com/paul8339/p/archive/2016/01)
- [2015年12月(3)](https://www.cnblogs.com/paul8339/p/archive/2015/12)
- [2015年11月(1)](https://www.cnblogs.com/paul8339/p/archive/2015/11)
- [2015年10月(4)](https://www.cnblogs.com/paul8339/p/archive/2015/10)
- [2014年7月(1)](https://www.cnblogs.com/paul8339/p/archive/2014/07)
- 更多

### [阅读排行榜](https://www.cnblogs.com/paul8339/most-viewed)

- [1\. iptables做TCP/UDP端口转发【转】(56382)](https://www.cnblogs.com/paul8339/p/14688156.html)
- [2\. 宝塔Linux面板新手安装教程【转】(44321)](https://www.cnblogs.com/paul8339/p/7065799.html)
- [3\. 诡异的Linux磁盘空间被占用问题，根目录满了，df和du占用不一样【转】(42628)](https://www.cnblogs.com/paul8339/p/6381946.html)
- [4\. Nginx实现404页面的几种方法【转】(38448)](https://www.cnblogs.com/paul8339/p/7389422.html)
- [5\. 有了SSL证书，如何在IIS环境下部署https？【转载】(38198)](https://www.cnblogs.com/paul8339/p/5879272.html)

### [评论排行榜](https://www.cnblogs.com/paul8339/most-commented)

- [1\. 有了SSL证书，如何在IIS环境下部署https？【转载】(13)](https://www.cnblogs.com/paul8339/p/5879272.html)
- [2\. 挂载cifs报错mount error(13): Permission denied（域账号访问时报错）(3)](https://www.cnblogs.com/paul8339/p/7199520.html)
- [3\. linux统计某个特定文件名的大小总和【原创】(2)](https://www.cnblogs.com/paul8339/p/9351675.html)
- [4\. nginx 实现mysql的负载均衡【转】(2)](https://www.cnblogs.com/paul8339/p/6934266.html)
- [5\. mysql my.cnf配置文件读取顺序，以及哪个优先级最高(1)](https://www.cnblogs.com/paul8339/p/19281761)

### [推荐排行榜](https://www.cnblogs.com/paul8339/most-liked)

- [1\. 有了SSL证书，如何在IIS环境下部署https？【转载】(8)](https://www.cnblogs.com/paul8339/p/5879272.html)
- [2\. ansible报错Using a SSH password instead of a key is not possible because Host Key checking is enabled and sshpass does not support this(3)](https://www.cnblogs.com/paul8339/p/9106086.html)
- [3\. Mysql Group Replication 简介及单主模式组复制配置【转】(3)](https://www.cnblogs.com/paul8339/p/7426880.html)
- [4\. 14个最常见的Kafka面试题及答案【转】(3)](https://www.cnblogs.com/paul8339/p/7412512.html)
- [5\. keepalived配置文件详解【转】(2)](https://www.cnblogs.com/paul8339/p/17203142.html)

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)

- 1  [kubelet的功能、作用是什么？（重点...](https://www.cnblogs.com/paul8339/p/18844452#autoid-0-0-0 "kubelet的功能、作用是什么？（重点，经常会问）")
- 2  [kube-api-server的端口是多...](https://www.cnblogs.com/paul8339/p/18844452#autoid-1-0-0 "kube-api-server的端口是多少？各个pod是如何访问kube-api-server的？")
- 3  [pod是什么？](https://www.cnblogs.com/paul8339/p/18844452#autoid-2-0-0 "pod是什么？")
- 4  [pause容器作用是什么？（经常问）...](https://www.cnblogs.com/paul8339/p/18844452#autoid-3-0-0 "pause容器作用是什么？（经常问）")
- 5  [pod的重启策略有哪些？（经常问）...](https://www.cnblogs.com/paul8339/p/18844452#autoid-4-0-0 "pod的重启策略有哪些？（经常问）")
- 6  [pod的镜像拉取策略有哪几种？（经常问）...](https://www.cnblogs.com/paul8339/p/18844452#autoid-5-0-0 "pod的镜像拉取策略有哪几种？（经常问）")
- 7  [pod的存活探针有哪几种？（必须记住3种...](https://www.cnblogs.com/paul8339/p/18844452#autoid-6-0-0 "pod的存活探针有哪几种？（必须记住3种探测方式，重点，经常问）")
- 8  [pod的就绪探针有哪几种？（必须记住3种...](https://www.cnblogs.com/paul8339/p/18844452#autoid-7-0-0 "pod的就绪探针有哪几种？（必须记住3种探测方式，重点，经常问）")