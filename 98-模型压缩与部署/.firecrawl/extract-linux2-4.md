# 面试题：Linux 系统基础(一) - 讲文张字- 博客园

URL: https://www.cnblogs.com/zhangwencheng/p/18280930

# [面试题：Linux 系统基础 (一)](https://www.cnblogs.com/zhangwencheng/p/18280930 "发布于 2024-07-03 09:41")

[合集 \- 面试题(2)](https://www.cnblogs.com/zhangwencheng/collections/17904)

1.面试题：Linux 系统基础 (一)2024-07-03

[2.面试题：Linux 系统基础 (二)2025-03-31](https://www.cnblogs.com/zhangwencheng/p/18285198)

收起

**目录**

- [Linux系统中如何管理用户和组？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label0)
- [Linux系统中常见的文件权限有哪些？如何修改它们？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label1)
- [Linux系统中的进程管理包括哪些基本操作？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label2)
- [Linux系统中的网络配置包括哪些基本操作？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label3)
- [Linux中如何使用cron和crontab进行任务调度？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label4)
- [Linux系统中的日志管理主要涉及哪些方面？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label5)
- [Linux系统中的inode是什么，它有什么作用？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label6)
- [Linux中的Swap空间是什么，它是如何工作的？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label7)
- [Linux系统中如何查看和监控系统性能？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label8)
- [Linux中的权限控制列表（ACL）是什么，如何使用它们？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label9)
- [Linux系统中的“僵尸进程”是什么，如何处理它们？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label10)
- [Linux中的I/O调度器有哪些类型，它们各自的特点是什么？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label11)
- [Linux系统中的RAID是什么，它有哪些常见类型？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label12)
- [Linux系统中的NFS是什么，它的主要用途是什么？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label13)
- [Linux中的SSH是什么，如何安全地使用SSH？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label14)
- [Linux系统中的Kernel Panic是什么，通常由什么原因引起？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label15)
- [Linux系统中如何查看和配置静态路由？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label16)
- [Linux中如何监控磁盘空间和使用情况？](https://www.cnblogs.com/zhangwencheng/p/18280930#_label17)

**正文**

### Linux系统中如何管理用户和组？

Linux系统中用户和组的管理通常包括以下几个方面：

1、 **创建用户和组**： 使用`useradd`和`groupadd`命令创建新用户和新组。

2、 **修改用户和组信息**： 使用`usermod`和`groupmod`命令来修改用户和组的信息。

3、 **删除用户和组**： 使用`userdel`和`groupdel`命令来删除用户和组。

4、 **管理密码**： 使用`passwd`命令来管理用户密码。

这些基本命令和操作使得Linux系统管理员能够有效地管理系统的用户和权限。

### Linux系统中常见的文件权限有哪些？如何修改它们？

1、Linux系统中的文件权限主要包括读（r）、写（w）和执行（x）权限。权限可以针对文件的所有者、所属组以及其他用户设置。

2、修改文件权限的常用命令是`chmod`。例如，`chmod u+x filename`命令将给文件所有者增加执行权限。

3、另外，`chown`和`chgrp`命令可用于更改文件的所有者和所属组。

### Linux系统中的进程管理包括哪些基本操作？

Linux系统中的进程管理主要包括以下操作：

1、 **查看进程**： 使用`ps`和`top`命令查看当前系统中的进程。

2、 **控制进程**： 使用`kill`、`pkill`、和`killall`命令来终止进程。

3、 **后台和前台控制**： 使用`bg`将进程移到后台，使用`fg`将进程带回前台。

4、 **调整优先级**： 使用`nice`和`renice`命令调整进程的优先级。

### Linux系统中的网络配置包括哪些基本操作？

Linux系统中的网络配置主要涉及以下操作：

1、 **查看和配置网络接口**： 使用`ifconfig`或`ip`命令查看和配置网络接口。

2、 **管理路由表**： 使用`route`或`ip route`命令查看和修改路由表。

3、 **配置DNS**： 编辑`/etc/resolv.conf`文件来配置DNS服务器。

4、 **测试网络连接**： 使用`ping`和`traceroute`命令测试网络连通性。

### Linux中如何使用cron和crontab进行任务调度？

在Linux中，cron是一个基于时间的任务调度器，用于定期执行任务。

1、`crontab`命令用来创建、修改、删除或列出cron作业。

2、编辑`crontab文件`时，需要指定执行时间和相应的命令。

3、格式通常为`“分钟 小时 日期 月份 星期 命令”`。

### Linux系统中的日志管理主要涉及哪些方面？

Linux系统的日志管理主要包括以下几个方面：

1、 **日志文件存储**： 系统和应用日志通常存储在`/var/log`目录。

2、 **日志级别**： 日志有不同的级别，如`INFO`、`ERROR`、`DEBUG`等。

3、 **查看日志**： 使用如`cat`、`less`、`grep`等命令查看日志文件。

4、 **日志轮换**： 使用`logrotate`等工具进行日志文件的轮换和管理。

5、 **系统日志服务**： 如`syslogd`或`rsyslog`服务，负责日志的收集和处理。

### Linux系统中的inode是什么，它有什么作用？

在Linux系统中，inode（索引节点）是文件系统的一个重要概念。每个文件或目录都有一个与之对应的inode，它包含了关于文件的元数据，但不包含文件名或文件数据。inode的作用包括：

1、 **存储属性**： `inode`存储了文件的属性，如大小、权限、所有者、创建时间、最后访问和修改时间等。

2、 **链接信息**： `inode`存储了文件数据块的位置信息，这是文件系统查找文件内容的关键。

3、 **管理硬链接**： 通过`inode`，多个文件名（硬链接）可以指向同一个文件。

inode是文件系统中非常关键的部分，对于文件的管理和访问至关重要。

### Linux中的Swap空间是什么，它是如何工作的？

1、`Swap空间`是Linux系统中的一种虚拟内存管理机制。它的主要作用是：

2、当物理内存不足时，系统可以将内存中不常用的数据页交换到`Swap空间`，从而为需要更多内存的进程释放空间。

3、`Swap空间`可以是一个专用的分区（`Swap分区`）或一个文件（`Swap文件`）。

4、`Swap`的工作原理是基于页面置换算法，当系统物理内存紧张时，它会将内存中的数据页移动到`Swap空间`，从而为新的数据页腾出空间。

### Linux系统中如何查看和监控系统性能？

在Linux系统中，查看和监控系统性能的常用方法包括：

1、`top`命令： 实时显示系统中各个进程的资源占用情况。

2、`vmstat`命令： 报告虚拟内存统计信息。

3、`iostat`命令： 监控系统输入输出设备和CPU使用情况。

4、`netstat`命令： 显示网络连接、路由表、接口统计等网络信息。

5、`free`命令： 显示系统内存使用情况。

这些工具和命令能帮助系统管理员有效监控和诊断系统性能问题。

### Linux中的权限控制列表（ACL）是什么，如何使用它们？

Linux中的权限控制列表（ACL）提供了比传统权限设置更详细的权限控制。使用ACL可以对单个用户或组设置特定的权限。基本的ACL操作包括：

1、使用`getfacl`命令查看文件或目录的ACL。

2、使用`setfacl`命令设置或修改文件或目录的ACL。

例如，`setfacl -m u:username:rw file`命令会给特定用户对文件的读写权限。ACL是高级文件权限设置的有效工具。

### Linux系统中的“僵尸进程”是什么，如何处理它们？

在Linux系统中，僵尸进程（Zombie Process）是指已经完成执行但其父进程尚未读取其退出状态的进程。它们的特点和处理方法如下：

1、 **特点**： 僵尸进程已经释放了大部分资源，但在进程表中保留一个条目，包含退出码等信息。

2、 **处理**： 通常，父进程会通过调用`wait()`或`waitpid()`函数来读取子进程的退出状态，从而移除僵尸进程。如果父进程未正确处理，僵尸进程将保持在系统中。

3、 **解决方法**： 可以尝试手动终止父进程，这通常会导致它的所有子进程被init进程接管并清理。如果不想终止父进程，可能需要修改并重启有问题的应用以防止产生更多僵尸进程。

僵尸进程通常不会导致严重问题，但如果数量过多，可能会消耗系统资源。

### Linux中的I/O调度器有哪些类型，它们各自的特点是什么？

Linux中的I/O调度器负责管理对磁盘的读写请求。主要类型及其特点包括：

1、`CFQ（完全公平队列）`调度器： 为每个进程提供公平的磁盘时间。适用于多任务和通用系统。

2、`Deadline`调度器： 确保请求在一定时间内完成，优先处理即将到期的请求。适用于需要快速响应的系统。

3、`NOOP（无操作）`调度器： 是一个简单的FIFO队列，适用于高级存储系统（如SSD）。

4、`Anticipatory`调度器： 基于启发式方法，预测下一个即将到来的I/O请求，减少磁盘寻道时间。

根据系统的具体需求和硬件配置，选择合适的I/O调度器可以显著提高系统性能。

### Linux系统中的RAID是什么，它有哪些常见类型？

RAID（冗余磁盘阵列的独立磁盘）是一种存储技术，它将多个磁盘组合成一个单一的逻辑单元，以提高性能和/或数据冗余。常见的RAID类型包括：

1、`RAID 0（条带化）`： 将数据分散存储在两个或更多的硬盘上，提高读写速度，但不提供冗余。

2、`RAID 1（镜像）`： 数据完全复制到两个或更多硬盘上，提供冗余，但容量只有单个硬盘的大小。

3、`RAID 5（带奇偶校验的条带化）`： 分布式奇偶校验，提供数据冗余和改善性能。

4、`RAID 10（镜像和条带化的组合）`： 结合了RAID 0和RAID 1的特点，提供冗余和性能的平衡。

选择合适的RAID级别取决于具体的性能和数据冗余需求。

### Linux系统中的NFS是什么，它的主要用途是什么？

NFS（Network File System）是一种分布式文件系统协议，允许一个系统上的用户访问另一个系统上的文件，就像访问本地文件一样。其主要用途包括：

1、 **共享存储**： 在网络中的多个机器间共享文件和目录。

2、 **简化数据管理**： 中央管理数据，而非在每个客户端单独存储。

3、 **提高可用性和灵活性**： 数据可以从网络的任何地方访问。

NFS广泛应用于企业和学术环境，提供了简单而有效的跨平台文件共享解决方案。

### Linux中的SSH是什么，如何安全地使用SSH？

SSH（Secure Shell）是一种网络协议，用于安全地访问远程计算机。安全使用SSH的方法包括：

1、 **使用密钥认证**： 相对于密码认证，使用SSH密钥对提供更高的安全性。

2、 **禁用根登录**： 修改SSH配置，禁止远程根用户登录。

3、 **更改默认端口**： 将SSH从默认的端口22更改为其他端口，减少自动化攻击的可能性。

4、 **使用防火墙限制访问**： 仅允许可信的IP地址访问SSH端口。

5、 **使用Fail2ban**： 自动阻止频繁尝试登录失败的IP地址。

遵循这些最佳实践可以显著提高通过SSH访问系统的安全性。

### Linux系统中的Kernel Panic是什么，通常由什么原因引起？

Kernel Panic是Linux系统中的一种致命错误，通常指内核遇到了无法安全恢复的问题。它可以由以下原因引起：

1、 **硬件故障**： 如内存损坏、硬盘故障等。

2、 **驱动程序错误**： 不兼容或有缺陷的硬件驱动可能导致内核崩溃。

3、 **文件系统损坏**： 严重的文件系统损坏可能导致内核无法正常读写数据。

4、 **内核自身的Bug**： 内核代码中的错误也可能导致崩溃。

处理Kernel Panic通常涉及检查硬件问题、更新驱动程序、检查文件系统，或升级内核。

### Linux系统中如何查看和配置静态路由？

在Linux系统中，静态路由可以通过以下命令查看和配置：

**查看路由**：使用`route -n`或`ip route`命令查看当前的路由表。

**配置静态路由**：使用route add或ip route add命令添加新的静态路由。

例如，`sudo ip route add 192.168.1.0/24 via 192.168.0.1 dev eth0`命令会添加一条通过网关192.168.0.1到达192.168.1.0/24网络的路由。

这些命令允许管理员手动管理网络流量的路由。

### Linux中如何监控磁盘空间和使用情况？

在Linux中监控磁盘空间和使用情况的常用命令包括：

1、`df`命令： 显示文件系统的总空间、已用空间、可用空间和挂载点。

2、`du`命令： 估算文件或目录的磁盘使用量。

3、`iotop`或`iostat`命令： 实时监控磁盘I/O使用情况。

定期检查磁盘空间和使用情况有助于预防空间不足问题。

\\*\\*\\*\\*\\*\\*\\*\\*\\*\\* 如果您认为这篇文章还不错或者有所收获，请点击右下角的【推荐】/【赞助】按钮，因为您的支持是我继续创作分享的最大动力！ \*\*\*\*\*\*\*\*\*\*

作者： [讲文张字](https://www.cnblogs.com/zhangwencheng)

出处： [https://www.cnblogs.com/zhangwencheng](https://www.cnblogs.com/zhangwencheng)

版权：本文版权归作者和博客园共有，欢迎转载，但未经作者同意必须保留此段声明，且在文章页面明显位置给出
[原文链接](https://www.cnblogs.com/zhangwencheng/p/18280930#)

合集:
[面试题](https://www.cnblogs.com/zhangwencheng/collections/17904)

分类:
[面试题](https://www.cnblogs.com/zhangwencheng/category/2407536.html)

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/1080590/20230619172255.png)](https://home.cnblogs.com/u/zhangwencheng/)

[讲文张字](https://home.cnblogs.com/u/zhangwencheng/)

[粉丝 \- 77](https://home.cnblogs.com/u/zhangwencheng/followers/) [关注 \- 1](https://home.cnblogs.com/u/zhangwencheng/followees/)

+加关注

1

0

[«](https://www.cnblogs.com/zhangwencheng/p/18252574) 上一篇： [CentOS 7 内核升级最新记录(yum及编译) 2024-08](https://www.cnblogs.com/zhangwencheng/p/18252574 "发布于 2024-06-24 10:01")

[»](https://www.cnblogs.com/zhangwencheng/p/18261074) 下一篇： [Zabbix LTS URL 健康监测](https://www.cnblogs.com/zhangwencheng/p/18261074 "发布于 2024-07-12 10:32")

posted @
2024-07-03 09:41 [讲文张字](https://www.cnblogs.com/zhangwencheng)
阅读(1216)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2Fzhangwencheng%2Fp%2F18280930&targetId=18280930&targetType=0)

[刷新页面](https://www.cnblogs.com/zhangwencheng/p/18280930#) [返回顶部](https://www.cnblogs.com/zhangwencheng/p/18280930#top)

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
[讲文张字](https://home.cnblogs.com/u/zhangwencheng/)

园龄：
[9年4个月](https://home.cnblogs.com/u/zhangwencheng/ "入园时间：2016-12-14")

粉丝：
[77](https://home.cnblogs.com/u/zhangwencheng/followers/)

关注：
[1](https://home.cnblogs.com/u/zhangwencheng/followees/)

+加关注

### 搜索

### 最新随笔

- [1\. MySQL 数据库日志总结(二)](https://www.cnblogs.com/zhangwencheng/p/19393122)
- [2\. MySQL 数据库日志总结(一)](https://www.cnblogs.com/zhangwencheng/p/19388036)
- [3\. MySQL 数据库事务知识](https://www.cnblogs.com/zhangwencheng/p/19168425)
- [4\. K8s Pod 资源访问控制策略](https://www.cnblogs.com/zhangwencheng/p/18864734)
- [5\. K8s Pod 多种数据存储方式](https://www.cnblogs.com/zhangwencheng/p/18864383)
- [6\. K8s 1.29.3 on Docker](https://www.cnblogs.com/zhangwencheng/p/18866490)
- [7\. Docker 国内镜像加速地址-长期更新](https://www.cnblogs.com/zhangwencheng/p/18909645)
- [8\. K8s Pod 资源如何访问](https://www.cnblogs.com/zhangwencheng/p/18844133)
- [9\. K8s Pod 控制器介绍及应用示例](https://www.cnblogs.com/zhangwencheng/p/18793889)
- [10\. 面试题：Linux 系统基础 (二)](https://www.cnblogs.com/zhangwencheng/p/18285198)

### 积分与排名

- 积分 \-
177570

- 排名 \-
7776


### 合集  (17)

- [Kubernetes(10)](https://www.cnblogs.com/zhangwencheng/collections/24788)
- [Docker(4)](https://www.cnblogs.com/zhangwencheng/collections/20523)
- [CI/CD(3)](https://www.cnblogs.com/zhangwencheng/collections/10894)
- [Redis(7)](https://www.cnblogs.com/zhangwencheng/collections/8648)
- [JumpServer(3)](https://www.cnblogs.com/zhangwencheng/collections/8649)
- [Zabbix(6)](https://www.cnblogs.com/zhangwencheng/collections/8654)
- [Ansible(2)](https://www.cnblogs.com/zhangwencheng/collections/8657)
- [MySQL(9)](https://www.cnblogs.com/zhangwencheng/collections/8650)
- [MongoDB(3)](https://www.cnblogs.com/zhangwencheng/collections/8652)
- [Nginx(3)](https://www.cnblogs.com/zhangwencheng/collections/8651)
- [LB&HA(2)](https://www.cnblogs.com/zhangwencheng/collections/8653)
- [Apache(2)](https://www.cnblogs.com/zhangwencheng/collections/8655)
- [System(14)](https://www.cnblogs.com/zhangwencheng/collections/8656)
- [Huawei R/S(35)](https://www.cnblogs.com/zhangwencheng/collections/8658)
- [Huawei FW(6)](https://www.cnblogs.com/zhangwencheng/collections/8660)
- [Huawei Wlan(4)](https://www.cnblogs.com/zhangwencheng/collections/8661)
- [面试题(2)](https://www.cnblogs.com/zhangwencheng/collections/17904)
- 更多

### 友情链接

- [Mysql 官方文档](https://dev.mysql.com/doc/)
- [Mysql rpm下载](https://repo.mysql.com/yum)
- [Mysql源码下载](https://downloads.mysql.com/archives/community/)
- [Nginx 官方文档](http://nginx.org/en/docs/)
- [Nginx rpm下载](http://nginx.org/packages/centos/7/x86_64/RPMS/)
- [Nginx源码下载](http://nginx.org/download/)
- [Ansible 官方文档](https://docs.ansible.com/)
- [Zabbix 官方文档](https://www.zabbix.com/manuals)
- [ProxySQL 官方文档](https://proxysql.com/documentation/)
- [HAProxy 官方文档](https://www.haproxy.com/documentation/)
- [Rpm包下载](https://pkgs.org/)
- [MongoDB 官方文档](https://www.mongodb.com/docs/)
- [Git 源码下载](https://mirrors.edge.kernel.org/pub/software/scm/git/)
- [GitLab 官方文档](https://docs.gitlab.com/)
- [鸟哥Linux私房菜](http://linux.vbird.org/)
- [DockerHub 官方地址](https://hub.docker.com/)
- [Kubernetes 官方文档](https://kubernetes.io/docs/home/)
- 更多

### [阅读排行榜](https://www.cnblogs.com/zhangwencheng/most-viewed)

- [1\. Docker 国内镜像加速地址-长期更新(19801)](https://www.cnblogs.com/zhangwencheng/p/18909645)
- [2\. 配置通过Console口登录交换机/路由器(16638)](https://www.cnblogs.com/zhangwencheng/p/13864162.html)
- [3\. Ubuntu 22.04/24.04 LTS 用 sed 快速换国内源(8732)](https://www.cnblogs.com/zhangwencheng/p/18472769)
- [4\. 交换机通过Loopback Detection检测(接口自环)(8228)](https://www.cnblogs.com/zhangwencheng/p/14033243.html)
- [5\. WLAN-AC+AP射频一劳永逸的调优方式(7418)](https://www.cnblogs.com/zhangwencheng/p/14416250.html)
- [6\. 配置OSPF与BFD联动(5583)](https://www.cnblogs.com/zhangwencheng/p/14153409.html)
- [7\. 配置交换机之间直连链路聚合-LACP模式(5564)](https://www.cnblogs.com/zhangwencheng/p/13903131.html)
- [8\. Linux下使用Ansible处理批量操作(4950)](https://www.cnblogs.com/zhangwencheng/p/14921603.html)
- [9\. HUAWEI交换机如何判断环路故障(4915)](https://www.cnblogs.com/zhangwencheng/p/15852971.html)
- [10\. Ubuntu 22.04 LTS 在线/离线安装 Docker(4653)](https://www.cnblogs.com/zhangwencheng/p/18472429)
- [11\. 交换机通过Loopback Detection检测(设备所在网络环路)(4506)](https://www.cnblogs.com/zhangwencheng/p/14033981.html)
- [12\. 交换机基于接口划分VLAN（汇聚层设备作为网关）(3892)](https://www.cnblogs.com/zhangwencheng/p/13959795.html)
- [13\. CentOS7下搭建 JumpServer 堡垒机(3774)](https://www.cnblogs.com/zhangwencheng/p/17014608.html)
- [14\. 交换机基于接口划分VLAN（接入层设备作为网关）(3645)](https://www.cnblogs.com/zhangwencheng/p/13955023.html)
- [15\. Linux下Rsyslog日志远程集中式管理(3630)](https://www.cnblogs.com/zhangwencheng/p/14862190.html)

### [推荐排行榜](https://www.cnblogs.com/zhangwencheng/most-liked)

- [1\. Kubernetes 知识梳理及集群搭建(11)](https://www.cnblogs.com/zhangwencheng/p/18672481)
- [2\. Docker 知识梳理及其 CentOS7.9 在线/离线安装使用(9)](https://www.cnblogs.com/zhangwencheng/p/18392444)
- [3\. Nginx + Keepalived 高可用集群部署(5)](https://www.cnblogs.com/zhangwencheng/p/17182896.html)
- [4\. Docker 国内镜像加速地址-长期更新(4)](https://www.cnblogs.com/zhangwencheng/p/18909645)
- [5\. 解读 Redis 常见命令(4)](https://www.cnblogs.com/zhangwencheng/p/17667905.html)
- [6\. CentOS7下搭建 JumpServer 堡垒机(4)](https://www.cnblogs.com/zhangwencheng/p/17014608.html)
- [7\. Linux下Nginx基础应用(4)](https://www.cnblogs.com/zhangwencheng/p/15006686.html)
- [8\. GitLab 简述及安装部署(3)](https://www.cnblogs.com/zhangwencheng/p/17848971.html)
- [9\. Redis 哨兵模式的原理及其搭建(3)](https://www.cnblogs.com/zhangwencheng/p/17717584.html)
- [10\. Redis 持久化 (RDB和AOF) 梳理(3)](https://www.cnblogs.com/zhangwencheng/p/17715096.html)
- [11\. MongoDB 基础知识梳理(3)](https://www.cnblogs.com/zhangwencheng/p/17239034.html)
- [12\. Linux下MySQL多实例部署记录(3)](https://www.cnblogs.com/zhangwencheng/p/15045074.html)
- [13\. 如何在Linux下部署Samba服务？(3)](https://www.cnblogs.com/zhangwencheng/p/14782027.html)
- [14\. 配置通过Console口登录交换机/路由器(3)](https://www.cnblogs.com/zhangwencheng/p/13864162.html)
- [15\. 掌握 K8s Pod 基础应用 (二)(2)](https://www.cnblogs.com/zhangwencheng/p/18762763)

[返回顶部](https://www.cnblogs.com/zhangwencheng/p/18280930#top)

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)

![](<Base64-Image-Removed>)

# 喜欢请打赏

- 支付宝![支付宝](<Base64-Image-Removed>)
- 微信![微信](<Base64-Image-Removed>)

扫描二维码打赏

![](https://files-cdn.cnblogs.com/files/zhangwencheng/zhifubao.bmp)

支付宝打赏

[了解更多](https://github.com/greedying/tctip)