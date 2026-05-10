# 网传Linux运维面试题解答（二）

URL: https://developer.aliyun.com/article/517729

[开发者社区](https://developer.aliyun.com/) [安全](https://developer.aliyun.com/group/security/) [文章](https://developer.aliyun.com/group/security/article/) 正文

# 网传Linux运维面试题解答（二）

2017-11-121802

版权
举报

版权声明：

本文内容由阿里云实名注册用户自发贡献，版权归原作者所有，阿里云开发者社区不拥有其著作权，亦不承担相应法律责任。具体规则请查看《
[阿里云开发者社区用户服务协议](https://developer.aliyun.com/article/768092)》和
《 [阿里云开发者社区知识产权保护指引](https://developer.aliyun.com/article/768093)》。如果您发现本社区中有涉嫌抄袭的内容，填写
[侵权投诉表单](https://yida.alibaba-inc.com/o/right) 进行举报，一经查实，本社区将立刻删除涉嫌侵权内容。


**简介：**

题目来源： [http://2358205.blog.51cto.com/2348205/1688323](http://2358205.blog.51cto.com/2348205/1688323)

      http://mofansheng.blog.51cto.com/8792265/1627907

大部分都自己做的，部分参考了原帖博主的答案

1、Linux开机流程

**BIOS开机自检→MBR引导→加载GRUB→加载linux内核→运行init进程，读取/etc/inittab→执行/etc/rc.d/rc.sysinit脚本→执行/etc/rc.d/rc脚本，运行/etc/rc.d/rcX.d中的脚本，X表示inittab中指定的运行级别→执行/etc/rc.d/rc.local脚本→启动mingetty进程，等待用户登陆**

2、TCP三次握手，四次挥手

**说明：SYN(synchronous建立联机) ACK(acknowledgement 确认) PSH(push传送) FIN(finish结束) RST(reset重置) URG(urgent紧急)**

**Sequence number(顺序号码) Acknowledge number(确认号码)**

**第一次握手————建立连接时，客户端发送syn包（syn=j）到服务器，并进入syn\_send状态，等待服务器确认；**

**第二次握手————服务器收到syn包，必须确认客户的syn（ack=j+1），同时自己也发送一个syn包（syn=k），即syn+ack包，此时服务器进入syn\_recv状态；**

**第三次握手————客户端收到服务器的syn+ack包，向服务器发送确认包ack（ack=k+1），此包发送完毕，客户端和服务器进入established状态，完成三次握手，客户端与服务器开始传送数据。**

**第一次挥手————客户端发送一个FIN，用来关闭客户端到服务器的数据传送，客户端进入FIN\_WAIT\_1状态；**

**第二次挥手————服务器收到FIN后，发送一个ACK包给客户端，确认序号为收到序号+1（与SYN相同，一个FIN占用一个序号），服务器进入CLOSE\_WAIT状态；**

**第三次挥手————服务器发送一个FIN，用来关闭服务器到客户端的数据传送，服务器进入LAST\_ACK状态；**

**第四次挥手————客户端收到FIN后，客户端进入TIME\_WAIT状态，接着发送一个ACK给服务器，确认序号为收到序号+1，服务器进入CLOSED状态，完成四次挥手。**

3、如何将本地80端口的请求转发到8080端口,当前主机IP为192.168.16.1,其中本地网卡eth0

**iptables -t nat -A PREROUTING -o eth0 -d 192.168.16.1 -p tcp --dport 80 -j REDIRECT --to-ports 8080**

4、什么是NAT,常见分为那几种，DNAT与SNAT有什么不同，应用事例有那些？

**NAT（Network Address Translation，网络地址转换）是将IP数据包头中的IP地址转换为另一个IP地址的过程。分为DNAT （目的网络地址转换）和SNAT（源网络地址转换）**

**SNAT主要是用于内网主机通过路由器或网关访问外网**

**DNAT将外部地址和端口的访问映射到内部地址和端口**

5、包过滤防火墙与代理应用防火墙有什么区别，能列举几种相应的产品吗？

**过滤防火墙工作在网络层，它只对IP包的源地址、目标地址及相应端口进行处理，因此速度比较快，能够处理的并发连接比较多，缺点是对应用层的攻击无能为力。**

**代理服务器防火墙工作在应用层，它将收到的IP包还原成高层协议的通讯数据，比如http连接信息，因此能够对基于高层协议的攻击进行拦截。缺点是处理速度比较慢，能够处理的并发数比较少。代理服务器是防火墙技术的发展方向，众多厂商都在提高处理速度的同时基于代理开发防火墙的更高级防护功能。**

6、iptables是否支持time时间控制用户行为，如有请写出具体操作步骤。

**iptables是支持时间控制的，以下摘抄自man iptables：**

**EXAMPLES. To match on weekends, use:**

**-m time --weekdays Sa,Su**

**Or, to match (once) on a national holiday block:**

**-m time --datestart 2007-12-24 --datestop 2007-12-27**

**Since  the stop time is actually inclusive, you would need the following stop time to not match the first second**

**of the new day:**

**-m time --datestart 2007-01-01T17:00 --datestop 2007-01-01T23:59:59**

**During lunch hour:**

**-m time --timestart 12:30 --timestop 13:30**

**The fourth Friday in the month:**

**-m time --weekdays Fr --monthdays 22,23,24,25,26,27,28**

7、说出你知道的几种linux/unix发行版本。

**Linux：RHEL、CentOS、Fedora、SuSE、Debian、Ubuntu等**

**Unix：FreeBSD、Solaris、AIX、Mac OS X等**

8、列出linux常见打包工具并写相应解压缩参数(至少三种)

**tar**

**gzip**

**bzip2**

9、计划每星期天早8点服务器定时发送一封内容为：test的邮件。发信人：user1@ab.com 收信人：test1@example.com，如何实现？

**crontab -e**

**00 08 \* \* 7  echo "test" \| /bin/mail -r user1@ab.com -s test test1@example.com &>/dev/null**

10、我们都知道，dns既采用了tcp协议，又采用了udp协议，什么时候采用tcp协议？什么时候采用udp协议？为什么要这么设计？

**首先了解一下TCP与UDP传送字节的长度限制： UDP报文的最大长度为512字节，而TCP则允许报文长度超过512字节。当DNS查询超过512字节时，协议的TC标志出现删除标志，这时则使用TCP发送。通常传统的UDP报文一般不会大于512字节。**

**区域传送时使用TCP：**

**辅域名服务器会定时（一般时3小时）向主域名服务器进行查询以便了解数据是否有变动。如有变动，则会执行一次区域传送，进行数据同步。区域传送将使用TCP而不是UDP，因为数据同步传送的数据量比一个请求和应答的数据量要多得多，并且** **TCP是一种可靠的连接，保证了数据的准确性。**

**域名解析时使用UDP：**

**客户端向DNS服务器查询域名，一般返回的内容都不超过512字节，用UDP传输即可。不用经过TCP三次握手，这样DNS服务器负载更低，响应更快。虽然从理论上说，客户端也可以指定向DNS服务器查询的时候使用TCP，但事实上，很多DNS服务器进行配置的时候，仅支持UDP查询包。**

11、一个EXT3的文件分区，当使用touch test.file命令创建一个新文件时报错，报错的信息是提示磁盘已满，但是采用df -h命令查看磁盘大小时，只使用了，60%的磁盘空间，为什么会出现这个情况，说说你的理由。

**df -i 查看inode使用情况**

**一般是inode满了造成的**

12、我们都知道FTP协议有两种工作模式，说说它们的大概的一个工作流程？

**主动模式FTP：**

**命令连接：客户端 >1024端口 -> 服务器 21端口**

**数据连接：客户端 >1024端口 <- 服务器 20端口**

**被动模式FTP：**

**命令连接：客户端 >1024端口 -> 服务器 21端口**

**数据连接：客户端 >1024端口 -> 服务器 >1024端口**

**下面是主动与被动FTP优缺点的简要总结：**

**主动FTP对FTP服务器的管理有利，但对客户端的管理不利。因为FTP服务器企图与客户端的高位随机端口建立连接，而这个端口很有可能被客户端的防火墙阻塞掉。被动FTP对FTP客户端的管理有利，但对服务器端的管理不利。因为客户端要与服务器端建立两个连接，其中一个连到一个高位随机端口，而这个端口很有可能被服务器端的防火墙阻塞掉。**

**解决办法：使用被动模式，可以把数据连接的端口固定（大于1024小于65535），在服务器端的防火墙上开放该端口**

13、编写个shell脚本将当前目录下大于100K的文件转移到/tmp目录下

**find .  -size +100K  xargs -I {} mv {} /tmp**

14、apache有几种工作模式，分别介绍下其特点，并说明什么情况下采用不同的工作模式？

**Apache主要有两种工作模式：prefork（预派生，默认安装模式）和worker（支持混合的多线程多进程的多路处理模块，可以在编译的时候加参数--with-mpm=worker指定为worker模式）**

**prefork的特点：**

**1、prefork MPM 使用多个子进程，每个子进程只有一个线程。每个进程在某个确定的时间只能维持一个连接。这种模式可以不必在请求到来时再产生新的进程，从而减小了系统开销。**

**2、可以防止意外的内存泄漏，但这种模式消耗的内存比较大；**

**3、在服务器负载下降的时候会自动减少子进程数**

**worker的特点：**

**worker MPM 使用多个子进程，每个子进程有多个线程。每个线程在某个确定的时间只能维持一个连接。通常来说，在一个高流量的HTTP服务器上，Worker MPM是个比较好的选择，因为Worker MPM的内存使用比Prefork MPM要低得多。但worker MPM也由不完善的地方，假如一个线程崩溃，整个进程就会连同其任何线程一起"死掉".由于线程共享内存空间，所以一个程式在运行时必须被系统识别为"每 个线程都是安全的"。**

15、编写shell脚本获取本机的IP/netmask。

**ifconfig \| awk -F "\[ :\]+" 'NR==2 {print $4}'**

16、简述DDOS攻击的原理,有没有解决办法？有，如何解决？

**分布式服务拒绝攻击就是用一台主服务器来控制N台肉鸡对目标服务器进行合理的资源请求，导致服务器资源耗尽而不能进行正常的服务。**

**几种流行的DDOS攻击方式：SYN/ACK FLOOD攻击、TCP全连接攻击、CC攻击(百科：攻击者借助代理服务器生成指向受害主机的合法请求，实现DDOS,和伪装就叫:cc(ChallengeCollapsar)。CC主要是用来攻击页面的。)**

**一个简单的测试： 首先是网站如果打不开的话，可以尝试着用3389连接一下服务器看看，然后还可以用PING命令来测试，再一种方式就是用telnet来登录80端口看看，看会不会出现黑屏。如果这些方式测试都连接不上的话，那就说明受到DDOS攻击了。**

**然后如果除了80端口之外的其他端口连接都正常，PING命令测试也正常，但就是80端口访问不了，然后看看IIS是否正常，可以把80端口改成其他端口测试，如果可以正常访问，那就说明很可能受到CC攻击。**

**防御DDOS攻击：**

**<1>要有充足的网络带宽和稳定安全的机房：选择口碑好、服务好、安全防护好点的机房，网络带宽直接决定了能抗受攻击的能力。**

**<2>软硬设备的防护：硬件DDOS防火墙黑洞、冰盾都不错，软件如web服务器都有相应的ddos防护模块，iptables,做单IP的并发限制，流量限制，syn及部分攻击限制。**

**<3>网站架构优化，避免单点提供服务，集群，冗余，负载均衡、缓存技术的架设。**

**<4>服务器系统自身的优化及安全参数调配**

**<5>采用高性能的网络设备**

17、如何查看占用80端口的进程，并清理该端口进程。

**lsof -i :80**

**pkill 或 kill**

18、如何查看当前用户的家目录是什么?

**echo $HOME**

19、如果设置 umask 为 001 , 那么用户默认创建的目录和文件的权限是什么样子的？

**目录：776  文件：666**

20、出于安全考虑，如何实现让别人ping不通你的在线的服务器。

**有多种方法**

**防火墙上用ACL封ICMP协议或者在服务器上使用iptables封icmp**

**或者在服务器上修改内核参数：echo 1 > /proc/sys/net/ipv4/icmp\_echo\_ignore\_all**

21、怎样防止他人在服务器前通过按下 ctrl+alt+del 强行重启系统（提示，仔细看/etc/inittab）

**vim /etc/init/control-alt-delete.conf**

**exec /sbin/shutdown -r now "Control-Alt-Delete pressed"    这一项注释掉，保存退出**

22、当在对服务器进行大负荷操作的时候，你不希望现在有普通用户登录上来，你该怎么做？不能剪网线。

**touch /etc/nologin**

**创建一个nologin文件，此文件为特殊文件，创建之后所有普通用户不能登录；系统维护结束后删除此文件，用户可以恢复登录；只限于shell登录用户，本身shell为/sbin/nologin 的用户本身就无法登录shell，不受影响；**

23、你新建了一批用户，出于安全考虑，要求这些用户在第一次登录的时候 就必须要更改密码，怎么实现？

**查看密码和账户过期信息：chage -l username**

**将密码设置为过期，用户登陆必须要更改密码：** **chage -d0 username** **或 passwd -e username**

24、如何把一个目录下的所有文件(不含目录)权限改为644？

**find ./ ! -type d -exec chmod 644 \\;**

25、请实现下面这个需求：只允许使用普通账户登陆，而普通账户登录后，可以不输入密码就能sudo切换到root账户，root是不允许远程登录的。

**vim /etc/ssh/sshd\_config**

**PermitRootLogin        设置为no，禁止root远程登录**

**/etc/init.d/sshd reload   重新加载sshd配置文件生效**

**visudo，添加一行：user    ALL=(root)    NOPASSWD: /bin/su**

26、 如何使文件只能写不能删除？ 如何使文件不能被删除、重命名、设定链接接、写入、新增数据？

**chattr +a  只能向文件中添加数据，而不能删除**

**chattr +i 文件不能被删除、改名、设定链接关系，同时不能写入或新增内容**

27、用ls 查看目录或者文件时，第二列的数值表示什么意思？如果一个目录的这列的值为3，那么这个3是如何得到的？

**第二列的数值表示硬链接数，默认情况下，新建一个目录，该目录就会包含一个指向自身的目录“.”和指向其上一级目录的父目录“..”，该数值是2，若分别在新建的目录里建一个文件和一个目录，那么再次查询时，就会发现该数值变为了3，由此类推..**

**本文转自 kuSorZ 51CTO博客，原文链接:http://blog.51cto.com/kusorz/1841122**

文章标签：

[云防火墙](https://developer.aliyun.com/label/article_de-product-3-cfw)

[容器服务Kubernetes版](https://developer.aliyun.com/label/article_de-product-3-csk)

[DDoS防护](https://developer.aliyun.com/label/article_de-product-3-ddos)

[云解析DNS](https://developer.aliyun.com/label/article_de-product-3-dns)

[NAT网关](https://developer.aliyun.com/label/article_de-product-3-nat)

[网络协议](https://developer.aliyun.com/label/article_de-3-100112)

[网络安全](https://developer.aliyun.com/label/article_de-3-100098)

[测试技术](https://developer.aliyun.com/label/article_de-3-100022)

[Linux](https://developer.aliyun.com/label/article_de-3-100077)

[安全](https://developer.aliyun.com/label/article_de-3-100244)

关键词：

[Linux运维](https://www.aliyun.com/sswb/259131.html)

[运维linux](https://www.aliyun.com/sswb/545241.html)

[运维面试](https://www.aliyun.com/sswb/555883.html)

[Linux面试](https://www.aliyun.com/sswb/258736.html)

[运维面试题](https://www.aliyun.com/sswb/553775.html)

[![](https://ucc.alicdn.com/avatar/avatar3.jpg?x-oss-process=image/resize,h_150,m_lfit)](https://developer.aliyun.com/profile/gtdqzn4uqwgsg)

[科技小能手](https://developer.aliyun.com/profile/gtdqzn4uqwgsg)

+关注

[23586文章](https://developer.aliyun.com/profile/gtdqzn4uqwgsg/article_1)

目录

0

0

0

507

分享

相关文章

[1239147665459462](https://developer.aliyun.com/profile/tormpzfmxjtle)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [安全](https://developer.aliyun.com/label/sc/de-3-100244) [Linux](https://developer.aliyun.com/label/sc/de-3-100077)

[试试Linux设备命令行运维工具——Wowkey](https://developer.aliyun.com/article/1659797)

WowKey 是一款专为 Linux 设备设计的命令行运维工具，提供自动化、批量化、标准化、简单化的运维解决方案。它简单易用、高效集成且无依赖，仅需 WIS 指令剧本文件、APT 账号密码文件和 wowkey 命令即可操作。通过分离鉴权内容与执行内容，WowKey 让运维人员专注于决策，摆脱繁琐的交互与执行细节工作，大幅提升运维效率与质量。无论是健康检查、数据采集还是配置更新，WowKey 都能助您轻松应对大规模设备运维挑战。立即从官方资源了解更多信息：https://atsight.top/training。

[1239147665459462](https://developer.aliyun.com/profile/tormpzfmxjtle)

54166

[1239147665459462](https://developer.aliyun.com/profile/tormpzfmxjtle)

\|

[数据采集](https://developer.aliyun.com/label/sc/de-3-100053) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [安全](https://developer.aliyun.com/label/sc/de-3-100244)

[Linux设备命令行运维工具WowKey问答](https://developer.aliyun.com/article/1659802)

WowKey 是一款用于 Linux 设备运维的工具，可通过命令行手动或自动执行指令剧本，实现批量、标准化操作，如健康检查、数据采集、配置更新等。它简单易用，只需编写 WIS 指令剧本和 APT 帐号密码表文件，学习成本极低。支持不同流派的 Linux 系统，如 RHEL、Debian、SUSE 等，只要使用通用 Shell 命令即可通吃Linux设备。

[1239147665459462](https://developer.aliyun.com/profile/tormpzfmxjtle)

34715

[郑小健](https://developer.aliyun.com/profile/t6pqrjtpbleqs)

\|

[存储](https://developer.aliyun.com/label/sc/de-3-100262) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [安全](https://developer.aliyun.com/label/sc/de-3-100244)

[深入解析操作系统控制台：阿里云Alibaba Cloud Linux（Alinux）的运维利器](https://developer.aliyun.com/article/1649416)

本文将详细介绍阿里云的Alibaba Cloud Linux操作系统控制台的功能和优势。

[郑小健](https://developer.aliyun.com/profile/t6pqrjtpbleqs)

51867

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [监控](https://developer.aliyun.com/label/sc/de-3-100072) [网络协议](https://developer.aliyun.com/label/sc/de-3-100112)

[Linux运维工程师必知：如何在 Linux 中使用网络命令netstat？](https://developer.aliyun.com/article/1631925)

【10月更文挑战第21天】

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

95412

[![Linux运维工程师必知：如何在 Linux 中使用网络命令netstat？](https://ucc.alicdn.com/z3pojg2spmpe4/developer-article1631925/20241031/4f6f37a69cd24504b8714d7324791eb4.png?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/1631925)

[栈江湖](https://developer.aliyun.com/profile/qcazreabeg5vk)

\|

[Prometheus](https://developer.aliyun.com/label/sc/de-3-100271) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [监控](https://developer.aliyun.com/label/sc/de-3-100072)

[Prometheus+Grafana+NodeExporter：构建出色的Linux监控解决方案，让你的运维更轻松](https://developer.aliyun.com/article/1646391)

本文介绍如何使用 Prometheus + Grafana + Node Exporter 搭建 Linux 主机监控系统。Prometheus 负责收集和存储指标数据，Grafana 用于可视化展示，Node Exporter 则采集主机的性能数据。通过 Docker 容器化部署，简化安装配置过程。完成安装后，配置 Prometheus 抓取节点数据，并在 Grafana 中添加数据源及导入仪表盘模板，实现对 Linux 主机的全面监控。整个过程简单易行，帮助运维人员轻松掌握系统状态。

[栈江湖](https://developer.aliyun.com/profile/qcazreabeg5vk)

238733

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

[aliyun5279339750-12408](https://developer.aliyun.com/profile/p3z7tuwaikleu)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [监控](https://developer.aliyun.com/label/sc/de-3-100072) [安全](https://developer.aliyun.com/label/sc/de-3-100244)

[盘点Linux服务器运维管理面板](https://developer.aliyun.com/article/1640202)

随着云计算和大数据技术的迅猛发展，Linux服务器在运维管理中扮演着越来越重要的角色。传统的Linux服务器管理方式已经无法满足现代企业的需求，因此，高效、安全、易用的运维管理面板应运而生。

[aliyun5279339750-12408](https://developer.aliyun.com/profile/p3z7tuwaikleu)

210333

[蓝易云](https://developer.aliyun.com/profile/3c4vysbj27yje)

\|

[缓存](https://developer.aliyun.com/label/sc/de-3-100261) [运维](https://developer.aliyun.com/label/sc/de-3-100073) [监控](https://developer.aliyun.com/label/sc/de-3-100072)

[【运维必备知识】Linux系统平均负载与top、uptime命令详解](https://developer.aliyun.com/article/1637061)

系统平均负载是衡量Linux服务器性能的关键指标之一。通过使用 \`top\`和 \`uptime\`命令，可以实时监控系统的负载情况，帮助运维人员及时发现并解决潜在问题。理解这些工具的输出和意义是确保系统稳定运行的基础。希望本文对Linux系统平均负载及相关命令的详细解析能帮助您更好地进行系统运维和性能优化。

[蓝易云](https://developer.aliyun.com/profile/3c4vysbj27yje)

132733

[Dylaniou](https://developer.aliyun.com/profile/zyajrazmomlru)

\|

[运维](https://developer.aliyun.com/label/sc/de-3-100073) [Java](https://developer.aliyun.com/label/sc/de-3-100001) [Linux](https://developer.aliyun.com/label/sc/de-3-100077)

[【运维基础知识】Linux服务器下手写启停Java程序脚本start.sh stop.sh及详细说明](https://developer.aliyun.com/article/1626568)

\### 启动Java程序脚本 \`start.sh\`
此脚本用于启动一个Java程序，设置JVM字符集为GBK，最大堆内存为3000M，并将程序的日志输出到\`output.log\`文件中，同时在后台运行。

\### 停止Java程序脚本 \`stop.sh\`
此脚本用于停止指定名称的服务（如\`QuoteServer\`），通过查找并终止该服务的Java进程，输出操作结果以确认是否成功。

[Dylaniou](https://developer.aliyun.com/profile/zyajrazmomlru)

124311

## 热门文章

## 最新文章

[1\\
\\
Linux使用ss命令查看socket状态\\
\\
4490](https://developer.aliyun.com/article/462955)
[2\\
\\
Linux命令sha1sum的详细解析\\
\\
10](https://developer.aliyun.com/article/1562877)
[3\\
\\
使用shell在Linux系统下下载cmip6文件出现报错：No ESG Credentials found in /Users/daniele/.esg/credentials.pem\\
\\
9](https://developer.aliyun.com/article/1120602)
[4\\
\\
让你提高效率的 Linux 技巧\\
\\
578](https://developer.aliyun.com/article/682661)
[5\\
\\
Linux的SCP使用笔记\\
\\
649](https://developer.aliyun.com/article/398837)
[6\\
\\
FMS在linux下安装时的问题处理\\
\\
3](https://developer.aliyun.com/article/565497)
[7\\
\\
Linux系统Logrotate服务介绍\\
\\
5](https://developer.aliyun.com/article/508748)
[8\\
\\
自制Linux重命名命令\\
\\
569](https://developer.aliyun.com/article/315837)
[9\\
\\
Linux备份策略(第二版)\\
\\
629](https://developer.aliyun.com/article/455368)
[10\\
\\
如何在 Linux 中清除缓冲区和缓存内存？\\
\\
13](https://developer.aliyun.com/article/1625027)

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

[Linux高级网络应用 - 网络管理与配置实战](https://edu.aliyun.com/course/314182)
[Linux Web服务器Nginx搭建与配置](https://edu.aliyun.com/course/314194)
[Linux服务器运维基本操作](https://edu.aliyun.com/course/314200)
[Linux企业运维实战 - 入门及常用命令](https://edu.aliyun.com/course/314053)
[Linux Shell 编程入门与实战](https://edu.aliyun.com/course/314056)
[Linux网络进阶 - TCP/IP协议及OSI七层模型](https://edu.aliyun.com/course/314065)

## 相关电子书

[更多](https://developer.aliyun.com/ebook/)

[Alibaba Cloud Linux 3 发布](https://developer.aliyun.com/ebook/6684)
[ECS系统指南之Linux系统诊断](https://developer.aliyun.com/ebook/7000)
[ECS运维指南 之 Linux系统诊断](https://developer.aliyun.com/ebook/448)

## 推荐镜像

[更多](https://developer.aliyun.com/mirror/)

[mxlinux-iso](https://developer.aliyun.com/mirror/mxlinux-iso)
[archlinuxarm](https://developer.aliyun.com/mirror/archlinuxarm)
[archlinuxcn](https://developer.aliyun.com/mirror/archlinuxcn)

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

4008013260 [售前咨询](https://smartservice.console.aliyun.com/pre-sale/chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F517729) [售后在线](https://smartservice.console.aliyun.com/service/robot-chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F517729)

### 其他服务

[我要建议](https://www.aliyun.com/connect/home) [我要投诉](https://www.aliyun.com/complaint)

![登录插画](https://img.alicdn.com/imgextra/i2/O1CN015QIT9m1FmmyUntYlQ_!!6000000000530-2-tps-320-200.png)

登录以查看您的控制台资源

管理云资源

状态一览

快捷访问

[快捷注册](https://account.aliyun.com/register/qr_register.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F517729) [登录阿里云](https://account.aliyun.com/login/login.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F517729)