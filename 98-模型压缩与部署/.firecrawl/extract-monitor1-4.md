# 常见运维面试题目 - Jeff.Chen的技术博客

URL: https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/

- [常见运维面试题目](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_1)
- [配管](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_2)
- [负载均衡](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_3)
- [Web服务](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#web)
- [存储](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_4)
- [变更与发布](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_5)
- [网络](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_6)
- [中间件](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_7)
- [数据库](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_8)
- [常用bash](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#bash)
- [操作系统](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_9)
- [个人学习](https://chenqing24.github.io/%E4%B8%9A%E5%8A%A1%E6%96%B9%E6%A1%88/monitor_system/ops_interview/#_10)

# 常见运维面试题目

## 配管

现有100台服务器，该如何管理？
\\* 设定跳板机，使用统一账号登录，便于安全
\\* 使用ansiable、puppet进行系统的统一调度与配置
\\* 建立服务器的系统、配置、应用的cmdb信息管理。便于查阅各种信息记录

## 负载均衡

LVS、Nginx、HAproxy有什么区别和选择？
\\* LVS： 是基于四层的转发，限端口
\\* HAproxy： 是基于四层和七层的转发
\\* Nginx： 是WEB服务器，缓存服务器，又是反向代理服务器，可以做七层的转发
\\* 大并发量的时候我们就要选择LVS；并发量没那么大选择HAproxy或者Nginx足已

Keepalived的工作原理？
\\* 一个虚拟路由器中，只有作为MASTER的VRRP路由器会一直发送VRRP通告
\\* 当MASTER不可用时(BACKUP收不到通告信息)，多台BACKUP中优先级最高的这台会被抢占为MASTER(<1s)，以保证服务的连续性
\\* VRRP包使用了加密协议进行加密。BACKUP不会发送通告信息

LVS三种模式？
\\* NAT模式：把客户端发来的数据包的IP头的目的地址，在负载均衡器上换成其中一台RS的IP地址，并发至此RS来处理；RS处理完后把数据交给负载均衡器,负载均衡器再把数据包原IP地址改为自己的IP
\\* 优点：只有负载均衡器需要一个合法的IP地址
\\* 缺点：扩展性有限，负载均衡器将成为整个系统的瓶颈（大量的数据包都交汇在负载均衡器）
\\* TUN隧道模式：把客户端发来的数据包，封装一个新的IP头标记(仅目的IP)发给RS；RS（内核必须支持IPTUNNEL协议）收到后,先把数据包的头解开,还原数据包,处理后,直接返回给客户端,不需要再经过LB
\\* 优点：能处理很巨大的请求量，LB不是瓶颈
\\* 缺点：RS节点需要合法IP，所有的服务器支持”IP Tunneling”协议
\\* DR直接路由模式：LB和RS都使用同一个IP对外服务，只有DR对ARP请求进行响应（所有RS对这个IP的ARP请求保持静默）；DR收到数据包后根据调度算法,找出对应的RS,把目的MAC地址改为RS的MAC，并将请求分发给这台RS；RS直接返回数据（IP一样）；LB和RS在同一台交换机下
\\* 优点：能处理很巨大的请求量，应答包通过单独的路由方法返回给客户端
\\* 缺点：LB的网卡必须与物理网卡在一个物理网段上

常用的Nginx模块?
\\* rewrite模块，实现重写
\\* access模块：来源控制
\\* ssl模块：安全加密
\\* ngx\_http\_gzip\_module：网络传输压缩模块
\\* ngx\_http\_proxy\_module 模块实现代理
\\* ngx\_http\_upstream\_module模块实现定义后端服务器列表

## Web服务

Tomcat8005、8009、8080三个端口的含义？
\\* 8005：关闭使用
\\* 8009：AJP端口，即容器使用，如Apache能通过AJP协议访问Tomcat的8009端口
\\* 8080：应用使用

CDN？
\\* 内容分发网络，通过在现有的Internet中增加一层新的网络架构，将网站的内容发布到最接近用户的网络边缘，使用户可就近取得所需的内容，提高用户访问网站的速度

## 存储

raid0 raid1 raid5 工作原理及特点？
\\* RAID 0，可以是一块盘和N个盘组合，优点读写快，缺点：没有冗余，一块坏了数据就全没有了
\\* RAID 1，只能2块盘，盘的大小可以不一样，以小的为准，100%的冗余，缺点：浪费资源
\\* RAID 5 ，3块盘，容量计算10\*（n-1）,损失一块盘，读性能一般，写不好
\\* 单台服务器：系统盘，RAID1
\\* 数据库服务器：主库：RAID10 从库 RAID5

## 变更与发布

灰度发布？
\\* 黑与白之间，能够平滑过渡的一种发布方式，如AB Test，可以保证整体系统的稳定，在初始灰度的时候就可以发现、调整问题，以保证其影响度

## 网络

DNS进行域名解析的过程？
\\* 本机的host文件-->本地设置的DNS服务器-->网络中找根服务器-->一级域名服务器.cn-->二级域名服务器.com.cn-->三级域名服务器.baidu.com.cn，正好有这个网站www.baidu.com，然后发给请求的服务器，保存一份之后，再发给客户端

## 中间件

RabbitMQ？

\\* 消息中间件是在消息的传息过程中保存消息的容器；目的是提供路由并保证消息的传递；如果发送消息时接收者不可用消息队列不会保留消息，直到可以成功地传递为止，有时限

## 数据库

mysql如何减少主从复制延迟？可能原因？
\\* 从库硬件比主库差
\\* 慢SQL语句过多
\\* 网络延迟
\\* 主库读写压力大，建议前端加Cache
\\* 多台slave来分摊读请，其中1台专用于备份

重置mysql root密码？
\\* 本机sh下：`mysqladmin –u root –p password "新密码"`
\\* mysql>环境中：`Update  mysql.user  set  password=password('新密码')  where  user='root';flush privileges;`或者`grant  all  on  *.*  to   root@'localhost' identified by '新密码'；`
\\* 忘记root密码时：
\\* 停止服务`service  mysqld  stop`
\\* 安全模式`mysql/bin/mysqld_safe  --skip-grant-table &`
\\* 用空密码的root用户登录`mysql -u root`，后同上

mysql数据备份?
\\* `mysql/bin/mysqldump`：支持热备，逻辑备份，速度不快，适合数据比较小场景
\\* LVM快照：只能进行泠备份
\\* tar包备份

## 常用bash

- 分析nginx访问日志，找出访问页面数量在前十位的ip: `cat access.log | awk '{print $1}' | uniq -c | sort -rn | head -10`
- 监听主机为192.168.1.1，tcp端口为80的数据，同时将输出结果保存输出到tcpdump.log: `tcpdump 'host 192.168.1.1 and port 80' > tcpdump.log`
- 本地80 端口的请求转发到8080 端口，当前主机IP 为192.168.2.1: `iptables -A PREROUTING -d 192.168.2.1 -p tcp -m tcp -dport 80 -j DNAT-to-destination 192.168.2.1:8080`
- 实时抓取并显示当前系统中tcp 80端口的网络数据: `tcpdump -nn tcp port 80`
- 查看http的并发请求数与其TCP连接状态: `netstat -n | awk '/^tcp/ {++S[$NF]} END {for(a in S) print a, S[a]}'`
- 命令取出 linux 中 eth0 的 IP: `ifconfig eth0|awk 'NR==2'|awk -F ":" '{print $2}'|awk '{print $1}'`
- 每天晚上 12 点，打包站点目录/var/www/html 备份到/data 目录下:

``cd /var/www/ && /bin/tar zcf /data/html-`date +%m-%d%H`.tar.gz html/``,使用crontab

## 操作系统

优化Linux系统？
\\* 不用root，添加普通用户，通过sudo授权
\\* 更改默认的远程连接SSH服务端口及禁止root用户远程
\\* 定时自动更新服务器时间
\\* 配置国内yum源
\\* 关闭selinux及iptables，但是工作场景如果有外网IP一定要打开
\\* 调整文件描述符的数量
\\* 精简开机启动服务（crond rsyslog network sshd）
\\* 内核参数优化（/etc/sysctl.conf）
\\* 更改字符集，支持中文，但建议还是用英文字符集，防止乱码
\\* 锁定关键系统文件
\\* 清空/etc/issue，去除系统及内核版本登录前的屏幕显示

## 个人学习