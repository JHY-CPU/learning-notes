# Linux运维工程师面试题全面汇总（2023） - 阿里云开发者社区

URL: https://developer.aliyun.com/article/1363622

[开发者社区](https://developer.aliyun.com/) [开发与运维](https://developer.aliyun.com/group/othertech/) [文章](https://developer.aliyun.com/group/othertech/article/) 正文

# Linux运维工程师面试题全面汇总（2023）

2023-10-312014发布于天津

版权
举报

版权声明：

本文内容由阿里云实名注册用户自发贡献，版权归原作者所有，阿里云开发者社区不拥有其著作权，亦不承担相应法律责任。具体规则请查看《
[阿里云开发者社区用户服务协议](https://developer.aliyun.com/article/768092)》和
《 [阿里云开发者社区知识产权保护指引](https://developer.aliyun.com/article/768093)》。如果您发现本社区中有涉嫌抄袭的内容，填写
[侵权投诉表单](https://yida.alibaba-inc.com/o/right) 进行举报，一经查实，本社区将立刻删除涉嫌侵权内容。


本文涉及的产品

RDS DuckDB + QuickBI 企业套餐，8核32GB + QuickBI 专业版

RDS MySQL DuckDB 分析主实例，基础系列 4核8GB

RDS MySQL DuckDB 分析主实例，集群系列 4核8GB

**简介：** Linux运维工程师面试题全面汇总（2023）

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_c892e7e88b0644c1b5180ec24024f129.jpeg?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

* * *

### **一、linux**

#### **1.linux系统启动流程**

- 第一步：开机自检，加载BIOS
- 第二步：读取ＭＢＲ
- 第三步：Boot Loader　grub引导菜单
- 第四步：加载kernel内核
- 第五步：init进程依据inittab文件夹来设定运行级别
- 第六步：init进程执行rc.sysinit
- 第七步：启动内核模块
- 第八步：执行不同运行级别的脚本程序
- 第九步：执行/etc/rc.d/rc.lo

#### **2.linux文件类型**

|     |     |
| --- | --- |
| 文件属性 | 文件类型 |
| - | 常规文件，即file |
| d | 目录文件 |
| b | block device 即块设备文件，如硬盘;支持以block为单位进行随机访问 |
| c | character device 即字符设备文件，如键盘支持以character为单位进行线性访问 |
| l | symbolic link 即符号链接文件，又称软链接文件 |
| p | pipe 即命名管道文件 |
| s | socket 即套接字文件，用于实现两个进程进行通信 |

#### **3.centos6和7怎么将源码安装的程序添加到开机自启动？**

- 通用方法：编辑/etc/rc.d/rc.local文件，在文件末尾添加启动服务命令
- centos6

①进入到/etc/rc.d/init.d目录下；

②新建一个服务启动脚本，脚本中指定chkconfig参数；

③添加执行权限；

④执行chkconfig --add 添加服务自启动；
- centos7

①进入到/usr/lib/systemd/system目录下；

②新建自定义服务文件，文件中包含\[Unit\] \[Service\] \[Install\]相关配置，然后添加下执行权限；

③执行systemctl enable 服务名称；

#### **4.简述lvm，如何给使用lvm的/分区扩容？**

- 功能：可以对磁盘进行动态管理。动态按需调整大小
- 概念：

①PV - 物理卷：物理卷在逻辑卷管理中处于最底层，它可以是实际物理硬盘上的分区，也可以是整个物理硬盘，也可以是raid设备。

②VG - 卷组：卷组建立在物理卷之上，一个卷组中至少要包括一个物理卷，在卷组建立之后可动态添加物理卷到卷组中。一个逻辑卷管理系统工程中可以只有一个卷组，也可以拥有多个卷组。

③LV - 逻辑卷：逻辑卷建立在卷组之上，卷组中的未分配空间可以用于建立新的逻辑卷，逻辑卷建立后可以动态地扩展和缩小空间。系统中的多个逻辑卷可以属于同一个卷组，也可以属于不同的多个卷组。

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_b1c52d5a7f9043b49ef12223c1e64e3d.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

- 给/分区扩容步骤：

①添加磁盘

②使用fdisk命令对新增加的磁盘进行分区

③分区完成后修改分区类型为lvm

④使用pvcreate创建物理卷

⑤使用vgextend命令将新增加的分区加入到根目录分区中

⑥使用lvextend命令进行扩容

⑦使用xfs\_growfs调整卷分区大小

#### **5.为何du和df统计结果不一致？**

- 用户删除了大量的文件被删除后，在文件系统目录中已经不可见了，所以du就不会再统计它。
- 然而如果此时还有运行的进程持有这个已经被删除的文件句柄，那么这个文件就不会真正在磁盘中被删除，分区超级块中的信息也就不会更改，df仍会统计这个被删除的文件。
- 可通过 lsof命令查询处于deleted状态的文件，被删除的文件在系统中被标记为deleted。如果系统有大量deleted状态的文件，会导致du和df统计结果不一致。

#### **6.如何升级内核？**

- 方法一

```
# 添加第三方yum源进行下载安装。
Centos 6 YUM源：http://www.elrepo.org/elrepo-release-6-6.el6.elrepo.noarch.rpm
Centos 7 YUM源：http://www.elrepo.org/elrepo-release-7.0-2.el7.elrepo.noarch.rpm
# 先导入elrepo的key，然后安装elrepo的yum源：
rpm -import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
rpm -Uvh http://www.elrepo.org/elrepo-release-7.0-2.el7.elrepo.noarch.rpm
# 查看可用的内核相关包
yum --disablerepo="*" --enablerepo="elrepo-kernel" list available
yum -y --enablerepo=elrepo-kernel install
```

AI 代码解读

- 方法二

```
# 通过下载kernel image的rpm包进行安装。
官方 Centos 6: http://elrepo.org/linux/kernel/el6/x86_64/RPMS/
官方 Centos 7: http://elrepo.org/linux/kernel/el7/x86_64/RPMS/
# 获取下载链接进行下载安装即可
wget https://elrepo.org/linux/kernel/el7/x86_64/RPMS/kernel-lt-4.4.185-1.el7.elrepo.x86_64.rpm
rpm -ivh kernel-lt-4.4.185-1.el7.elrepo.x86_64.rp
# 查看默认启动顺序
[root@localhost ~]# awk -F\' '$1=="menuentry " {print $2}' /etc/grub2.cfg
CentOS Linux (5.2.2-1.el7.elrepo.x86_64) 7 (Core)
CentOS Linux (4.4.182-1.el7.elrepo.x86_64) 7 (Core)
CentOS Linux (3.10.0-957.21.3.el7.x86_64) 7 (Core)
CentOS Linux (3.10.0-957.10.1.el7.x86_64) 7 (Core)
CentOS Linux (3.10.0-327.el7.x86_64) 7 (Core)
CentOS Linux (0-rescue-e34fb4f1527b4f2d9fc75b77c016b6e7) 7 (Core)
由上面可以看出新内核(4.12.4)目前位置在0，原来的内核(3.10.0)目前位置在1
# 设置默认启动
[root@localhost ~]# grub2-set-default 0　　// 0代表当前第一行，也就是4.12.4版本
# 重启验证
```

AI 代码解读

**7.nginx日志访问量前十的ip怎么统计？**

```
awk '{array[$1]++}END{for (ip in array)print ip,array[ip]}' access.log |sort -k2 -rn|head
```

AI 代码解读

**8.如何删除/var/log/下.log结尾的30天前的日志？**

```
find /var/log/ -type f -name .*.log -mtime 30|xargs rm -f
```

AI 代码解读

**9.ansible有哪些模块？功能是什么？**

|     |     |
| --- | --- |
| 模块 | 功能 |
| copy | 拷贝文件到被控端 |
| cron | 定时任务 |
| fetch | 拷贝被控端文件到本地 |
| file | 文件模块 |
| group | 用户组模块 |
| user | 用户模块 |
| hostname | 主机名模块 |
| script | 脚本模块 |
| service | 服务启动模块 |
| command | 远程执行命令模块 |
| shell | 远程执行命令模块，command高级用法 |
| yum | 安装包组模块 |
| setup | 查看主机系统信息 |

**10.nginx为什么比apache快？**

- nginx采用epoll模型
- apache采用select模型

**11\. 四层负载和七层负载区别是什么？**

- 四层基于IP+端口进行转发
- 七层就是基于URL等应用层信息的负载均衡

**12\. lvs有哪些工作模式？哪个性能高？**

- dr：直接路由模式，请求由 LVS 接受，由真实提供服务的服务器直接返回给用户，返回的时候不经过 LVS。（ **性能最高**）
- tun：隧道模式，客户端将访问vip报文发送给LVS服务器。LVS服务器将请求报文重新封装，发送给后端真实服务器。后端真实服务器将请求报文解封，在确认自身有vip之后进行请求处理。后端真实服务器在处理完数据请求后，直接响应客户端。
- nat：网络报的进出都要经过 LVS 的处理。LVS 需要作为 RS 的网关。当包到达 LVS 时，LVS 做目标地址转换（DNAT），将目标 IP 改为 RS 的 IP。RS 接收到包以后，仿佛是客户端直接发给它的一样。RS 处理完，返回响应时，源 IP 是 RS IP，目标 IP 是客户端的 IP。这时 RS 的包通过网关（LVS）中转，LVS 会做源地址转换（SNAT），将包的源地址改为 VIP，这样，这个包对客户端看起来就仿佛是 LVS 直接返回给它的。客户端无法感知到后端 RS 的存在。
- fullnat模式：fullnat模式和nat模式相似，但是与nat不同的是nat模式只做了两次地址转换，fullnat模式却做了四次。

**13\. tomcat各个目录含义，如何修改端口，如何修改内存数？**

- bin 存放tomcat命令
- conf 存放tomcat配置文件
- lib 存放tomcat运行需要加载的jar包
- log 存在Tomcat运行产生的日志
- temp 运行过程中产生的临时文件
- webapps 站点目录
- work 存放tomcat运行时的编译后的文件
- conf/server.xml 修改端口号
- bin/catalina.sh 修改jvm内存

**14\. nginx反向代理时，如何使后端获取真正的访问来源ip？**

```
# 在location配置段添加以下内容：
proxy_set_header Host $http_host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;
```

AI 代码解读

**15\. nginx负载均衡算法有哪些？**

- rr 轮训
- weight 加权轮训
- ip\_hash 静态调度算法
- fair 动态调度算法
- url\_hash url哈希
- leat\_conn 最小连接数

**16\. 如何进行压力测试？**

例如：模拟10个用户，对百度首页发起总共100次请求。

```
# 测试命令：
ab -n 100 -c 10 https://www.baidu.com/index.htm
```

AI 代码解读

**17\. curl命令如何发送https请求？如何查看response头信息？如何发送get和post表单信息？**

- 发送https请求：

```
curl --tlsv1 'https://www.bitstamp.net/api/v2/transactions/btcusd/'
```

AI 代码解读

- response头信息 ：curl -I
- get：curl 请求地址?key1=value1&key2=value2&key3=value3
- post：curl -d “key1=value1&key2=value2&key3=value3”

### **二、mysql**

#### **1\. 索引的为什么使查询加快？有啥缺点？**

默认的方式是根据搜索条件进行全表扫描，遇到匹配条件的就加入搜索结果集合。如果我们对某一字段增加索引，查询时就会先去索引列表中一次定位到特定值的行数，大大减少遍历匹配的行数，所以能明显增加查询的速度

缺点：

- 创建索引和维护索引要耗费时间，这种时间随着数据量的增加而增加
- 索引需要占物理空间，除了数据表占用数据空间之外，每一个索引还要占用一定的物理空间，如果需要建立聚簇索引，那么需要占用的空间会更大
- 以表中的数据进行增、删、改的时候，索引也要动态的维护，这就降低了整数的维护速度

#### **2\. sql语句左外连接 右外连接 内连接 全连接区别**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_b66470acbf624b43b349a5d394ef4c44.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

### **3\. mysql数据备份方式，如何恢复？你们的备份策略是什么？**

- 物理完全备份

```
备份所有数据库文件：/var/lib/mysql/*
备份所有binlog文件:  /var/lib/mysql/mysql-bin.*
备份选项文件: /etc/my.cnf
```

AI 代码解读

- mysqldump逻辑备份

```
mysqldump -uroot -p --all-databases > /backup/mysqldump/all.db
```

AI 代码解读

- 物理备份恢复

```
#先把原来的数据目录改名
mv /var/lib/mysql /var/lib/mysql.old
cp -a /backups/mysql /var/lib
```

AI 代码解读

- 逻辑备份数据恢复

```
mysql > use db_name
mysql > source /backup/mysqldump/db_name.db
```

AI 代码解读

### **4\. 如何配置数据库主从同步，实际工作中是否遇到数据不一致问题？如何解决？**

为每个服务器配置唯一值的server-id

- 主库

```
开启binlog日志
创建主从复制用户
查看master的状态
```

AI 代码解读

- 从库

```
change master to设置主库信息
start slave开始复制
```

AI 代码解读

### **5\. mysql约束有哪些？**

- 非空约束
- 唯一约束
- 主键约束
- 外键约束

### **6\. 二进制日志（binlog）用途？**

BINLOG记录数据库的变更过程。例如创建数据库、建表、修改表等DDL操作、以及数据表的相关DML操作，这些操作会导致数据库产生变化，开启binlog以后导致数据库产生变化的操作会按照时间顺序以“事件”的形式记录到binlog二进制文件中。

### **7\. mysql数据引擎有哪些？**

- 常用的 myisam、innodb
- 区别：

（1）InnoDB 支持事务，MyISAM 不支持，这一点是非常之重要。事务是一种高级的处理方式，如在一些列增删改中只要哪个出错还可以回滚还原，而 MyISAM就不可以了；

（2）MyISAM 适合查询以及插入为主的应用，InnoDB 适合频繁修改以及涉及到安全性较高的应用；

（3）InnoDB 支持外键，MyISAM 不支持；

（4）MyISAM 是默认引擎，InnoDB 需要指定；

（5）InnoDB 不支持 FULLTEXT 类型的索引；

（6）InnoDB 中不保存表的行数，如 select count( _) from table 时，InnoDB；需要扫描一遍整个表来计算有多少行，但是 MyISAM 只要简单的读出保存好的行数即可。注意的是，当 count()_ 语句包含 where 条件时 MyISAM 也需要扫描整个表；

（7）对于自增长的字段，InnoDB 中必须包含只有该字段的索引，但是在 MyISAM表中可以和其他字段一起建立联合索引；

（8）清空整个表时，InnoDB 是一行一行的删除，效率非常慢。MyISAM 则会重建表；

（9）InnoDB 支持行锁（某些情况下还是锁整表，如 update table set a=1 where user like ‘%lee%’

### **8\. 如何查询mysql数据库存放路径？**

- myisam

```
.frm文件：保护表的定义
.myd：保存表的数据
.myi：表的索引文件
```

AI 代码解读

### **9\. mysql数据库文件后缀名有哪些？用途什么？**

- myisam

```
.frm文件：保护表的定义
.myd：保存表的数据
.myi：表的索引文件
```

AI 代码解读

- innodb

```
.frm：保存表的定义
.ibd：表空间
```

AI 代码解读

### **10\. 如何修改数据库用户的密码？**

- mysql8之前

```
set password for 用户名@localhost = password('新密码');
mysqladmin -u用户名 -p旧密码 password 新密码
update user set password=password('123') where user='root' and host='localhost';
```

AI 代码解读

- mysql8之后

```
# mysql8初始对密码要求高，简单的字符串不让改。先改成:MyNewPass@123;
alter user 'root'@'localhost' identified by 'MyNewPass@123';
# 降低密码难度
set global validate_password.policy=0;
set global validate_password.length=4;
# 修改成简易密码
alter user 'root'@'localhost'IDENTIFIED BY '1111';
```

AI 代码解读

### **11\. 如何修改用户权限？如何查看？**

- 授权：

```
grant all on *.* to user@'%' identified by 'passwd'
```

AI 代码解读

- 查看权限

```
show grants for user@'%';
```

AI 代码解读

### **三、nosql**

### **1\. redis数据持久化有哪些方式？**

- rdb
- aof

### **2\. redis集群方案有哪些？**

- 官方cluster方案
- twemproxy代理方案
- 哨兵模式
- codis

客户端分片

### **3\. redis如何进行数据备份与恢复？**

- 备份

```
redis 127.0.0.1:6379> SAVE
创建 redis 备份文件也可以使用命令 BGSAVE，该命令在后台执行。
```

AI 代码解读

- 还原

```
只需将备份文件 (dump.rdb) 移动到 redis 安装目录并启动服务即可
redis 127.0.0.1:6379> CONFIG GET dir
```

AI 代码解读

### **4\. MongoDB如何进行数据备份？**

```
mongoexport / mongoimport
mongodump  / mongorestore
```

AI 代码解读

### **5\. kafka为何比redis rabbitmq快？**

> [RabbitMQ，ZeroMQ，Kafka 是一个层级的东西吗？相互之间有哪些优缺点？ - 知乎](https://www.zhihu.com/question/22480085)

### **四、docker**

#### **1\. dockerfile有哪些关键字？用途是什么？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_585a296674254cae8bae2ca578be91d3.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### 2. **如何减小dockerfile生成镜像体积？**

- 尽量选取满足需求但较小的基础系统镜像，例如大部分时候可以选择debian:wheezy或debian:jessie镜像，仅有不足百兆大小；
- 清理编译生成文件、安装包的缓存等临时文件；
- 安装各个软件时候要指定准确的版本号，并避免引入不需要的依赖；
- 从安全角度考虑，应用要尽量使用系统的库和依赖；
- 如果安装应用时候需要配置一些特殊的环境变量，在安装后要还原不需要保持的变量值；

#### **3\. dockerfile中CMD与ENTRYPOINT区别是什么？**

- CMD 和 ENTRYPOINT 指令都是用来指定容器启动时运行的命令。
- 指定 ENTRYPOINT 指令为 exec 模式时，CMD指定的参数会作为参数添加到 ENTRYPOINT 指定命令的参数列表中。

#### **4\. dockerfile中COPY和ADD区别是什么？**

- COPY指令和ADD指令都可以将主机上的资源复制或加入到容器镜像中
- 区别是ADD可以从 远程URL中的资源不会被解压缩。
- 如果是本地的压缩包ADD进去会被解压缩

#### **5\. docker的cs架构组件有哪些？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_281c85ee7d224312923c24b3212f2afc.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **6\. docker网络类型有哪些？**

- host模式
- container模式
- none模式
- bridge模式

#### **7\. 如何配置docker远程访问？**

- vim /lib/systemd/system/docker.service
- 在ExecStart=后添加配置，注意，需要先空格后，再输入 -H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock

#### **8\. docker核心namespace CGroups 联合文件系统功能是什么？**

- namespace：资源隔离
- cgroup：资源控制
- 联合文件系统：支持对文件系统的修改作为一次提交来一层层的叠加，同时可以将不同目录挂载到同一个虚拟文件系统下

#### **9\. 命令相关：导入导出镜像，进入容器，设置重启容器策略，查看镜像环境变量，查看容器占用资源**

- 导入镜像 docker load -i xx.tar
- 导出镜像docker save -o xx.tar image\_name
- 进入容器docker exec -it 容器命令 /bin/bash
- 设置容器重启策略启动时 --restart选项
- 查看容器环境变量 docker exec {containerID} env
- 查看容器资源占用docker stats test2

#### **10\. 构建镜像有哪些方式？**

- dockerfile
- 容器提交为镜像

#### **11\. docker和vmware虚拟化区别？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_12f3fe9abefb4486a31facd8f4ee5954.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_ee7116dd55b0419b9eee36bd372ce0c6.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

### **五、kubernetes**

#### **1\. k8s的集群组件有哪些？功能是什么？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_3118300ddfc241a281389d94320ca6a6.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **2\. kubectl命令相关：如何修改副本数，如何滚动更新和回滚，如何查看pod的详细信息，如何进入pod交互？**

- 修改副本数

```
kubectl scale deployment redis --replicas=3
```

AI 代码解读

- 活动更新

```
kubectl set image deployments myapp-deploy myapp=myapp:v2
```

AI 代码解读

- 回滚

```
kubectl rollout undo deployments myapp-deploy
```

AI 代码解读

- 查看pod详细信息

```
kubectl describe pods/<pod-name>
```

AI 代码解读

- 进入pod交互

```
kubectl exec -it <pod-name> -c <container-name> bash
```

AI 代码解读

#### **3\. etcd数据如何备份？**

- etcdctl --endpoints=“ [https://192.168.32.129:2379](https://link.zhihu.com/?target=https%3A//192.168.32.129%3A2379), [https://192.168.32.130:2379](https://link.zhihu.com/?target=https%3A//192.168.32.130%3A2379),192.168.32.128:2379” --cacert=/etc/kubernetes/cert/ca.pem --key=/etc/etcd/cert/etcd-key.pem --cert=/etc/etcd/cert/etcd.pem snapshot save snashot1.db
- Snapshot saved at snashot1.db

#### **4\. k8s控制器有哪些？**

- 副本集（ReplicaSet）
- 部署（Deployment）
- 状态集（StatefulSet）
- Daemon集（DaemonSet）
- 一次任务（Job）
- 计划任务（CronJob）
- 有状态集（StatefulSet）

#### **5\. 哪些是集群级别的资源？**

- Namespace
- Node
- Role
- ClusterRole
- RoleBinding
- ClusterRoleBinding

#### **6\. pod状态有哪些？**

- Pending 等待中
- Running 运行中
- Succeeded 正常终止
- Failed 异常停止
- Unkonwn 未知状态

#### **7\. pod创建过程是什么？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_b3be03926ede4fc3a7303ce6c5a7e4b7.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **8\. pod重启策略有哪些？**

Pod的重启策略有3种，默认值为Always。

- Always ：容器失效时，kubelet 自动重启该容器；
- OnFailure ：容器终止运行且退出码不为0时重启；
- Never ：不论状态为何， kubelet 都不重启该容器

#### **9\. 资源探针有哪些？**

- ExecAction：在容器中执行一个命令，并根据其返回的状态码进行诊断的操作称为Exec探测，状态码为0表示成功，否则即为不健康状态。
- TCPSocketAction：通过与容器的某TCP端口尝试建立连接进行诊断，端口能够成功打开即为正常，否则为不健康状态。
- HTTPGetAction：通过向容器IP地址的某指定端口的指定path发起HTTP GET请求进行诊断，响应码为2xx或3xx时即为成功，否则为失败

#### **10\. requests和limits用途是什么？**

- “requests”属性定义其请求的确保可用值，即容器运行可能用不到这些额度的资源，但用到时必须要确保有如此多的资源可用
- ”limits”属性则用于限制资源可用的最大值，即硬限制

#### **11\. kubeconfig文件包含什么内容，用途是什么？**

包含集群参数（CA证书、API Server地址），客户端参数（上面生成的证书和私钥），集群context 信息（集群名称、用户名）。

#### **12\. RBAC中role和clusterrole区别，rolebinding和 clusterrolebinding区别？**

- Role 可以定义在一个 namespace 中，如果想要跨 namespace则可以创建ClusterRole，ClusterRole 具有与 Role相同的权限角色控制能力，不同的是 ClusterRole 是集群级别的
- RoleBinding 适用于某个命名空间内授权，而 ClusterRoleBinding 适用于集群范围内的授权

#### **13\. ipvs为啥比iptables效率高？**

IPVS模式与iptables同样基于Netfilter，但是ipvs采用的hash表，iptables采用一条条的规则列表。iptables又是为了防火墙设计的，集群数量越多iptables规则就越多，而iptables规则是从上到下匹配，所以效率就越是低下。因此当service数量达到一定规模时，hash查表的速度优势就会显现出来，从而提高service的服务性能

#### **14\. sc pv pvc用途，容器挂载存储整个流程是什么？**

- PVC：Pod 想要使用的持久化存储的属性，比如存储的大小、读写权限等。
- PV ：具体的 Volume 的属性，比如 Volume 的类型、挂载目录、远程存储服务器地址等。
- StorageClass：充当 PV 的模板。并且，只有同属于一个 StorageClass 的 PV 和 PVC，才可以绑定在一起。当然，StorageClass 的另一个重要作用，是指定 PV 的 Provisioner（存储插件）。这时候，如果你的存储插件支持 Dynamic Provisioning 的话，Kubernetes 就可以自动为你创建 PV 了。

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_0ea67bdf1ca041e88d68bd47dc9a1b82.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **15\. nginx ingress的原理本质是什么？**

- ngress controller通过和kubernetes api交互，动态的去感知集群中ingress规则变化，
- 然后读取它，按照自定义的规则，规则就是写明了哪个域名对应哪个service，生成一段nginx配置，
- 再写到nginx-ingress-controller的pod里，这个Ingress

controller的pod里运行着一个Nginx服务，控制器会把生成的nginx配置写入/etc/nginx.conf文件中，
- 然后reload一下使配置生效。以此达到域名分配置和动态更新的问题。

#### **16\. 描述不同node上的Pod之间的通信流程**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_3296378aa5614c79ad4a95dc75d3fbf3.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **17\. k8s集群节点需要关机维护，需要怎么操作**

- 进行pod驱逐：kubelet drain <node\_name>
- 检查node上是否无pod运行，切被驱逐的pod已经在其他节点运行正常
- 关机维护
- 开机启动相关服务（注意启动顺序）
- 解除node节点不可调度：kubectl uncordon node
- 创建测试pod，并使用节点标签测试节点可以被正常调度

#### **18\. canal和flannel区别**

- Flannel（简单、使用居多）：基于Vxlan技术（叠加网络+二层隧道），不支持网络策略
- Calico（较复杂，使用率少于Flannel）：也可以支持隧道网络，但是是三层隧道（IPIP），支持网络策略
- Calico项目既能够独立地为Kubernetes集群提供网络解决方案和网络策略，也能与flannel结合在一起，由flannel提供网络解决方案，而Calico此时仅用于提供网络策略。

### **六、prometheus**

#### **1\. prometheus对比zabbix有哪些优势？**

> [https://blog.csdn.net/wangyiyungw/article/details/85774969](https://link.zhihu.com/?target=https%3A//blog.csdn.net/wangyiyungw/article/details/85774969)\*\*

#### **2\. prometheus组件有哪些，功能是什么？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_e14d7598da9c4575bcf2d1c8b6efdd68.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **3\. 指标类型有哪些？**

- Counter（计数器）
- Guage（仪表盘）
- Histogram（直方图）
- Summary（摘要）

#### **4\. 在应对上千节点监控时，如何保障性能**

- 降低采集频率
- 缩小历史数据保存天数，
- 使用集群联邦和远程存储

#### **5\. 简述从添加节点监控到grafana成图的整个流程**

- 被监控节点安装exporter
- prometheus服务端添加监控项
- 查看prometheus web界面——status——targets
- grafana创建图表

#### **6\. 在工作中用到了哪些exporter**

- node-exporter监控linux主机
- cAdvisor监控容器
- MySQLD Exporter监控mysql
- Blackbox Exporter网络探测
- Pushgateway采集自定义指标监控
- process exporter进程监控

### **七、ELK**

#### **1\. Elasticsearch的数据如何备份与恢复？**

> [https://www.cnblogs.com/tcy1/p/13492361.html](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/tcy1/p/13492361.html) [https://blog.csdn.net/moxiaomomo/article/details/78401400?locationNum=8&fps=1](https://link.zhihu.com/?target=https%3A//blog.csdn.net/moxiaomomo/article/details/78401400%3FlocationNum%3D8%26fps%3D1)

#### **2\. 你们项目中使用的logstash过滤器插件是什么？实现哪些功能？**

- date 日期解析
- grok 正则匹配解析
- overwrite 写某个字段
- dissect 分隔符解析
- mutate 对字段做处理
- json 解析
- geoip 地理位置解析
- ruby 修改logstash event

#### **3\. 是否用到了filebeat的内置module？用了哪些？**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_c0592ad76ad74bbcaab5c6ed47d0a994.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **4\. elasticsearch分片副本是什么？你们配置的参数是多少？**

> [https://juejin.cn/post/6844903862088777736](https://link.zhihu.com/?target=https%3A//juejin.cn/post/6844903862088777736)

### **八、运维开发**

**1\. 备份系统中所有容器镜像**

```
#备份镜像列表
```

AI 代码解读

#### **2\. 编写脚本，定时备份某个库，然后压缩，发送异机**

- 公共部分定义函数，如获取时间戳，配置报警接口
- 多使用if判断是否存在异常并处理，如数据库大，检测任务是否完成。检测生成文件大小是否是空文件

#### **3\. 批量获取所有主机的系统信息**

- 使用python的paramiko库，ssh登陆主机执行查询操作
- 使用shell脚本批量ssh登陆主机并执行命令
- 使用ansible的setup模块获取主机信息
- prometheus的node\_exporter收集主机资源信息

#### **4\. django的mtv模式流程**

![](https://ucc.alicdn.com/pic/developer-ecology/6h2qt5rxnv6ci_65359728c312451684ccc49c612179e9.png?x-oss-process=image%2Fresize%2Cw_1400%2Cm_lfit%2Fformat%2Cwebp)

#### **5\. python如何导出、导入环境依赖包**

- 导出环境

```
pip freeze >> requirements.txt
```

AI 代码解读

- 导入环境

```
pip install -r requirement.txt
```

AI 代码解读

#### **6\. python创建，进入，退出，查看虚拟环境**

- 安装软件包

```
pip3 install virtualenv
```

AI 代码解读

- 检测安装是否成功

```
virtualenv --version
```

AI 代码解读

- 创建虚拟环境
- cd到要创建虚拟环境的目录

```
cd github/test/venv/
```

AI 代码解读

- 创建虚拟环境

```
virtualenv test
```

AI 代码解读

- 激活虚拟环境

```
source test/bin/activate(activate路径)
```

AI 代码解读

- 退出虚拟环境

```
deactivate
```

AI 代码解读

#### **7\. flask和django区别，应用场景**

- Django功能大而全，Flask只包含基本的配置 Django的一站式解决的思路，能让开发者不用在开发之前就在选择应用的基础设施上花费大量时间。Django有模板，表单，路由，认证，基本的数据库管理等等内建功能。与之相反，Flask只是一个内核，默认依赖于两个外部库：Jinja2 模板引擎和 Werkzeug WSGI 工具集，其他很多功能都是以扩展的形式进行嵌入使用。
- Flask 比 Django 更灵活 用Flask来构建应用之前，选择组件的时候会给开发者带来更多的灵活性 ，可能有的应用场景不适合使用一个标准的ORM(Object-Relational Mapping 对象关联映射)，或者需要与不同的工作流和模板系统交互

#### **8\. 列举常用的git命令**

- $ git init
- $ git config
- $ git add
- $ git commit
- $ git branch
- $ git checkout
- $ git tag
- $ git push
- $ git status
- $ git log

#### **9\. git gitlab jenkins的CICD流程如何配置**

- 开发者git提交代码至gitlab仓库
- jenkins从gitlab拉取代码，触发镜像构建
- 镜像上传至harbor私有仓库
- 镜像下载至执行机器
- 镜像运行

### **九、日常工作**

#### **1\. 在日常工作中遇到了什么棘手的问题，如何排查**

- redis弱口令导致中挖矿病毒，排查，优化
- k8s中开发的程序在用户上传文件时开启进程，未及时关闭，导致节点超出最大进程数

#### **2\. 日常故障处理流程**

- 查看报警内容，快速定位大致故障主机，服务，影响范围
- 告知运维经理故障，并开始排查
- 如果需要修改配置文件，重启服务器等操作，告知相关开发人员
- 完成故障处理

#### **3\. 修改线上业务配置文件流程**

- 先告知运维经理和业务相关开发人员
- 在测试环境测试，并备份之前的配置文件
- 测试无误后修改生产环境配置
- 观察生产环境是否正常，是否有报警
- 完成配置文件更改

#### **4\. 业务pv多少？集群规模多少？怎么保障业务高可用？**

### **十、开放性问题**

#### **1\. 你认为初级运维工程师和高级运维工程师的区别？**

#### **2\. 你认为未来运维发展方向?**

> **注：文章转自IT运维技术圈，如有侵权请联系删除。**

文章标签：

[容器服务Kubernetes版](https://developer.aliyun.com/label/article_de-product-3-csk)

[云解析DNS](https://developer.aliyun.com/label/article_de-product-3-dns)

[云数据库 Tair（兼容 Redis）](https://developer.aliyun.com/label/article_de-product-3-kvstore)

[云数据库 RDS MySQL 版](https://developer.aliyun.com/label/article_de-product-3-mysql)

[NAT网关](https://developer.aliyun.com/label/article_de-product-3-nat)

[可观测监控 Prometheus 版](https://developer.aliyun.com/label/article_de-product-3-prometheus)

[日志服务](https://developer.aliyun.com/label/article_de-product-3-sls)

[容器](https://developer.aliyun.com/label/article_de-3-100018)

[关系型数据库](https://developer.aliyun.com/label/article_de-3-100067)

[Perl](https://developer.aliyun.com/label/article_de-3-100007)

[Python](https://developer.aliyun.com/label/article_de-3-100008)

[应用服务中间件](https://developer.aliyun.com/label/article_de-3-100033)

关键词：

[运维工程师](https://www.aliyun.com/sswb/835893.html)

[Linux工程师](https://www.aliyun.com/sswb/563058.html)

[运维面试](https://www.aliyun.com/sswb/555883.html)

[运维工程师面试](https://www.aliyun.com/sswb/1367769.html)

[Linux面试](https://www.aliyun.com/sswb/258736.html)

相关实践学习

深入解析Docker容器化技术

Docker是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的Linux机器上，也可以实现虚拟化，容器是完全使用沙箱机制，相互之间不会有任何接口。Docker是世界领先的软件容器平台。开发人员利用Docker可以消除协作编码时“在我的机器上可正常工作”的问题。运维人员利用Docker可以在隔离容器中并行运行和管理应用，获得更好的计算密度。企业利用Docker可以构建敏捷的软件交付管道，以更快的速度、更高的安全性和可靠的信誉为Linux和Windows Server应用发布新功能。 在本套课程中，我们将全面的讲解Docker技术栈，从环境安装到容器、镜像操作以及生产环境如何部署开发的微服务应用。本课程由黑马程序员提供。 &nbsp; &nbsp; 相关的阿里云产品：容器服务 ACK 容器服务 Kubernetes 版（简称 ACK）提供高性能可伸缩的容器应用管理能力，支持企业级容器化应用的全生命周期管理。整合阿里云虚拟化、存储、网络和安全能力，打造云端最佳容器化应用运行环境。 了解产品详情: https://www.aliyun.com/product/kubernetes

[![](https://ucc.alicdn.com/avatar/0fc429d454bd4b198588fb8bf43c7ce6.jpg?x-oss-process=image/resize,h_150,m_lfit)](https://developer.aliyun.com/profile/6h2qt5rxnv6ci)

![](https://ucc.alicdn.com/pic/ucc-admin/88c34b916d704521b87d41daa9a77107.png?x-oss-process=image%2Fresize%2Ch_80%2Cm_lfit%2Fformat%2Cwebp)

[征服Bug](https://developer.aliyun.com/profile/6h2qt5rxnv6ci)

+关注

[210文章](https://developer.aliyun.com/profile/6h2qt5rxnv6ci/article_1) [2问答](https://developer.aliyun.com/profile/6h2qt5rxnv6ci/ask_1)

目录

1

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

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

\|

[域名解析](https://developer.aliyun.com/label/sc/de-3-100240) [网络协议](https://developer.aliyun.com/label/sc/de-3-100112) [安全](https://developer.aliyun.com/label/sc/de-3-100244)

[网络工程师需要掌握的10个Linux网络命令，收藏！](https://developer.aliyun.com/article/1634274)

【10月更文挑战第27天】

[wljslmz](https://developer.aliyun.com/profile/z3pojg2spmpe4)

92711

[![网络工程师需要掌握的10个Linux网络命令，收藏！](https://ucc.alicdn.com/z3pojg2spmpe4/developer-article1634274/20241106/247763b4b7fa44ebb946273b585b1fe4.jpeg?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/1634274)

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

[工程师阿龙](https://developer.aliyun.com/profile/zh2uunrcvveuk)

\|

[Ubuntu](https://developer.aliyun.com/label/sc/de-3-100079) [Linux](https://developer.aliyun.com/label/sc/de-3-100077) [Shell](https://developer.aliyun.com/label/sc/de-3-100011)

[这7个重要的Linux命令，每一位Linux工程师都必须盘它！](https://developer.aliyun.com/article/1588392)

这7个重要的Linux命令，每一位Linux工程师都必须盘它！

[工程师阿龙](https://developer.aliyun.com/profile/zh2uunrcvveuk)

13000

[良许Linux](https://developer.aliyun.com/profile/m5tbckv3o4hdi)

\|

[Ubuntu](https://developer.aliyun.com/label/sc/de-3-100079) [Linux](https://developer.aliyun.com/label/sc/de-3-100077)

[Linux工程师如何不被工作打扰，专心与女神约会？](https://developer.aliyun.com/article/885252)

作为 Linux 工程师，很多工作是在命令行下完成的。有时候我们执行一个命令，可能需要很长时间才能完成，比如 sudo apt-get update ，或者编译一个需要花费半小时的系统，如果我们啥也不干就干等着结果，那效率未免也太低了。

[良许Linux](https://developer.aliyun.com/profile/m5tbckv3o4hdi)

24700

[![Linux工程师如何不被工作打扰，专心与女神约会？](https://ucc.alicdn.com/pic/developer-ecology/0b9e6f4ba911416b874d8758226ed9be.png?x-oss-process=image/format,webp/resize,h_160,m_lfit)](https://developer.aliyun.com/article/885252)

## 热门文章

## 最新文章

[1\\
\\
清华裴丹分享AIOps落地路线图，看智能运维如何落地生根\\
\\
8841](https://developer.aliyun.com/article/272155)
[2\\
\\
实战：阿里巴巴 DevOps 转型后的运维平台建设\\
\\
7268](https://developer.aliyun.com/article/573219)
[3\\
\\
一文读懂智能化运维监控如何赋能IT可观察性\\
\\
194](https://developer.aliyun.com/article/880685)
[4\\
\\
运维工程师到底在作什么？从何学起，掌握哪些知识？\\
\\
630](https://developer.aliyun.com/article/629379)
[5\\
\\
如何改变运维在数据中心中的地位\\
\\
653](https://developer.aliyun.com/article/231571)
[6\\
\\
智能化运维：AI在故障预测与自愈系统中的应用\\
\\
8](https://developer.aliyun.com/article/1528974)
[7\\
\\
使用运维编排服务配置多台实例的免密登陆\\
\\
1](https://developer.aliyun.com/article/779618)
[8\\
\\
函数计算自动化运维实战3 -- 事件触发自动创建快照\\
\\
2](https://developer.aliyun.com/article/737564)
[9\\
\\
运维.Linux下执行定时任务（中：Cron的常用替代方案）\\
\\
8](https://developer.aliyun.com/article/1580964)
[10\\
\\
Linux运维 第二阶段 （三）软件安装\\
\\
700](https://developer.aliyun.com/article/509811)

[1\\
\\
面试性能测试总被刷？学员真实遇到的高频问题全解析！\\
\\
476](https://developer.aliyun.com/article/1686192)
[2\\
\\
提供一些准备Java八股文面试的建议\\
\\
587](https://developer.aliyun.com/article/1685410)
[3\\
\\
Redis常见面试题全解析\\
\\
733](https://developer.aliyun.com/article/1684426)
[4\\
\\
C++面试周刊(3):面试不慌,这样回答指针与引用，青铜秒变王者\\
\\
849](https://developer.aliyun.com/article/1678826)
[5\\
\\
Python面试题精选及解析\\
\\
503](https://developer.aliyun.com/article/1677659)
[6\\
\\
字节面试： MySQL 百万级 导入发生的 “死锁” 难题如何解决？“2序4拆”，彻底攻克\\
\\
686](https://developer.aliyun.com/article/1672034)
[7\\
\\
Redis数据类型面试给分情况\\
\\
463](https://developer.aliyun.com/article/1671774)
[8\\
\\
Java 面试实操指南与最新技术结合的实战攻略\\
\\
628](https://developer.aliyun.com/article/1671524)
[9\\
\\
MyBatis场景面试题\\
\\
444](https://developer.aliyun.com/article/1671340)
[10\\
\\
大厂RAG面试题：24个RAG八股文。偷偷背下来，毒打面试官 ！\\
\\
1652](https://developer.aliyun.com/article/1671240)

相关商品

## 相关课程

[更多](https://edu.aliyun.com/explore/)

[Linux MySQL服务器搭建与应用](https://edu.aliyun.com/course/314187)
[Linux用户和组管理](https://edu.aliyun.com/course/314195)
[计算机基础与Linux入门](https://edu.aliyun.com/course/314208)
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

热门文章

最新文章

下一篇

[\[网络安全\] Dirsearch 工具的安装、使用详细教程](https://developer.aliyun.com/article/1395854)

目录

- [一、linux](https://developer.aliyun.com/article/1363622#slide-0)
- [1.linux系统启动流程](https://developer.aliyun.com/article/1363622#slide-1)
- [2.linux文件类型](https://developer.aliyun.com/article/1363622#slide-2)
- [3.centos6和7怎么将源码安装的程序添加到开机自启动？](https://developer.aliyun.com/article/1363622#slide-3)
- [4.简述lvm，如何给使用lvm的/分区扩容？](https://developer.aliyun.com/article/1363622#slide-4)
- [5.为何du和df统计结果不一致？](https://developer.aliyun.com/article/1363622#slide-5)
- [6.如何升级内核？](https://developer.aliyun.com/article/1363622#slide-6)
- [二、mysql](https://developer.aliyun.com/article/1363622#slide-7)
- [1\. 索引的为什么使查询加快？有啥缺点？](https://developer.aliyun.com/article/1363622#slide-8)
- [2\. sql语句左外连接 右外连接 内连接 全连接区别](https://developer.aliyun.com/article/1363622#slide-9)
- [3\. mysql数据备份方式，如何恢复？你们的备份策略是什么？](https://developer.aliyun.com/article/1363622#slide-10)
- [4\. 如何配置数据库主从同步，实际工作中是否遇到数据不一致问题？如何解决？](https://developer.aliyun.com/article/1363622#slide-11)
- [5\. mysql约束有哪些？](https://developer.aliyun.com/article/1363622#slide-12)
- [6\. 二进制日志（binlog）用途？](https://developer.aliyun.com/article/1363622#slide-13)
- [7\. mysql数据引擎有哪些？](https://developer.aliyun.com/article/1363622#slide-14)
- [8\. 如何查询mysql数据库存放路径？](https://developer.aliyun.com/article/1363622#slide-15)
- [9\. mysql数据库文件后缀名有哪些？用途什么？](https://developer.aliyun.com/article/1363622#slide-16)
- [10\. 如何修改数据库用户的密码？](https://developer.aliyun.com/article/1363622#slide-17)
- [11\. 如何修改用户权限？如何查看？](https://developer.aliyun.com/article/1363622#slide-18)
- [三、nosql](https://developer.aliyun.com/article/1363622#slide-19)
- [1\. redis数据持久化有哪些方式？](https://developer.aliyun.com/article/1363622#slide-20)
- [2\. redis集群方案有哪些？](https://developer.aliyun.com/article/1363622#slide-21)
- [3\. redis如何进行数据备份与恢复？](https://developer.aliyun.com/article/1363622#slide-22)
- [4\. MongoDB如何进行数据备份？](https://developer.aliyun.com/article/1363622#slide-23)
- [5\. kafka为何比redis rabbitmq快？](https://developer.aliyun.com/article/1363622#slide-24)
- [四、docker](https://developer.aliyun.com/article/1363622#slide-25)
- [1\. dockerfile有哪些关键字？用途是什么？](https://developer.aliyun.com/article/1363622#slide-26)
- [2.如何减小dockerfile生成镜像体积？](https://developer.aliyun.com/article/1363622#slide-27)
- [3\. dockerfile中CMD与ENTRYPOINT区别是什么？](https://developer.aliyun.com/article/1363622#slide-28)
- [4\. dockerfile中COPY和ADD区别是什么？](https://developer.aliyun.com/article/1363622#slide-29)
- [5\. docker的cs架构组件有哪些？](https://developer.aliyun.com/article/1363622#slide-30)
- [6\. docker网络类型有哪些？](https://developer.aliyun.com/article/1363622#slide-31)
- [7\. 如何配置docker远程访问？](https://developer.aliyun.com/article/1363622#slide-32)
- [8\. docker核心namespace CGroups 联合文件系统功能是什么？](https://developer.aliyun.com/article/1363622#slide-33)
- [9\. 命令相关：导入导出镜像，进入容器，设置重启容器策略，查看镜像环境变量，查看容器占用资源](https://developer.aliyun.com/article/1363622#slide-34)
- [10\. 构建镜像有哪些方式？](https://developer.aliyun.com/article/1363622#slide-35)
- [11\. docker和vmware虚拟化区别？](https://developer.aliyun.com/article/1363622#slide-36)
- [五、kubernetes](https://developer.aliyun.com/article/1363622#slide-37)
- [1\. k8s的集群组件有哪些？功能是什么？](https://developer.aliyun.com/article/1363622#slide-38)
- [2\. kubectl命令相关：如何修改副本数，如何滚动更新和回滚，如何查看pod的详细信息，如何进入pod交互？](https://developer.aliyun.com/article/1363622#slide-39)
- [3\. etcd数据如何备份？](https://developer.aliyun.com/article/1363622#slide-40)
- [4\. k8s控制器有哪些？](https://developer.aliyun.com/article/1363622#slide-41)
- [5\. 哪些是集群级别的资源？](https://developer.aliyun.com/article/1363622#slide-42)
- [6\. pod状态有哪些？](https://developer.aliyun.com/article/1363622#slide-43)
- [7\. pod创建过程是什么？](https://developer.aliyun.com/article/1363622#slide-44)
- [8\. pod重启策略有哪些？](https://developer.aliyun.com/article/1363622#slide-45)
- [9\. 资源探针有哪些？](https://developer.aliyun.com/article/1363622#slide-46)
- [10\. requests和limits用途是什么？](https://developer.aliyun.com/article/1363622#slide-47)
- [11\. kubeconfig文件包含什么内容，用途是什么？](https://developer.aliyun.com/article/1363622#slide-48)
- [12\. RBAC中role和clusterrole区别，rolebinding和 clusterrolebinding区别？](https://developer.aliyun.com/article/1363622#slide-49)
- [13\. ipvs为啥比iptables效率高？](https://developer.aliyun.com/article/1363622#slide-50)
- [14\. sc pv pvc用途，容器挂载存储整个流程是什么？](https://developer.aliyun.com/article/1363622#slide-51)
- [15\. nginx ingress的原理本质是什么？](https://developer.aliyun.com/article/1363622#slide-52)
- [16\. 描述不同node上的Pod之间的通信流程](https://developer.aliyun.com/article/1363622#slide-53)
- [17\. k8s集群节点需要关机维护，需要怎么操作](https://developer.aliyun.com/article/1363622#slide-54)
- [18\. canal和flannel区别](https://developer.aliyun.com/article/1363622#slide-55)
- [六、prometheus](https://developer.aliyun.com/article/1363622#slide-56)
- [1\. prometheus对比zabbix有哪些优势？](https://developer.aliyun.com/article/1363622#slide-57)
- [2\. prometheus组件有哪些，功能是什么？](https://developer.aliyun.com/article/1363622#slide-58)
- [3\. 指标类型有哪些？](https://developer.aliyun.com/article/1363622#slide-59)
- [4\. 在应对上千节点监控时，如何保障性能](https://developer.aliyun.com/article/1363622#slide-60)
- [5\. 简述从添加节点监控到grafana成图的整个流程](https://developer.aliyun.com/article/1363622#slide-61)
- [6\. 在工作中用到了哪些exporter](https://developer.aliyun.com/article/1363622#slide-62)
- [七、ELK](https://developer.aliyun.com/article/1363622#slide-63)
- [1\. Elasticsearch的数据如何备份与恢复？](https://developer.aliyun.com/article/1363622#slide-64)
- [2\. 你们项目中使用的logstash过滤器插件是什么？实现哪些功能？](https://developer.aliyun.com/article/1363622#slide-65)
- [3\. 是否用到了filebeat的内置module？用了哪些？](https://developer.aliyun.com/article/1363622#slide-66)
- [4\. elasticsearch分片副本是什么？你们配置的参数是多少？](https://developer.aliyun.com/article/1363622#slide-67)
- [八、运维开发](https://developer.aliyun.com/article/1363622#slide-68)
- [2\. 编写脚本，定时备份某个库，然后压缩，发送异机](https://developer.aliyun.com/article/1363622#slide-69)
- [3\. 批量获取所有主机的系统信息](https://developer.aliyun.com/article/1363622#slide-70)
- [4\. django的mtv模式流程](https://developer.aliyun.com/article/1363622#slide-71)
- [5\. python如何导出、导入环境依赖包](https://developer.aliyun.com/article/1363622#slide-72)
- [6\. python创建，进入，退出，查看虚拟环境](https://developer.aliyun.com/article/1363622#slide-73)
- [7\. flask和django区别，应用场景](https://developer.aliyun.com/article/1363622#slide-74)
- [8\. 列举常用的git命令](https://developer.aliyun.com/article/1363622#slide-75)
- [9\. git gitlab jenkins的CICD流程如何配置](https://developer.aliyun.com/article/1363622#slide-76)
- [九、日常工作](https://developer.aliyun.com/article/1363622#slide-77)
- [1\. 在日常工作中遇到了什么棘手的问题，如何排查](https://developer.aliyun.com/article/1363622#slide-78)
- [2\. 日常故障处理流程](https://developer.aliyun.com/article/1363622#slide-79)
- [3\. 修改线上业务配置文件流程](https://developer.aliyun.com/article/1363622#slide-80)
- [4\. 业务pv多少？集群规模多少？怎么保障业务高可用？](https://developer.aliyun.com/article/1363622#slide-81)
- [十、开放性问题](https://developer.aliyun.com/article/1363622#slide-82)
- [1\. 你认为初级运维工程师和高级运维工程师的区别？](https://developer.aliyun.com/article/1363622#slide-83)
- [2\. 你认为未来运维发展方向?](https://developer.aliyun.com/article/1363622#slide-84)

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

4008013260 [售前咨询](https://smartservice.console.aliyun.com/pre-sale/chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363622) [售后在线](https://smartservice.console.aliyun.com/service/robot-chat?entrance=201&referrer=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363622)

### 其他服务

[我要建议](https://www.aliyun.com/connect/home) [我要投诉](https://www.aliyun.com/complaint)

![登录插画](https://img.alicdn.com/imgextra/i2/O1CN015QIT9m1FmmyUntYlQ_!!6000000000530-2-tps-320-200.png)

登录以查看您的控制台资源

管理云资源

状态一览

快捷访问

[快捷注册](https://account.aliyun.com/register/qr_register.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363622) [登录阿里云](https://account.aliyun.com/login/login.htm?oauth_callback=https%3A%2F%2Fdeveloper.aliyun.com%2Farticle%2F1363622)