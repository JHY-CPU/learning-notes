# Linux 运维工程师面试题总结- 野鹤% - 博客园

URL: https://www.cnblogs.com/yehewang/p/16834780.html

# [Linux 运维工程师面试题总结](https://www.cnblogs.com/yehewang/p/16834780.html "发布于 2022-10-28 09:36")

Linux 运维工程师面试题总结

一、有文件 file1

**1**、查询 **file1** 里面空行的所在行号

答：awk ‘{if($0~/^$/)print NR}’ file or grep -n ^$ file \|awk ‘BEGIN{FS=”:”}{print $1}’

**2**、查询 **file1** 以 **abc** 结尾的行答：grep abc$ file1

**3**、打印出 **file1** 文件第 **1** 到第 **3** 行答：sed -n ’1,3p’ file1

head -3 file1

二、如何将本地 **80** 端口的请求转发到 **8080** 端口，当前主机 **IP** 为 **192.168.2.1**

答：Iptables -A PREROUTING -d 124.42.60.109 -p tcp -m tcp –dport 80 -j DNAT –to-destination 10.0.0.18:9000

三、crontab

在 **11** 月份内，每天的早上 **6** 点到 **12** 点中，每隔 **2** 小时执行一次 **/usr/bin/httpd.sh** 怎么实现

答：0 6-12/2 \* 11 \* /usr/bin/httpd.sh

四、编写个 **shell** 脚本将 **/usr/local/test** 目录下大于 **100K** 的文件转移到 **/tmp** 目录下

#!/bin/bash

for file in \`ls /root\` do

if \[ -f $file \]; then

if \[ \`ls -l $file\|awk '{print $5}'\` -gt 10000 \]; then mv $file /tmp/

fi fi done

五、简述 **raid0** **raid1** **raid5** 三种工作模式的工作原理及特点。

RAID 0：连续以位或字节为单位分割数据，并行读/写于多个磁盘上，因此具有很高的数据传输率，但它没有数据冗余，因此并不能算是真正的 RAID 结构。RAID 0 只是单纯地提高 性能，并没有为数据的可靠性提供保证，而且其中的一个磁盘失效将影响到所有数据。因此， RAID 0 不能应用于数据安全性要求高的场合。

RAID 1：它是通过磁盘数据镜像实现数据冗余，在成对的独立磁盘上产生互为备份的数据。当原始数据繁忙时，可直接从镜像拷贝中读取数据，因此 RAID 1 可以提高读取性能。

RAID 1 是磁盘阵列中单位成本最高的，但提供了很高的数据安全性和可用性。当一个磁盘失效时，系统可以自动切换到镜像磁盘上读写 ，而不需要重组失效的数据。简单来说就是：镜象结构，类似于备份模式，一个数据被复制到两块硬盘上。

RAID10:高可靠性与高效磁盘结构

一个带区结构加一个镜象结构，因为两种结构各有优缺点，因此可以相互补充。主要用于容量不大，但要求速度和差错控制的数据库中。

RAID5：分布式奇偶校验的独立磁盘结构，它的奇偶校验码存在于所有磁盘上，任何一个 硬盘损坏，都可以根据其它硬盘上的校验位来重建损坏的数据。支持一块盘掉线后仍然正常运行。

六、 **oracle** 数据库备份方式

物理备份：开启网络监听，备份数据库文件。

RMAN 备份：通过表空间文件在 RMAN 模式对 ORACLE 数据备份。七、如何查看占用端口 **8080** 的进程

答：lsof -i:8080

八、请写出 **apache2.X** 版本的两种工作模式，以及各自工作原理。如何查看 **apache** 当前所支持的模块，并且查看是工作在哪种模式下？

答案：

prefork(多进程，每个进程产生子进程)和 worker（多进程，每个进程生成多个线程) prefork的工作原理是，控制进程在最初建立“StartServers”个子进程后，为了满足 MinSpareServers 设置的需要创建一个进程，等待一秒钟，继续创建两个，再等待一秒钟，继续创建四个?? 如此按指数级增加创建的进程数， 最多达到每秒 32 个，直到满足 MinSpareServers 设置的值为止。这就是预派生（prefork）的由来。这种模式可以不必 在请求到来时再产生新的进程，从而减小了系统开销以增加性能。

worker 是 2.0 版中全新的支持多线程和多进程混合模型的 MPM。由于使用线程来处

理，所以可以处理相对海量的请求，而系统资源的开销要小于基于进程的服务器。但是， worker 也使用了多进程，每个进程又生成多个线程，以获得基于进程服务器的稳定性。这种 MPM 的工作方式将是Apache 2.0 的发展趋势。

可以通过命令httpd -l 可以查看 apache 当前的模块，如果带有 worker.c 就是工作在 worker

模式下，如果有 prefork.c 就是工作在 prefork.c 的模式下。九、你使用过监控软件吗？说说其特点

答：使用 nagios 对服务器进行监控，其特点可实时实现手机短信、电子邮件、MSN、飞信报警。 使用 cacti 对流量进行监控。

十、你对现在运维工程师的理解和以及对其工作的认识

运维工程师在公司当中责任重大，需要保证时刻为公司及客户提供最高、最快、最稳定、最安全的服务。运维工程师的一个小小的失误，很有可能会对公司及客户造成重大损失，因此运维工程师的工作需要严谨及富有创新精神。

十一、 **linux** 下常用的 **DNS** 服务软件是什么，举出几种常用的 **DNS** 记录，如果域名 **abc.com** 配置好了一台邮件服务器 **,IP** 地址为 **202.106.0.20**，我该如何做相关的解析？是否了解 **bind** 的智能解析，如果了解请简述一下其原理

答案：

1) 常用的 DNS 软件是 bind

2) A 记录 地址记录

MX 记录 邮件交换记录

CNAME 记录 别名域记录

3) 修改 abc.com 域名的配置文件，增加以下记录

IN MX 10 mail.abc.com. mail IN A 202.106.0.20

4) bind 根据请求解析客户端的 IP 地址，做出不同的解析，其原理是在配置文件中，设定了 view，在每个 view 都有客户端的 IP 地址段，bind 服务器根据请求解析客户端的 IP 地址，匹配不同的view,再根据该view 的配置，到相应的配置文件进行查询，将结果返回给请求的客户端。

十二、通过 apache 访问日志 access.log 统计 IP 和每个地址访问的次数，按访问量列出 前

10 名。

日志格式样例如下

192.168.1.247 – - \[02/Jul/2010:23:44:59 +0800\] “GET / HTTP/1.1″ 200 19 答案：

cat access\_log \| awk ‘{print $1}’ \| uniq -c\|sort -rn\|head -10

//这个别的方法也能统计,但有些命令是必要的 awk , sort,uniq ,主要看是否这些命令都 使用了。

十三、如何用 **mysql** 命令进行备份和恢复？以 **test** 库为例，创建一个备份，并再用此备份进行恢复。

mysqldump -u root -p test > test.sql mysql -u root -p test < test.sql

//主要考对方msqldump > test.sql 和 mysql < test.sql

十四、你认为在系统调优方面都包括哪些工作，以linux 为例，请简明阐述，并举一些参数为例。

答案：

系统调优包括内核参数优化和应用优化 2 个方面，对方只要从这两方面来说，就可以了，尽量能有些经验的阐述。

有个文件如下： [http://a.domain.com/1.html](http://a.domain.com/1.html) [http://b.domain.com/1.html](http://b.domain.com/1.html) [http://c.domain.com/1.html](http://c.domain.com/1.html) [http://a.domain.com/2.html](http://a.domain.com/2.html) [http://b.domain.com/2.html](http://b.domain.com/2.html) [http://a.domain.com/3.html](http://a.domain.com/3.html)

要求：得到主机名（和域名），并统计哪个网址出现的次数，并排序。可以 shell 或 C。 得

到的结果应该是:

3 a.domain.com

2 b.domain.com

1 c.domain.com

\[root@mail ~\]# awk ‘BEGIN{FS=”/”}{arr\[$3\]++}END{for(i in arr) print arr\[i\],i}’ list\| sort -r 答案

3 a.domain.com

2 b.domain.com

1 c.domain.com

挂载windows 的共享目录？

mount.cifs //IP/SHARE linux 的目录 --verbose -o user=username <--这个用户是 windows 下的用户--verbose 这个参数可以不加，它是显示过程的

例如 mount.cifs //10.1.1.246/gongxiang /mnt --verbose -o user=gao

或者是 mount -t cifs

umount /mnt 或 umount.cifs /mnt -l <--取消挂载图形界面：smb://IP A B 网络是通的，最少列出五种传输文件的服务

nfs ,ftp,scp ,rsync,samba,http://

1\. 假设 Apache 产生的日志文件名为 access\_log, 在 apache 正在运行时, 执行命令 mv access\_log access\_log.bak,执行完后,请问新的 apache 的日志会打印到哪里,为什么? 新的日志会打印在 access\_log.bak 中，因为 apache 启动时会找到 access\_log 文件， 随时准备向文件中加入日志信息，

虽然此时文件被改名，但是由于服务正在运行，因为它的 inode 节点的位置没有变，程序 打开的fd 仍然会指向原来那个 inode，

不会因为文件名的改变而改变。apache  会继续向已改名的文件中追加日志，但是若重启

apache 服务，系统会检查 access\_log

文件是否存在，若不存在则创建。

2\. 在 Shell 环境下,如何查看远程 Linux 系统运行了多少时间? 2、监控主机执行： ssh user@被监控主机 ip "uptime"

这样得到了被监控主机的 uptime

3\. 处理以下文件内容,将域名取出并进行计数排序,如处理: [http://www.baidu.com/index.html](http://www.baidu.com/index.html) [http://www.baidu.com/1.html](http://www.baidu.com/1.html) [http://post.baidu.com/index.html](http://post.baidu.com/index.html) [http://mp3.baidu.com/index.html](http://mp3.baidu.com/index.html) [http://www.baidu.com/3.html](http://www.baidu.com/3.html) [http://post.baidu.com/2.html](http://post.baidu.com/2.html)

得到如下结果:

域名的出现的次数 域名

3 [www.baidu.com](http://www.baidu.com/)

2 post.baidu.com

1 mp3.baidu.com

可以使用 bash/perl/php/c 任意一种

3、\[root@localhost shell\]# cat file \| sed -e ' s/http:\\/\\///' -e ' s/\\/.\*//' \| sort \| uniq -c \| sort -rn

3 [www.baidu.com](http://www.baidu.com/)

2 post.baidu.com

1  mp3.baidu.com

\[root@codfei4 shell\]# awk -F/ '{print $3}' file \|sort -r\|uniq -c\|awk '{print $1"\\t",$2}' 3 [www.baidu.com](http://www.baidu.com/)

2  post.baidu.com

1 mp3.baidu.com

4\. 如果得到随机的字串,长度和字串中出现的字符表可定义, 并将字串倒序显示, 如 把 0123456789 作为基准的字串字符表,产生一个 6 位的字串 642031,打印出的字串为 130246,可使用 bash/perl/php/c 任意一种.

4、\[root@localhost ~\]# awk -v count=6 'BEGIN

{srand();str="0123456789";len=length(str);for(i=count;i>0;i--) marry\[i\]=substr(str,int(rand()\*len),1);for(i=count;i>0;i--) printf("%c",marry\[i\]);printf("\\n");for

(i=0;i<=count;i++) printf("%c",marry\[i\]);printf("\\n")}' 838705

507838

**5.** 如何查看当前 **Linux** 系统的状态 **,** 如 **CPU** 使用 **,** 内存使用 **,** 负载情况等 **.**

5、Linux 系统中“/proc”是个伪文件目录,不占用系统空间，及时的反应出内存现在使用的进程情况      其中许多文件都保存系统运行状态和相关信息

对于“/proc”中文件可使用文件查看命令浏览其内容，文件中包含系统特定信息： cpuinfo

主机 CPU 信息

filesystems 文件系统信息

meninfo 主机内存信息 version Linux 内存版本信息 diskstatus 磁盘负载情况

另外 top 命令可以动态的显示当前系统进程用户的使用情况,而且是动态的显示出来，尤其是在该命令显示出来的对上方对系统的情况进行汇总.

free 命令呢可以查看真实使用的内存 一般用 free -m

使用 lsof 、ps -aux 可以查看详细的每个进程的使用状况

dmesg 也是常用来查看系统性能的命令

#题目：有 10 台被监控主机、一台监控机，在监控机上编写脚本，一旦某台被监控机器/ 分区适用率大于 80%， 就发邮件报警放到 crontab 里面， 每 10 分钟检查一次 #测试机器：虚拟机 Linux as 4

#1.首先建立服务器间的信任关系。拿两台机器做测试本机 ip:192.168.1.6

\[root@codfei ~\]# ssh-keygen -t rsa Generating public/private rsa key pair.

Enter file in which to save the key (/root/.ssh/id\_rsa):

/root/.ssh/id\_rsa already exists.

Overwrite (y/n)? y (以为我是第 2 次建立关系所以此处覆盖原来的文件) Enter passphrase (empty for no passphrase):(直接回车无须输入密钥) Enter same passphrase again:

Your identification has been saved in /root/.ssh/id\_rsa. Your public key has been saved in /root/.ssh/id\_rsa.pub. The key fingerprint is:

04:37:13:2a:4b:10:af:c1:2b:03:3f:6b:27:ce:b9:62 root@codfei \[root@codfei ~\]# cd .ssh/

\[root@codfei .ssh\]# ll

-rw------ 1 root root 883 Apr 25 17:51 id\_rsa

-rw-r--r-- 1 root root 221 Apr 25 17:51 id\_rsa.pub

-rw-r--r-- 1 root root 442 Apr 25 17:37 known\_hosts id\_rsa 是密钥文件，id\_rsa.pub 是公钥文件。

\[root@codfei  .ssh\]#  scp  id\_rsa.pub192.168.1.4:/root/.ssh/192.168.1.6  root@192.168.1.4's

password:

id\_rsa.pub 100% 221 0.2KB/s 00:00

这里把公钥文件取名为本机的 ip 地址就是为了以后和更多的机器建立信任关系不发生混淆。

现在登陆到 192.168.1.4 机器

\[root@codfei ~\]# cd .ssh/

\[root@codfei .ssh\]# cat 192.168.1.6 >> authorized\_keys

然后回到 192.168.1.6 机器直接

\[root@codfei .ssh\]# ssh 192.168.1.4

Last login: Wed Aug 8 12:14:42 2007 from 192.168.1.6

这样就可以了，里面偶尔涉及到权限问题。一般./ssh 文件夹是 755 authorized\_keys 为 600

或者 644

####脚本如下#######################

#!/bin/bash #SCRIPT:df\_check.sh

#Writeen by codfei Mon Sep 3 07:25:28 CST 2007 #PURPOSE:This script is used to monitor for full filesystems. #######################Begining#################### ####################

FSMAX="80"

remote\_user='root' #####完全可以不用 root

remote\_ip=(192.168.1.5    192.168.1.6    192.168.1.7   192.168.1.8    192.168.1.9    192.168.1.10

192.168.1.11 192.168.1.12 192.168.1.13 192.168.1.14 ) ----> 这里填写你要监控的主机 ip

ip\_num='0'

while \[ "$ip\_num" -le "$(expr ${#remote\_ip\[@\]} - 1)" \] do

read\_num='1'

ssh "$remote\_user"@"${remote\_ip\[$ip\_num\]}" df -h > /tmp/diskcheck\_tmp grep '^/dev/\*'

/tmp/diskcheck\_tmp\|awk '{print $5}'\|sed 's/\\%//g' >

/tmp/diskcheck\_num\_tmp

while \[ "$read\_num" -le $(wc -l < /tmp/diskcheck\_num\_tmp) \] do

size=$(sed -n "$read\_num"'p' /tmp/diskcheck\_num\_tmp) if \[ "$size" -gt "$FSMAX" \]

then

$(grep '^/dev/\*' /tmp/diskcheck\_tmp\|sed -n $read\_num'p' >

/tmp/disk\_check\_mail)

$(echo ${remote\_ip\[$ip\_num\]} >> /tmp/disk\_check\_mail)

$(mail -s "diskcheck\_alert" admin < /tmp/disk\_check\_mail) fi

read\_num=$(expr $read\_num + 1) done

ip\_num=$(expr $ip\_num + 1) done

#############over################################ ################让脚本每十分钟执行一次#############

在 cron 表中加入

0/10 \* \* \* \* /home/codfei/diskcheck.sh 2>&1 ################################################ ##########################

比如， ext2 文件系统， 如果异常死机，开机如何修复文件系统？如果异常关机，比如断电，通知机房的人开机之后，

我们需要远程修复、检查文件系统除了/分区之外， 其他的分区： umount /home

fsck -y /home

/ 分区需要开机之后， 由机房的人来扫描

随后我们再登录并扫描/home 等其他分区如何查看一个进程所使用的文件句柄？ 看这里面 /proc/进程号/fd/

的个数就行了

简单的比如如何查看 apache 进程数 \[root@localhost fd\]# ps -ef\|grep httpd\|wc -l 1

如何统计 apache 的每秒访问数？

tail access\_log \| awk '{print $1,$4}'

\[root@localhost logs\]# grep -c \`date -d '3 second ago' +%T\` access\_log 0 ################################################

1、/proc/sys 子目录的作用

该子目录的作用是报告各种不同的内核参数，并让您能交互地更改其中的某些。与 /proc 中所有其他文件不同，该目录中的某些文件可以写入，不过这仅针对 root。

其中的目录以及文件的详细列表将占据过多的篇幅，而且该目录的内容是依赖于系统的，而大部分的文件也仅仅对某些特殊的应用程序有用。然而，以下是该子目录的两个最常见的用途：

允许路由：即便是 Mandrakelinux 默认的内核也是允许路由的，您必需显式允许它这么 做。为此，您只要以 root 身份键入以下命令：

$ echo 1 >/proc/sys/net/ipv4/ip\_forward

如果您要禁用路由，请将上述命令中的 1 改为 0。

阻止 IP 欺骗：IP 欺骗会让人认为某个来自于外部的某个数据包是来自于它到达的那个接口。这一技术常被骇客(cracker)所使用。您可以让内核阻止这种入侵。请键入：

$ echo 1 >/proc/sys/net/ipv4/conf/all/rp\_filter

这样，这种攻击就不再可能了。

这些改变仅当系统运行时有效。在系统重新启动之后，它们会改会它们的默认值。要在启动时就改动这些值，您可以将您在 shell 提示符后键入的命令添加到 /etc/rc.d/rc.local 中 以免每次都键入它们。另一个方法是修改

/etc/sysctl.conf

2、将一个文本的奇数行和偶数行合并，第 2 行和第 3 行合并

\[root@localhost bin\]# cat 1

48 Oct 3bc1997 lpas 68.00 lvx2a 138

484 Jan 380sdf1 usp 78.00 deiv 344

483 nov 7pl1998 usp 37.00 kvm9d 644

320 aug der9393 psh 83.00 wiel 293

231 jul sdf9dsf sdfs 99.00 werl 223

230 nov 19dfd9d abd 87.00 sdiv 230

219 sept 5ap1996 usp 65.00 lvx2c 189

216 Sept 3zl1998 usp 86.00 kvm9e 234 \[root@localhost bin\]# sed '$!N;s/\\n/ /g' 1

48 Oct 3bc1997 lpas 68.00 lvx2a 138 484 Jan 380sdf1 usp 78.00 deiv 344 483 nov 7pl1998 usp

37.00 kvm9d 644 320 aug der9393 psh 83.00 wiel 293 231 jul sdf9dsf sdfs 99.00 werl 223 230 nov 19dfd9d abd 87.00 sdiv 230 219 sept 5ap1996 usp 65.00 lvx2c 189 216 Sept 3zl1998 usp 86.00 kvm9e 234

\[root@localhost bin\]# sed -n -e 2p -e 3p 1\|sed '$!N;s/\\n/ /'

484 Jan 380sdf1 usp 78.00 deiv 344 483 nov 7pl1998 usp 37.00 kvm9d 644 3、read 命令 5 秒后自动退出

\[root@localhost bin\]# read -t 5

4、自动ftp 上传

#!/bin/sh

ftp -n<<END\_FTP open 192.168.1.4

user codfei duibuqi //用户名 codfei 密码 duibuqi binary

prompt off //关闭提示 mput test //上传 test close

bye END\_FTP

自动 ssh 登陆 从 A 到 B 然后再到 c #!/usr/bin/expect -f

set timeout 30 spawn ssh codfei@B expect "password:" send "pppppp\\r" expect "\]\*"

send "ssh codfei@C\\r" expect "password:" send "pppppp\\r" interact

5、#打印第一个域 \[root@localhost bin\]# cat 3 eqeqedadasdD eqeqdadfdfDD fdsfdsfQWEDD DSADASDSADSA

\[root@localhost bin\]# \[root@localhost bin\]#

\[root@localhost bin\]# awk -F "" '{print $1}' 3 e

e f D

6、实现字符串翻转 \[root@localhost bin\]# cat 8 qweqewqedadaddas \[root@localhost bin\]# rev 8 saddadadeqweqewq

########################################第 2 次电面

7、sed awk grep 哪个最好

我答的是 哪个掌握的精通，都很好，但是还是问我哪个最好，我只能说 awk 了，对于行操作和列操作都可以操作的很好。

8、grep -E -P 是什么意思

我说的是-E, --extended-regexp 采用规则表示式去解释样式。 -P 不太清楚

9、请介绍一下你对运维这个工作的理解，和应该具备的素质。 shell 脚本编程部分：

1. 从 a.log 文件中提取包含“WARNING”或”FATAL”,同时不包含“IGNOR”的行，然后，提取以“：”分割的第五个字段？
2. 添加一个新组为 class01,然后，添加属于这个组的 30 个用户，用户名的形式为 stdXX, 其中，XX 从 01 到 30？
3. 在每个月的第一天备份并压缩/etc 目录下的所有内容，存放在/root/backup 目录里， 且文件名为如下形式 yymmdd\_etc,yy 为年，mm 为月，dd 为日。shell 程序 fileback 存放在

/usr/bin 目录下？

1. 用 shell 编程，判断一文件是不是字符设备文件，如果是将其拷贝到/dev 目录下？ 参考答案：

#!/bin/bash directory=/dev

for file in anaconda-ks.cfg install.log install.log.syslog do

if \[ -f $file \] then

cp $file $directory/$file.bak

echo " HI, $LOGNAME $file is backed up already in $directory !!" fi

done

1. 某系统管理员需要每天做一定的重复工作，编制一个解决方案： (1).从下午 4：50 删除/abc 目录下的全部子目录和全部文件；

(2).  从早上 8：00～下午 6：00 每小时读取/xyz 目录下 x1 文件中每行第一个域的全部数 据加入到/backup 目录下的 back01.txt 文件内；

(3).  每逢周一下午 5：50 将/data 目录下的所有目录和文件归档并压缩为文件

backup.tar.gz;

(4).  在下午 5：55 将 IDE 接口的 CD－ROM 缷载（假设 CD－ROM 的设备名为 hdc）; (5).在早上 8：00 前开机后启动。

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

－－－－－－－－－－－－－－－－－－

1、简述Apache 两种工作模式，以及它们之间的区别。

答案：最主要的两种模式是prefork 模式与worker 模式。prefork 每个子进程只有一个线 程，效率高但消耗内存大，是 unix 下默认的模式；worker 模式每个子进程有多个线程， 内存消耗低，但一个线程崩溃会牵连其它同子进程的线程。

2、用 iptables 添加一个规则允许 192.168.0.123 访问本机 3306 端口 iptables -I INPUT 1 -p tcp -m tcp --dport 3306 -s 192.168.0.123 -j ACCEPT 3、如何对一台 Linux 服务器进行系统性能调优，列举出参数。

4、DNS 服务器的工作原理。

5、修改第一块网卡的路径是什么。

/etc/sysconfig/network-scripts/ifcfg-eth0

7、使用 shell，建立 class1 用户组，再批量建立 stu1--stu30 的用户，并指定用户组为 class1。 vi autoaddusr

#!/usr/bin/php -q

<?php

exec("groupadd class1"); for($i=1; $i<=30; $i++){ exec("useradd -G class1 stu".$i);

}

?>

chmod +x autoaddusr

./autoaddusr

8、个人对该工作的未来如何规划，需要加强哪些能力。首先，我有一颗真诚的心，遇事沉着冷静，不急不躁；

其次，我有相应的专业知识和工作经验。一年多的系统管理经历锻炼了我在这个行业的业务能力，并对行业前景和发展动态有相应的了解；

最后，我会用踏实的作风在今后的工作中证明我自己的能力！

9、日常监控都需要监控哪些？

1) 硬件： CPU：/proc/cpuinfo 内存：/proc/meminfo硬盘：fdisk -l

2) 系统：

负载：/proc/loadavg

uptime 查看实时 load average、swap虚拟内存：vmstat（参数-s；2 4） SUID,用户,进程

系统日志：tail -f /var/log/messages

logwatch --print --range Today --service SSHD --service pam\_unix 3)网络：Host\_Alive,Ping,端口,连接

1\. 如何将本地 80 端口的请求转发到 8080 端口,当前主机 IP 为 192.168.16.1,其中本地 网卡

eth0:

答：

#iptables -t nat -A PREROUTING -d 192.168.16.1 -p tcp --dport 80 -j DNAT --to 192.168.16.1:8080

或者：

#iptables -t nat -A PREROUTING -i eth0 -d 192.168.16.1 -p tcp -m tcp --dport 80 -j REDIRECT --to- ports 8080

2\. 什么是NAT,常见分为那几种，DNAT 与 SNAT 有什么不同，应用事例有那些？

3\. 包过滤防火墙与代理应用防火墙有什么区别，能列举几种相应的产品吗？ 4.iptables 是否支持 time 时间控制用户行为，如有请写出具体操作步骤

5\. 说出你知道的几种linux/unix 发行版本

6\. 列出 linux 常见打包工具并写相应解压缩参数(至少三种)

7\. 计划每星期天早 8 点服务器定时重启,如何实现？

8\. 列出作为完整邮件系统的软件,至少二类

9，当用户在浏览器当中输入一个网 g 站，说说计算机对 dns 解释经过那些流程？注：本机跟本地 dns 还没有缓存。

答： a.用户输入网址到浏览器 b.浏览器发出 DNS 请求信息

c. 计算机首先查询本机 HOST  文件，看是否存在，存在直接返回结果，不存在，继续下一 步

d. 计算机按照本地 DNS 的顺序，向合法 dns 服务器查询 IP 结果，

1. 合法 dns 返回 dns 结果给本地 dns，本地 dns 并缓存本结果，直到 TTL 过期，才再次 查询此结果

f.返回 IP 结果给浏览器

g. 浏览器根据 IP 信息，获取页面

10，我们都知道，dns 既采用了 tcp 协议，又采用了 udp 协议，什么时候采用 tcp 协议？什么时候采用 udp 协议？为什么要这么设计？

答：这个题需要理解的东西比较的多，分一下几个方面

a，从数据包大小上分：UDP 的最大包长度是 65507 个字节，响应 dns 查询的时候数据包长度超过 512 个字节，而返回的只要前 512 个字节，这时名字解释器通常使用 TCP 从发 原来的请求。

b，从协议本身来分：大部分的情况下使用UDP 协议，大家都知道 UDP 协议是一种不可靠的协议，dns 不像其它的使用 UDP 的 Internet 应用 (如：TFTP，BOOTP 和 SNMP 等)， 大部分集中在局域网，dns 查询和响应需要经过广域网，分组丢失和往返时间的不确定性在 广域网比局域网上更大，这就要求 dns 客户端需要好的重传和超时算法，这时候使用 TCP 11，一个 EXT3 的文件分区，当使用 touch test.file 命令创建一个新文件时报错，报错的 信息是提示磁盘已满，但是采用 df -h 命令查看磁盘大小时，只使用了，60%的磁盘空间， 为什么会出现这个情况，说说你的理由。

答：两种情况，一种是磁盘配额问题，另外一种就是 EXT3 文件系统的设计不适合很多小 文件跟大文件的一种文件格式，出现很多小文件时，容易导致 inode 耗尽了。

12，我们都知道 FTP  协议有两种工作模式，说说它们的大概的一个工作流程？

FTP 两种工作模式：主动模式（Active FTP）和被动模式（Passive FTP）

在主动模式下，FTP 客户端随机开启一个大于 1024 的端口 N 向服务器的 21 号端口发起连接，然后开放N+1 号端口进行监听，并向服务器发出 PORT N+1 命令。

服务器接收到命令后，会用其本地的 FTP 数据端口（通常是 20）来连接客户端指定的端口 N+1，进行数据传输。

在被动模式下，FTP 客户端随机开启一个大于 1024 的端口 N 向服务器的 21 号端口发起连接，同时会开启 N+1 号端口。然后向服务器发送 PASV 命令，通知服务器自己处于被动模式。服务器收到命令后，会开放一个大于 1024 的端口 P 进行监听，然后用 PORT P 命 令通知客户端，自己的数据端口是 P。客户端收到命令后，会通过

N+1 号端口连接服务器的端口 P，然后在两个端口之间进行数据传输。

总的来说，主动模式的 FTP 是指服务器主动连接客户端的数据端口，被动模式的 FTP 是指服务器被动地等待客户端连接自己的数据端口。

被动模式的FTP 通常用在处于防火墙之后的 FTP 客户访问外界 FTp 服务器的情况，因为在这种情况下，防火墙通常配置为不允许外界访问防火墙之

后主机，而只允许由防火墙之后的主机发起的连接请求通过。

因此，在这种情况下不能使用主动模式的 FTP  传输，而被动模式的 FTP  可以良好的工作。

1. 编写个 shell 脚本将当前目录下大于 10K 的文件转移到/tmp 目录下

#/bin/sh #Programm :

\# Using for move currently directory to /tmp

for FileName in \`ls -l \|awk '$5>10240 {print $9}'\` do

mv $FileName /tmp done

ls -al /tmp echo "Done! "

1. apache 有几种工作模式，分别介绍下其特点，并说明什么情况下采用不同的工作模 式？ apache 主要有两种工作模式：prefork(apache 的默认安装模式)和 worker(可以在编译 的时候加参数--with-mpm-worker 选择工作模式)

prefork 的特点是：(预派生)

1\. 这种模式可以不必在请求到来时再产生新的进程，从而减小了系统开销

2\. 可以防止意外的内存泄漏

3\. 在服务器负载下降的时候会自动减少子进程数

worker 的特点是：支持混合的多线程多进程的多路处理模块

如果对于一个高流量的 HTTP 服务器，worker MPM 是一个比较好的选择，因为 worker MPM

占用的内存要比 prefork 要小。

1. 名词解释 HDLC,VTP,OSPF,RIP,DDOS,system

V,GNU,netscreen,ssh,smartd,apache,WAIT\_TIME 等等

1. 编写 shell 脚本获取本机的网络地址。比如：本机的 ip 地址是：

192.168.100.2/255.255.255.0，那么它的网络地址是

192.168.100.1/255.255.255.0

方法一：

1. #!/bin/bash
2. #This script print ip and network
3. file="/etc/sysconfig/network-scripts/ifcfg-eth0"
4. if \[ -f $file \] ;then
5. IP=\`grep "IPADDR" $file\|awk -F"=" '{ print $2 }'\`
6. MASK=\`grep "NETMASK" $file\|awk -F"=" '{ print $2 }'\`
7. echo "$IP/$MASK"
8. exit 1
1. fi

方法二：

1. #!/bin/bash
2. #This programm will printf ip/network 3. #
1. IP=\`ifconfig eth0 \|grep 'inet ' \|sed 's/^.\*addr://g'\|sed 's/ Bcast.\*$//g'\`
2. NETMASK=\`ifconfig eth0 \|grep 'inet '\|sed 's/^.\*Mask://g'\`
3. echo "$IP/$NETMASK"
4. exit

17.在命令行下发一邮件，发件人：123@abc.com,收信人： [abc@xyz.com](mailto:abc@xyz.com)

二简述题：

1.linux 下如何改 IP,主机名，DNS 2.linux 下如何添加路由

3\. 简述 linux 下编译内核的意义与步骤

4\. 简述 Linux 启动过程

5\. 简述 DDOS 攻击的原理

6\. 简述 Tcp 三次握手的过程

7\. 简述 VPN，常见有哪几种？

8.

三：设计题：

1\. 系统设计

请考虑以下系统的设计. 您可以翻阅资料，查询任何您有帮助的资料、指南等。 您有的资源：

8 台安装 Linux (2.6 内核) 的双网卡 PC 服务器以及相关开源软件，交换机 Apache 2.2.x Tomcat 5.5.X

数据库系统

最多 8 个 Internet IP 地址,请您设计一个系统：

1、使用双 apache web server 前端；

2、采用 AJP 连接后段的３台 Tomcat 应用服务器，这些 tomcat 被配置成 cluster, 因此 需要考虑 apache 对后端的分配， 分配采用完全平衡的方法

； 配置使用 cookie 来实现 session stickness;

3、１台数据库服务器只有 tomcat 才需要连接，也不需要对 Internet 提供服务。

4、考虑系统的安全性和维护方便性；

5、通过rewrite 规则配置把下属 URL 规则改写成友好的 URL [http://server/webapp/getinfo?id=XXXX&name=YYYY](http://server/webapp/getinfo?id=XXXX&name=YYYY) –> [http://server/getinfo/YYYY/XXXX](http://server/getinfo/YYYY/XXXX)

您需要提交

1、服务器规划，包括：

＊ 网络结构图

＊ 每台机器的 IP 地址分配

＊ 每台机器上运行的关键软件

＊ 您从安全性和维护性方面的考虑

2、Apache 的以下配置文件给我们：

＊ extra/http-proxy-ajp.conf

＊ extra/http-rewrite.conf

2\. 你可以采取任何设备和不同操作系统服务器设计对两台 WWW 服务器和两台 FTP 服务器做负载均衡，用网络拓扑图表示并加以说明！（方法越多

越好）

第一种方法: DNS 轮巡 www1 IN A 192.168.1.1 www2 IN A 192.168.1.2 www3 IN A 192.168.1.3 ftp1 IN A 192.1.1.4

ftp2 IN A 192.1.1.5

ftp3 IN A 192.1.1.6

www IN CNAME www1 www IN CNAME www2 www IN CNAME www3 ftp IN CNAME ftp1

ftp IN CNAME ftp2 ftp IN CNAME ftp3

＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝题空面试题＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

＝＝＝＝＝＝＝＝＝＝＝

Linux 面试题一．填空题：

1. 在 Linux 系统 中，以 文件 方式访问设备 。
2. Linux 内核引导时，从文件/etc/fstab 中读取要加载的文件系统 。
3. Linux 文件系统中每个文件用 i 节点 来标识。
1. 全部磁盘块由四个部分组成，分别为引导块 、专用块 、 i 节点表块 和 数据 存储块 。
2. 链接分为：硬链接 和 符号链接 。
3. 超级块包含了i 节点表 和 空闲块表 等重要的文件系统信息。
4. 某文件的权限为：d-rw-\_r--\_r--，用数值形式表示该权限，则该八进制数为：644 ， 该文件属性是目录 。
      1. 前台起动的进程使用 Ctrl+c 终止。
      2. 静态 路由 设定后，若 网络 拓扑结构发生变化，需由系统 管理 员 修改路由的 设 置 。
         1. 网络管理的重要任务是：控制 和 监控 。
         2. 安装 Linux 系统对硬盘分区时，必须有两种分区类型：文件系统分区 和 交换分区 。
         3. 编写的 Shell 程序运行前必须赋予该脚本文件执行 权限。
         4. 系统管理的任务之一是能够在分布式 环境中实现对程序和数据的安全保护、备份、恢复和更新。
            1. 系统交换分区是作为系统虚拟存储器 的一块区域 。
            2. 内核分为进程管理系统 、 内存管理系统 、 I/O 管理系统 和文件管理系统 等四个子系统。
               1. 内核配置是系统管理员在改变系统配置硬件 时要进行的重要操作。
               2. 在安装 Linux 系统中，使用 netconfig 程序对网络进行配置，该安装程序会一步步提 示用户 输入主机名、域名、域名 服务 器 、IP 地址、网关地址 和子网掩码 等必要信息。
                  1. 唯一标识每一个用户的是用户 ID 和用户名 。

20 .RIP 协议 是最为普遍的一种内部协议，一般称为动态路由信息协议 。

1. 在 Linux 系统中所有内容都被表示为文件，组织文件的各种方法称为文件系统 。
2. DHCP 可以实现动态 IP 地址分配。
1. 系统网络管理员的管理对象是服务器 、用户 和服务器的进程 以及系统的各种资源。
2. 网络管理通常由监测、传输和管理 三部分组成，其中管理部分是整个网络管理的中心。 25.当想删除本系统用不上的设备驱动程序 时必须编译内核，当内核不支持系统上的设备驱动程序 时，必须对内核升级 。

26 Ping 命令可以测试网络中本机系统是否能到达一台远程主机 ，所以常常用于测试网络的 连通性 。

1. vi 编辑器具有两种工作模式： 命令 模式 和 输入模式 。
1. 可以用 ls –al 命令来观察文件的权限，每个文件的权限都用 10 位表示，并分为四段，

其中第一段占 1 位，表示 文件类型 ，第二段占 3 位，表示文件所有者 对该文件的权限。

1. 进程与程序的区别在于其动态性，动态的产生和终止，从产生到终止进程可以具有的 基本状态为：运行态 、 就绪态 和 等待态（阻塞态） 。

30\. DNS 实际上是分布在 internet 上的主机信息的 数据库 ，其作用是实现 IP 地址和主 机名 之间的转换。

1. Apache 是实现 WWW 服务器功能 的 应用 程序，即通常所说的“浏览 web 服务器”，在服务器端为用户提供浏览 web 服务 的就是 apache 应用程序。

32.在 Linux 系统上做备份可以有两种类型：系统备份 和用户备份 。其中前者是指对 操 作系统 的备份，后者是指对应用程序和用户文件的备份 。

1. CD-ROM 标准的文件系统类型是 iso9660 。
1. 当 lilo.conf 配置完毕后，使之生效，应运行的命令及参数是 lilo 。
2. 在使用 ls 命令时，用八进制形式显示非打印字符应使用参数-b 。
3. Linux 使用支持 Windows 9.x/2000 长文件名的文件系统的类型是 vfat 。
      1. 设定限制用户使用磁盘 空间 的命令是 quota 。

38 在 Linux 系统中，用来存放系统所需要的配置文件和子目录的目录是/etc 。

1. 硬连接只能建立对文件 链接。符号链接可以跨不同文件系统创建。
2. 套接字文件的属性位是 s 。
3. 结束后台进程的命令是 kill 。
4. 进程的运行有两种方式，即独立运行和使用父进程运行 。
5. Links 分为硬链接和符号链接 。
1. 在超级用户下显示Linux 系统中正在运行的全部进程，应使用的命令及参数是ps -aux 。
2. 管道文件的属性位是 p 。
3. 将前一个命令的标准输出作为后一个命令的标准输入，称之为管道 。
4. 为脚本程序指定执行权的命令及参数是 chmod a+x filename 。
5. 进行远程登录的命令是 telnet 。
6. 欲发送 10 个分组报文测试与主机 abc.tuu.edu.cn 的连通性，应使用的命令和参数是：

ping abc.tuu.edu.cn –c 10 。

1. DNS 服务器的进程命名为 named，当其启动时，自动装载 /etc 目录下的

named.conf 文件中定义的 DNS 分区数据库文件。

1. Apache 服务器进程配置文件是 httpd.conf 。

52.在 Linux 系统中，压缩文件后生成后缀为.gz 文件的命令是 gzip 。

1. 在用 vi 编辑文件时，将文件内容存入 test.txt 文件中，应在命令模式下键入 ：w test.txt 。

54 可以在标准输出上显示整年日历的命令及参数是 cal -y 。

1. 在 shell 编程时，使用方括号表示测试条件的规则是：方括号两边必须有空格 。
2. 检查已安装的文件系统/dev/had5 是否正常，若检查有错，则自动修复，其命令及参 数是 fsck –a /dev/had 5 。
1. 在 Windows9.x 环境下共享 Unix/Linux 中的用户目录的一个工具是 Samba 服务器 。
2. 系统管理员的职责是进行系统资源管理、系统性能管理、设备管理、安全管理和系统 性能监测 。

59 在 Linux 系统中，测试 DNS 服务器是否能够正确解析 域名的的客户端命令，使用命 令

nslookup 。

1. 在 Linux 系统下，第二个 IDE 通道的硬盘（从盘）被标识为 hdb 。
2. 当系统管理员需升级内核版本和改变系统硬件配置时，应重新编译内核 。
3. 如果只是要修改系统的 IP 地址，应修改/etc/rc.d/rc.inet1 配置文件。
4. 当 LAN 内没有条件建立 DNS 服务器，但又想让局域网内的用户可以使用计算机名互 相访问时，应配置/etc/hosts 文件。
1. 在vi 编辑环境下，使用 Esc 键 进行模式转换。
2. Slackware Linux 9.0 通常使用 ext3 文件系统，系统的全部磁盘块由四 部分组成。
3. 将/home/stud1/wang 目录做归档压缩，压缩后生成 wang.tar.gz 文件，并将此文 件保存到/home 目录下，实现此任务的tar 命令格式tar zcvf /home/wang.tar.gz /home/stud1/wang 。
      1. 管道就是将前一个命令的 标准输出 作为后一个命令的标准输入 。
      2. 在使用手工的方法配置网络时，可通过修改/etc/HOSTNAME 文件来改变主机名，若 要配置该计算机的域名解析客户端，需配置/etc/resolv.conf 文件。
         1. 启动进程有手动启动和调度启动两种方法，其中调度启动常用的命令为 at 、 batch 和

crontab 。

70\. test.bns.com.cn 的域名是 bns.com.cn ，如果要配置一域名服务器，应在 named.conf 文件中定义 DNS 数据库的工作目录。

71\. Sendmail 邮件系统使用的两个主要协议是：SMTP 和 POP ，前者用来发送邮件,后 者用来接收邮件。

1. DHCP 是动态主机配置协议的简称，其作用是：为网络中的主机分配 IP  地址 。
1. 目前代理服务器使用的软件包有很多种，教材中使用的是 squid 。
2. rm 命令可删除文件或目录，其主要差别就是是否使用递归开关-r 或-R 。

75.mv 命令可以移动文件和目录，还可以为文件和目录重新命名。

1. 路由选择 协议（RIP）的跳数表示到达目的地之前必须通过的网关 数，RIP 接受的最 长距离是 15 跳 。

77\. ping 命令用于测试网络的连通性，ping 命令通过 ICMP 协议（internet 控制信息协 议）来实现。

78.nfs 协议 用于实现 Unix （/linux）主机之间的文件系统共享。

1. 在 Linux 操作系统中，设备都是通过特殊的文件 来访问。

80\. shell 不仅是用户命令的解释器 ，它同时也是一种功能强大的编程语言。 bash 是 Linux

的缺省 shell。

1. 用>;>; 符号将输出重定向内容附加在原文的后面。
2. 增加一个用户的命令是：adduser 或 useradd 。

83 进行字符串查找，使用 grep 命令。

1. 使用\\* 每次匹配若干个字符。

85./sbin 目录用来存放系统管理员使用的管理程序。二．单项选择题:

1. 下面的网络协议中，面向连接的的协议是： A 。

A 传输控制协议 B 用户数据报协议 C 网际协议

D 网际控制报文协议

1. 在/etc/fstab 文件中指定的文件系统加载参数中，D 参数一般用于CD-ROM 等移动设 备。

A defaults B sw

C rw 和 ro D noauto

1. Linux 文件权限一共 10 位长度，分成四段，第三段表示的内容是 C 。

A    文件类型

B    文件所有者的权限

C    文件所有者所在组的权限

D 其他用户的权限

1. 终止一个前台进程可能用到的命令和操作 B 。

A kill

B <CTRL>;+C

C shut down D halt

1. 在使用 mkdir 命令创建新的目录时，在其父目录不存在时先创建父目录的选项是 D 。

A -m

B -d

C -f

D -p

1. 下面关于 i 节点描述错误的是 A 。（inode 是一种数据结构，vfs 中描述文件的相关参数？？）

A i 节点和文件是一一对应的

B i 节点能描述文件占用的块数

C i 节点描述了文件大小和指向数据块的指针

D 通过 i 节点实现文件的逻辑结构和物理结构的转换

1. 一个文件名字为rr.Z，可以用来解压缩的命令是： D 。

A tar B gzip C compress D uncompress

1. 具有很多 C 语言的功能，又称过滤器的是 C 。

A csh B tcsh

C awk （awk 详解 ）

D sed

1. 一台主机要实现通过局域网与另一个局域网通信，需要做的工作是 C 。

A    配置域名服务器

B    定义一条本机指向所在网络的路由

C    定义一条本机指向所在网络网关的路由

D 定义一条本机指向目标网络网关的路由

1. 建立动态路由需要用到的文件有 D 。

A /etc/hosts

B /etc/HOSTNAME C /etc/resolv.conf D /etc/gateways

1. 局域网的网络地址 192.168.1.0/24，局域网络连接其它网络的网关地址是 192.168.1.1。主机 192.168.1.20 访问 172.16.1.0/24 网络时，其路由设置正确的是 B 。

A route add –net 192.168.1.0 gw 192.168.1.1 netmask 255.255.255.0 metric 1

B route add –net 172.16.1.0 gw 192.168.1.1 netmask 255.255.255.255 metric 1

C route add –net 172.16.1.0 gw 172.16.1.1 netmask 255.255.255.0 metric 1

D route add default 192.168.1.0 netmask 172.168.1.1 metric 1

1. 下列提法中，不属于 ifconfig 命令作用范围的是 D 。

A 配置本地回环地址 B 配置网卡的 IP 地址 C 激活网络适配器

D 加载网卡到内核中

1. 下列关于链接描述，错误的是 B 。

A    硬链接就是让链接文件的 i 节点号指向被链接文件的 i 节点

B    硬链接和符号连接都是产生一个新的 i 节点

C    链接分为硬链接和符号链接

D 硬连接不能链接目录文件

1. 在局域网络内的某台主机用 ping 命令测试网络连接时发现网络内部的主机都可以连同，而不能与公网连通，问题可能是 C。

A    主机 IP 设置有误

B    没有设置连接局域网的网关

C    局域网的网关或主机的网关设置有误

D 局域网 DNS 服务器设置有误

1. 下列文件中，包含了主机名到 IP 地址的映射关系的文件是： B 。

A /etc/HOSTNAME B /etc/hosts

C /etc/resolv.conf D /etc/networks

1. 不需要编译内核的情况是 D 。 A 删除系统不用的设备驱动程序时 B 升级内核时

C 添加新硬件时

D 将网卡激活

1. 在 shell 中变量的赋值有四种方法，其中，采用 name=12 的方法称 A 。

A    直接赋值

B    使用 read 命令 C 使用命令行参数 D 使用命令的输出

1. D 命令可以从文本文件的每一行中截取指定内容的数据。

A cp B dd C fmt D cut

1. 下列不是 Linux 系统进程类型的是 D 。

A 交互进程 B 批处理进程 C 守护进程

D 就绪进程（进程状态）

1. 配置 Apache 1.3.19 服务器需要修改的配置文件为   A           A httpd.conf

B access.conf C srm.conf

D named.conf

1. 内核不包括的子系统是 D 。

A 进程管理系统 B 内存管理系统 C I/O 管理系统 D 硬件管理系统

1. 在日常管理中，通常 CPU 会影响系统性能的情况是： A 。

A CPU 已满负荷地运转 B CPU 的运行效率为 30% C CPU 的运行效率为 50% D CPU 的运行效率为 80%

1. 若一台计算机的内存为 128MB，则交换分区的大小通常是 C 。

A 64MB B 128MB C 256MB D 512MB

1. 在安装 Linux  的过程中的第五步是让用户选择安装方式，如果用户希望安装部分组件

（软件程序），并在选择好后让系统自动安装，应该选择的选项是 D 。

A full

B expert C newbie D menu

1. Linux 有三个查看文件的命令，若希望在查看文件内容过程中可以用光标上下移动来查看文件内容，应使用 C 命令。

A cat

B more C less

D menu

1. 下列信息是某系统用 ps –ef 命令列出的正在运行的进程， D 进程是运行Internet 超级服务器，它负责监听 Internet sockets 上的连接，并调用合适的服务器来处理接收的 信息。 A root 1 4.0 0.0 344 204? S 17:09 0:00 init

B root 2 0.0 0.1 2916 1520? S 17:09 0:00 /sbin/getty

C root 3 0.0 0.2 1364 632? S 17:09 0:00 /usr/sbin/syslogd

D root 4 0.0 1344 1204? S 17:09 0:10 /usr/sbin/inetd

1. 在 TCP/IP 模型中，应用层包含了所有的高层协议，在下列的一些应用协议中， B 是 能够实现本地与远程主机之间的文件传输工作。

A telnet B FTP

C SNMP D NFS

1. 当我们与某远程网络连接不上时，就需要跟踪路由查看，以便了解在网络的什么位置 出现了问题，满足该目的的命令是 C 。

A ping

B ifconfig

C traceroute D netstat

1. 对名为 fido 的文件用chmod 551 fido 进行了修改，则它的许可权是 D 。

A -rwxr-xr-x B -rwxr--r--

C -r--r--r--

D -r-xr-x—x

1. 在 i 节点表中的磁盘地址表中，若一个文件的长度是从磁盘地址表的第 1 块到第 11

块，则该文件共占有 B 块号。

A 256

B 266

C 11

D 256×10

1. 用 ls –al 命令列出下面的文件列表， D 文件是符号连接文件。

A -rw-rw-rw- 2 hel-s users 56 Sep 09 11:05 hello

B -rwxrwxrwx 2 hel-s users 56 Sep 09 11:05 goodbey

C drwxr--r-- 1 hel users 1024 Sep 10 08:10 zhang

Dl rwxr--r-- 1 hel users 2024 Sep 12 08:12 cheng

1. DNS 域名系统主要负责主机名和 A 之间的解析。

A IP 地址

B MAC 地址 C 网络地址 D 主机别名

1. WWW 服务器是在 Internet 上使用最为广泛，它采用的是 B 结构。

A 服务器/工作站

B B/S

C 集中式

D 分布式

1. Linux 系统通过 C 命令给其他用户发消息。

A less

B mesg y C write

D echo to

\[ 注：mesg \[y\|n\] 所有使用者 决定是否允许其他人传讯息到自己的终端机介面 \] 35．NFS 是 C 系统。

A    文件

B    磁盘

C    网络文件

D 操作

1. B  命令可以在 Linux  的安全系统中完成文件向磁带备份的工作。

A cp B tr

C dir D cpio

\[注：如果用 echo $PATH 或者 echo $LD\_LIBRARY\_PATH 等类似的命令来显示路径\
\
信息的话，我们看到的将会是一大堆用冒号连接在一起的路径， tr 命令可以把这些冒号转换为回车，这样，这些路径就具有很好的可读性了：\
\
echo $PATH \| tr ":" "\\n" \]

1. Linux 文件系统的文件都按其作用分门别类地放在相关的目录中，对于外部设备文件，一般应将其放在 C 目录中。

A /bin B /etc C /dev D /lib

1. 在重新启动 Linux 系统的同时把内存中的信息写入硬盘，应使用 D 命令实现。

A # reboot B # halt

C # reboot

D # shutdown –r now

1. 网络管理具备以下几大功能：配置管理、 A 、性能管理、安全管理和计费管理等

A    故障 管理

B    日常备份管理

C    升级管理

D 发送邮件

1. 关于代理服务器的论述，正确的是 A 。

A    使用 internet 上已有的公开代理服务器，只需配置客户端。

B    代理服务器只能代理客户端 http 的请求。

C    设置好的代理服务器可以被网络上任何主机使用。 D 使用代理服务器的客户端没有自己的 ip 地址。 41.关闭 linux 系统（不重新启动）可使用命令 B 。 A Ctrl+Alt+Del

B halt

C shutdown -r now D reboot

1. 实现从 IP 地址到以太网 MAC 地址转换的命令为： C 。

A ping

B ifconfig C arp

D traceroute

1. 在vi 编辑器中的命令模式下，键入 B 可在光标当前所在行下添加一新行。

A <a>; B <o>; C <I>; D A

1. 在vi 编辑器中的命令模式下，删除当前光标处的字符使用 A 命令。

A <x>;

B <d>;<w>; C <D>;

D <d>;<d>;

1. 在 vi 编辑器中的命令模式下，重复上一次对编辑的文本进行的操作，可使用 C 命令。

A    上箭头 B 下箭头 C <.>;

D <\*>;

1. 用命令 ls -al 显示出文件 ff 的描述如下所示，由此可知文件 ff 的类型为 A 。 -rwxr-xr-

\- 1 root root 599 Cec 10 17:12 ff A 普通文件

B    硬链接

C    目录

D 符号链接

1. 删除文件命令为： D 。

A mkdir B rmdir C mv

D rm

1. 在下列的名称中，不属于 DNS 服务器类型的是：     C         A Primary Master Server

B Secondary Master Server C samba

D Cache\_only Server

1. 网络管理员对 WWW 服务器进行访问、控制存取和运行等控制，这些控制可在 A 文 件中体现。

A httpd.conf B lilo.conf

C inetd.conf D resolv.conf

1. 邮件转发代理也称邮件转发服务器，它可以使用 SMTP 协议，也可以使用 C 协议。

A FTP B TCP

C UUCP D POP

1. 启动 samba 服务器进程，可以有两种方式：独立启动方式和父进程启动方式，其中 前者是在 C 文件中以独立进程方式启动。

A /usr/sbin/smbd B /usr/sbin/nmbd Crc.samba

D /etc/inetd.conf

1. DHCP 是动态主机配置协议的简称，其作用是可以使网络管理员通过一台服务器来管 理一个网络系统，自动地为一个网络中的主机分配                                 D           地址。

A 网络

B MAC C TCP D IP

1. 为了保证在启动服务器时自动启动 DHCP 进程，应将 A 文件中的 dhcpd=no 改为

dhcpd=yes。 Arc.inet1

B lilo.conf

C inetd.conf D httpd.conf

\[注：英文原义：RC\
\
中文释义：含有程序（应用程序甚至操作系统）启动指令的脚本文件\
\
注解：这一文件在操作系统启动时会自动执行，它含有要运行的指令（命令或其它脚本）列表。\]

1. 对文件进行归档的命令为 D 。

A dd B cpio C gzip D tar

1. 改变文件所有者的命令为 C 。

A chmod B touch C chown D cat

1. 在给定文件中查找与设定条件相符字符串的命令为： A 。

A grep B gzip C find D sort

1. 建立一个新文件可以使用的命令为 D 。

A chmod B more C cp

D touch (指令改变档案的时间记录。)

1. 在下列命令中，不能显示文本文件内容的命令是： D 。

A more B less C tail

D join

1. 在使用匿名登录 ftp 时，用户名为 B 。

A users

B anonymous C root

D guest

1. 在实际操作中，想了解命令 logname 的用法，可以键入 D 得到帮助。

A logname --man B logname/？

C help logname

D logname --help 6

1．如果 LILO 被安装在 MBR，使用 A 命令即可卸载 LILO。 A lilo –u

B lilo –c C lilo –v D lilo -V

1. 当用命令 ls –al 查看文件和目录时，欲观看卷过屏幕的内容，应使用组合键 D 。

A Shift+Home B Ctrl+ PgUp C Alt+ PgDn D Shift+ PgUp

1. mc 是 UNIX 风格操作系统的 C 。

A    文件编辑器/程序编译器

B    配置网络的窗口工具

C    目录浏览器/文件管理器

D Samba 服务器管理工具

64．i 节点是一个 D 长的表 ，表中包含了文件的相关信息。

A 8 字节

B 16 字节

C 32 字节

D 64 字节

1. 文件权限读、写、执行的三种标志符号依次是 A 。

A rwx B xrw C rdx D srw

1. Linux 文件名的长度不得超过 C 个字符。

A 64

B 128

C 256

D 512

1. 进程有三种状态： C 。 A 准备态、执行态和退出态 B 精确态、模糊态和随机态 C 运行态、就绪态和等待态 D 手工态、自动态和自由态
1. 从后台启动进程，应在命令的结尾加上符号 A 。

A & B @ C #

D $

1. B 不是邮件系统的组成部分。

A 用户代理 B 代理服务器 C 传输代理 D 投递代理

1. 在 Shell 脚本中，用来读取文件内各个域的内容并将其赋值给 Shell 变量的命令是 D 。

A fold B join C tr

D read

1. crontab 文件由六个域组成 ，每个域之间用空格分割，其排列如下： B 。

A MIN HOUR DAY MONTH YEAR COMMAND

B MIN HOUR DAY MONTH DAYOFWEEK COMMAND C COMMAND HOUR DAY MONTH DAYOFWEEK

D COMMAND YEAR MONTH DAY HOUR MIN

crontab 命令：实现程序定时运行

1. 用 ftp 进行文件传输时，有两种模式： C 。

A Word 和 binary

B .txt 和 Word Document C ASCII 和 binary

D ASCII 和 Rich Text Format

1. 某文件的组外成员的权限为只读；所有者有全部权限；组内的权限为读与写，则该文 件的权限为 D 。

A 467

B 674

C 476

D 764

1. 在 DNS 系统测试时，设 named 进程号是 53，命令 D 通知进程重读配置文件。

A kill –USR2 53 B kill –USR1 53 C kill -INT 63

D kill –HUP 53

1. Apache 服务器默认的接听连接端口号是 C 。

A 1024

B 800

C 80 (http)

D 8

1. PHP 和 MySQL 的联合使用 解决 了 C 。

A    在 Proxy 上处理数据库的访问问题

B    在 WWW 服务器上处理黑客的非法访问问题

C    在 WWW 服务器上处理数据库的访问问题

D 在 Sendmail 邮件系统上处理数据库的访问问题

1. OpenSSL 是一个 A 。

A    加密软件

B    邮件系统

C    数据库管理系统

D 嵌入式脚本编程语言

1. Samba 服务器的配置文件是 D 。

A httpd.conf B inetd.conf C rc.samba D smb.conf

1. 关于 DNS 服务器，叙述正确的是 D 。

A DNS 服务器配置不需要配置客户端

B 建立某个分区的DNS 服务器时只需要建立一个主 DNS 服务器 C 主 DNS 服务器需要启动 named 进程，而辅 DNS 服务器不需要 D DNS 服务器的root.cache 文件包含了根名字服务器的有关信息 80．退出交互模式的shell，应键入 C 。

A <Esc>;

B ^q C exit D quit

81 ．将 Windows C: 盘(hda1) 安装在 Linux 文件系统的/winsys 目录下， 命令是 B 。

Aroot@l04.edu.cn:~#mount dev/had1 /winsys Broot@l04.edu.cn:~#mount /dev/had1 /winsys Croot@l04.edu.cn:~#mount /dev/had1 winsys Droot@l04.edu.cn:~#mount dev/had1 winsys

1. 设超级用户root 当前所在目录为：/usr/local，键入 cd 命令后，用户当前所在目录 为

B 。

A /home B /root

C /home/root D /usr/local

1. 字符设备文件类型的标志是 B 。

A p B c C s D l

1. 将光盘CD-ROM（hdc）安装到文件系统的/mnt/cdrom 目录下的命令是 C 。

A mount /mnt/cdrom

B mount /mnt/cdrom /dev/hdc C mount /dev/hdc /mnt/cdrom D mount /dev/hdc

1. 将光盘/dev/hdc 卸载的命令是 C 。

A umount /dev/hdc B unmount /dev/hdc

C umount /mnt/cdrom /dev/hdc

D unmount /mnt/cdrom /dev/hdc

1. 在/home/stud1/wang 目录下有一文件 file，使用 D 可实现在后台执行命令，此命 令将

file 文件中的内容输出到 file.copy 文件中。

A cat file >;file.copy B cat >;file.copy

C cat file file.copy & D cat file >;file.copy &

1. 在 DNS 配置文件中，用于表示某主机别名的是： B 。

A NS

B CNAME C NAME D CN

1. 可以完成主机名与IP 地址的正向解析和反向解析任务的命令是： A 。

Anslookup B arp

C ifconfig D dnslook

1. 下列变量名中有效的 shell 变量名是： C 。

A -2-time B \_2$3

C trust\_no\_1 D 2004file

1. qmail 是 B 。 A 收取邮件的协议 B 邮件服务器的一种 C 发送邮件的协议 D 邮件队列
1. 已知某用户 stud1，其用户目录为/home/stud1。分页显示当前目录下的所有文件的

文件或目录名、用户组、用户、文件大小、文件或目录权限、文件创建时间等信息的命令是

D 。

A more ls –al B more –al ls C more < ls –al D ls –al \| more

1. 关于进程调度命令， B 是不正确的。at--定期执行程序的调度命令

A    当日晚 11 点执行 clear 命令，使用 at 命令：at 23:00 today clear

B    每年 1 月 1 日早上 6 点执行 date 命令，使用 at 命令：at 6am Jan 1 date C 每日晚 11 点执行 date 命令，crontab 文件中应为：0 23 \* \* \* date

D 每小时执行一次clear 命令，crontab 文件中应为：0 \*/1 \* \* \* clear

1. 系统中有用户 user1 和 user2，同属于 users 组。在 user1 用户目录下有一文件 file1，它拥有 644 的权限，如果 user2 用户想修改 user1 用户目录下的 file1 文件，应拥有 B 权限。

A 744

B 664

C 646

D 746

1. 如果想配置一台匿名ftp 服务器，应修改 C 文件。

A /etc/gateway B /etc/ftpservers C /etc/ftpusers

D /etc/inetd.conf

1. Samba 服务器的进程由 B 两部分组成 。

A named 和 sendmail Bsmbd 和 nmbd

C bootp 和 dhcpd D httpd 和 squid

1. 要配置NFS 服务器，在服务器端主要配置 C 文件。

A /etc/rc.d/rc.inet1 B /etc/rc.d/rc.M

C /etc/exports D /etc/rc.d/rc.S

1. 为保证在启动服务器时自动启动 DHCP 进程，应对 B 文件进行编辑。

A /etc/rc.d/rc.inet2 B /etc/rc.d/rc.inet1 C /etc/dhcpd.conf D /etc/rc.d/rc.S

1. 在配置代理服务器时，若设置代理服务器的工作缓存为 64MB，配置行应为 D 。

A cache 64MB

B cache\_dir ufs /usr/local/squid/cache 10000 16 256 C cache\_ mgr 64MB

Dcache\_ mem 64MB

1. 安全管理涉及的问题包括保证网络管理工作可靠进行的安全问题和保护网络用户及网络管理对象问题。 C 属于安全管理的内容。

A    配置设备的工作参数

B    收集与网络性能有关的数据

C    控制和维护访问权限

D 监测故障

1. 以下命令对中，正确的是： B 。

A ls 和 sl

B cat 和 tac

C more 和erom D exit 和 tixe

cat 是显示文件夹的命令，这个大家都知道，tac 是 cat 的倒写，意思也和它是相反的。cat是从第一行显示到最后一行，而 tac 是从最后一行显示到第一行，而 rev 则是从最后一个字符显示到第一个字符

1. B 命令是在vi 编辑器中执行存盘退出。

A :q B ZZ

C :q! D :WQ

1. 下列关于/etc/fstab 文件描述，正确的是 D 。

A fstab 文件只能描述属于linux 的文件系统

B CD\_ROM 和软盘必须是自动加载的

C fstab 文件中描述的文件系统不能被卸载

D 启动时按 fstab 文件描述内容加载文件系统

1. 通过文件名存取文件时，文件系统内部的操作过程是通过 C 。

A    文件在目录中查找文件数据存取位置。

B    文件名直接找到文件的数据，进行存取操作。

C    文件名在目录中查找对应的 I 节点，通过 I 节点存取文件数据。

D    文件名在中查找对应的超级块，在超级块查找对应 i 节点，通过 i 节点存取文件数据 105．Linux 将存储设备和输入/输出设备均看做文件来操作， C 不是以文件的形式出现。 A 目录

B 软链接

C i 节点表

D 网络适配器

1. 关于 i 节点和超级块，下列论述不正确的是 B 。

A i 节点是一个长度固定的表

B 超级块在文件系统的个数是唯一的

C i 节点包含了描述一个文件所必需的全部信息

D 超级块记录了 i 节点表和空闲块表信息在磁盘中存放的位置

1. D 设备是字符设备。

A hdc B fd0 C hda1

D tty1(A,B,C 为块设备)

1. B 目录存放着 Linux 的源代码。

A /etc

B /usr/src C /usr

D /home

1. 关于文件系统的安装和卸载，下面描述正确的是 A 。

A    如果光盘未经卸载，光驱是打不开的

B    安装文件系统的安装点只能是/mnt 下

C    不管光驱中是否有光盘，系统都可以安装 CD-ROM  设备

D mount /dev/fd0 /floppy 此命令中目录/floppy 是自动生成的

1. B 不是进程和程序的区别。

A    程序是一组有序的静态指令，进程是一次程序的执行过程

B    程序只能在前台运行，而进程可以在前台或后台运行

C    程序可以长期保存，进程是暂时的

D 程序没有状态，而进程是有状态的

1. 文件exer1 的访问权限为 rw-r--r--，现要增加所有用户的执行权限和同组用户的写 权限，下列命令正确的是 A 。

A chmod a+x g+w exer1 B chmod 765 exer1 C chmod o+x exer1 D chmod g+w exer1

1. 有关归档和压缩命令，下面描述正确的是 C 。

A 用 uncompress 命令解压缩由 compress 命令生成的后缀为.zip 的压缩文件

B unzip 命令和 gzip 命令可以解压缩相同类型的文件

C tar 归档且压缩的文件可以由 gzip 命令解压缩 D tar 命令归档后的文件也是一种压缩文件 113．不是 shell 具有的功能和特点的是 C 。

A 管道 B 输入输出重定向 C 执行后台进程 D 处理程序命令

114．下列对 shell 变量 FRUIT 操作，正确的是： C 。

A    为变量赋值：$FRUIT=apple B 显示变量的值：fruit=apple C 显示变量的值：echo $FRUIT

D 判断变量是否有值：\[ -f “$FRUIT” \]

三．简答题：

1. 简述 Linux 文件系统通过 i 节点把文件的逻辑结构和物理结构转换的工作过程。 参考答案：

Linux 通过 i 节点表将文件的逻辑结构和物理结构进行转换。

i 节点是一个 64 字节长的表，表中包含了文件的相关信息，其中有文件的大小、文件所有者、文件的存取许可方式以及文件的类型等重要信息。在 i 节点表中最重要 的内容是磁盘地址表 。在磁盘地址表中有 13 个块号，文件将以块号在磁盘地址表中出现的顺序依次读取相应的块。Linux 文件系统通过把 i 节点和文件名进行 连接，当需要读取该文件时，文件系统在当前目录表中查找该文件名对应的项，由此得到该文件相对应的 i 节点号，通过该 i 节点的磁盘地址表把分散存放的文件物 理块连接成文件的逻辑结构。

1. 简述进程的启动、终止的方式以及如何进行进程的查看。参考答案：

在 Linux  中启动一个进程有手工启动和调度启动两种方式：

（1） 手工启动

用户在输入端发出命令，直接启动一个进程的启动方式。可以分为：

①前台启动：直接在 SHELL 中输入命令进行启动。

②后台启动：启动一个目前并不紧急的进程，如打印进程。

（2） 调度启动

系统管理员根据系统资源和进程占用资源的情况，事先进行调度安排，指定任务运行的时间和场合，到时候系统会自动完成该任务。

经常使用的进程调度命令为：at、batch、crontab。

1. 简述 DNS 进行域名解析的过程。参考答案：

首先，客户端发出DNS  请求翻译 IP 地址或主机名。DNS 服务器在收到客户机的请求后：

（1） 检查 DNS  服务器的缓存，若查到请求的地址或名字，即向客户机发出应答信息；

（2） 若没有查到，则在数据库中查找，若查到请求的地址或名字，即向客户机发出应答信息；

（3） 若没有查到，则将请求发给根域 DNS 服务器，并依序从根域查找顶级域，由顶级查 找二级域，二级域查找三级，直至找到要解析的地址或名字，即向客户机所在网络的 DNS 服务器发出应答信息，DNS 服务器收到应答后现在缓存中存储，然后，将解析结果发给客 户

机。

（4） 若没有找到，则返回错误信息。

1. 系统管理员的职责包括那些？管理的对象是什么？参考答案：

系统管理员的职责是进行系统资源管理、设备管理、系统性能管理、安全管理和系统性能监测。管理的对象是服务器、用户、服务器的进程及系统的各种资源等。

1. 简述安装Slackware Linux 系统的过程。参考答案：

（1）对硬盘重新分区。 （2）启动 Linux 系统（用光盘、软盘等）。

（3）建立 Linux 主分区和交换分区。（4）用 setup 命令安装 Linux 系统。

（5）格式化 Linux 主分区和交换分区（6）安装 Linux 软件包

（7）安装完毕，建立从硬盘启动 Linux 系统的 LILO 启动程序，或者制作一张启动 Linux 系统的软盘。重新启动 Linux 系统。

1. 什么是静态路由，其特点是什么？什么是动态路由，其特点是什么？参考答案：

静态路由是由系统管理员设计与构建的路由表规定的路由。适用于网关数量有限的场合，且网络拓朴结构不经常变化的网络。其缺点是不能动态地适用网络状况的变化，当网络状况变化后必须由网络管理员修改路由表。

动态路由是由路由选择协议而动态构建的，路由协议之间通过交换各自所拥有的路由信息实时更新路由表的内容。动态路由可以自动 学习 网络的拓朴结构，并更新路由表。其缺点是路由广播更新信息将占据大量的网络带宽。

87．进程的查看和调度分别使用什么命令？参考答案：

进程查看的命令是 ps 和 top。

进程调度的命令有 at，crontab，batch，kill。

8．当文件系统受到破坏时，如何检查和修复系统？参考答案：

成功修复文件系统的前提是要有两个以上的主文件系统，并保证在修复之前首先卸载将被修复的文件系统。

使用命令fsck 对受到破坏的文件系统进行修复。fsck 检查文件系统分为 5 步，每一步检查系统不同部分的连接特性并对上一步进行验证和修改。在执行 fsck 命令时，检查首先从超级块开始，然后是分配的磁盘块、路径名、目录的连接性、链接数目以及空闲块链表、i-node。 9．解释 i 节点在文件系统中的作用。

参考答案：

在 linux 文件系统中，是以块为单位存储信息的，为了找到某一个文件在存储空间中存放的位置，用 i 节点对一个文件进行索引。I 节点包含了描述一个文件所必须的全部信息。所以 i 节点是文件系统管理的一个数据结构。

1. 什么是符号链接，什么是硬链接？符号链接与硬链接的区别是什么？参考答案：

链接分硬链接和符号链接。

符号链接可以建立对于文件和目录的链接。符号链接可以跨文件系统，即可以跨磁盘分区。符号链接的文件类型位是 l，链接文件具有新的i 节点。

硬链接不可以跨文件系统。它只能建立对文件的链接，硬链接的文件类型位是－，且硬链接文件的 i 节点同被链接文件的 i 节点相同。

1. 在对 linux 系统分区进行格式化时需要对磁盘簇（或 i 节点密度）的大小进行选择，请说明选择的原则。

参考答案：

磁盘簇（或 i 节点密度）是文件系统调度文件的基本单元。磁盘簇的大小，直接影响系统调度磁盘空间效率。当磁盘分区较大时，磁盘簇也应选得大些；当分区较小时，磁盘簇应选得小些。通常使用经验值。

1

2．简述网络文件系统NFS，并说明其作用。参考答案：

网络文件系统是应用层的一种应用服务，它主要应用于 Linux 和 Linux 系统、Linux 和 Unix系统之间的文件或目录的共享。对于用户而言可以通过 NFS 方便的访问远地的文件系统，使之成为本地文件系统的一部分。采用 NFS 之后省去了登录的过程，方便了用户访问系统资源。

1. 某/etc/fstab 文件中的某行如下：

/dev/had5 /mnt/dosdata msdos defaults,usrquota 1 2

请解释其含义。参考答案:

（1）第一列：将被加载的文件系统名；（2）第二列：该文件系统的安装点；

（3）第三列：文件系统的类型；（4）第四列：设置参数；

（5） 第五列：供备份程序确定上次备份距现在的天数；

（6） 第六列：在系统引导时检测文件系统的顺序。

1. Apache 服务器的配置文件 httpd.conf 中有很多内容，请解释如下配置项：

（1）MaxKeepAliveRequests 200 （2）UserDir public\_html

（3）DefaultType text/plain （4）AddLanguare en.en

（5） DocumentRoot“/usr/local/httpd/htdocs”

（6） AddType application/x-httpd-php.php.php.php4

参考答案:

（1）允许每次连接的最大请求数目，此为 200；（2）设定用户放置网页的目录；

（3） 设置服务器对于不认识的文件类型的预设格式；

（4） 设置可传送语言的文件给浏览器；（5）该目录为 Apache 放置网页的地方；

（6）服务器选择使用 php4。

1. 某 Linux 主机的/etc/rc.d/rc.inet1 文件中有如下语句，请修正错误，并解释其内容。

/etc/rc.d/rc.inet1：

??

ROUTE add –net default gw 192.168.0.101 netmask 255.255.0.0 metric 1

ROUTE add –net 192.168.1.0 gw 192.168.0.250 netmask 255.255.0.0 metric 1 参考答案:修正错误:

（1）ROUTE 应改为小写：route；（2）netmask 255.255.0.0 应改为:netmask 255.255.255.0；

（3） 缺省路由的子网掩码应改为:netmask 0.0.0.0；

（4） 缺省路由必须在最后设定,否则其后的路由将无效。解释内容:

（1）route：建立静态路由表的命令；（2）add：增加一条新路由；

（3）-net 192.168.1.0：到达一个目标网络的网络地址；

（4）default：建立一条缺省路由；（5）gw 192.168.0.101：网关地址；

（6）metric 1：到达目标网络经过的 路由器 数（跳数）。

1. 试解释 apache 服务器以下配置的含义：

（1）port 1080 （2）UserDir userdoc

（3） DocumentRoot “/home/htdocs”

（4） <Directory /home/htdocs/inside>; Options Indexes FollowSymLinks AllowOverride None

Order deny,allow deny from all

allow from 192.168.1.5

</Directory>;

（5） Server Type Standlone

参考答案：

Apache 服务器配置行含义如下：

（1） 将 apache 服务器的端口号设定为 1080；

（2） 设定用户网页目录为 userdoc；

（3） 设定 apache 服务器的网页根目录:/home/htdocs；

（4） 在此 apache 服务器上设定一个目录/home/htdocs/inside，且此目录只允许 IP 地 址为

192.168.1.5 的主机访问；

（5） 定义 apache 服务器以独立进程的方式运行。

1. 简述使用 ftp 进行文件传输时的两种登录方式？它们的区别是什么？常用的 ftp 文件传输命令是什么？

参考答案：

（1）                       ftp 有两种登录方式：匿名登录和授权登录。使用匿名登录时，用户名为：anonymous，密码为：任何合法 email 地址；使用授权登录时，用户名为用户在远程系统中的用户帐号，密码为用户在远程系统中的用户密码。

区别：使用匿名登录只能访问 ftp 目录下的资源，默认配置下只能下载；而授权登录访问的权限大于匿名登录，且上载、下载均可。

（2）                       ftp 文件传输有两种文件传输模式：ASCII 模式和 binary 模式。ASCII 模式用来传 输文本文件，其他文件的传输使用 binary 模式。

（3）                       常用的 ftp 文件传输命令为：bin、asc、put、get、mput、mget、prompt、bye四．编程与应用题：

1. 用 Shell 编程，判断一文件是不是字符设备文件，如果是将其拷贝到 /dev 目录下。 参考程序：

#!/bin/sh FILENAME=

echo “Input file name：” read FILENAME

if \[ -c "$FILENAME" \]

then

cp $FILENAME /dev fi

1. 请下列shell 程序加注释，并说明程序的功能和调用方法：#!/bin/sh

#!/bin/sh #

\# /etc/rc.d/rc.httpd #

\# Start/stop/restart the Apache web server. #

\# To make Apache start automatically at boot, make this # file executable: chmod 755 /etc/rc.d/rc.httpd

#

case "$1" in 'start')

/usr/sbin/apachectl start ;; 'stop')

/usr/sbin/apachectl stop ;; 'restart')

/usr/sbin/apachectl restart ;;

\*)

echo "usage $0 start\|stop\|restart" ;; esac

参考答案：

（1） 程序注释

#!/bin/sh 定义实用的shell #

\# /etc/rc.d/rc.httpd 注释行，凡是以星号开始的行均为注释行。

#

\# Start/stop/restart the Apache web server. #

\# To make Apache start automatically at boot, make this # file executable: chmod 755 /etc/rc.d/rc.httpd

#

case "$1" in #case 结构开始，判断“位置参数”决定执行的操作。本程序携带一个“位置 参数”，即$1

'start') #若位置参数为 start

/usr/sbin/apachectl start ;; #启动 httpd 进程

'stop') #若位置参数为 stop

/usr/sbin/apachectl stop ;; #关闭 httpd 进程

'restart') #若位置参数为 stop

/usr/sbin/apachectl restart ;; #重新启动 httpd 进程

\*) #若位置参数不是start、stop 或 restart 时

echo "usage $0 start\|stop\|restart" ;; #显示命令提示信息：程序的调用方法 esac #case 结构结束

（2） 程序的功能是启动，停止或重新启动 httpd 进程

（3） 程序的调用方式有三种：启动，停止和重新启动。

1. 设计一个 shell 程序，添加一个新组为 class1，然后添加属于这个组的 30 个用户，用 户

名的形式为stdxx，其中 xx 从 01 到 30。参考答案：

#!/bin/sh i=1

groupadd class1 while \[ $i -le 30 \] do

if \[ $i -le 9 \] ;then USERNAME=stu0${i}

else USERNAME=stu${i} fi

useradd $USERNAME mkdir /home/$USERNAME

chown -R $USERNAME /home/$USERNAME chgrp -R class1 /home/$USERNAME i=$(($i+1))

done

1. 编写 shell 程序，实现自动删除 50 个账号的功能。账号名为 stud1 至 stud50。 参考程序：

#!/bin/sh i=1

while \[ $i -le 50 \] do

userdel -r stud${i} i=$(($i+1 ))

done

1. 某系统管理员需每天做一定的重复工作，请按照下列要求，编制一个解决 方案 ：

（1） 在下午 4 :50 删除/abc 目录下的全部子目录和全部文件；

（2） 从 8:00～下午 6:00 每小时读取/xyz 目录下 x1 文件中每行第一个域的全部数据 加入到/backup 目录下的 bak01.txt 文件内；

（3）  ） 每逢星期一下午 5:50 将/data 目录下的所有目录和文件归档并压缩为文件：

backup.tar.gz；

（4） 在下午 5:55 将 IDE 接口的 CD-ROM 卸载（假设：CD-ROM 的设备名为 hdc）；

（5） 在早晨 8:00 前开机后启动。参考答案:

解决方案：

（1）用vi 创建编辑一个名为 prgx 的 crontab 文件；

prgx 文件的内容：

50 16 \* \* \* rm -r /abc/\*

（2）、0 8-18/1 \* \* \* cut -f1 /xyz/x1 >;>; /backup/bak01.txt

（3）    、50 17 \* \* \* tar zcvf backup.tar.gz /data

（4）    、55 17 \* \* \* umount /dev/hdc

（5）    、由超级用户登录，用 crontab 执行 prgx 文件中的内容：

root@xxx:#crontab prgx；在每日晨 8:00 之前开机后即可自动启动 crontab。

－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

－－－－－－－

1. 设计一个 shell 程序，在每月第一天备份并压缩/etc 目录的所有内容，存放在/root/bak目录里，且文件名为如下形式 yymmdd\_etc，yy 为年，mm 为月，dd 为日。Shell 程序 fileback存放在/usr/bin 目录下。

参考答案：

（1） 编写 shell 程序 fileback： #!/bin/sh

DIRNAME=\`ls /root \| grep bak\` if \[ -z "$DIRNAME" \] ; then mkdir /root/bak

cd /root/bak fi

YY=\`date +%y\` MM=\`date +%m\` DD=\`date +%d\`

BACKETC=$YY$MM$DD\_etc.tar.gz tar zcvf $BACKETC /etc

echo "fileback finished!"

（2） 编写任务定时器：

echo "0 0 1 \* \* /bin/sh /usr/bin/fileback" >; /root/etcbakcron crontab /root/etcbakcron

或使用 crontab -e 命令添加定时任务：

0 1 \* \* \* /bin/sh /usr/bin/fileback

1. 有一普通用户想在每周日凌晨零点零分定期备份/user/backup 到/tmp 目录下，该用 户应如何做？

参考答案：（1）第一种方法：

用户应使用crontab –e 命令创建 crontab 文件。格式如下：

0 0 \* \* sun cp –r /user/backup /tmp

（2）第二种方法：

用户先在自己目录下新建文件 file，文件内容如下： 0 \* \* sun cp –r /user/backup /tmp

然后执行 crontab file 使生效。

8\. 设计一个 Shell 程序，在/userdata 目录下建立 50 个目录，即 user1～user50，并设 置每个目录的权限，其中其他用户的权限为：读；文件所有者的权限为：读、写、执行；文 件所有者所在组的权限为：读、执行。

参考答案: 建立程序 Pro16 如下：

#!/bin/sh i=1

while \[ i -le 50 \] do

if \[ -d /userdata \];then

mkdir -p -m 754 /userdata/user$i 加上-m 754 就不用写下面那一句了 -p 是递归建 立目录

#chmod 754 /userdata/user$i echo "user$i"

let "i = i + 1" （或 i=$(($i+1)) else

mkdir /userdata

mkdir -p -m /userdata/user$i #chmod 754 /userdata/user$i echo "user$i"

let "i = i + 1" （或 i=$（（$i＋1）） fi

done

五、多选题

1. 关于硬链接的描述正确的（BE）。 A 跨文件系统

B    不可以跨文件系统

D 可以做目录的连接

C    为链接文件创建新的i 节点 E 链接文件的 i 节点同被链接文件的 i 节点

1. 在网站发布用户 wang 的个人网页时，需要创建用户网页目录，假定用户网页目录设定为 web

（用户目录在/home 目录下），如下描述正确的是（BCE）

A 存放用户网页的绝对路径/wang/web B 存放用户网页的目录～wang/

C 存放用户网页的绝对路径/home/wang/web D 存放用户网页的绝对路径/home/web E 在本机访问用户wang 的个人网页的 URL 地址 [http://localhost/](http://localhost/) ～wang/

1. 在一台 WWW 服务器上将端口号设定为 8000，默认的网页文件 index.html，服务器 网页的根目录/www。在本机访问服务器时，正确的用法是（BDE）

A    浏览器访问该服务器的URL 地址 [http://localhost/](http://localhost/)

B    浏览器访问该服务器的URL  地址 http://localhost:8000/

C    浏览器访问该服务器的用户 li 网页 URL 地址 [http://localhost/~li](http://localhost/~li)

D 浏览器访问该服务器的用户 li 网页 URL 地址 http://localhost:8000/~li E 浏览器访问该服务器的 URL 地址 localhost:8000/

1. 在 shell 编程中关于$2 的描述正确的是（CE）

A    程序后携带了两个位置参数

B    宏替换

C    程序后面携带的第二个位置参数

D 携带位置参数的个数 E 用$2 引用第二个位置参数

1. 某文件的权限是 \- r w x r - - r- -，下面描述正确的是(CD) A 文件的权限值是 755

B    文件的所有者对文件只有读权 限

C    文件的权限值是 744

D 其他用户对文件只有读权限 E 同组用户对文件只有写权限

1. 关于 OpenSSH  的作用的描述正确的是（ACE）

A 开放源代码的安全加密程序

B OpenSSH 常用于为 http 协议加密

C OpenSSH  用于提高远程登录访问的安全性

D 它和 telnet 实用同样的端口号

E OpenSSH 是免费下载的应程序

1. 关于NFS  服务器描述正确的是（BC）

A    网络中实现 Windows 系统之间文件系统共享的应用软件

B    网络中实现 Linux  系统之间文件系统共享的应用软件

C    网络中实现 Unix 系统之间文件系统共享的应用软件

D 网络中实现 Windows 系统和 Unix 之间文件系统共享的应用软件 E 网络中实现 Windows 系统和 Linux 之间文件系统共享的应用软件 8．关于 sed 描述正确的是（ABD）

A sed 是 Linux 系统中的流编辑器 B sed 是 UNIX 系统中的流编辑器 C sed 网络文件系统的类型

D 利用管道对标准输入/标准输入的数据进行编辑和组合

E sed 是 NFS 的应用程序

1. 关于限制磁盘限额，描述正确的是（ABD）

A    使用 edquota 可以监控系统所有用户使用的磁盘空间，并在接近极限时提示用户

B    用户组的磁盘限额是用户组内所有用户予设磁盘空间总和

C    单个用户的磁盘限额就是该用户所在用户组内所有磁盘限额的总合

D 在 Linux 系统下限制用户使用的磁盘空间可以使用 edquota

E 用户组的磁盘限额就是该用户组内拥有最大磁盘限额值的用户的磁盘限额

1. 关于建立系统用户的正确描述是（）

A    在 Linux 系统下建立用户使用 adduser 命令

B    每个系统用户分别在/etc/passwd 和/etc/shadow 文件中有一条记录

C    访问每个用户的工作目录使用命令“cd /用户名”

D 每个系统用户在默认状态下的工作目录在/home/用户名

E 每个系统用户在/etc/fstab 文件中有一条记录

lspci \|grep Ethernet ==查看机器双网卡 mii-tool 查看网线是否接号

用户进程、系统进程、IO 进程、空闲的比例" 如果 idle 时常处于 0，则需要检查引起大量 CPU 消耗的原因

内存使用情况 "vmstat：观察 free 值

top： 观察 memory 项" 低于 50 时，值得注意交换区使用情况 "vmstat：观察 pi、po 值

top： 观察 paging/paging space 项 free： 观察 Swap 行的值" 当空闲值低时，值得 注意

I/O 情况 "sar -u:观察 io 占用系统情况

iostat -d：观察哪块盘 io 较多

top:观察 io 最多的进程" IO 值过高的进程将会严重影响到整机的性能，要对高 IO 的进程重点监控，检查

系统进程 ps aux 有无多个相同的进程名

df -h：检查空间使用达到 90％的文件系统 尤其是使用情况

系统日志 last：观察最近的主机登录情况，查看 var/log/messages 文件内容， 对不明 主机进行检查

网络状况 ping：查看到其他主机的 time 值是否小于 10ms 无频繁丢包

top 查询 CPU， 内存， 系统进程情况 ( CPU 内存瓶颈), 某个进程

cat /etc/redhat-release <---看本机系统是什么版本的 smartmontools-5.38-2.el5

smartctl --all /dev/sda 检测磁盘有没有坏块

smartctl -i /dev/sda vim /etc/smartd.conf

/dev/sda -a -d sat 把硬盘注册为 sat 57 行

/dev/sda -d scsi -s L/../../3/18 打开注释 65 行

service smartd restart 磁盘有问题的话，会发邮件给管理员的

service sendmail restart yum istall -y sysstat

iostat 看当前磁盘读写的情况 iostat 2 10 查询当前状态 ( 磁盘 i/o ) sar 2 10 查询当前状态 ( service sysstat start )

sar -r ( 内存 ) sar -u ( cpu )

sar -P ( cpu ) --> sar -P 0 \|\| sar -P ALL sar -b ( i/o )

sar -n DEV ( 网络设备 ) sar -f

service sysstat on

checkconfig sysstat on 每十分钟搜集一次信息

sar -f /var/log/sa/sa13 -s 10:10:00 -e 11:10:00 查看昨天（今天是 14 号）10：

10：00 到 11：10：00 的系统情况

针对日志 /var/log/sa/\* 查询之前的日志信息

sar -n DEV

tty 看自己的

mpstat 2 查看 cpu 状态

vmstat 2 10

mpstat 2 10 <- cpu 每二秒显示一次，共显示十次

vmstat 2 10 <- 整体资源

free ltrace

pmap 进程号 是看这个进程占了多少内存

pgrep httpd 查询这个服务的所有进程号 killall httpd 杀掉所有 ＜－－不怎么安全 ps aux \| grep mysql \|xargs kill -9 杀掉所有前面查出来的所有进程 ＜－－推荐用这个 pstree ps nice renice 不建议把系统资源的优先级提高

ldd /bin/ls 显示当前这个命令运行时所需要的库文件

yum install strace -y

strace +服务名称 分析出当前的命令执行时所找的库文件的路径

lspci \| grep Ethernet 查本机有哪些网卡设备

dmesg 查看本机的设备信息

mii-tool 看本机网卡是否连接正常

iptraf 查看本机当前的流量

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/3019638/20221027195432.png)](https://home.cnblogs.com/u/yehewang/)

[野鹤%](https://home.cnblogs.com/u/yehewang/)

[粉丝 \- 0](https://home.cnblogs.com/u/yehewang/followers/) [关注 \- 1](https://home.cnblogs.com/u/yehewang/followees/)

+加关注

0

0

[«](https://www.cnblogs.com/yehewang/p/16833520.html) 上一篇： [Linux笔记](https://www.cnblogs.com/yehewang/p/16833520.html "发布于 2022-10-27 19:46")

posted @
2022-10-28 09:36 [野鹤%](https://www.cnblogs.com/yehewang)
阅读(1082)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2Fyehewang%2Fp%2F16834780.html&targetId=16834780&targetType=0)

[刷新页面](https://www.cnblogs.com/yehewang/p/16834780.html#) [返回顶部](https://www.cnblogs.com/yehewang/p/16834780.html#top)

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

浏览器不兼容canvas

昵称：
[野鹤%](https://home.cnblogs.com/u/yehewang/)

园龄：
[3年6个月](https://home.cnblogs.com/u/yehewang/ "入园时间：2022-10-27")

粉丝：
[0](https://home.cnblogs.com/u/yehewang/followers/)

关注：
[1](https://home.cnblogs.com/u/yehewang/followees/)

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

- [我的随笔](https://www.cnblogs.com/yehewang/p/ "我的博客的随笔列表")
- [我的评论](https://www.cnblogs.com/yehewang/MyComments.html "我的发表过的评论列表")
- [我的参与](https://www.cnblogs.com/yehewang/OtherPosts.html "我评论过的随笔列表")
- [最新评论](https://www.cnblogs.com/yehewang/comments "我的博客的评论列表")
- [我的标签](https://www.cnblogs.com/yehewang/tag/ "我的博客的标签列表")

### 随笔档案

- [2022年10月(2)](https://www.cnblogs.com/yehewang/p/archive/2022/10)

### [阅读排行榜](https://www.cnblogs.com/yehewang/most-viewed)

- [1\. Linux 运维工程师面试题总结(1082)](https://www.cnblogs.com/yehewang/p/16834780.html)
- [2\. Linux笔记(184)](https://www.cnblogs.com/yehewang/p/16833520.html)

01. 1ЗвездаVitas
02. 2左手指月萨顶顶
03. 3黎明前的黑暗周子健
04. 4男儿当自强 (主题音乐)黄霑
05. 5花香周传雄
06. 6黄昏周传雄
07. 7处处吻杨千嬅
08. 8关不上的窗周传雄
09. 9野狼disco宝石Gem
10. 10人间不值得黄诗扶
11. 11时光的残片（翻自 赵小磊）少女榨汁机
12. 12把孤独当作晚餐刘旭阳
13. 13把孤独当晚餐却难以下咽李袁杰 / 刘旭阳
14. 14美丽的神话胡歌 / 白冰
15. 15阆中之恋姚贝娜 / 沙宝亮
16. 16Sound Of My DreamKate Lesing
17. 17Theme to "Braveheart"Martin Winch
18. 18天各一方曾路得
19. 19Dancer In The Dark (Original Mix)GRABOTE / Marc Philippe
20. 20Hearts & Rubies (Prod. lord Wavy & Yung Kenshin)Lord Wavy

Theme to "Braveheart" \- Martin Winch

00:00 / 00:00

An audio error has occurred, player will skip forward in 2 seconds.

作曲 : James Horner

纯音乐，请欣赏

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)