# 常见Linux面试题汇总（2025）_linux常见面试题 - CSDN博客

URL: https://blog.csdn.net/weixin_42358373/article/details/145717877

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

# 常见Linux面试题汇总（2025）

最新推荐文章于 2026-02-27 16:12:36 发布

原创 [![](https://csdnimg.cn/release/blogv2/dist/pc/img/identityVipNew.png)](https://mall.csdn.net/vip) 于 2025-02-19 21:13:49 发布·1.9k 阅读

·![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png)
18


·![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png)
23


文章标签：

[#linux](https://so.csdn.net/so/search/s.do?q=linux&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#java](https://so.csdn.net/so/search/s.do?q=java&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#服务器](https://so.csdn.net/so/search/s.do?q=%E6%9C%8D%E5%8A%A1%E5%99%A8&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art)

## 常见Linux面试题  汇总

### 一、基础概念  类

#### （一）文件系统相关

1. **Linux文件系统的基本结构是怎样的？**
   - Linux文件系统采用树形结构，根目录为“/”。主要有以下几种类型的目录：
     - `/bin`：存放常用用户命令，如ls、cp等。
     - `/sbin`：存放系统管理命令，如ifconfig、fdisk等，通常只有超级用户可以执行。
     - `/home`：用户的主目录，每个用户在该目录下有自己的子目录。
     - `/var`：存放可变数据，如日志文件（/var/log）、邮件（/var/mail）等。
     - `/tmp`：临时文件存放目录，系统重启时可能会清空。
     - `/usr`：包含大量的系统软件资源，如二进制文件（/usr/bin）、库文件（/usr/lib）等。
2. **什么是inode？它在文件系统中有什么作用？**
   - inode（索引节点）是Unix/Linux文件系统中用于存储文件元数据的一种数据结构。
   - 作用：
     - 它包含了文件的权限、所有者、组、大小、修改时间等重要信息。
     - 每个文件都有一个唯一的inode编号，通过这个编号可以在文件系统中定位和管理文件。即使文件名改变，只要inode编号不变，文件在系统中的位置和相关属性就不变。
3. **如何查看文件系统的使用情况？**
   - 可以使用`df -h`命令。其中：

     - `-h`参数表示以人类可读的格式（如KB、MB、GB等）显示结果。
     - 输出结果包含文件系统的挂载点、总容量、已使用容量、可用容量和使用率等信息。

#### （二）用户和权限管理

4. **如何在Linux中创建新用户？**
   - 使用`useradd`命令。例如，创建一个名为`newuser`的用户：`useradd newuser`。
   - 可以添加更多参数来定制用户的属性，如`-d`指定用户的家目录（`useradd -d /home/newhome newuser`），`-s`指定用户的默认shell（`useradd -s /bin/bash newuser`）。
5. **如何修改用户的密码？**
   - 使用`passwd`命令。如果要修改当前用户的密码，直接输入`passwd`，然后按照提示输入新密码即可。
   - 如果要修改其他用户的密码（需要有足够的权限，如root用户），可以使用`passwd username`，然后输入新密码。
6. **Linux中的文件权限有哪些类型？如何设置？**
   - 文件权限分为读（r）、写（w）、执行（x）三种类型。
   - 对于文件所有者、所属组和其他用户分别有不同的权限设置。
   - 可以使用`chmod`命令设置权限。

     - 符号模式：例如，给文件所有者添加执行权限，`chmod u+x file`（u表示所有者，+表示添加权限，x表示执行权限）。
     - 数字模式：用三位数字表示权限，每一位分别对应所有者、所属组和其他用户的权限。数字的计算方式为：读（4）+写（2）+执行（1）。例如，设置文件权限为所有者可读写执行，所属组可读执行，其他用户可读，`chmod 754 file`。

#### （三）  进程管理

7. **如何查看当前系统中的进程？**
   - 可以使用`ps -ef`命令。

     - `-e`表示显示所有进程，`-f`表示显示完整的格式信息，包括UID（用户ID）、PID（进程ID）、PPID（父进程ID）、C（CPU使用率）、STIME（启动时间）、TTY（终端类型）、TIME（累计CPU时间）、CMD（命令）等。
   - 也可以使用`top`命令，它会动态地显示系统中各个进程的资源占用情况，如CPU使用率、内存使用率等，并且可以按照不同的指标进行排序。
8. **如何杀死一个进程？**
   - 可以使用`kill`命令。

     - 首先需要知道进程的PID。例如，要杀死PID为1234的进程，可以使用`kill 1234`。
     - 如果进程没有响应正常的终止信号，可以使用`kill -9 1234`，其中`-9`表示发送SIGKILL信号，强制终止进程，但这种方式可能会导致数据丢失等问题，应谨慎使用。

### 二、命令操作类

#### （一）  文本处理   命令

9. **如何查看文件的内容？**
   - `cat`命令：用于一次性查看整个文件的内容，例如`cat file.txt`。
   - `less`命令：适合查看较大的文件，它支持分页查看，可以使用上下箭头键翻页，按`q`键退出，如`less largefile.txt`。
   - `head`命令：默认显示文件的前10行内容，如`head file.txt`，也可以使用`-n`参数指定显示的行数，如`head -n 5 file.txt`。
   - `tail`命令：默认显示文件的后10行内容，如`tail file.txt`，同样可以使用`-n`参数指定行数，也可用于实时查看文件的末尾内容（如查看日志文件的最新更新），如`tail -f logfile.txt`。
10. **如何查找文件中的特定字符串？**

- `grep`命令：例如，在文件`file.txt`中查找包含字符串“abc”的行，可以使用`grep "abc" file.txt`。
- 可以使用`-r`参数进行递归查找（在目录及其子目录下的所有文件中查找），如`grep -r "abc" /home/user/dir`。
- 还可以使用`-i`参数进行忽略大小写的查找，如`grep -i "abc" file.txt`。

#### （二）文件操作命令

11. **如何复制文件和目录？**

- 复制文件：使用`cp`命令。例如，将`file1.txt`复制为`file2.txt`，可以使用`cp file1.txt file2.txt`。
- 复制目录：需要使用`-r`（递归）参数。例如，将`dir1`复制为`dir2`，可以使用`cp -r dir1 dir2`。

12. **如何移动或重命名文件和目录？**

- 移动文件或目录：使用`mv`命令。例如，将`file1.txt`移动到`/home/user/dir`目录下，可以使用`mv file1.txt /home/user/dir`。
- 重命名文件或目录：也是使用`mv`命令。例如，将`oldname.txt`重命名为`newname.txt`，可以使用`mv oldname.txt newname.txt`。

#### （三）网络相关命令

13. **如何查看网络连接状态？**

- `netstat -tunlp`命令：

  - `-t`表示显示TCP连接，`-u`表示显示UDP连接，`-n`表示以数字形式显示地址和端口（不进行域名解析等），`-l`表示只显示监听状态的连接，`-p`表示显示进程信息（需要有足够权限）。
- `ss -tunlp</`

![](https://csdnimg.cn/release/blogv2/dist/pc/img/lock.png)最低0.47元/天 解锁文章![](https://img-home.csdnimg.cn/images/20240516053626.png)

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png)

确定要放弃本次机会？


福利倒计时

_:_ _:_

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png)立减 ¥

普通VIP年卡可用

[立即使用](https://mall.csdn.net/vip)

[![](https://profile-avatar.csdnimg.cn/1fe2f52fe7d446b7b49581e437bb02c0_weixin_42358373.jpg!1)\\
全息架构师](https://blog.csdn.net/weixin_42358373)

关注关注

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png)
18

点赞

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png)
踩

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png)
23




收藏







觉得还不错?

一键收藏
![](https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png)

- [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png)\\
0](https://blog.csdn.net/weixin_42358373/article/details/145717877#commentBox)
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


专栏目录

[45个经典 _Linux_ _面试题_（非常详细）零基础入门到精通，收藏这一篇就够了](https://blog.csdn.net/weixin_53312997/article/details/147069273)

[小陈的博客](https://blog.csdn.net/weixin_53312997)

04-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1902


[（文件系统分配其中的一些磁盘块用来记录它自身的一些数据，如 i 节点，磁盘分布图，间接块，超级块等。$ chmod 751 file 给 file 的属主分配读、写、执行 _(_ 7 _)_ 的权限，给 file 的所在组分配读、执行 _(_ 5 _)_ 的权限，给其他用户分配执行 _(_ 1 _)_ 的权限。 _linux_ 命令’hash’管理着一个内置的哈希表，记录了已执行过的命令的完整路径, 用该命令可以打印出你所使用过的命令以及执行的次数。创建文件：典型的如 touch，vi 也可以创建文件，其实只要向一个不存在的文件输出，都会创建文件。](https://blog.csdn.net/weixin_53312997/article/details/147069273)

[_Linux_ _面试题_ 大全（含答案）](https://blog.csdn.net/weixin_67430601/article/details/142831113)

[weixin\_67430601的博客](https://blog.csdn.net/weixin_67430601)

10-10![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3242


[SE _Linux_（Security-Enhanced _Linux_）是 _Linux_ 的一个安全子系统，提供了访问控制安全策略机制，用于限制进程访问系统资源。在 _Linux_ 系统中，通过安装SSH服务（如OpenSSH）并配置其监听端口（默认为22），用户可以安全地通过SSH客户端从远程计算机访问 _Linux_ _服务器_。答案： _Linux_ 中的文件权限是操作系统用来控制特定用户或用户组可以对文件或目录执行的操作的机制。是一种强大的文本分析工具，它逐行扫描文件，并根据定义的模式进行数据处理。](https://blog.csdn.net/weixin_67430601/article/details/142831113)

参与评论您还未登录，请先登录后发表或查看评论

[超级经典的 _linux_ _面试题_ 总结 _(_ 非常详细 _)_,零基础入门到精通,收藏这一篇就...](https://blog.csdn.net/Libra1313/article/details/148287568)

4-17

[在 _Linux_ 内核环境下,申请大块内存的成功率随着系统运行时间的增加而减少,虽然可以通过vmalloc系列调用申请物理不连续但虚拟地址连续的内存,但毕竟其使用效率不高且在32位系统上vmalloc的内存地址空间有限。所以,一般的建议是在系统启动阶段申请大块内存,但是其成功的概率也只是比较高而已,而不是100%。如果程序真的比较在意...](https://blog.csdn.net/Libra1313/article/details/148287568)

[_linux_ _常见_ _面试题_\_ _linux_ _面试题_ csdn](https://blog.csdn.net/Javatwx/article/details/79774275)

5-4

[_Linux_ 下安装Tomcat _服务器_ 步骤和使用到的命令: 步骤: 1.上传并安装,解压Tomcat _服务器_ 2.配置JDK环境变量 _(_ 编辑etc下的profile文件,步骤:cd /etc/然后vim profile _)_ 3.修改tomcat _服务器_ 的启动端口 _(_ vim server.xml _)_ 4.Tomcat的启动 _(_ 关闭 _)_: cd bin ./startup.sh _(_./shutdown.sh _)_ ...](https://blog.csdn.net/Javatwx/article/details/79774275)

[_Linux_ 命令高频 _面试题_\\
\\
最新发布](https://blog.csdn.net/dzh0622/article/details/158464355)

[se-tester的专栏](https://blog.csdn.net/dzh0622)

02-27![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
493


[本文介绍了10个常用 _Linux_ 命令面试问题及其解答，涵盖文件操作、文本处理、权限管理和系统监控等方面。主要内容包括：使用ls -a查看隐藏文件、less分页浏览文件内容、find搜索文件、grep查找文本、chmod修改权限、ps/top查看进程、ping测试网络、df/du检查磁盘空间、tar压缩解压文件以及I/O重定向操作。文章强调实际操练的重要性，建议通过虚拟机模拟场景加深理解，并推荐查阅man手册提升命令熟练度。这些基础命令的掌握是 _Linux_ 系统管理的核心技能。](https://blog.csdn.net/dzh0622/article/details/158464355)

[超级经典的 _linux_ _面试题_ 总结 _(_ 非常详细），零基础入门到精通，收藏这一篇就够了](https://devpress.csdn.net/v1/article/detail/148287568)

[Libra1313的博客](https://blog.csdn.net/Libra1313)

05-28![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3964


[1、什么是堆内存和栈内存？答： 我们可以从几个方面来看它们之间的区别栈：由编译器自动分配和释放，一般存放函数的参数、局部变量、临时变量、函数返回地址等堆：堆内存是由程序员进行分配和释放的，也称为动态内存分配。如果没有手动free，在程序结束时有可能由操作系统自动释放（但是仅限于有回收机制的语言， 像C/C++就必须进行手动释放，不然就有可能造成内存泄漏）栈：由于栈上的空间是自动分配自动回收的，所以栈上数据的生命周期只是在函数的运行过程中存在，运行后就释放掉了堆：堆上的数据，只要程序员不释放空间，就可以一直访](https://devpress.csdn.net/v1/article/detail/148287568)

[_linux_ 驱动面试 _常见_ 题目\_ _linux_ 驱动 _面试题_](https://blog.csdn.net/linux_devices_driver/article/details/6589004)

3-19

[1\. _linux_ 内核里面,内存申请有哪几个函数,各自的区别? Kmalloc _(_ _)_ \_\_get\_free\_page _(_ _)_ mempool\_create _(_ _)_ 2\. IRQ和FIQ有什么区别,在CPU里面是是怎么做的? 3. int \*a; char \*b; a和b本身是什么类型? a、b里面本身存放的只是一个地址,难道是这两个地址有不同么?](https://blog.csdn.net/linux_devices_driver/article/details/6589004)

[2万字系统总结，带你实现 _Linux_ 命令自由?还不赶紧进来学习](https://blog.csdn.net/Java_Pluto/article/details/115242190)

[Java\_Pluto的博客](https://blog.csdn.net/Java_Pluto)

03-26![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2272


[2万字系统总结，带你实现 _Linux_ 命令自由？\\
##前言\\
_Linux_ 的学习对于一个程序员的重要性是不言而喻的。前端开发相比后端开发，接触 _Linux_ 机会相对较少，因此往往容易忽视它。但是学好它却是程序员必备修养之一。\\
如果本文对你有所帮助，请点个???? 吧。\\
作者使用的是阿里云 _服务器_ ECS （最便宜的那种） CentOS 7.7 64位。当然你也可以在自己的电脑安装虚拟机，虚拟机中再去安装 CentOS 系统（这就完全免费了）。至于它的安装教程可以去谷歌搜索下，相关教程非常多。\\
## _Linux_ 基](https://blog.csdn.net/Java_Pluto/article/details/115242190)

[50 个最热门的 _Linux_ 面试问题及答案](https://yoagoa.blog.csdn.net/article/details/140169816)

[技术探索驿站](https://blog.csdn.net/xiefeng240601)

07-05![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[_Linux_ 是一个开源的类 Unix 操作系统内核，广泛用于 _服务器_、台式机和嵌入式系统。本文整理了50个 _面试题_。](https://yoagoa.blog.csdn.net/article/details/140169816)

[100道 _Linux_ 系统 _面试题_（含答案）](https://tingyu.blog.csdn.net/article/details/135007114)

[a342874650的专栏](https://blog.csdn.net/a342874650)

12-15![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3万+


[通过解答 _Linux_ _面试题_，读者可以更深入地了解和掌握 _Linux_ 的知识和技能。这些问题涵盖了 _Linux_ 操作系统的各个方面，包括文件系统、进程管理、网络管理、权限管理、软件包管理等等。通过回答这些问题，读者可以更好地理解 _Linux_ 操作系统的内部机制、概念和原理，提高自己的技术水平和解决问题的能力。同时，这些问题也能够帮助读者更好地准备和应对实际的 _Linux_ 系统管理和应用开发工作。总之，通过解答 _Linux_ _面试题_，读者可以增进对 _Linux_ 的认识，提高自己的专业素养和竞争力。](https://tingyu.blog.csdn.net/article/details/135007114)

[精心整理- _2025_ 年Android大厂 _面试题_ 面试资料大全-有这份就足够了（共67页）.zip](https://download.csdn.net/download/goodxianping/92452499)

12-11

[第2章 _Java_ 核心基础 _面试题_ _汇总_.pdf 第3章 _Java_ 深入泛型与注解 _面试题_ _汇总_.pdf 第4章 _Java_ 并发编程 _面试题_ _汇总_.pdf 第5章 _Java_ 虚拟机原理 _面试题_ _汇总_.pdf 第6章 _Java_ 反射类加载与动态代理 _面试题_ _汇总_.pdf 第7章 网络编程...](https://download.csdn.net/download/goodxianping/92452499)

[_Linux_ 嵌入式方向的 _常见_ _面试题_](https://blog.csdn.net/sanqima/article/details/132175130)

[sanqima的专栏](https://blog.csdn.net/sanqima)

08-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1698


[libudev是udev是一种开源实现库，它能根据系统中硬件设备的状态来动态的更新设备文件，包括设备文件的创建、删除等。使用udev后，在/dev目录下就直包含系统中真正存在的设备。udev同时提供了建设接口，当设备的状态发生改变时，监视接口可以发送对应的事件给应用程序。主要管理/dev目录下的设备节点，同时，也接替devfs、hotplug热插拔的功能，处理添加硬件、删除硬件、加载firmware，以及用户空间的行为。](https://blog.csdn.net/sanqima/article/details/132175130)

[_linux_ 运维 _面试题_ _汇总_](https://download.csdn.net/download/weixin_42058895/10535747)

07-11

[_linux_ 运维 _面试题_ _汇总_，包括计算机网络， _linux_ 系统 _常见_ 命令，常用系统服务，集群搭建等当面](https://download.csdn.net/download/weixin_42058895/10535747)

[_Linux_ _面试题_（34道）](https://blog.csdn.net/qq_43161404/article/details/122884901)

[qq\_43161404的博客](https://blog.csdn.net/qq_43161404)

02-16![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3万+


[1、 _Linux_ 中各个目录的作用\\
1）/ 根目录\\
2）/boot 引导程序，内核等存放的目录\\
3）/sbin 超级用户可以使用的命令的存放目录\\
4）/bin 普通用户可以使用的命令的存放目录\\
5）/lib 根目录下的所程序的共享库目录\\
6）/dev 设备文件目录\\
在 _Linux_ 中设备都是以文件形式出现，这里的设备可以是硬盘，键盘，鼠标，网卡，终端，等设备，通过访问这些文件可以访问到相应的设备。\\
7）/home 普通用户的家目录\\
8）/root 用户root的$HOME目录\\
9）/etc 全局的配置](https://blog.csdn.net/qq_43161404/article/details/122884901)

[48 个 _Linux_ 面试问题和答案\\
\\
热门推荐](https://blog.csdn.net/taoxicun/article/details/122915066)

[太极淘的博客](https://blog.csdn.net/taoxicun)

02-13![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
4万+


[你在准备 _Linux_ 面试吗？我们准备了一些 _常见_ 的 _Linux_ 面试问题及其答案。\\
\\
如果您是初学者（具有一定的 _Linux_ 知识或获得认证）或具有专业的 _Linux_ 管理经验，那么下面的问答有助于您准备面试。\\
\\
1.什么是 _Linux_ 及基本组件？\\
\\
_Linux_ 是一个基于 _Linux_ 内核的免费开源操作系统。它是广泛使用的操作系统之一，尤其是在 _服务器_ 世界和开发人员中。它的用途从托管网站和应用程序延伸到成为智能手机、平板电脑和电视等智能设备的核心操作系统。\\
\\
_Linux_ 由 4 个基本组件构成：\\
\\
Kerne](https://blog.csdn.net/taoxicun/article/details/122915066)

[_Linux_ 面试 _常见_（55题）](https://blog.csdn.net/l453521934/article/details/131321943)

[l453521934的博客](https://blog.csdn.net/l453521934)

06-21![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[mkdir命令可以创建一个新的目录。例如，在当前目录下创建一个名为“test”的目录：mkdir test\\
2\. 删除目录：rmdirrmdir命令可以删除一个空目录。例如，要删除名为“test”的空目录，rmdir test\\
如果目录不为空，可以使用rm -r命令来递归删除目录及其所有子目录和文件。rm -r\\
3\. 列出目录内容：ls使用命令可以列出目录中的所有文件和子目录ls\\
ls -l命令显示文件和文件夹的权限、所有者、大小、创建日期等详细信息。ls -l\\
4\. 更改目录：cd使用命令可以更改当前工作](https://blog.csdn.net/l453521934/article/details/131321943)

[建议收藏 100 道 _Linux_ _面试题_ 附答案](https://devpress.csdn.net/v1/article/detail/117719299)

[田维常](https://blog.csdn.net/o9109003234)

06-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2万+


[关注公众号“ _Java_ 后端技术全栈”回复“000”获取程序员必备电子书大家好，我是老田，今天我给大家分享100道 _Linux_ _面试题_。本文一共 3万多字，分别从 _Linux_ 概述、磁盘、目录、文件...](https://devpress.csdn.net/v1/article/detail/117719299)

[《吐血整理》 _Linux_ _面试题_ Top100@面试官你好，我精通 _Linux_！嘿嘿~](https://devpress.csdn.net/v1/article/detail/103920352)

[陈哈哈的菜园子](https://blog.csdn.net/qq_39390545)

01-10![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[_Linux_ 初学者面试问题\\
\\
_Linux_ 基本面试问题\\
\\
1.什么是 _Linux_？\\
\\
回答： _Linux_ 是基于 _Linux_ 内核的操作系统。它是一个开源操作系统，可以在不同的硬件平台上运行。它为用户提供了免费的低成本操作系统。这是一个用户友好的环境，他们可以在其中轻松修改和创建源代码的变体。\\
\\
2.谁发明了 _Linux_？解释 _Linux_ 的历史？\\
\\
回答：Linus Torvalds创建了 _Linux_。莱纳斯·...](https://devpress.csdn.net/v1/article/detail/103920352)

[_Linux_ 经典 _面试题_（你想找的都在这里）](https://blog.csdn.net/wyttRain/article/details/115257976)

[wyttRain的博客](https://blog.csdn.net/wyttRain)

09-13![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
7617


[1、 _Linux_ 内存管理\\
2、 _Linux_ 中断机制\\
3、 _Linux_ 同步机制\\
4、 _Linux_ 总线模型](https://blog.csdn.net/wyttRain/article/details/115257976)

[面试必备的 _Linux_ 常用命令](https://blog.csdn.net/wangyuxiang946/article/details/133880048)

[wangyuxiang946的博客](https://blog.csdn.net/wangyuxiang946)

12-13![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[面试必备的 _Linux_ 常用命令大全](https://blog.csdn.net/wangyuxiang946/article/details/133880048)

[面试必备， _Linux_ _面试题_ 和答案！](https://devpress.csdn.net/v1/article/detail/130199438)

[人生不怕起点低，就怕没追求](https://blog.csdn.net/AI_Green)

04-17![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[答案：du 显示目录或文件的大小df 显示每个所在的文件系统的信息，默认是显示所有文件系统。（文件系统分配其中的一些磁盘块用来记录它自身的一些数据，如 i 节点，磁盘分布图，间接块，超级块等。这些数据对大多数用户级的程序来说是不可见的，通常称为 Meta Data。） du 命令是用户级的程序，它不考虑 Meta Data，而 df命令则查看文件系统的磁盘分配图并考虑 Meta Data。df 命令获得真正的文件系统数据，而 du 命令只查看文件系统的部分情况。](https://devpress.csdn.net/v1/article/detail/130199438)

[_Linux_ _面试题_](https://blog.csdn.net/weixin_67732682/article/details/142066922)

[weixin\_67732682的博客](https://blog.csdn.net/weixin_67732682)

09-09![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3100


[IP/URL 散列算法是一种根据客户端 IP 地址或 URL 来分配请求的负载均衡算法，这样相同的IP或者URL就会负载到相同的 _服务器_ 上。说明：表示请求的资源已经被永久移动到了新的位置，浏览器会自动重定向到新的位置，并且以后的所有请求都应该直接访问新的位置。加权轮询算法是轮询算法的一种改进，只不过在负载时会根据 _服务器_ 的权重来分配请求，权重越大，分配的请求就会越多。加权轮询算法是轮询算法的一种改进，只不过在负载时会根据 _服务器_ 的权重来分配请求，权重越大，分配的请求就会越多。直白讲就是随速度快，随就干的多。](https://blog.csdn.net/weixin_67732682/article/details/142066922)

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

[![](https://profile-avatar.csdnimg.cn/1fe2f52fe7d446b7b49581e437bb02c0_weixin_42358373.jpg!1)](https://blog.csdn.net/weixin_42358373)

全息架构师

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

![](https://blog.csdn.net/weixin_42358373/article/details/145717877)

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