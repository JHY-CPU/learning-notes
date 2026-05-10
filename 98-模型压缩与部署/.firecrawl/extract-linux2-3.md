# 经典48道linux命令面试题总结（解析在篇尾） - 牛客

URL: https://www.nowcoder.com/discuss/352910181216423936

# 面试必备：经典48道linux命令面试题总结（解析在篇尾）

烫

![](https://uploadfiles.nowcoder.com/images/20220508/174636739_1652011563993/D1944EF3D7C6D55B3F21D153F7761C88)

# 目录

**1、绝对路径用什么符号表示？当前目录、上层目录用什么表示？主目录用什么表示? 切换目录用什么命令？**

**2、怎么查看当前进程？怎么执行退出？怎么查看当前路径？**

**3、怎么清屏？怎么退出当前命令？怎么执行睡眠？怎么查看当前用户 id？查看指定帮助用什么命令？？**

**4、Ls命令执行什么功能？ 可以带哪些参数，有什么区别？**

**5、查看文件有哪些命令？**

**6、列举几个常用的Linux命令。**

**7、你平时是怎么查看日志的？**

**8、建立软链接(快捷方式)，以及硬链接的命令**

**9、目录创建用什么命令？创建文件用什么命令？复制文件用什么命令？**

**10、查看文件内容有哪些命令可以使用？**

**11、随意写文件命令？怎么向屏幕输出带空格的字符串，比如”hello world”?**

**12、终端是哪个文件夹下的哪个文件？黑洞文件是哪个文件夹下的哪个命令？**

**13、移动文件用哪个命令？改名用哪个命令？**

**14、复制文件用哪个命令？如果需要连同文件夹一块复制呢？如果需要有提示功能呢？**

**15、删除文件用哪个命令？如果需要连目录及目录下文件一块删除呢？删除空文件夹用什么命令？**

**16、Linux下命令有哪几种可使用的通配符？分别代表什么含义？**

**17、用什么命令对一个文件的内容进行统计？(行号、单词数、字节数)**

**18、Grep命令有什么用？ 如何忽略大小写？ 如何查找不含该串的行?**

**19、Linux中进程有哪几种状态？在ps显示出来的信息中分别用什么符号表示的？**

**20、怎么使一个命令在后台运行?**

**21、利用ps怎么显示所有的进程? 怎么利用ps？**

**22、哪个命令专门用来查看后台任务?**

**23、把后台任务调到前台执行使用什么命令?把停下的后台任务在后台执行起来用什么命令?**

**24、终止进程用什么命令? 带什么参数?**

**25、怎么查看系统支持的所有信号？**

**26、搜索文件用什么命令? 格式是怎么样的?**

**27、查看当前谁在使用该主机用什么命令? 查找自己所在的终端信息用什么命令?**

**28、使用什么命令查看用过的命令列表?**

**29、使用什么命令查看磁盘使用空间？空闲空间呢?**

**30、使用什么命令查看网络是否连通?**

**31、使用什么命令查看IP地址及接口信息？**

**32、查看各类环境变量用什么命令?**

**33、通过什么命令指定命令提示符?**

**34、查找命令的可执行文件是去哪查找的? 怎么对其进行设置及添加?**

**35、通过什么命令查找执行命令?**

**36、怎么对命令进行取别名？**

**37、du和df的定义，以及区别？**

**38、awk详解。**

**39、当你需要给命令绑定一个宏或者按键的时候，应该怎么做呢？**

**40、如果一个Linux新手想要知道当前系统支持的所有命令的列表，他需要怎么做？**

**41、如果你的助手想要打印出当前的目录栈，你会建议他怎么做？**

**42、你的系统目前有许多正在运行的任务，在不重启机器的条件下，有什么方法可以把所有正在运行的进程移除呢？**

**43、bash shell中的hash命令有什么作用？**

**44、哪一个bash内置命令能够进行数\*\*\*算。**

**45、怎样一页一页地查看一个大文件的内容呢？**

**46、数据字典属于哪一个用户的？**

**47、怎样查看一个linux命令的概要与用法？假设你在/bin 目录中偶然看到一个你从没见过的的命令，怎样才能知道它的作用和用法呢？**

**48、使用哪一个命令可以查看自己文件系统的磁盘空间配额呢？**

## **详解解析**

**### 1、绝对路径用什么符号表示？当前目录、上层目录用什么表示？主目录用什么表示? 切换目录用什么命令？   答：     绝对路径： 如/etc/init.d     当前目录和上层目录：./ …/     主目录： ~/     切换目录：cd     2、怎么查看当前进程？怎么执行退出？怎么查看当前路径？   答：     查看当前进程：ps     执行退出：exit     查看当前路径：pwd     3、怎么清屏？怎么退出当前命令？怎么执行睡眠？怎么查看当前用户 id？查看指定帮助用什么命令？？   答：     清屏：clear     退出当前命令：ctrl+c彻底退出     执行睡眠 ：ctrl+z挂起当前进程fg恢复后台查看当前用户id：”id“：查看显示目前登陆账户的uid和gid及所属分组及用户名     查看指定帮助：如man adduser这个很全 而且有例子；adduser–help这个告诉你一些常用参数；info adduesr；     4、Ls命令执行什么功能？ 可以带哪些参数，有什么区别？   答：     ls执行的功能： 列出指定目录中的目录，以及文件哪些参数以及区别：a所有文件l详细信息，包括大小字节数，可读可写可执行的权限等     5、查看文件有哪些命令？   答：     vi文件名\#编辑方式查看，可修改     cat文件名\#显示全部文件内容     more文件名\#分页显示文件内容     less文件名\#与more相似，更好的是可以往前翻页     tail文件名\#仅查看尾部，还可以指定行数     head文件名\#仅查看头部,还可以指定行数     6、列举几个常用的Linux命令。   答：     列出文件列表：ls【参数 -a -l】     创建目录和移除目录：mkdir rmdir     用于显示文件后几行内容：tail，例如： tail -n 1000：显示最后1000行     打包：tar -xvf     打包并压缩：tar -zcvf     查找字符串：grep     显示当前所在目录：pwd创建空文件：touch     编辑器：vim vi     7、你平时是怎么查看日志的？   答：     Linux查看日志的命令有多种：tail、cat、tac、head、echo等，本文只介绍几种常用的方法。     1、tail   最常用的一种查看方式     命令格式: tail\[必要参数\]\[选择参数\]\[文件\]     -f 循环读取     -q 不显示处理信息     -v 显示详细的处理信息     -c<数目> 显示的字节数     -n<行数> 显示行数     -q, --quiet, --silent 从不输出给出文件名的首部     -s, --sleep-interval=S 与-f合用,表示在每次反复的间隔休眠S秒     例如：          [复制代码](https://www.nowcoder.com/discuss/352910181216423936\#)      |     |     | | --- | --- | | 1 | `tail -n``10``test.log 查询日志尾部最后``10``行的日志; tail -n +``10``test.log 查询``10``行之后的所有日志; tail -fn``10``test.log 循环实时查看最后``1000``行记录(最常用的)` |**

**一般还会配合着grep搜索用，例如;**

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `tail -fn``1000``test.log | grep``'关键字'` |

如果一次性查询的数据量太大,可以进行翻页查看，例如 ：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `tail -n``4700``aa.log |more -``1000``可以进行多屏显示(ctrl + f 或者 空格键可以快捷键）` |

2、head

跟tail是相反的head是看前多少行日志

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `head -n``10``test.log 查询日志文件中的头``10``行日志; head -n -``10``test.log 查询日志文件除了最后``10``行的其他所有日志;` |

head其他参数参考tail

3、cat

cat 是由第一行到最后一行连续显示在屏幕上

一次显示整个文件：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `$ cat filename` |

从键盘创建一个文件：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `$cat > filename` |

将几个文件合并为一个文件：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `$cat file1 file2 > file 只能创建新文件,不能编辑已有文件` |

将一个日志文件的内容追加到另外一个：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `$cat -n textfile1 > textfile2` |

清空一个日志文件；

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `$cat : >textfile2` |

注意：\> 意思是创建，>>是追加。千万不要弄混了。

cat其他参数参考tail

4、more

more命令是一个基于vi编辑器文本过滤器，它以全屏幕的方式按页显示文本文件的内容，支持vi中的关键字定位操作。more名单中内置了若干快捷键，常用的有H（获得帮助信息），Enter（向下翻滚一行），空格（向下滚动一屏），Q（退出命令）。more命令从前向后读取文件，因此在启动时就加载整个文件。

该命令一次显示一屏文本，满屏后停下来，并且在屏幕的底部出现一个提示信息，给出至今己显示的该文件的百分比：–More–（XX%）

more的语法：more文件名

Enter 向下n行，需要定义，默认为1行

Ctrl f 向下滚动一屏

空格键 向下滚动一屏

Ctrl b返回上一屏

= 输出当前行的行号

:f 输出文件名和当前行的行号

v 调用vi编辑器

!命令调用Shell，并执行命令

q退出more

5、sed

这个命令可以查找日志文件特定的一段 , 根据时间的一个范围查询，可以按照行号和时间范围查询按照行号

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `sed -n``'5,10p'``filename这样你就可以只查看文件的第``5``行到第``10``行。` |

按照时间段

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `sed -n``'/2014-12-17 16:17:20/,/2014-12-17 16:17:36/p'``test.log` |

6、less

less命令在查询日志时，一般流程是这样的

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `less log.log shift + G 命令到文件尾部 然后输入 ？加上你要搜索的关键字例如 ？``1213``按 n 向上查找关键字 shift+n 反向查找关键字 less与more类似，使用less可以随意浏览文件，而more仅能向前移动，不能向后移动，而且 less 在查看 之前不会加载整个文件。 less log2013.log 查看文件 ps -ef | less ps查看进程信息并通过less分页显示 history | less 查看命令历史使用记录并通过less分页显示 less log2013.log log2014.log 浏览多个文件` |

常用命令参数：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `less与more类似，使用less可以随意浏览文件，而more仅能向前移动，不能向后移动，而且 less 在查看 之前不会加载整个文件。 less log2013.log 查看文件 ps -ef | less ps查看进程信息并通过less分页显示 history | less 查看命令历史使用记录并通过less分页显示 less log2013.log log2014.log 浏览多个文件常用命令参数： -b <缓冲区大小> 设置缓冲区的大小 -g 只标志最后搜索的关键词 -i 忽略搜索时的大小写 -m 显示类似more命令的百分比 -N 显示每行的行号 -o <文件名> 将less 输出的内容在指定文件中保存起来 -Q 不使用警告音 -s 显示连续空行为一行 /字符串：向下搜索``"字符串"``的功能 ?字符串：向上搜索``"字符串"``的功能 n：重复前一个搜索（与 / 或 ? 有关） N：反向重复前一个搜索（与 / 或 ? 有关） b 向后翻一页 h 显示帮助界面 q 退出less命令` |

一般本人查日志配合应用的其他命令

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1 | `history``// 所有的历史记录 history | grep XXX // 历史记录中包含某些指令的记录 history | more // 分页查看记录 history -c // 清空所有的历史记录 !! 重复执行上一个命令 查询出来记录后选中 : !323` |

### 8、建立软链接(快捷方式)，以及硬链接的命令

答：

[复制代码](https://www.nowcoder.com/discuss/352910181216423936#)

|     |     |
| --- | --- |
| 1<br>2 | `软链接： ln -s slink source`<br>`硬链接： ln link source` |

9、目录创建用什么命令？创建文件用什么命令？复制文件用什么命令？

答：

创建目录： mkdir

创建文件：典型的如touch，vi也可以创建文件，其实只要向一个不存在的文件输出，都会创建文件复制文件： cp7文件权限修改用什么命令？格式是怎么样的？

文件权限修改： chmod

格式如下：

chmodu+xfile 给 file 的属主增加执行权限 chmod 751 file 给 file 的属主分配读、写、执行(7)的权限，给 file 的所在组分配读、执行(5)的权限，给其他用户分配执行(1)的权限chmodu=rwx,g=rx,o=xfile 上例的另一种形式 chmod =r file 为所有用户分配读权限chmod444file 同上例 chmod a-wx,a+r file 同上例$ chmod -R u+r directory 递归地给 directory 目录下所有文件和子目录的属主分配读的权限

10、查看文件内容有哪些命令可以使用？

答：

vi文件名 #编辑方式查看，可修改

cat文件名 #显示全部文件内容

more文件名 #分页显示文件内容

less文件名#与 more 相似，更好的是可以往前翻页

tail 文件名 #仅查看尾部，还可以指定行数

head 文件名 #仅查看头部,还可以指定行数

篇幅原因，不能一一展现。以上完整题目答案资料，及更多java大小厂面经真题分享： 点赞 评论 ：学习

![](https://uploadfiles.nowcoder.com/images/20220508/174636739_1652012300778/95906EEC9C18E9AABB8DA235B4999DB9)

[#笔试题目#](https://www.nowcoder.com/creation/subject/c475208ba36c4020820bf4f3805720b9) [#面经#](https://www.nowcoder.com/creation/subject/928d551be73f40db82c0ed83286c8783) [#Java#](https://www.nowcoder.com/creation/subject/a85612c5fb4a412e885dd11f1293c822) [#Linux#](https://www.nowcoder.com/creation/subject/5196ff3737b149e783ad7c8740100733) [#技术栈#](https://www.nowcoder.com/creation/subject/f35a28e47b3c43ae86764e8c38136348) [#题解#](https://www.nowcoder.com/creation/subject/42d4c11aceda45ccbd16b2e4c118eb6d) [#读书笔记#](https://www.nowcoder.com/creation/subject/9b21d71df1ac413a81e49368da00984e)

提示

订阅专刊

43
点赞成功，聊一聊 >
44161

161

11

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

浏览
1.6w

邀请牛友回答

换一批关 闭

收到1人送花1朵

送花成功，捎句话 >

![](<Base64-Image-Removed>)

1

快捷表情

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763718/D9FDAE9918A39C99254A9D8D179628E5)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763890/5072FC474BC4CF9234FABC22E54A999A)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763465/6F6CA9EC40A6F04C7838E4DE94A77241)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763434/7A0C3C39D0D8037360A2B600921D52C5)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763573/A95184503DF1D65798194F12FCEDE5C5)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763930/8B36D115CE5468E380708713273FEF43)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763673/6409638369766F7FC4FBE09BD8BF58AB)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763988/A36B51811C2F08F8B587B123A106E546)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763618/25C98751B489394CFB21CE09AE55BC97)

![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553764153/6078915537BAB7E95CE12422FD944756)

﻿

畅所欲言吧～

共0张，最多还能上传9张

图片

最近使用

热门话题

话题加载中...

话题

表情

- 默认



emoji



动作篇



校招顺利



打工篇









😀



😁



😂



🤣



😃



😄



😅



😆



😉



😊



😋



😎



😍



😘



🥰



😗



😙



😚



🙂



🤗



🤩



🤔



🤨



😐



😑



😶



🙄



😏



😣



😥



😮



🤐



😯



😪



😫



😴



😌



😛



😜



😝



🤤



😒



😓



😔



😕









1/2


同时转发到我的动态

评论

全部评论(43)

推荐最新楼层

![](https://static.nowcoder.com/fe/file/oss/1681101031872EGDPQ.png)

暂无评论，快来抢首评~

![](<Base64-Image-Removed>)

相关推荐

![](<Base64-Image-Removed>)

![头像](<Base64-Image-Removed>)

[千影逐风](https://www.nowcoder.com/users/501543961) [![](https://static.nowcoder.com/fe/file/images/common/honorLevel/level_mini_7.png)](https://www.nowcoder.com/users/501543961)

05-06 20:54

[门头沟学院 Java](https://www.nowcoder.com/users/501543961)

发私信

取消发送

[从手足无措到慢慢踏实](https://www.nowcoder.com/discuss/881824224908738560?sourceSSR=post)

[刷到这个话题，一下子就想起了我实习第一天的样子，紧张到手心冒汗，连喝水都不敢大声，现在回头看，又好笑又怀念。我是双非计算机本科，在杭州一家互联网公司做后端开发实习，实习第一天，我提前半个小时就到了公司楼下，在楼下便利店坐了20分钟，反复对着手机里的自我介绍稿默念，生怕进去说错话，连呼吸都在紧张。到了上班时间，我给mentor发了消息，他下来接我，带我进了办公区。第一关就是认人，他带着我挨个工位介绍组里的同事，“这是张哥，负责架构的”“这是李姐，负责业务开发”“这是王哥，... 查看更多](https://www.nowcoder.com/discuss/881824224908738560?sourceSSR=post)

[实习第一天，你在干什么](https://www.nowcoder.com/creation/subject/47640d0bfa2748798b2be0da6f141fd0?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

1评论收藏

分享

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

屏蔽该作者全部动态
关注作者
举报

举报

取 消确 定

取消确定

设置该动态仅作者可见

取消确认举报

指定话题只能指定一个话题

取消确定

指定圈子一条动态只能指定一个圈子

取消确定

取消确认编辑

取消确认编辑

![头像](<Base64-Image-Removed>)

[我wyn实名上网](https://www.nowcoder.com/users/270936111) [![](https://static.nowcoder.com/fe/file/images/common/honorLevel/level_mini_2.png)](https://www.nowcoder.com/users/270936111)

05-06 22:50

[东北农业大学 C++](https://www.nowcoder.com/users/270936111)

发私信

取消发送

[c++简历求指导](https://www.nowcoder.com/discuss/881853272561106944?sourceSSR=post)

[投了十几个无人回应... 查看更多](https://www.nowcoder.com/discuss/881853272561106944?sourceSSR=post)

14收藏

分享

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

屏蔽该作者全部动态
关注作者
举报

举报

取 消确 定

取消确定

设置该动态仅作者可见

取消确认举报

指定话题只能指定一个话题

取消确定

指定圈子一条动态只能指定一个圈子

取消确定

取消确认编辑

取消确认编辑

![头像](<Base64-Image-Removed>)

[Orio\_](https://www.nowcoder.com/users/37367753) [![](<Base64-Image-Removed>)](https://www.nowcoder.com/users/37367753)

04-27 03:01

[早稲田大学 Java](https://www.nowcoder.com/users/37367753)

发私信

取消发送

[0约面已急哭](https://www.nowcoder.com/feed/main/detail/aa3b2cddcbf245129a0001670661041b?sourceSSR=post)

[是不是简历有什么问题。😭\\
阿里简历挂\\
拼多多笔试挂\\
腾讯 美团 字节 百度 得物 泡池子... 查看更多](https://www.nowcoder.com/feed/main/detail/aa3b2cddcbf245129a0001670661041b?sourceSSR=post)

![](https://uploadfiles.nowcoder.com/files/20250508/120063338_1746688133065/nx.png)牛客72191338...：可能是时间点的问题，四月底机会确实会相对少点，但佬这个学历摆在这，会有机会的![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763718/D9FDAE9918A39C99254A9D8D179628E5)

[简历中的项目经历要怎么写](https://www.nowcoder.com/creation/subject/7815f1f9e7964f90b835ef2366b49e49?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

9298

分享

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

屏蔽该作者全部动态
关注作者
举报

举报

取 消确 定

取消确定

设置该动态仅作者可见

取消确认举报

指定话题只能指定一个话题

取消确定

指定圈子一条动态只能指定一个圈子

取消确定

取消确认编辑

取消确认编辑

![头像](<Base64-Image-Removed>)

[mummer](https://www.nowcoder.com/users/888544631) [![](<Base64-Image-Removed>)](https://www.nowcoder.com/users/888544631)

04-29 09:43

[中国石油大学（华东） Java](https://www.nowcoder.com/users/888544631)

发私信

取消发送

[暑期实习&&日常实习毫无收获](https://www.nowcoder.com/feed/main/detail/036f6de71e3c4e92bbe1ba6d1ce85319?sourceSSR=post)

[牛友们谁能帮忙看一下为什么没约面试的 从四月初投到现在…..是我的简历有问题吗![](https://uploadfiles.nowcoder.com/images/20220815/318889480_1660553763930/8B36D115CE5468E380708713273FEF43)... 查看更多](https://www.nowcoder.com/feed/main/detail/036f6de71e3c4e92bbe1ba6d1ce85319?sourceSSR=post)

[我的求职进度条](https://www.nowcoder.com/creation/subject/81e9423ee42941edb230faabe67267b2?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

22收藏

分享

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

屏蔽该作者全部动态
关注作者
举报

举报

取 消确 定

取消确定

设置该动态仅作者可见

取消确认举报

指定话题只能指定一个话题

取消确定

指定圈子一条动态只能指定一个圈子

取消确定

取消确认编辑

取消确认编辑

![头像](<Base64-Image-Removed>)

[RockyT](https://www.nowcoder.com/users/571831313) [![](<Base64-Image-Removed>)](https://www.nowcoder.com/users/571831313)

05-06 23:10

[华南师范大学 深度学习](https://www.nowcoder.com/users/571831313)

发私信

取消发送

[26届本硕双2应届生已经对校招绝望了，想问下过两个月社招进各种各样的大厂外包要做什么准备](https://www.nowcoder.com/discuss/881858522718232576?sourceSSR=post)

[rt，bg是本硕双二，做计算成像相关方向，但是因为我水平太菜太菜太菜太菜太菜太菜太菜太菜太菜太菜太菜太菜太菜了，到现在也没有找到合适的工作。还剩不到两个月我觉得已经不可能找到工作了，打算毕业后刷刷代码然后社招应聘外包岗位了。想问下各位目前大厂外包有哪些，我目前知道的只有华为od，其他厂比如BAT，五大手机厂，或者与图像岗位有关的公司等等有投递途径吗？要进这些知名大厂的外包需要做刷多少代码题和背什么八股？#牛客AI配图神器#... 查看更多](https://www.nowcoder.com/discuss/881858522718232576?sourceSSR=post)

[第一份工作一定要去大厂吗](https://www.nowcoder.com/creation/subject/e3dfa7b81e5043bda88e86f3850d8082?entranceType_var=%E5%86%85%E5%AE%B9%E6%9D%A1%E7%9B%AE)

13收藏

分享

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

屏蔽该作者全部动态
关注作者
举报

举报

取 消确 定

取消确定

设置该动态仅作者可见

取消确认举报

指定话题只能指定一个话题

取消确定

指定圈子一条动态只能指定一个圈子

取消确定

取消确认编辑

取消确认编辑

43

点赞成功，聊一聊 >

44

161

标题

161

分享

- 转发到动态
- 复制链接
- 微信
- QQ
- 微博

分享到微信

分享给好友

暂不保存保存图片

评论

![](https://www.nowcoder.com/discuss/352910181216423936) 提到的真题

返回内容


招聘动态

[查看更多](https://www.nowcoder.com/jobs/school/schedule?pageSource=105)

[![](<Base64-Image-Removed>)\\
\\
“联宝杯”\\
\\
2026 大学生创新大赛](https://www.nowcoder.com/jump?type=ad&source=105&companyId=6267&url=https%3A%2F%2Fcompetition.nowcoder.com%2F226%2Fintroduce%3Fchannel%3D26lcfczpdt&entityId=13201)

[![](<Base64-Image-Removed>)\\
\\
上海人工智能实验室\\
\\
2026年春季校园招聘](https://www.nowcoder.com/jump?type=ad&source=105&companyId=13352&url=https%3A%2F%2Fwww.nowcoder.com%2Fjobs%2Fcompany-project%3FprojectId%3D2600&entityId=13205)

[![](<Base64-Image-Removed>)\\
\\
联想\\
\\
27届暑期实习](https://www.nowcoder.com/jump?type=ad&source=105&companyId=958&url=https%3A%2F%2Ftalent.lenovo.com.cn%2Fposition%3FprojectType%3D2&entityId=13182)

[![](<Base64-Image-Removed>)\\
\\
联想\\
\\
26届补录](https://www.nowcoder.com/jump?type=ad&source=105&companyId=958&url=https%3A%2F%2Ftalent.lenovo.com.cn%2Fposition%3FprojectType%3D1&entityId=13135)

[![](<Base64-Image-Removed>)\\
\\
27届校招宝典](https://www.nowcoder.com/jump?type=ad&source=105&companyId=1727&url=https%3A%2F%2Flink.zhiyeapp.com%2Fr%2FxEKxW1g0pC&entityId=13136)

[![](<Base64-Image-Removed>)\\
\\
厦门银行\\
\\
2026届春季校园招聘](https://www.nowcoder.com/jump?type=ad&source=105&companyId=2932&url=https%3A%2F%2Fxmccb.zhaopin.com%2F&entityId=13209)

[![](<Base64-Image-Removed>)\\
\\
快手\\
\\
27届实习超多转正机会](https://www.nowcoder.com/jump?type=ad&source=105&companyId=898&url=https%3A%2F%2Fcampus.kuaishou.cn%2Frecruit%2Fcampus%2Fe%2F%23%2Fcampus%2Fjobs%3Fcode%3DcampusaeZwiKwdB%26pageNum%3D1%26positionNatureCode%3Dintern&entityId=13150)

[![](<Base64-Image-Removed>)\\
\\
联宝科技](https://www.nowcoder.com/jump?type=ad&source=105&companyId=6267&url=https%3A%2F%2Fcompetition.nowcoder.com%2F226%2Fintroduce%3Fchannel%3D26lcfcqzzd&entityId=13203)

![](<Base64-Image-Removed>)

## 全站热榜

更多

- [![](<Base64-Image-Removed>)\\
\\
毕业了，有些话只能藏在心里了...毕业了，有些话只能藏在心里了\\
\\
3.0W](https://www.nowcoder.com/feed/main/detail/ab8b9756319640bbb404278c5357b7c9)
- [![](<Base64-Image-Removed>)\\
\\
毕业啦！我们要一起去广...毕业啦！我们要一起去广州打拼啦！\\
\\
2.8W](https://www.nowcoder.com/feed/main/detail/2edb812890b8454598ac7b613a281b13)
- [![](<Base64-Image-Removed>)\\
\\
字节 中国交易与广告 后端一面...字节 中国交易与广告 后端一面\\
\\
2.0W](https://www.nowcoder.com/feed/main/detail/068e351d710247498ff5912a43991dd2)
- [![](<Base64-Image-Removed>)\\
\\
大三下了 学校不放实习怎么办...大三下了 学校不放实习怎么办\\
\\
1.1W](https://www.nowcoder.com/feed/main/detail/93aca64499914701bb9e3cf0f8f1406f)
- [![](<Base64-Image-Removed>)\\
\\
华为暑期实习...华为暑期实习\\
\\
1.1W](https://www.nowcoder.com/feed/main/detail/5d025cf3aa0d4dd9a7aa839b140897bd)
- [![](<Base64-Image-Removed>)\\
\\
从腾讯到阿里感，谢一路...从腾讯到阿里感，谢一路走来的自己\\
\\
1.0W](https://www.nowcoder.com/feed/main/detail/158f919b01f343e5863c036355d73def)
- [![](<Base64-Image-Removed>)\\
\\
211本，130投0...211本，130投0面，agent应用开发，简历求助！\\
\\
9815](https://www.nowcoder.com/feed/main/detail/2b6facd1e7874bf48bb2f7fe3348a3e2)
- [![](<Base64-Image-Removed>)\\
\\
别人：阿里 字节 腾讯...别人：阿里 字节 腾讯\\
\\
8870](https://www.nowcoder.com/feed/main/detail/99094f4112244d39a33791ffc494e52c)
- [![](<Base64-Image-Removed>)\\
\\
从阿里被裁到快手升P6，...从阿里被裁到快手升P6，我花了四年\\
\\
8562](https://www.nowcoder.com/discuss/881637930345713664)
- [![](<Base64-Image-Removed>)\\
\\
爸，你说的对，四年真的很快...爸，你说的对，四年真的很快\\
\\
6438](https://www.nowcoder.com/feed/main/detail/d2eb0556e6a54a158a906d59fdf3390d)

![](<Base64-Image-Removed>)

## 创作者周榜

更多

正在热议

更多

[#如果春招能重来，我会\_\_\_#\\
\\
27788次浏览275人参与](https://www.nowcoder.com/creation/subject/c0a982e922424cc3877c9ee5a1fc877f?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#机械人还在等华为开奖吗？#\\
\\
338917次浏览1652人参与](https://www.nowcoder.com/creation/subject/57526a547e41496d9f30add0e8c38bd7?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#联宝杯大学生创新大赛，你的技术值得产业级答案#\\
\\
49421次浏览748人参与](https://www.nowcoder.com/creation/subject/6ac8bfccd3524a549cde7c76718fd292?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#这个offer值得去吗？#\\
\\
28815次浏览208人参与](https://www.nowcoder.com/creation/subject/48a5dbf5510349cf935fce0adc49df94?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#24秋招避雷总结#\\
\\
1019224次浏览7097人参与](https://www.nowcoder.com/creation/subject/15cdb5b8559a4137978419a283a1dd63?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#你会因为行情，降低找工作标准吗？#\\
\\
42968次浏览313人参与](https://www.nowcoder.com/creation/subject/20a3a2d6b72749858cebef61909cde69?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#在爱玛，骑向未来#\\
\\
19274次浏览379人参与](https://www.nowcoder.com/creation/subject/597a95bb19094530a548b413e80785df?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#大学最后一个寒假，我想……#\\
\\
103128次浏览846人参与](https://www.nowcoder.com/creation/subject/34e009504eed47809f40a9cffac5be0e?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#机械求职避坑tips#\\
\\
103525次浏览589人参与](https://www.nowcoder.com/creation/subject/b073a08183be4a90a30f974b66c24560?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#华为池子有多大#\\
\\
177516次浏览928人参与](https://www.nowcoder.com/creation/subject/7effda96005a409a9a43584f4341df16?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#刚入职就\_\_\_\_，这样正常吗？#\\
\\
148102次浏览708人参与](https://www.nowcoder.com/creation/subject/c3a3c1874c424429ae0a4aab56c9178c?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#记录我的毕业季#\\
\\
3531次浏览96人参与](https://www.nowcoder.com/creation/subject/74ac297129cb4b9e80f790bda0fb4b27?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#去年你投递实习了吗？#\\
\\
32754次浏览339人参与](https://www.nowcoder.com/creation/subject/42b46203df4044578a5cc4dc6d3a86f3?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#字节开奖#\\
\\
158363次浏览776人参与](https://www.nowcoder.com/creation/subject/aa208513b8f04422bf8764691080a6fc?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#做完笔试后你收到面试了吗？#\\
\\
65126次浏览309人参与](https://www.nowcoder.com/creation/subject/def4d7f9fa824da78b597b62b704c0ad?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#今年找实习到底有多难？#\\
\\
104757次浏览489人参与](https://www.nowcoder.com/creation/subject/d1bb7ec0151e4189b78c3b5c125ad422?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#秋招盘点:机械人值得去的企业#\\
\\
106390次浏览741人参与](https://www.nowcoder.com/creation/subject/5351add2a2cd41bb8f80675189beb2ec?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#远程面试的尴尬瞬间#\\
\\
363512次浏览2060人参与](https://www.nowcoder.com/creation/subject/6395c21e0d3e4acd97abdea130d0afb2?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#秋招前后对offer的期望对比#\\
\\
551489次浏览3464人参与](https://www.nowcoder.com/creation/subject/fcb9a4a4022c4abb8ff62db39cebe962?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#面试体验最好和最差的公司#\\
\\
47987次浏览197人参与](https://www.nowcoder.com/creation/subject/1688daf085774f6f8975ac80b677c8d7?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#有深度的简历长什么样？#\\
\\
60656次浏览771人参与](https://www.nowcoder.com/creation/subject/ca0132de61684b2db58626392a736e54?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F) [#金融财会交流会#\\
\\
150958次浏览498人参与](https://www.nowcoder.com/creation/subject/055c9b9db6254122bb529fa63abebb5f?entranceType_var=%E4%BE%A7%E8%BE%B9%E6%A0%8F)

![](https://uploadfiles.nowcoder.com/images/20180815/59_1534321710941_41A541F87AE349E1D829B1B0B95C955D)

扫描二维码，进入QQ群


![](https://static.nowcoder.com/fe/file/oss/1646799945943MMDZY.png)

扫描二维码，关注牛客公众号


每天登录，牛客都会送你一朵免费的花。

[牛客网](https://www.nowcoder.com/simple/home)

[牛客网在线编程](https://www.nowcoder.com/simple/question)

[牛客网题解](https://www.nowcoder.com/simple/question/comment)

[牛客企业服务](https://www.nowcoder.com/simple/hr)

![](https://static.nowcoder.com/fe/file/oss/1688615753578AHSOY.png)![](https://static.nowcoder.com/fe/file/oss/1688615782322IEGPA.png)

![求职者](https://static.nowcoder.com/fe/file/oss/1689329076959EQVSW.png)我是求职者


![我是求职者](https://static.nowcoder.com/fe/file/oss/1688615533943XCPPM.png)

![招聘者](https://static.nowcoder.com/fe/file/oss/1689329036834EOKNP.png)我是招聘方


![我是招聘者](https://static.nowcoder.com/fe/file/oss/1688615569777HLTAY.png)

校招快人一步

- 注册登录


- 密码登录


- 中国 +86
- 美国 +1
- 加拿大 +1
- 俄罗斯 +7
- 埃及 +20
- 南非 +27
- 希腊 +30
- 荷兰 +31
- 比利时 +32
- 法国 +33
- 西班牙 +34
- 匈牙利 +36
- 意大利 +39
- 罗马尼亚 +40
- 瑞士 +41
- 奥地利 +43
- 英国 +44
- 丹麦 +45
- 瑞典 +46
- 斯瓦尔巴岛 +47
- 波兰 +48
- 德国 +49
- 秘鲁 +51
- 墨西哥 +52
- 古巴 +53
- 阿根廷 +54
- 巴西 +55
- 智利 +56
- 哥伦比亚 +57
- 委内瑞拉 +58
- 马来西亚 +60
- 澳大利亚 +61
- 印度尼西亚 +62
- 菲律宾 +63
- 新西兰 +64
- 新加坡 +65
- 泰国 +66
- 日本 +81
- 韩国 +82
- 越南 +84
- 土耳其 +90
- 印度 +91
- 巴基斯坦 +92
- 阿富汗 +93
- 斯里兰卡 +94
- 缅甸 +95
- 伊朗 +98
- 南苏丹 +211
- 摩洛哥 +212
- 阿尔及利亚 +213
- 突尼斯 +216
- 利比亚 +218
- 冈比亚 +220
- 塞内加尔 +221
- 毛里塔尼亚 +222
- 马里 +223
- 几内亚 +224
- 科特迪瓦 +225
- 布基纳法索 +226
- 尼日尔 +227
- 多哥 +228
- 贝宁 +229
- 毛里求斯 +230
- 利比里亚 +231
- 塞拉利 +232
- 加纳 +233
- 尼日利亚 +234
- 乍得 +235
- 中非 +236
- 喀麦隆 +237
- 佛得角 +238
- 圣多美和普林西比 +239
- 赤道几内亚 +240
- 加蓬 +241
- 刚果 +242
- 刚果 +243
- 安哥拉 +244
- 几内亚比绍 +245
- 阿森松岛 +247
- 塞舌尔 +248
- 苏丹 +249
- 卢旺达 +250
- 埃塞俄比亚 +251
- 索马里 +252
- 吉布提 +253
- 肯尼亚 +254
- 坦桑尼亚 +255
- 乌干达 +256
- 布隆迪 +257
- 莫桑比克 +258
- 赞比亚 +260
- 马达加斯加 +261
- 留尼汪 +262
- 津巴布韦 +263
- 纳米尼亚 +264
- 马拉维 +265
- 莱索托 +266
- 博茨瓦纳 +267
- 斯威士兰 +268
- 马约特 +269
- 阿鲁巴 +297
- 法罗群岛 +298
- 格陵兰 +299
- 直布罗陀 +350
- 葡萄牙 +351
- 卢森堡 +352
- 爱尔兰 +353
- 冰岛 +354
- 阿尔巴尼亚 +355
- 马耳他 +356
- 塞浦路斯 +357
- 芬兰 +358
- 保加利亚 +359
- 立陶宛 +370
- 拉脱维亚 +371
- 爱沙尼亚 +372
- 摩尔多瓦 +373
- 亚美尼亚 +374
- 白俄罗斯 +375
- 安道尔 +376
- 摩纳哥 +377
- 圣马力诺 +378
- 乌克兰 +380
- 塞尔维亚和黑山 +381
- 黑山 +382
- 克罗地亚 +385
- 斯洛文尼亚 +386
- 波黑 +387
- 马其顿 +389
- 捷克 +420
- 斯洛伐克 +421
- 列支敦士登 +423
- 伯利兹 +501
- 危地马拉 +502
- 萨尔瓦多 +503
- 洪都拉斯 +504
- 尼加拉瓜 +505
- 哥斯达黎加 +506
- 巴拿马 +507
- 圣皮埃尔和密克隆 +508
- 海地 +509
- 瓜德罗普 +590
- 玻利维亚 +591
- 圭亚那 +592
- 厄瓜多尔 +593
- 法属圭亚那 +594
- 巴拉圭 +595
- 马提尼克 +596
- 苏里南 +597
- 乌拉圭 +598
- 荷属安的列斯 +599
- 东帝汶 +670
- 文莱 +673
- 巴布亚新几内亚 +675
- 汤加 +676
- 所罗门群岛 +677
- 瓦努阿图 +678
- 斐济 +679
- 帕劳 +680
- 库克群岛 +682
- 萨摩亚 +685
- 基里巴斯 +686
- 新喀里多尼亚 +687
- 法属波利尼西亚 +689
- 中国香港 +852
- 中国澳门 +853
- 柬埔寨 +855
- 老挝 +856
- 孟加拉国 +880
- 中国台湾 +886
- 马尔代夫 +960
- 黎巴嫩 +961
- 约旦 +962
- 叙利亚 +963
- 伊拉克 +964
- 科威特 +965
- 沙特阿拉伯 +966
- 也门 +967
- 阿曼 +968
- 巴勒斯坦 +970
- 阿拉伯联合酋长国 +971
- 以色列 +972
- 巴林 +973
- 卡塔尔 +974
- 不丹 +975
- 蒙古 +976
- 尼泊尔 +977
- 塔吉克斯坦 +992
- 土库曼斯坦 +993
- 阿塞拜疆 +994
- 格鲁吉亚 +995
- 吉尔吉斯斯坦 +996
- 乌兹别克斯坦 +998
- 巴哈马 +1242
- 巴巴多斯 +1246
- 安圭拉 +1264
- 安提瓜和巴布达 +1268
- 英属维尔京群岛 +1284
- 美属维尔京群岛 +1340
- 开曼群岛 +1345
- 百慕大 +1441
- 格林纳达 +1473
- 特克斯和凯科斯群岛 +1649
- 蒙特塞拉特 +1664
- 关岛 +1671
- 东萨摩亚 +1684
- 荷属圣马丁 +1721
- 圣卢西亚 +1758
- 多米尼克 +1767
- 圣文森特和格林纳丁斯 +1784
- 波多黎各 +1787
- 多米尼加 +1809
- 特立尼达和多巴哥 +1868
- 圣基茨和尼维斯 +1869
- 牙买加 +1876

发送验证码

注册/登录


同意
[《注册协议》](https://static.nowcoder.com/protocol/register.html)
和
[《隐私政策》](https://static.nowcoder.com/protocol/privacy-policy.html)

未注册手机验证后自动登录

or

扫码登录

![](https://static.nowcoder.com/fe/file/oss/1689128604878FSSFW.png)
微信
![](https://static.nowcoder.com/fe/file/oss/1689128504959WPAUF.png)
牛客
![](https://static.nowcoder.com/fe/file/oss/1689128414924WQVUP.png)
QQ
![](https://static.nowcoder.com/fe/file/oss/1689128559948AKPYS.png)
微博


扫码关注【牛客】即可登录

登录或完成注册即代表你

同意 [《注册协议》](https://static.nowcoder.com/protocol/register.html) 和 [《隐私政策》](https://static.nowcoder.com/protocol/privacy-policy.html)

牛客优聘APP 随时开聊