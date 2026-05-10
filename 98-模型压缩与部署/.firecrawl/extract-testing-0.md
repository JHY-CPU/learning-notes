# 测试面试题集| 自动化测试与性能测试篇（附答案） 原创 - CSDN博客

URL: https://blog.csdn.net/okcross0/article/details/152041194

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

# 测试面试题集 \| 自动化测试与性能测试篇（附答案）

最新推荐文章于 2026-05-05 22:37:37 发布

原创于 2025-09-24 15:04:12 发布·330 阅读

·![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png)
3


·![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png)
7
·

CC 4.0 BY-SA版权

版权声明：本文为博主原创文章，遵循 [CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/) 版权协议，转载请附上原文出处链接和本声明。


文章标签：

[#python](https://so.csdn.net/so/search/s.do?q=python&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#功能测试](https://so.csdn.net/so/search/s.do?q=%E5%8A%9F%E8%83%BD%E6%B5%8B%E8%AF%95&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#测试用例](https://so.csdn.net/so/search/s.do?q=%E6%B5%8B%E8%AF%95%E7%94%A8%E4%BE%8B&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#selenium](https://so.csdn.net/so/search/s.do?q=selenium&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#自动化](https://so.csdn.net/so/search/s.do?q=%E8%87%AA%E5%8A%A8%E5%8C%96&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#单元测试](https://so.csdn.net/so/search/s.do?q=%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#测试工具](https://so.csdn.net/so/search/s.do?q=%E6%B5%8B%E8%AF%95%E5%B7%A5%E5%85%B7&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art)

本系列文章总结归纳了一些软件测试工程师常见的面试题，主要来源于个人面试遇到的、网络搜集（完善）、工作日常讨论等，分为以下十个部分，供大家参考。如有错误的地方，欢迎指正。有更多的面试题或面试中遇到的坑，也欢迎补充分享。希望大家都能找到满意的工作，共勉之！~

##### 自动化测试相关

包含 Selenium、Appium 和接口测试。

01. **自动化代码中，用到了哪些设计模式？**

    - 单例模式

    - 工厂模式

    - PO模式

    - 数据驱动模式
02. **什么是断言？**

    - 检查一个条件，如果它为真，就不做任何事，用例通过。如果它为假，则会抛出 AssertError 并且包含错误信息。
03. **UI 自动化测试中，如何做集群？**

    - Selenium Grid，分布式执行用例

    - Appium 使用 STF 管理多设备

    - Docker+K8S 管理集群
04. **怎么对含有验证码的功能进行自动化测试？**

    - 万能验证码

    - 测试环境屏蔽验证

    - 其他操作不推荐
05. **如何优化和提高 Selenium 脚本的执行速度？**

    - 尽量使用 `by_css_selector()` 方法

    - `by_css_selector()` 方法的执行速度比 `by_id()` 方法的更快，因为源码中 `by_id()` 方法会被自动转成 `by_css_selector()` 方法处理；

    - 使用等待时，尽量使用显示等待，少用 `sleep()`，尽量不用隐式等待；

    - 尽量减少不必要的操作：可以直接访问页面的，不要通过点击操作访问；

    - 并发执行测试用例：同时执行多条测试用例，降低用例间的耦合；

    - 有些页面加载时间长，可以中断加载；
06. **接口测试能发现哪些问题？**

    - 可以发现很多在页面上操作发现不了的 bug；

    - 检查系统的异常处理能力；

    - 检查系统的安全性、稳定性；

    - 前端随便变，接口测好了，后端不用变；

    - 可以测试并发情况，一个账号，同时（大于 2 个请求）对最后一个商品下单，或不同账号，对最后一个商品下单；

    - 可以修改请求参数，突破前端页面输入限制（如金额）；
07. **Selenium 中隐藏元素如何定位？**

    - 如果单纯的定位的话，隐藏元素和普通不隐藏元素定位没啥区别，用正常定位方法就行了（这个很多面试官也搞不清楚）；

    - 元素的属性隐藏和显示，主要是 `type="hidden"` 和 `style="display: none;"` 属性来控制的，接下来在元素属性里面让它隐藏，隐藏元素可以正常定位到，只是不能操作（定位元素和操作元素是两码事，很多初学者傻傻分不清楚），操作元素是 `click,clear,send_keys` 这些方法；

    - JS 操作隐藏元素；
08. **如何判断一个页面上元素是否存在？**

    - 方法一：用 try…except…

    - 方法二：用 elements 定义一组元素方法，判断元素是否存在,存在返回 True,不存返回 False

    - 方法三：结合 WebDriverWait 和 expected\_conditions 判断（推荐）
09. **如何提高脚本的稳定性？**

    - 不要右键复制 xpath(十万八千里那种路径，肯定不稳定)，自己写相对路径，多用 id 为节点查找；

    - 定位没问题，第二个影响因素那就是等待了，sleep 等待尽量少用（影响执行时间）；

    - 定位元素方法重新封装，结合 WebDriverWait 和 expected\_conditions 判断元素方法，自己封装一套定位元素方法；
10. **如何定位动态元素？**

    - 动态元素有 2 种情况，一个是属性动态，比如 id 是动态的，定位时候，那就不要用 id 定位就是了；

    - 还有一种情况动态的，那就是这个元素一会在页面上方，一会在下方，飘忽不定的动态元素，定位方法也是一样，按 f12，根据元素属性定位（元素的 tag、name的步伐属性是不会变的，动的只是 class 属性和 styles 属性）；
11. **如何通过子元素定位父元素**

    - 使用element.parent方法
12. **平常遇到过哪些问题? ?如何解决的**

    - 可以把平常遇到的元素定位的一些坑说下，然后说下为什么没定位到，比如动态 id、有 iframe、没加等待等因素；
13. **一个元素明明定位到了，点击无效（也没报错），如果解决？**

    - 使用 JS 点击，Selenium 有时候点击元素是会失效；
14. **测试的数据你放在哪?**

    - 对于账号密码，这种管全局的参数，可以用命令行参数，单独抽出来，写的配置文件里（如 ini）；

    - 对于一些一次性消耗的数据，比如注册，每次注册不一样的数，可以用随机函数生成；

    - 对于一个接口有多组测试的参数，可以参数化，数据放 YAML,Text,JSON,Excel 都可以；

    - 对于可以反复使用的数据，比如订单的各种状态需要造数据的情况，可以放到数据库，每次数据初始化，用完后再清理；

    - 对于邮箱配置的一些参数，可以用 ini 配置文件；

    - 对于全部是独立的接口项目，可以用数据驱动方式，用 excel/csv 管理测试的接口数据；

    - 对于少量的静态数据，比如一个接口的测试数据，也就 2-3 组，可以写到 py脚本的开头，十年八年都不会变更的；
15. **什么是数据驱动，如何参数化？**

    - 参数化的思想是代码用例写好了后，不需要改代码，只需维护测试数据就可以了，并且根据不同的测试数据生成多个用例；
16. **其他接口都需要登录接口的信息，怎么去让这个登录的接口只在其他接口调用一次？**

    - 使用单例模式

    - 使用自定义缓存机制

    - 使用测试框架中的 setup 机制

    - pytest 中 fixture 机制
17. **接口产生的垃圾数据如何清理？**

    - 造数据和数据清理，需用 python 连数据库了，做增删改查的操作测试用例前置操作，setUp 做数据准备后置操作，tearDown 做数据清理
18. **怎么用接口案例去覆盖业务逻辑？**

    - 考虑不同的业务场景，一个接口走过的流程是什么样的，流程的逻辑是什么样的，什么样的参数会有什么样的结果，多场景覆盖；

##### 性能篇

1. **性能测试指标包括哪些**

   - 最大并发用户数，HPS（点击率）、事务响应时间、每秒事务数、每秒点击量、吞吐量、CPU 使用率、物理内存使用、网络流量使用等。

   - 前端需主要关注的点是：

     - 响应时间：用户从客户端发出请求，并得到响应，以及展示出来的整个过程的时间。

     - 加载速度：通俗的理解为页面内容显示的快慢。

     - 流量：所消耗的网络流量。
   - 后端需主要关注的是：

     - 响应时间：接口从请求到响应、返回的时间。

     - 并发用户数：同一时间点请求服务器的用户数，支持的最大并发数。

     - 内存占用：也就是内存开销。

     - 吞吐量（TPS）：Transaction Per Second, 每秒事务数。在没有遇到性能瓶颈时：TPS=并发用户数\*事务数/响应时间。

     - 错误率：失败的事务数/事务总数。

     - 资源使用率：CPU占用率、内存使用率、磁盘I/O、网络I/O。

     - 从性能测试分析度量的度角来看，主要可以从如下几个大的维度来收集考察性能指标：

     - 系统性能指标、资源性能指标、稳定性指标
2. **如果一个需求没有明确的性能指标，要如何开始进行性能测试？**

   - 先输出业务数据，如 pv、pu、时间段等，计算出大概的值，然后不断加压测到峰值
3. **介绍 JMeter 聚合报告包括哪些内容？**

   - 请求名、线程数、响应时间（50 95 99 最小 最大）错误率、吞吐量
4. **如果有一个页面特别卡顿，设想一下可能的原因？**

   - 后台：接口返回数据慢，查询性能等各种问题

   - 前端：使用 Chrome 工具调试，判断 JS 执行久或是其他问题

   - 网络问题
5. **说一说项目中的实际测试内容**

   - 根据自己项目中的经验实话实说，有没有经验很容易露馅。
6. **介绍一下 JMeter 进行性能测试的过程**

   - 结合自己的项目经验聊。大家也可以自行搜索。
7. **介绍一下 JMeter 和 LoadRunner 的区别**

   - 详细的不展开了，最重要的是相对来说 LoadRunner 的笨重、昂贵、闭源，理念和生态都落后，而 JMeter 是开源、可定制化开发，功能强大易用，并且在互联网大厂都已经有非常成熟的落地方案（主流的互联网公司基本都在使用 JMeter+ELK+Grafana+Influxdb 这套架构），可以说是进 BAT 大厂必备技能。还不会 JMeter 的同学建议抓紧补起来。

**感谢每一个认真阅读我文章的人，礼尚往来总是要有的，虽然不是什么很值钱的东西，如果你用得到的话可以直接拿走：**

![](https://i-blog.csdnimg.cn/direct/8cf625eb14784c33b25dfa889656ae7f.png)

这些资料，对于【 [软件测试](https://so.csdn.net/so/search?q=%E8%BD%AF%E4%BB%B6%E6%B5%8B%E8%AF%95&spm=1001.2101.3001.7020 "软件测试")】的朋友来说应该是最全面最完整的备战仓库，这个仓库也陪伴上万个测试工程师们走过最艰难的路程，希望也能帮助到你! **有需要的小伙伴可以点击下方小卡片领取**

![](https://i-blog.csdnimg.cn/direct/ca570a2666124f468c8be7aa0a81b098.png)

![](https://img-blog.csdnimg.cn/735f5f9ee698487982df5d6c52aecf96.jpeg)软件测试学习交流群（资源共享）

![](https://g.csdnimg.cn/extension-box/2.0.2/image/qq.png)QQ群名片

![](https://g.csdnimg.cn/extension-box/2.0.2/image/ic_move.png)

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png)

确定要放弃本次机会？


福利倒计时

_:_ _:_

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png)立减 ¥

普通VIP年卡可用

[立即使用](https://mall.csdn.net/vip)

[![](https://profile-avatar.csdnimg.cn/d7ce19eec3ba4267bca8a015d6c0cfe4_okcross0.jpg!1)\\
程序员威子](https://blog.csdn.net/okcross0)

关注关注

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png)
3

点赞

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png)
踩

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png)
7




收藏







觉得还不错?

一键收藏
![](https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png)

- [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png)\\
0](https://blog.csdn.net/okcross0/article/details/152041194#commentBox)
评论

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/share.png)分享




复制链接



分享到 QQ



分享到新浪微博









![](https://csdnimg.cn/release/blogv2/dist/pc/img/share/icon-wechat.png)扫一扫


- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/more.png)


![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报



![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/report.png)举报


[【软件 _测试_】接口 _自动化测试_ _面试题_ 及详细 _答案_](https://blog.csdn.net/m0_70618214/article/details/125685848)

[m0\_70618214的博客](https://blog.csdn.net/m0_70618214)

07-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
8255


[前言最近看到网上流传着各种面试经验及 _面试题_，往往都是一大堆技术题目贴上去，但是没有 _答案_。为此我业余时间整理了这份软件 _测试_ 基础常见的 _面试题_ 及详细 _答案_，望各路大牛发现不对的地方不吝赐教，留言即可。接口 _自动化测试_ _面试题_ 1、请结合你熟悉的项目，介绍一下你是怎么做 _测试_ 的？　　\-首先要自己熟悉项目，熟悉项目的需求、项目组织架构、项目研发接口等　　\-功能 \+ 接口 \+ _自动化_ \+ 性能 是怎么处理的？　　　　-第一步： 进行需求分析，需求评审，研发和 _测试_ 对需求达成统一的理解　　　　-第二步：架构师会输出接口规范；](https://blog.csdn.net/m0_70618214/article/details/125685848)

[软件 _测试_ _自动化_ _面试题_（含 _答案_）](https://blog.csdn.net/weixin_60870637/article/details/126974874)

[weixin\_60870637的博客](https://blog.csdn.net/weixin_60870637)

09-21![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[_自动化测试_ _面试题_](https://blog.csdn.net/weixin_60870637/article/details/126974874)

参与评论您还未登录，请先登录后发表或查看评论

[2024 _自动化测试_ _面试题_ _(_ 含 _答案_ _)_](https://devpress.csdn.net/v1/article/detail/138077064)

[m0\_58026506的博客](https://blog.csdn.net/m0_58026506)

04-22![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2142


[1、你会封装 _自动化测试_ 框架吗？ _自动化_ 框架主要的核心框架就是分层+PO模式：分别为：基础封装层BasePage，PO页面对象层，TestCase _测试用例_ 层。然后再加上日志处理模块，ini配置文件读取模块，unittest+ddt数据驱动模块，jenkins持续 _集_ 成模式组成。](https://devpress.csdn.net/v1/article/detail/138077064)

[2023面试 _自动化测试_ _面试题_【含 _答案_】](https://blog.csdn.net/m0_67695717/article/details/123574110)

[主要分享测试的学习资源，帮助快速了解测试行业，帮助想转行、进阶、小白成长为高级测试工程师。](https://blog.csdn.net/m0_67695717)

03-18![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2万+


[1、你做了几年的 _测试_、 _自动化测试_，说一下 _selenium_ 的原理是什么？\\
我做了五年的 _测试_，1年的 _自动化测试_；\\
_selenium_ 它是用 http 协议来连接 webdriver ，客户端可以使用 Java 或者 _Python_ 各种编程语言来实现；\\
2、什么项目适合做 _自动化测试_？\\
关键字：不变的、重复的、规范的\\
第一点，需求变化不能太频繁；\\
第二点，项目周期要足够长，如果 _自动化_ 代码还没有写完，公司就倒闭了，那也不需要 _自动化_ 了\\
第三点，脚本可以重复使用：在一些典型的场景，比如说 “冒烟 _测试_、回归 _测试_” 的地](https://blog.csdn.net/m0_67695717/article/details/123574110)

[java _自动化测试_ _面试题_\_ _自动化测试_ _面试题_](https://devpress.csdn.net/v1/article/detail/114097041)

[weixin\_39528289的博客](https://blog.csdn.net/weixin_39528289)

02-13![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
5164


[1、为什么做 _自动化_？解放手工劳动-UI回归 _测试_ 持续 _集_ 成中自动验证手工 _测试_ 无法实现-压力 并发 _测试_ 2、分层 _自动化测试_？概念应用场景形式UI _自动化_：模拟手工接口 _自动化_：没有界面 _单元测试_-白盒 _测试_ 6、如何保证脚本有效性元素定位有效：元素单独封装业务流程有效：封装独立方法 _测试_ 数据有效：保证数据库环境稳定，备份恢复，脚本灵活，实时提取数据，随机数。7、用例不稳定Sleep try catch8、UI _自动化_ 和...](https://devpress.csdn.net/v1/article/detail/114097041)

[_自动化测试_ _面试题_ 整理出炉 _附_ _答案_，建议收藏](https://blog.csdn.net/A18285759691/article/details/127639796)

[A18285759691的博客](https://blog.csdn.net/A18285759691)

11-01![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[简单来说，就是把页面作为对象，在使用中传递页面对象，来使用页面对象中相应的成员或者方法，能更好地体现面向对象语言（比如java或者 _python_）的面向对象和封装特性。](https://blog.csdn.net/A18285759691/article/details/127639796)

[【软件 _测试_】APP _自动化测试_ _面试题_，含 _答案_](https://blog.csdn.net/qq_40214204/article/details/124052206)

[qq\_40214204的博客](https://blog.csdn.net/qq_40214204)

04-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
3294


[1.Android APP 内存不足时， 系统如何结束进程获得内存？\\
系统优先结束被挂起（暂停）的进程，释放内存\\
2.APP _测试_ 常见的严重问题有哪些？ 分别引起的原因有哪些？\\
常见的有 crash、ANR（应用无响应、卡死），一般由设备碎片化、网络波动大、内存泄\\
漏、代码编写错误\\
3.请简单介绍你曾使用过的一款 APP _自动化测试_ 工具 ？\\
开放性问题，带点主观意见\\
1.对比其他熟悉的 _自动化_ 工具的优缺点\\
2. _自动化_ 的简要方案（简要的同时关键内容请具体）。（提示： appnium 等）\\
4.Android 测](https://blog.csdn.net/qq_40214204/article/details/124052206)

[软件 _测试_ 必问必背 _面试题_\\
\\
热门推荐](https://devpress.csdn.net/v1/article/detail/108661479)

[weixin\_45912307的博客](https://blog.csdn.net/weixin_45912307)

10-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
27万+


[软件 _测试_ 必问必背 _面试题_\\
01 软件 _测试_ 理论部分\\
1.1 _测试_ 概念\\
1\. 请你分别介绍一下 _单元测试_、 _集_ 成 _测试_、系统 _测试_、验收 _测试_、回归 _测试_ _单元测试_：完成最小的软件设计单元（模块）的验证工作，目标是确保模块被正确的编码\\
_集_ 成 _测试_：通过 _测试_ 发现 _与_ 模块接口有关的问题\\
系统 _测试_：是基于系统整体需求说明书的黑盒类 _测试_，应覆盖系统所有联合的部件\\
回归 _测试_：回归 _测试_ 是指在发生修改之后重新 _测试_ 先前的 _测试用例_ 以保证修改的正确性\\
验收 _测试_：这时相关的用户或独立 _测试_ 人员根据 _测试_ 计划和结果对系统进行 _测试_ 和接收。验收 _测试_ 包括Al](https://devpress.csdn.net/v1/article/detail/108661479)

[来一套 _自动化测试_ _面试题_（ _答案_ 版）](https://blog.csdn.net/weixin_38177508/article/details/127068862)

[大田的博客](https://blog.csdn.net/weixin_38177508)

09-27![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
4536


[在考虑异常时，通常我们都会想到正常情况，无效的情况，但是不一定能覆盖所有错误码，而接口定义返回的错误码可以帮助我们补充这一部分的用例，比如网络异常，无效的规则，无效的参数，无效的业务ID，无效的任务，服务器异常等，把errorcode的值都补充上去可以设计更多的用例。根据状态转换的分析：比如支付类业务，先支付成功，撤单后会退款，再次支付如果支付未成功，则是支付失败，状态之间的 切换是否正常，未按正常业务顺利进行操作时，状态怎么显示，是否可控，是否出现异常状态，空状态 业务怎么处理等。](https://blog.csdn.net/weixin_38177508/article/details/127068862)

[【软件 _测试_】接口 _自动化测试_ _面试题_（含 _答案_）](https://blog.csdn.net/qq_40214204/article/details/124052301)

[qq\_40214204的博客](https://blog.csdn.net/qq_40214204)

04-08![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
7756


[1.按你的理解，软件接口是什么？\\
答：\\
就是指程序中具体负责在不同模块之间传输或接受数据的并做处理的类或者函数。\\
2.HTTP 和 HTTPS 协议区别？\\
答：\\
https 协议需要到 CA（Certificate Authority，证书颁发机构）申请证书，一般免费证书\\
较少，因而需要一定费用；\\
http 是超文本传输协议，信息是明文传输，Https 协议是由 SSL+Http 协议构建的可进行加\\
密传输、身份认证的网络协议，比 http 协议安全；\\
http 和 https 使用的是完全不同的连接方式，](https://blog.csdn.net/qq_40214204/article/details/124052301)

精选资源 [_自动化测试_ _面试题_ 总结.docx](https://download.csdn.net/download/weixin_42384238/18451131)

05-07

[【 _自动化测试_ _面试题_ 总结】 一、Linux 在Linux操作系统中，熟悉常用命令是基础，例如： 1. \`grep\`：用于在文件中搜索特定文本，如\`grep 关键词 -C 10 文件名\`可查看关键词上下文。 2. \`tail\`：用于查看文件尾部，\`...](https://download.csdn.net/download/weixin_42384238/18451131)

精选资源 [1000道软件 _测试_ _面试题_ 及 _答案_ 解析， _自动化测试_、 _性能测试_、接口 _测试_、项目面试、自我介绍、软件 _测试_ _面试题_、接口 _自动化_、WEB _自动化_](https://download.csdn.net/download/weixin_44010641/89069625)

04-02

[软件 _测试_ _面试题_ 1000道软件 _测试_ _面试题_ 及 _答案_ 解析， _自动化测试_、 _性能测试_、接口 _测试_、项目面试、自我介绍、软件 _测试_ _面试题_、接口 _自动化_、WEB _自动化_](https://download.csdn.net/download/weixin_44010641/89069625)

[_Python_ _自动化测试_ 笔试 _面试题_ 精选](https://download.csdn.net/download/weixin_38628990/12854316)

09-17

[本 _篇_ 文章将聚焦于 _Python_ _自动化测试_ 中常见的 _面试题_，涵盖哈希、递归、分治等核心概念。 首先，哈希是一种高效的数据结构，它通过键值映射实现快速查找。在 _Python_ 中，字典和 _集_ 合是哈希数据结构的代表。哈希表的查找...](https://download.csdn.net/download/weixin_38628990/12854316)

精选资源 [件 _测试_ 1000道 _面试题_ 及 _答案_ 解析， _自动化测试_、 _性能测试_、接口 _测试_、项目面试等](https://download.csdn.net/download/weixin_44010641/89039764)

03-27

[软件 _测试_ _面试题_ 件 _测试_ 1000道 _面试题_ 及 _答案_ 解析， _自动化测试_、 _性能测试_、接口 _测试_、项目面试、自我介绍、软件 _测试_ _面试题_、接口 _自动化_、WEB _自动化_](https://download.csdn.net/download/weixin_44010641/89039764)

[【保姆级教程】RTX 4090 24G 部署 DeepSeek-V4-Flash 全攻略（INT4 量化 + 128K 上下文）\\
\\
最新发布](https://devpress.csdn.net/v1/article/detail/160801705)

[weixin\_67327688的博客](https://blog.csdn.net/weixin_67327688)

05-05![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
643


[本文介绍了在RTX 4090 24G显卡上部署DeepSeek-V4-Flash大模型的方案。由于硬件限制，必须采用INT4量化并限制上下文长度为128K以降低显存占用。关键步骤包括：1 _)_ 环境准备，要求vLLM≥0.6.6版本；2 _)_ 通过ModelScope下载模型；3 _)_ 配置核心参数如INT4量化、显存利用率等。部署支持单卡到4卡并行，性能随显卡数量提升，4卡可实现70-130 token/s的生成速度。](https://devpress.csdn.net/v1/article/detail/160801705)

[_Python_ 3 模块精讲：pymongo（第三方）超详细教程 ——MongoDB 连接 + 全 CURD 实战](https://blog.csdn.net/qq_28372005/article/details/160668192)

[qq\_28372005的博客](https://blog.csdn.net/qq_28372005)

04-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
566


[在 _Python_ _与_ MongoDB 数据库交互的技术生态中，pymongo 是官方推荐、生态最完善、兼容性最强的第三方驱动模块，它彻底打通了 _Python_ 代码 _与_ MongoDB 服务的通信链路，成为后端开发、数据爬虫、数据分析、AI 数据存储等场景的核心工具。](https://blog.csdn.net/qq_28372005/article/details/160668192)

[【 _python_ 因果库实战21】快餐业就业数据上比较效应估计量1](https://bqleng.blog.csdn.net/article/details/140910943)

[kaggle expert，全球排名前1000，清华计算机研究生，兴趣算法工程](https://blog.csdn.net/qq_32146369)

04-30![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
937


[摘要 本研究基于卡德和克鲁格关于最低工资对快餐业就业影响的经典研究，使用causallib软件包比较不同因果效应估计方法。分析显示，初始简单估计表明最低工资提高会减少就业，但校正协变量后，实际效应可能为中性或略有增加。研究数据来自新泽西州和宾夕法尼亚州快餐店，包含就业人数、工资等协变量，通过 _Python_ 实现数据加载和处理，为因果推断提供了标准化分析框架。](https://bqleng.blog.csdn.net/article/details/140910943)

[Claude Code安装配置](https://blog.csdn.net/w690333243/article/details/160727495)

[王人冉的博客](https://blog.csdn.net/w690333243)

05-04![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
39


[在此路径下配置你自己的settings.json文件内容，如果没有此文件，新建一个。查看安装的Claude Code版本号。settings.json参考。](https://blog.csdn.net/w690333243/article/details/160727495)

[介绍一下背包DP（ _Python_）](https://blog.csdn.net/zfjkdfhduhsdk/article/details/160719862)

[zfjkdfhduhsdk的博客](https://blog.csdn.net/zfjkdfhduhsdk)

05-02![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
80


[背包问题是一类经典的动态规划问题，分为0-1背包和完全背包两种主要类型。核心思想是通过状态转移方程逐步填充一个二维数组（或优化后的一维数组），记录不同容量下的最优解。](https://blog.csdn.net/zfjkdfhduhsdk/article/details/160719862)

[VASP官方教程 TRIQS DFT+DMFT计算教程](https://blog.csdn.net/icehoqion/article/details/160694004)

[DFT\_M的博客](https://blog.csdn.net/icehoqion)

05-01![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
213


[VASP可通过接口 _与_ 外部DMFT代码 _(_ 如TRIQS/solid\_dmft _)_ 协同工作通过求解杂质模型获得自能，进行电荷自洽的CSC DFT+DMFT循环，输出可经最大熵方法解析延拓至实频得到局域谱函数。observables\_imp0.dat：每轮迭代的关键可观测量，重点看G _(_ beta/2 _)_，该值趋近于 0，说明体系打开能隙，符合 NiO 莫特绝缘体的特性。conv\_obs0.dat：收敛判据，重点看δGimp，该值持续下降并稳定在 1e-4 以下，说明 DMFT 迭代收敛。运行 solid\_dmft。](https://blog.csdn.net/icehoqion/article/details/160694004)

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

![](https://blog.csdn.net/okcross0/article/details/152041194)

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

![](https://i-operation.csdnimg.cn/images/fb287ddc3c984e04a2021d439632f08c.png)

提问

QQ群名片![](https://g.csdnimg.cn/extension-box/2.0.2/image/ic_close.png)

![](https://i-blog.csdnimg.cn/direct/38cc177aeb4245b589ba0e2463c3a4af.png) QQ群 ID：974990200QQ扫码添加好友或搜索 ID

复制QQ群 ID

![](https://csdnimg.cn/release/blogv2/dist/pc/img/quoteClose1White.png)

![](https://i-blog.csdnimg.cn/direct/8cf625eb14784c33b25dfa889656ae7f.png)

![](https://i-blog.csdnimg.cn/direct/ca570a2666124f468c8be7aa0a81b098.png)

-100%+1:1还原