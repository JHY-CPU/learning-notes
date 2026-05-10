# DevOps & CI/CD 常见面试题原创- 自动化测试 - CSDN博客

URL: https://blog.csdn.net/qq_44534541/article/details/127349206

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

# DevOps & CI/CD 常见面试题

最新推荐文章于 2026-04-01 02:52:32 发布

原创于 2022-10-16 16:32:19 发布·2k 阅读

·![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newHeart2023Black.png)
4


·![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollect2.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarCollectionActive2.png)
17
·

CC 4.0 BY-SA版权

版权声明：本文为博主原创文章，遵循 [CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/) 版权协议，转载请附上原文出处链接和本声明。


文章标签：

[#devops](https://so.csdn.net/so/search/s.do?q=devops&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#ci/cd](https://so.csdn.net/so/search/s.do?q=ci%2Fcd&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art) [#运维](https://so.csdn.net/so/search/s.do?q=%E8%BF%90%E7%BB%B4&t=all&o=vip&s=&l=&f=&viparticle=&from_tracking_code=tag_word&from_code=app_blog_art)

[![](https://i-blog.csdnimg.cn/columns/default/20201014180756738.png?x-oss-process=image/resize,m_fixed,h_224,w_224)运维学习笔记专栏收录该内容](https://blog.csdn.net/qq_44534541/category_11797310.html "运维学习笔记")

342 篇文章

订阅专栏

![](https://i-operation.csdnimg.cn/images/a7311a21245d4888a669ca3155f1f4e5.png)本文介绍了DevOps的概念及其在软件开发中的重要性，涵盖了持续集成、持续交付和持续部署等核心实践，强调了自动化测试和版本控制的作用。


#### 1\. 什么是 DevOps

用最简单的术语来说，DevOps 是产品开发过程中开发（Dev）和运营（Ops）团队之间的灰色区域。DevOps 是一种在产品开发周期中强调沟通，集成和协作的文化。因此，它消除了软件开发团队和运营团队之间的孤岛，使他们能够快速，连续地集成和部署产品。

#### 2.什么是持续集成

持续集成（Continuous integration，缩写为 CI）是一种软件开发实践，团队开发成员经常集成他们的工作。利用自动测试来验证并断言其代码不会与现有代码库产生冲突。理想情况下，代码更改应该每天在 CI 工具的帮助下，在每次提交时进行自动化构建（包括编译，发布，自动化测试），从而尽早地发现集成错误，以确保合并的代码没有破坏主分支。

#### 3\. 什么是持续交付

持续交付（Continuous delivery，缩写为 CD）以及持续集成为交付代码包提供了完整的流程。在此阶段，将使用自动构建工具来编译工件，并使其准备好交付给最终用户。它的目标在于让软件的构建、测试与发布变得更快以及更频繁。这种方式可以减少软件开发的成本与时间，减少风险。

#### 4\. 什么是持续部署

持续部署（Continuous deployment）通过集成新的代码更改并将其自动交付到发布分支，从而将持续交付提升到一个新的水平。更具体地说，一旦更新通过了生产流程的所有阶段，便将它们直接部署到最终用户，而无需人工干预。因此，要成功利用连续部署，软件工件必须先经过严格建立的自动化测试和工具，然后才能部署到生产环境中。

#### 5\. 什么是持续测试及其好处

连续测试是一种在软件交付管道中尽早、逐步和适当地应用自动化测试的实践。在典型的 CI/CD 工作流程中，将小批量发布构建。因此，为每个交付手动执行测试用例是不切实际的。自动化的连续测试消除了手动步骤，并将其转变为自动化例程，从而减少了人工。因此，对于 DevOps 文化而言，自动连续测试至关重要。

持续测试的好处：

a. 确保构建的质量和速度。

b. 支持更快的软件交付和持续的反馈机制。

c. 一旦系统中出现错误，请立即检测。

d. 降低业务风险。在潜在问题变成实际问题之前进行评估。

#### 6\. 什么是版本控制及其用途？

版本控制（或源代码控制）是一个存储库，源代码中的所有更改都始终存储在这个代码仓库中。版本控件提供了代码开发的操作历史记录，追踪文件的变更内容、时间、人等信息忠实地了记录下来。版本控制是持续集成和持续构建的源头。

#### 7.什么是 Git？

Git 是一个分布式版本控制系统，可跟踪代码存储库中的更改。利用 GitHub 流，Git 围绕着一个基于分支的工作流，该工作流随着团队项目的不断发展而简化了团队协作。

#### 8.DevOps 为什么重要？DevOps 如何使团队在软件交付方面受益？

在当今的数字化世界中，组织必须重塑其产品部署系统，使其更强大，更灵活，以跟上竞争的步伐。

这就是 DevOps 概念出现的地方。DevOps 在为整个软件开发管道（从构思到部署，再到最终用户）产生移动性和敏捷性方面发挥着至关重要的作用。DevOps 是将不断更新和改进产品的更简化，更高效的流程整合在一起的解决方案。

#### 9.解释 DevOps 对开发人员有何帮助？

在没有 DevOps 的世界中，开发人员的工作流程将首先建立新代码，交付并集成它们，然后，操作团队有责任打包和部署代码。之后，他们将不得不等待反馈。而且如果出现问题，由于错误，他们将不得不重新执行一次，在项目中涉及的不同团队之间的无数手动沟通。

由于 CI/CD 实践已经合并并自动化了其余任务，因此应用 DevOps 可以将开发人员的任务简化为仅构建代码。随着流程变得更加透明并且所有团队成员都可以访问，将工程团队和运营团队相结合有助于建立更好的沟通和协作。

#### 10.为什么 DevOps 最近在软件交付方面变得越来越流行？

DevOps 在过去几年中受到关注，主要是因为它能够简化组织运营的开发，测试和部署流程，并将其转化为业务价值。

技术发展迅速。因此，组织必须采用一种新的工作流程-DevOps 和 Agile 方法-来简化和刺激其运营，而不能落后于其他公司。DevOps 的功能通过 Facebook 和Netflix 的持续部署方法所取得的成功得到了清晰体现，该方法成功地促进了其增长，而没有中断正在进行的运营。

#### 11.CI/CD 有什么好处？

CI 和 CD 的结合将所有代码更改统一到一个单一的存储库中，并通过自动化测试运行它们，从而在所有阶段全面开发产品，并随时准备部署。

CI/CD 使组织能够按照客户期望的那样快速，高效和自动地推出产品更新。简而言之，精心规划和执行良好的 CI/CD 管道可加快发布速度和可靠性，同时减轻产品的代码更改和缺陷。这最终将导致更高的客户满意度。

#### 12.持续交付有什么好处？

通过手动发布代码更改，团队可以完全控制产品。在某些情况下，该产品的新版本将更有希望：具有明确业务目的的促销策略。

通过自动执行重复性和平凡的任务，IT 专业人员可以拥有更多的思考能力来专注于改进产品，而不必担心集成进度。

#### 14.持续部署有哪些好处？

通过持续部署，开发人员可以完全专注于产品，因为他们在管道中的最后任务是审查拉取请求并将其合并到分支。通过在自动测试后立即发布新功能和修复，此方法可实现快速部署并缩短部署持续时间。

客户将是评估每个版本质量的人。新版本的错误修复更易于处理，因为现在每个版本都以小批量交付。

#### 15.SCM 团队在 DevOps 中扮演什么角色？

软件配置管理（SCM）是跟踪和保留开发环境记录的实践，包括在操作系统中进行的所有更改和调整。

在 DevOps 中，将 SCM 作为代码构建在基础架构即代码实践的保护下。SCM 为开发人员简化了任务，因为他们不再需要手动管理配置过程。现在，此过程以机器可读的形式构建，并且会自动复制和标准化。

#### 16.DevOps 使用哪些工具？描述你使用任何这些工具的经验

在典型的 DevOps 生命周期中，有不同的工具来支持产品开发的不同阶段。因此，用于 DevOps 的最常用工具可以分为 6 个关键阶段：

持续开发：Git, SVN, Mercurial, CVS, Jira

持续整合：Jenkins, Bamboo, CircleCI

持续交付：Nexus, Archiva, Tomcat

持续部署：Puppet, Chef, Docker

持续监控：Splunk, ELK Stack, Nagios

连续测试：Selenium，Katalon Studio

#### 17.CI/CD 的一些核心组件是什么？

稳定的 CI/CD 管道需要用作版本控制系统的存储库管理工具。这样开发人员就可以跟踪软件版本中的更改。

在版本控制系统中，开发人员还可以在项目上进行协作，在版本之间进行比较并消除他们犯的任何错误，从而减轻对所有团队成员的干扰。

连续测试和自动化测试是成功建立无缝 CI / CD 管道的两个最关键的关键。自动化测试必须集成到所有产品开发阶段（包括单元测试，集成测试和系统测试），以涵盖所有功能，例如性能，可用性，性能，负载，压力和安全性。

#### 18.CI/CD 的一些常见做法是什么？

以下是建立有效的 CI / CD 管道的一些最佳实践：

- 发展 DevOps 文化
- 实施和利用持续集成
- 以相同的方式部署到每个环境
- 失败并重新启动管道
- 应用版本控制
- 将数据库包含在管道中
- 监控你的持续交付流程
- 使你的 CD 流水线流畅

#### 19.敏捷和 DevOps 之间有哪些主要区别？

基本上，DevOps 和敏捷是相互补充的。敏捷更加关注开发新软件和以更有效的方式管理复杂过程的价值和原则。同时，DevOps 旨在增强由开发人员和运营团队组成的不同团队之间的沟通，集成和协作。

它需要采用敏捷方法和 DevOps 方法来形成无缝工作的产品开发生命周期：敏捷原理有助于塑造和引导正确的开发方向，而 DevOps 利用这些工具来确保将产品完全交付给客户。

#### 20.DevOps 和持续交付之间有什么区别？

DevOps 更像是一种组织和文化方法，可促进工程团队和运营团队之间的协作和沟通。

同时，持续交付是成功将 DevOps 实施到产品开发工作流程中的重要因素。持续交付实践有助于使新发行的版本更加乏味和可靠，并建立更加无缝和短的流程。DevOps 的主要目的是有效地结合 Dev 和 Ops 角色，消除所有孤岛，并实现独立于持续交付实践的业务目标。

另一方面，如果已经有 DevOps 流程，则连续交付效果最佳。因此，它扩大了协作并简化了组织的统一产品开发周期。

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-newWhite.png)

确定要放弃本次机会？


福利倒计时

_:_ _:_

![](https://csdnimg.cn/release/blogv2/dist/pc/img/vip-limited-close-roup.png)立减 ¥

普通VIP年卡可用

[立即使用](https://mall.csdn.net/vip)

[![](https://profile-avatar.csdnimg.cn/1e3d338237c84b3f86c16521dbc34b46_qq_44534541.jpg!1)\\
会飞的土拨鼠呀](https://ncayu.blog.csdn.net/)

关注关注

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/tobarThumbUpactive.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/like.png)
4

点赞

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/unlike.png)
踩

- ![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect-active.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/collect.png)![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCollectActive.png)
17




收藏







觉得还不错?

一键收藏
![](https://csdnimg.cn/release/blogv2/dist/pc/img/collectionCloseWhite.png)

- [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/comment.png)\\
0](https://blog.csdn.net/qq_44534541/article/details/127349206#commentBox)
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

[【 _运维_ _面试_】 _DevOps_ _&_ _&_ _CI/CD_ _常见_ _面试题_](https://zmedu.blog.csdn.net/article/details/118960748)

[互联网老辛](https://blog.csdn.net/xinshuzhan)

07-21![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
4688


[文章目录1\. 什么是 _DevOps_ 2\. 什么是持续集成3. 什么是持续交付4. 什么是持续部署5. 什么是持续测试及其好处6. 什么是版本控制及其用途？7. 什么是 Git？8. 解释 _DevOps_ 对开发人员有何帮助？9\. _CI/CD_ 有什么好处？10\. 持续交付有什么好处？11. 持续部署有哪些好处？12. 如何有效实施 _DevOps_ 13\. _DevOps_ 使用哪些工具？描述你使用任何这些工具的经验14\. 有哪些 _常见_ 的 _CI/CD_ 服务器15\. 持续集成和持续交付之间的区别是什么？\\
1\. 什么是DevO](https://zmedu.blog.csdn.net/article/details/118960748)

[_DevOps_ _&_ _CI/CD_ _常见_ _面试题_ 汇总](https://blog.csdn.net/qq_25221835/article/details/120556128)

[trouble is a friend](https://blog.csdn.net/qq_25221835)

09-29![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2245


[_DevOps_ 术语和定义\\
1\. 什么是 _DevOps_\\
答：用最简单的术语来说， _DevOps_ 是产品开发过程中开发（Dev）和运营（Ops）团队之间的灰色区域。 _DevOps_ 是一种在产品开发周期中强调沟通，集成和协作的文化。因此，它消除了软件开发团队和运营团队之间的孤岛，使他们能够快速，连续地集成和部署产品。\\
\\
2\. 什么是持续集成\\
答：持续集成（Continuous integration，缩写为 CI）是一种软件开发实践，团队开发成员经常集成他们的工作。利用自动测试来验证并断言其代码不会与现有代码库产生冲](https://blog.csdn.net/qq_25221835/article/details/120556128)

参与评论您还未登录，请先登录后发表或查看评论

[_CI/CD_ _面试题_ 整合](https://blog.csdn.net/m0_56444183/article/details/124521814)

[不要因为走的太远而忘了为什么出发](https://blog.csdn.net/m0_56444183)

05-01![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
5572


[文章目录一、 _DevOps_ 术语和定义1.什么是 _DevOps_ 2\. 什么是持续集成3. 什么是持续交付4. 什么是持续部署5. 持续测试及其优点6. 什么是版本控制及其用途?7. 什么是Git?二、实施 _DevOps_ 的原因1\. _DevOps_ 为什么重要？2\. _DevOps_ 对开发人员有何帮助？3\. 为什么 _DevOps_ 变得越来越流行?4\. _CI/CD_ 有什么好处？5\. 持续交付有什么好处?6. 持续部署有哪些好处?三、如何有效实施 _DevOps_ 1\. _DevOps_ 工作流程2\. _DevOps_ 的核心操作是什么？3\. 在实施Dev](https://blog.csdn.net/m0_56444183/article/details/124521814)

[别再纠结了！Ollama和LM Studio到底怎么选？一张图帮你搞定（附保姆级安装避坑指南）\\
\\
最新发布](https://blog.csdn.net/weixin_30338481/article/details/95006313)

[weixin\_30338481的博客](https://blog.csdn.net/weixin_30338481)

04-01![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
411


[本文详细对比了Ollama和LM Studio两大工具的核心差异与应用场景，帮助用户根据需求快速决策。提供决策流程图、深度功能拆解及跨平台安装避坑指南，特别适合开发者和创意工作者选择适合自己的大语言模型工具。](https://blog.csdn.net/weixin_30338481/article/details/95006313)

[【 _CI/CD_】软件测试 _面试_ 相关的 _CI/CD_ 问题](https://blog.csdn.net/u013177528/article/details/137922717)

[u013177528的博客](https://blog.csdn.net/u013177528)

04-18![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1098


[在软件测试 _面试_ 中， _面试_ 官可能会询问与持续集成（CI）和持续交付（CD）相关的问题，以评估应聘者对 _CI/CD_ 的理解和实践经验。准备这些问题的答案时，最好结合自己的实际经验和具体案例来说明。这将有助于展示你对 _CI/CD_ 的深入理解以及在实际工作中应用 _CI/CD_ 的能力。](https://blog.csdn.net/u013177528/article/details/137922717)

[_DevOps_ 与 _CI/CD_ _常见_ _面试_ 问题汇总](https://blog.csdn.net/wx17343624830/article/details/132805972)

[伤心的辣条](https://blog.csdn.net/wx17343624830)

09-11![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1470


[01 您能告诉我们 _DevOps_ 和Agile(敏捷)之间的根本区别吗？\\
答：尽管 _DevOps_ 与敏捷方法（这是最流行的SDLC\[Software Development Life Cycle\]方法之一）有一些相似之处，但两者在软件开发方面都是根本不同的方法。以下是两者之间的各种基本差异：](https://blog.csdn.net/wx17343624830/article/details/132805972)

[直通 _CI/CD_ _面试_：典型问题深度解读](https://blog.csdn.net/buxiangxueyun/article/details/147091354)

[buxiangxueyun的博客](https://blog.csdn.net/buxiangxueyun)

04-09![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1996


[_CI/CD_ 代表持续集成（Continuous Integration）和持续交付（Continuous Delivery）。•持续集成是指开发人员频繁将代码集成到主干中，通常每天多次。CI 通过自动化测试和构建过程，确保集成后的代码质量并减少集成问题。•持续交付是指将经过自动化测试的代码自动部署到生产环境的准备状态，使得代码能够随时交付给用户。 _CI/CD_ 提高了软件开发的效率，减少了人为错误，确保更高质量的代码，并使得发布更频繁、可靠。•。](https://blog.csdn.net/buxiangxueyun/article/details/147091354)

[_Devops_ _&_ _CI/CD_ 2022年最新 _常见_ _面试题_ 汇总](https://blog.csdn.net/m0_71742635/article/details/128040294)

[m0\_71742635的博客](https://blog.csdn.net/m0_71742635)

11-25![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
435


[_DevOps_ 是产品开发过程中开发（Dev）和运营（Ops）团队之间的灰色区域。 _DevOps_ 是一种在产品开发周期中强调沟通，集成和协作的文化。因此，它消除了软件开发团队和运营团队之间的孤岛，使他们能够快速，连续地集成和部署产品。](https://blog.csdn.net/m0_71742635/article/details/128040294)

精选资源 [云原生训练营 \_ _DevOps_ _&_ amp; CI\_CD _常见_ _面试题_ 汇总.pdf](https://download.csdn.net/download/weixin_45308597/20585295)

07-28

[_DevOps_ 文化的实施，使开发人员可以专注于编写代码，而 _CI/CD_ 实践自动处理其他流程，减少了重复工作，增加了透明度，促进了团队间的沟通与协作。 最近 _DevOps_ 在软件交付领域变得流行的原因在于其能够简化并加速组织的...](https://download.csdn.net/download/weixin_45308597/20585295)

[_devops_ _&_ amp; _&_ amp; _ci/cd_ _常见_ _面试题_](https://wenku.csdn.net/answer/efda00c98bcd1efe5bbe51ed0664f3dc)

06-06

[_CI/CD_ 是 _DevOps_ 中的两个重要概念，CI（Continuous Integration）指持续集成，CD（Continuous Delivery/Deployment）指持续交付/部署。CI是指在代码提交到版本控制系统后，自动进行编译、测试和打包等操作，以确保代码...](https://wenku.csdn.net/answer/efda00c98bcd1efe5bbe51ed0664f3dc)

[【 _CI/CD_】软件测试 _面试_ _常见_ _CI/CD_ 问题的参考答案](https://blog.csdn.net/u013177528/article/details/137922982)

[u013177528的博客](https://blog.csdn.net/u013177528)

04-18![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1139


[作为软件测试专家，我深知 _CI/CD_ 的重要性，并一直致力于通过自动化测试和持续改进来提高软件的质量和交付速度。](https://blog.csdn.net/u013177528/article/details/137922982)

[2022年最新 _DevOps_ 和CI CD _常见_ _面试题_ 汇总](https://download.csdn.net/download/xiaoli8748/88731041)

01-11

[2022年最新 _DevOps_ 和CI CD _常见_ _面试题_ 汇总](https://download.csdn.net/download/xiaoli8748/88731041)

[_CI/CD_ _面试题_ 及答案](https://blog.csdn.net/weixin_42795092/article/details/147820000)

[weixin\_42795092的博客](https://blog.csdn.net/weixin_42795092)

05-09![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1479


[回答 _CI/CD_ _面试题_ 时，建议结合具体工具（如 Jenkins、GitLab _CI/CD_）和实际项目经验，突出自动化、质量保障和持续改进的思路。理解各种部署策略（蓝绿、金丝雀）和高级概念（GitOps）能体现技术深度。](https://blog.csdn.net/weixin_42795092/article/details/147820000)

[_面试_ 篇：（三十一）前端工程化与 _CI/CD_ \- 2024 年 _面试_ 项目实战](https://blog.csdn.net/mmc123125/article/details/143651326)

[一名热衷于技术的全栈开发者，专注于前端与后端的全面技术探索。在这里，我将分享我在技术领域的学习与成长，助力更多开发者的进步。](https://blog.csdn.net/mmc123125)

11-09![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1318


[前端工程化是指将开发流程中的各个环节（如模块化、构建、测试、部署等）通过工具和自动化手段进行系统化管理，达到提升开发效率、降低错误率、提高代码质量的目的。它可以优化团队协作，确保不同开发人员的代码风格一致性，并通过工具提高开发速度和可靠性。](https://blog.csdn.net/mmc123125/article/details/143651326)

[_DevOps_ _&_ amp； CI CD _常见_ _面试题_\_ _cicd_ _面试_，2024年最新吊打 _面试_ 官](https://blog.csdn.net/2301_78398209/article/details/137761739)

[2301\_78398209的博客](https://blog.csdn.net/2301_78398209)

04-15![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1120


[Git 是一个分布式版本控制系统，可跟踪代码存储库中的更改。利用 GitHub 流，Git 围绕着一个基于分支的工作流，该工作流随着团队项目的不断发展而简化了团队协作。最近很多小伙伴找我要Linux学习资料，于是我翻箱倒柜，整理了一些优质资源，涵盖视频、电子书、PPT等共享给大家！](https://blog.csdn.net/2301_78398209/article/details/137761739)

[测试工程师 _面试_ 热门问题（六）](https://blog.csdn.net/hai40587/article/details/140369488)

[陈辰学长的博客](https://blog.csdn.net/hai40587)

07-12![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1223


[持续集成和持续部署是现代软件开发中不可或缺的实践，它们与测试的关系密不可分。通过自动化测试和部署，持续集成和持续部署能够加速软件的开发和交付过程，提高软件的质量和稳定性，并减少人工干预的时间和成本。这种自动化的流程为开发人员和测试人员提供了更好的合作方式，共同推动软件的持续迭代和交付。](https://blog.csdn.net/hai40587/article/details/140369488)

[_DevOps_ _&_ amp； CI CD _常见_ _面试题_\_ _cicd_ _面试_(1)，腾讯Linux _运维_ 开发 _面试_ 记录](https://blog.csdn.net/2301_78398209/article/details/137761735)

[2301\_78398209的博客](https://blog.csdn.net/2301_78398209)

04-15![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
977


[_DevOps_ 的功能通过 Facebook 和Netflix 的持续部署方法所取得的成功得到了清晰体现，该方法成功地促进了其增长，而没有中断正在进行的运营。在没有 _DevOps_ 的世界中，开发人员的工作流程将首先建立新代码，交付并集成它们，然后，操作团队有责任打包和部署代码。而且如果出现问题，由于错误，他们将不得不重新执行一次，在项目中涉及的不同团队之间的无数手动沟通。自动化测试必须集成到所有产品开发阶段（包括单元测试，集成测试和系统测试），以涵盖所有功能，例如性能，可用性，性能，负载，压力和安全性。](https://blog.csdn.net/2301_78398209/article/details/137761735)

[百家互联网QA _面试题_--develop/ _CICD_/容器化](https://blog.csdn.net/sun_qian_li/article/details/106031543)

[后街女孩的博客](https://blog.csdn.net/sun_qian_li)

05-10![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
2456


[1、简述一下对Jenkins的认识\\
Jenkins可以构建一个自动化的持续集成环境可以使用它来“自动化”编译、打包、分发部署应用，它兼容ant、maven、gradle等多种第三方构建工具，同时与svn、git能无缝集成，也支持直接与知名源代码托管网站，如github、bitbucket直接集成。\\
2、Jenkins的功能有哪些？\\
\\
1.定时拉取代码并编译\\
2.静态代码分析\\
3.定时打包发布测试版\\
4.自定义额外的操作，如跑单元测试等\\
5.出错提醒\\
\\
3、什么是持续集成？\\
持续集成是一种软件开发实践，即团队](https://blog.csdn.net/sun_qian_li/article/details/106031543)

[【2023】 _DevOps_、SRE、 _运维_ 开发 _面试_ 宝典之 _CI/CD_ 相关 _面试题_\\
\\
热门推荐](https://jiangxl.blog.csdn.net/article/details/129294969)

[江晓龙的博客](https://blog.csdn.net/weixin_44953658)

03-02![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1万+


[用最简单的术语来说， _DevOps_ 是产品开发过程中开发（Dev）和运营（Ops）团队之间的灰色区域。 _DevOps_ 是一种在产品开发周期中强调沟通，集成和协作的文化。因此，它消除了软件开发团队和运营团队之间的孤岛，使他们能够快速，连续地集成和部署产品。持续集成（Continuous integration,缩写为Cl）是一种软件开发实践，团队开发成员经常集成他们的工作。利用自动测试来验证并断言其代码不会与现有代码库产生冲突。](https://jiangxl.blog.csdn.net/article/details/129294969)

[_CI/CD_ 实战 _面试_ 宝典：从构建到高可用性的全面解析](https://blog.csdn.net/u010282639/article/details/139553282)

[小渣渣的学习与运维日常](https://blog.csdn.net/u010282639)

06-09![](https://csdnimg.cn/release/blogv2/dist/pc/img/readCountWhite.png)
1706


[然后，安装和配置Jenkins，创建Job，配置构建触发器，编写Jenkinsfile定义构建、测试、打包和部署的步骤。通过这些步骤，我们搭建了一个完整的 _CI/CD_ pipeline，实现了代码的自动化构建、测试和部署，提高了开发和 _运维_ 效率。在 _CI/CD_ pipeline中，我们会集成代码扫描和容器镜像扫描工具（如SonarQube、Trivy、Clair），在构建阶段检查代码和镜像中的已知漏洞和安全问题。同时，我们调整了CI服务器的资源配置，增加了CPU和内存，确保构建过程有足够的资源。](https://blog.csdn.net/u010282639/article/details/139553282)

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

[![](https://profile-avatar.csdnimg.cn/1e3d338237c84b3f86c16521dbc34b46_qq_44534541.jpg!1)](https://ncayu.blog.csdn.net/)

会飞的土拨鼠呀

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

![](https://blog.csdn.net/qq_44534541/article/details/127349206)

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