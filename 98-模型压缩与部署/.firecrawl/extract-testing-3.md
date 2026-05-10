# 金九银十：软件测试工程师高频面试题及答案解析（2025版） - 博客园

URL: https://www.cnblogs.com/hebendexiaomao/p/19093043

# [金九银十：软件测试工程师高频面试题及答案解析（2025版）](https://www.cnblogs.com/hebendexiaomao/p/19093043 "发布于 2025-09-15 15:24")

![金九银十：软件测试工程师高频面试题及答案解析（2025版）](https://img2024.cnblogs.com/blog/2951280/202509/2951280-20250915152348296-2067500121.png)
金三银四是软件测试工程师职业发展的黄金时期，充分准备面试是成功的关键。本文提供的面试题和答案解析涵盖了软件测试的各个方面，希望能为你的求职之路提供有力支持。


金九银十招聘季如期而至，作为软件测试工程师的你，是否已经做好充分的面试准备？为了帮助大家在2025年的求职市场中脱颖而出，我们精心整理了一份软件测试工程师高频面试题及详细答案解析。

无论你是初入行的新人，还是经验丰富的资深工程师，这份指南都将为你提供宝贵的参考。让我们直接进入正题！

## 一、软件测试基础概念

1\. 什么是软件测试？软件测试的主要目的是什么？

软件测试是通过手动或自动方式运行系统或应用程序，以发现软件缺陷、验证软件是否满足特定要求的过程。

主要目的包括：

- 发现软件中的缺陷和错误

- 验证软件是否满足业务需求和规格说明

- 评估软件质量，建立使用信心

- 预防缺陷，降低开发成本

- 确保软件符合行业标准和法规要求


2\. 黑盒测试、白盒测试和灰盒测试有什么区别？

- 黑盒测试：只关注输入和输出，不关心内部代码结构。测试人员基于需求规格说明书设计测试用例。

- 白盒测试：需要了解内部代码结构和实现细节。测试人员基于代码逻辑设计测试用例。

- 灰盒测试：结合黑盒和白盒测试的特点。测试人员部分了解系统内部结构，但测试设计仍主要基于外部功能。


3\. 解释一下回归测试是什么？为什么要进行回归测试？

回归测试是在修改代码后重新执行先前已经通过的测试用例，以确保修改没有引入新的错误或导致其他功能出现问题。

进行回归测试的原因：

- 确保代码修改不会破坏现有功能

- 验证缺陷修复是否有效

- 保证软件质量在持续开发过程中保持稳定

- 支持敏捷开发和持续集成流程


## 二、测试方法与技术

1\. 等价类划分和边界值分析是什么？请举例说明

等价类划分是将输入数据划分为若干等价类，从每个类中选取代表性数据作为测试用例。边界值分析是专注于输入边界条件的测试技术。

例如，测试一个接受1-100数字输入的字段：

- 等价类划分：有效类(1-100)、无效类(小于1、大于100)

- 边界值分析：测试0、1、2、99、100、101这些边界值


2\. 什么是探索性测试？它在什么情况下最有用？

探索性测试是一种非预设脚本的测试方法，测试人员同时设计、执行和学习测试过程。

最有用的情况：

- 需求文档不完整或经常变更

- 需要快速理解系统功能

- 时间紧迫，需要快速发现关键缺陷

- 作为脚本测试的补充，发现意想不到的问题


3\. 如何测试一个电梯系统？请设计测试用例

功能测试：

- 呼叫电梯：按下上下按钮，验证电梯响应

- 选择楼层：进入电梯选择目标楼层，验证到达正确楼层

- 超载报警：超过载重限制时是否发出警报

- 紧急停止：测试紧急停止按钮功能


性能测试：

- 多用户同时呼叫时的响应时间

- 电梯在不同负载下的运行速度


安全测试：

- 门开关安全性：确保门不会夹人

- 断电应急措施：停电时电梯是否安全停靠


用户体验测试：

- 按钮响应灵敏度

- 显示屏信息清晰度

- 语音提示准确性


## 三、自动化测试专题

1\. Selenium、Cypress和Playwright各有什么优缺点？

- Selenium：


  - 优点：生态成熟、支持多种语言、跨浏览器能力强

  - 缺点：配置复杂、执行速度相对慢、需要额外等待处理


- Cypress：


  - 优点：安装简单、执行速度快、实时重载、调试方便

  - 缺点：只支持JavaScript、同时只能测试一个域名、浏览器支持有限


- Playwright：

  - 优点：支持多种语言、跨浏览器(Chromium/WebKit/Firefox)、自动等待、移动端模拟

  - 缺点：相对较新、社区规模小于Selenium

2\. 什么是Page Object模式？它有什么优点？

Page Object模式是一种设计模式，将页面元素定位和元素操作封装在一个类中。

优点：

- 提高代码可维护性：元素定位与测试逻辑分离

- 减少代码重复：公共操作可以复用

- 增强测试可读性：测试用例更接近业务语言

- 易于维护：UI变更只需修改Page Object类


3\. 如何提高自动化测试的稳定性和可靠性？

- 使用显式等待而非固定等待

- 添加重试机制处理偶发性失败

- 设计独立于环境的测试用例

- 实施截图和日志记录机制

- 定期维护和更新选择器

- 使用数据驱动分离测试逻辑与测试数据

- 建立失败测试分析流程


## 四、性能测试专题

1\. 描述一下性能测试的主要类型和目的

- 负载测试：评估系统在预期负载下的性能表现

- 压力测试：确定系统在极端负载下的稳定性和极限处理能力

- 耐力测试：验证系统在长时间运行下的稳定性和资源使用情况

- 尖峰测试：检查系统在突然增加的负载下的表现

- 容量测试：确定系统能够处理的最大用户数或数据量


2\. 什么是吞吐量、响应时间和并发用户数？它们之间的关系是什么？

- 响应时间：系统对请求作出响应所需的时间

- 吞吐量：单位时间内系统处理的请求数量

- 并发用户数：同时向系统发送请求的用户数量


关系：在系统资源充足的情况下，吞吐量随并发用户数增加而增加，响应时间保持稳定。当达到系统瓶颈时，吞吐量趋于平稳，响应时间开始显著增加。

3\. 如何分析和定位性能瓶颈？

- 前端性能：使用浏览器开发者工具分析页面加载时间、资源大小

- 网络传输：检查网络延迟、带宽限制、CDN效果

- 应用服务器：分析代码效率、数据库查询、缓存使用

- 数据库：检查慢查询、索引使用、连接池配置

- 系统资源：监控CPU、内存、磁盘I/O、网络I/O使用情况

- 外部服务：评估第三方服务的响应时间和可用性


## 五、API测试专题

1\. REST和SOAP的主要区别是什么？

- 协议：REST基于HTTP，SOAP通常基于HTTP或SMTP

- 数据格式：REST使用JSON/XML/纯文本，SOAP只使用XML

- 标准：REST无官方标准，SOAP有WSDL等严格标准

- 性能：REST通常更轻量高效，SOAP消息更冗长

- 安全性：SOAP内置WS-Security等安全标准，REST依赖HTTPS

- 状态：REST是无状态的，SOAP可支持有状态操作


2\. 如何测试一个API接口？考虑哪些方面？

功能测试：

- 验证API输入参数验证和错误处理

- 测试各种HTTP方法(GET/POST/PUT/DELETE)

- 验证响应数据和状态码


性能测试：

- 测试API响应时间

- 评估API吞吐量和并发处理能力

- 检查API在负载下的稳定性


安全测试：

- 验证身份认证和授权机制

- 测试SQL注入和XSS等安全漏洞

- 检查敏感数据泄露


可靠性测试：

- 测试API的容错能力和错误恢复

- 验证超时处理和重试机制


3\. 什么是GraphQL？它与RESTful API相比有什么优势？

GraphQL是一种用于API的查询语言和运行时环境，允许客户端精确请求需要的数据。

优势：

- 减少过度获取：客户端可以精确指定需要的数据字段

- 单一请求获取多个资源：避免多次往返请求

- 强类型系统：API具有明确的类型系统，便于工具开发

- API演进无需版本号：可以添加新字段而不影响现有查询

- 自文档化：内置 introspection 系统，便于文档生成


## 六、测试策略与流程

1\. 描述一下你在项目中是如何制定测试策略的？

制定测试策略的步骤：

- 分析项目需求和业务目标

- 确定测试范围、目标和退出标准

- 识别风险区域并确定测试优先级

- 选择适当的测试类型和技术

- 分配测试资源和制定时间表

- 确定测试环境和数据需求

- 建立缺陷管理流程和报告机制

- 规划测试自动化策略和工具

- 制定回归测试策略


2\. 什么是Shift-Left测试？它有什么好处？

Shift-Left测试是将测试活动提前到软件开发早期阶段的一种实践。

好处：

- 早期发现缺陷，降低修复成本

- 提高开发人员对质量的关注度

- 减少项目后期发现重大问题的风险

- 缩短交付周期，提高交付效率

- 促进开发和测试团队之间的协作


3\. 如何在敏捷团队中有效开展测试工作？

- 参与迭代规划会议，提前理解需求

- 编写测试用例同时进行，而非开发完成后

- 持续进行回归测试，确保现有功能不受影响

- 与开发人员紧密合作，参与代码审查

- 自动化重复测试任务，提高效率

- 每日站会分享测试进展和遇到的问题

- 迭代回顾中提出改进测试过程的建议


## 七、软技能与团队协作

1\. 当你发现一个缺陷但开发人员认为不是问题时，你会怎么做？

- 客观描述问题，提供清晰的复现步骤和实际结果

- 参考需求文档或设计规范，说明期望行为

- 提供截图、日志或屏幕录制作为证据

- 邀请产品经理或业务分析师澄清需求

- 组织三方会议（测试、开发、产品）讨论解决方案

- 保持专业态度，专注于解决问题而非指责

- 必要时将问题上报给项目经理或技术负责人


2\. 如何向非技术人员解释一个复杂的技术问题？

- 使用类比和比喻，将技术概念与日常生活联系起来

- 避免使用技术术语，或先解释术语含义

- 专注于业务影响，而非技术细节

- 使用可视化工具：图表、流程图或示意图

- 分步骤解释，从高层次概念开始逐步深入

- 确认对方理解程度，鼓励提问

- 提供实际例子说明问题的影响和解决方案


3\. 你是如何保持软件测试技能更新的？

- 定期阅读测试相关博客、论坛和技术文章

- 参加行业会议、线上研讨会和技术分享

- 参与开源项目或自己创建测试工具实践

- 学习新技术和工具，获得相关认证

- 与技术社区保持联系，与其他测试人员交流

- 在工作中主动尝试新方法和工具

- 定期总结分享学习成果，撰写技术博客或内部分享


## 八、场景与行为问题

1\. 描述一次你发现的最复杂的缺陷，你是如何定位和报告的？

回答时应包括：

- 项目背景和缺陷的复杂性

- 使用的工具和方法来定位问题

- 与开发团队协作解决问题的过程

- 缺陷对项目的影响和最终解决方案

- 从这次经历中学到的经验教训


2\. 如果你被分配到一个时间非常紧迫的项目，你会如何安排测试工作？

- 优先测试高风险和高业务价值的功能

- 与团队沟通，争取更多资源或调整优先级

- 专注于核心功能测试，非核心功能可考虑简化测试

- 最大化利用自动化测试，减少重复手动工作

- 采用探索性测试快速发现重要问题

- 定期向团队报告测试进展和风险

- 在项目结束后进行复盘，优化未来测试流程


3\. 你如何衡量测试工作的有效性？使用哪些指标？

- 缺陷相关：缺陷密度、缺陷发现率、缺陷修复时间、缺陷重开率

- 测试覆盖：需求覆盖率、代码覆盖率、测试用例执行覆盖率

- 测试效率：测试用例设计效率、测试执行速度、自动化测试 ROI

- 质量评估：逃逸缺陷数量、用户反馈问题数量、上线后故障率

- 过程改进：测试周期时间、测试成本占比、测试活动分布


注意：强调不单独使用任何单一指标，而是结合多个指标综合评估测试效果。

## 九、2025年软件测试趋势

1\. AI在软件测试中的应用有哪些？

- 智能测试用例生成：基于需求或用户行为自动生成测试用例

- 缺陷预测：通过历史数据预测可能产生缺陷的代码区域

- 视觉测试：使用计算机视觉技术验证UI正确性

- 测试优化：智能选择和高优先级测试用例，减少测试套件规模

- 日志分析：自动分析系统日志，发现异常模式和潜在问题

- 自愈自动化：自动修复因UI变化而失败的自动化测试脚本


2\. 测试工程师在未来需要具备哪些新技能？

- AI和机器学习基础知识

- 大数据测试和处理能力

- 云原生和容器化技术理解

- 安全测试和DevSecOps实践

- 性能工程而不仅仅是性能测试

- 编程和脚本能力（Python/Java/JavaScript）

- CI/CD管道构建和维护技能

- 业务分析和数据可视化能力


## 十、反问面试官的问题

面试最后环节，你可以问面试官一些问题，这不仅帮助你了解公司，也展示你的专业性和对职位的兴趣：

- 团队目前的测试策略和流程是怎样的？

- 测试团队与开发团队是如何协作的？

- 公司对测试自动化的投入程度如何？

- 这个职位面临的最大挑战是什么？

- 团队如何保持技术更新和学习？

- 公司的产品质量文化和测试在其中的角色？

- 这个职位的职业发展路径是怎样的？


## 结语

金三银四是软件测试工程师职业发展的黄金时期，充分准备面试是成功的关键。本文提供的面试题和答案解析涵盖了软件测试的各个方面，希望能为你的求职之路提供有力支持。

记住，技术知识固然重要，但解决问题的能力、学习能力和团队协作能力同样不可或缺。祝你在2025年的金九银十招聘季中取得理想offer！

你有遇到过的有趣面试问题吗？欢迎在评论区分享你的面试经历和经验！

本文原创于【程序员二黑】公众号，转载请注明出处！

欢迎大家关注笔者的公众号：程序员二黑，专注于软件测试干活分享，全套测试资源可免费分享！

最后如果你想学习软件测试，欢迎加入笔者的交流群：785128166，里面会有很多资源和大佬答疑解惑，我们一起交流一起学习！

标签:
[Python](https://www.cnblogs.com/hebendexiaomao/tag/Python/), [web测试](https://www.cnblogs.com/hebendexiaomao/tag/web%E6%B5%8B%E8%AF%95/), [功能测试](https://www.cnblogs.com/hebendexiaomao/tag/%E5%8A%9F%E8%83%BD%E6%B5%8B%E8%AF%95/), [接口测试](https://www.cnblogs.com/hebendexiaomao/tag/%E6%8E%A5%E5%8F%A3%E6%B5%8B%E8%AF%95/), [软件测试](https://www.cnblogs.com/hebendexiaomao/tag/%E8%BD%AF%E4%BB%B6%E6%B5%8B%E8%AF%95/), [软件测试工程师](https://www.cnblogs.com/hebendexiaomao/tag/%E8%BD%AF%E4%BB%B6%E6%B5%8B%E8%AF%95%E5%B7%A5%E7%A8%8B%E5%B8%88/), [性能测试](https://www.cnblogs.com/hebendexiaomao/tag/%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95/), [自动化测试](https://www.cnblogs.com/hebendexiaomao/tag/%E8%87%AA%E5%8A%A8%E5%8C%96%E6%B5%8B%E8%AF%95/)

免责声明：本内容来自平台创作者，博客园系信息发布平台，仅提供信息存储空间服务。


好文要顶关注我收藏该文微信分享

[![](https://pic.cnblogs.com/face/2951280/20221216202433.png)](https://home.cnblogs.com/u/hebendexiaomao/)

[程序员二黑](https://home.cnblogs.com/u/hebendexiaomao/)

[粉丝 \- 71](https://home.cnblogs.com/u/hebendexiaomao/followers/) [关注 \- 0](https://home.cnblogs.com/u/hebendexiaomao/followees/)

+加关注

0

0

[升级成为会员](https://cnblogs.vip/)

[«](https://www.cnblogs.com/hebendexiaomao/p/19088592) 上一篇： [测试工程师的核心竞争力是什么？绝不是点点点](https://www.cnblogs.com/hebendexiaomao/p/19088592 "发布于 2025-09-12 21:34")

[»](https://www.cnblogs.com/hebendexiaomao/p/19095228) 下一篇： [面试官视角：什么样的测试工程师能拿到年薪50W+？](https://www.cnblogs.com/hebendexiaomao/p/19095228 "发布于 2025-09-16 17:09")

posted @
2025-09-15 15:24 [程序员二黑](https://www.cnblogs.com/hebendexiaomao)
阅读(1241)
评论(0)

收藏 [举报](https://report.cnblogs.com/?targetLink=https%3A%2F%2Fwww.cnblogs.com%2Fhebendexiaomao%2Fp%2F19093043&targetId=19093043&targetType=0)

[刷新页面](https://www.cnblogs.com/hebendexiaomao/p/19093043#) [返回顶部](https://www.cnblogs.com/hebendexiaomao/p/19093043#top)

登录后才能查看或发表评论，立即 登录 或者
[逛逛](https://www.cnblogs.com/) 博客园首页

[![](https://img2024.cnblogs.com/blog/35695/202604/35695-20260423213336272-1914399152.webp)](https://www.volcengine.com/activity/codingplan?utm_campaign=hw&utm_content=hw&utm_medium=devrel_tool_web&utm_source=OWO&utm_term=cnblogs)

- [从 305 GB 到 7.4 GB：大模型 KVCache 架构演进全景](https://www.cnblogs.com/cswuyg/p/19981922)
- [DeepSeek V4模型的Agent能力实测](https://www.cnblogs.com/zhayujie/p/19935607/deepseek-v4-eval)
- [C# 15 类型系统改进：Union Types](https://www.cnblogs.com/hez2010/p/19891530/union-types-in-csharp-15)
- [你能被装进一个文件里吗？——7 万人把同事“蒸馏”成了 AI](https://www.cnblogs.com/wmyskxz/p/19854791)
- [别再吹牛了，100% Vibe Coding 存在无法自洽的逻辑漏洞！](https://www.cnblogs.com/mengxiang2/p/19796426)

### 公告

我自己创建的测试学习交流群： **[735745871](https://jq.qq.com/?_wv=1027&k=3Az9HK3x)**。

众多免费分享如：自动化、接口、性能、敏捷等这些成为高级测试工程师必备的知识体系。欢迎加入，共同成长！

昵称：
[程序员二黑](https://home.cnblogs.com/u/hebendexiaomao/)

园龄：
[3年8个月](https://home.cnblogs.com/u/hebendexiaomao/ "入园时间：2022-08-10")

粉丝：
[71](https://home.cnblogs.com/u/hebendexiaomao/followers/)

关注：
[0](https://home.cnblogs.com/u/hebendexiaomao/followees/)

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

- [我的随笔](https://www.cnblogs.com/hebendexiaomao/p/ "我的博客的随笔列表")
- [我的评论](https://www.cnblogs.com/hebendexiaomao/MyComments.html "我的发表过的评论列表")
- [我的参与](https://www.cnblogs.com/hebendexiaomao/OtherPosts.html "我评论过的随笔列表")
- [最新评论](https://www.cnblogs.com/hebendexiaomao/comments "我的博客的评论列表")
- [我的标签](https://www.cnblogs.com/hebendexiaomao/tag/ "我的博客的标签列表")

### [我的标签](https://www.cnblogs.com/hebendexiaomao/tag/)

- [软件测试工程师(128)](https://www.cnblogs.com/hebendexiaomao/tag/%E8%BD%AF%E4%BB%B6%E6%B5%8B%E8%AF%95%E5%B7%A5%E7%A8%8B%E5%B8%88/)
- [软件测试(128)](https://www.cnblogs.com/hebendexiaomao/tag/%E8%BD%AF%E4%BB%B6%E6%B5%8B%E8%AF%95/)
- [自动化测试(128)](https://www.cnblogs.com/hebendexiaomao/tag/%E8%87%AA%E5%8A%A8%E5%8C%96%E6%B5%8B%E8%AF%95/)
- [接口测试(128)](https://www.cnblogs.com/hebendexiaomao/tag/%E6%8E%A5%E5%8F%A3%E6%B5%8B%E8%AF%95/)
- [性能测试(128)](https://www.cnblogs.com/hebendexiaomao/tag/%E6%80%A7%E8%83%BD%E6%B5%8B%E8%AF%95/)
- [功能测试(128)](https://www.cnblogs.com/hebendexiaomao/tag/%E5%8A%9F%E8%83%BD%E6%B5%8B%E8%AF%95/)
- [web测试(128)](https://www.cnblogs.com/hebendexiaomao/tag/web%E6%B5%8B%E8%AF%95/)
- [Python(102)](https://www.cnblogs.com/hebendexiaomao/tag/Python/)

### 随笔档案

- [2025年11月(3)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/11)
- [2025年10月(12)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/10)
- [2025年9月(25)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/09)
- [2025年8月(22)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/08)
- [2025年7月(4)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/07)
- [2025年6月(2)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/06)
- [2025年3月(1)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/03)
- [2025年2月(3)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/02)
- [2025年1月(7)](https://www.cnblogs.com/hebendexiaomao/p/archive/2025/01)
- [2024年12月(5)](https://www.cnblogs.com/hebendexiaomao/p/archive/2024/12)
- [2024年11月(1)](https://www.cnblogs.com/hebendexiaomao/p/archive/2024/11)
- [2023年11月(3)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/11)
- [2023年10月(6)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/10)
- [2023年9月(3)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/09)
- [2023年8月(3)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/08)
- [2023年7月(2)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/07)
- [2023年6月(5)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/06)
- [2023年5月(2)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/05)
- [2023年4月(3)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/04)
- [2023年3月(6)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/03)
- [2023年2月(7)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/02)
- [2023年1月(1)](https://www.cnblogs.com/hebendexiaomao/p/archive/2023/01)
- [2022年12月(2)](https://www.cnblogs.com/hebendexiaomao/p/archive/2022/12)
- [2022年8月(1)](https://www.cnblogs.com/hebendexiaomao/p/archive/2022/08)

### [阅读排行榜](https://www.cnblogs.com/hebendexiaomao/most-viewed)

- [1\. 基于Python的Selenium详细教程(5210)](https://www.cnblogs.com/hebendexiaomao/p/18700094)
- [2\. pytest接口自动化测试框架搭建的全过程(5022)](https://www.cnblogs.com/hebendexiaomao/p/17512666.html)
- [3\. 软件测试（测试用例）—写用例无压力(4910)](https://www.cnblogs.com/hebendexiaomao/p/18673728)
- [4\. 集成测试最全详解，看完必须懂了(4581)](https://www.cnblogs.com/hebendexiaomao/p/17548929.html)
- [5\. Postman如何导出接口的几种方法？(3817)](https://www.cnblogs.com/hebendexiaomao/p/18671470)

### [评论排行榜](https://www.cnblogs.com/hebendexiaomao/most-commented)

- [1\. 当面试问你接口测试时，不要再说不会了(2)](https://www.cnblogs.com/hebendexiaomao/p/17798102.html)
- [2\. pytest接口自动化测试框架搭建的全过程(2)](https://www.cnblogs.com/hebendexiaomao/p/17512666.html)
- [3\. 八年 “自动化测试” 老鸟，写给 3-5 年测试员的几点建议，满满硬货指导(2)](https://www.cnblogs.com/hebendexiaomao/p/17106749.html)
- [4\. 基于Python的Selenium详细教程(1)](https://www.cnblogs.com/hebendexiaomao/p/18700094)
- [5\. 软件测试（测试用例）—写用例无压力(1)](https://www.cnblogs.com/hebendexiaomao/p/18673728)

### [推荐排行榜](https://www.cnblogs.com/hebendexiaomao/most-liked)

- [1\. pytest接口自动化测试框架搭建的全过程(3)](https://www.cnblogs.com/hebendexiaomao/p/17512666.html)
- [2\. 想自学软件测试？一般人我还是劝你算了吧。。。(2)](https://www.cnblogs.com/hebendexiaomao/p/17247859.html)
- [3\. 在职阿里6年，一个29岁女软件测试工程师的心声(2)](https://www.cnblogs.com/hebendexiaomao/p/17006976.html)
- [4\. 软件测试工程师必看：手把手教你用Fiddler进行数据抓包与分析(1)](https://www.cnblogs.com/hebendexiaomao/p/19054793)
- [5\. 软件测试基本流程和方法：从入门到精通(1)](https://www.cnblogs.com/hebendexiaomao/p/18946723)

点击右上角即可分享

![微信分享提示](https://img2023.cnblogs.com/blog/35695/202309/35695-20230906145857937-1471873834.gif)