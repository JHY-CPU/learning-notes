# 监控常见面试题 - Egon林海峰

URL: https://egonlin.com/?p=1736

**Egon的技术星球**

夜间模式暗黑模式

字体

Sans SerifSerif

阴影

浅阴影深阴影

滤镜

关闭日落暗化灰度

圆角

主题色

0%

## 一、简述常见的监控软件？

```bash

    Cacti：是一套基于PHP、MySQL、SNMP及RRDTool开发的网络流量监测图形分析工具。



    Zabbix：Zabbix是一个企业级的高度集成开源监控软件，提供分布式监控解决方案。可以用来监控设备、服务等可用性和性能。

    Open-falcon：open-falcon是一款用golang和python写的监控系统，由小米启动这个项目。

    Prometheus：Prometheus是由SoundCloud开发的开源监控报警系统和时序列数据库(TSDB)。Prometheus使用Go语言开发，是Google BorgMon监控系统的开源版本。


















```

## 二、简述Prometheus及其主要特性？

### 1、简述

```bash

    Prometheus是一个已加入CNCF的开源监控报警系统和时序列数据库项目，通过不同的组件完成数据的采集，数据的存储和告警。


















```

### 2、主要特性

#### 1.多维数据模型

```bash

时间序列数据通过 metric 名和键值对来区分。

所有的 metrics 都可以设置任意的多维标签。

数据模型更随意，不需要刻意设置为以点分隔的字符串。

可以对数据模型进行聚合，切割和切片操作。

支持双精度浮点类型，标签可以设为全 unicode。


















```

#### 2.灵活的查询语句（PromQL）

```bash

可以利用多维数据完成复杂的查询


















```

#### 3.集成度高

> Prometheus server 是一个单独的二进制文件，不依赖（任何分布式）存储，支持 local 和 remote 不同模型

#### 4.数据拉取

> 采用 http 协议，使用 pull 模式，拉取数据，或者通过中间网关推送方式采集数据

#### 5.目标发现

> 监控目标，可以采用服务发现或静态配置的方式

#### 6.高效

> 一个 Prometheus server 可以处理数百万的 metrics适用于以机器为中心的监控以及高度动态面向服务架构的监控

## 三、简述Prometheus主要组件及其功能？

#### 1.Prometheus Server

## 查看更多

联系管理员微信tutu19192010，注册账号

| | Username: |
| --- |
|  |
| Password: |
|  |

Log in |  |

[Register](https://egonlin.com/wp-login.php?action=register)

[Forgotten username or password?](https://egonlin.com/wp-login.php?action=lostpassword)

[linux](https://egonlin.com/?tag=linux) [prometheus](https://egonlin.com/?tag=prometheus) [SRE](https://egonlin.com/?tag=sre) [zabbix](https://egonlin.com/?tag=zabbix) [监控](https://egonlin.com/?tag=%e7%9b%91%e6%8e%a7) [面试](https://egonlin.com/?tag=%e9%9d%a2%e8%af%95)

上一篇 [jenkins常见面试题](https://egonlin.com/?p=1733)

下一篇 [mysql常见面试题](https://egonlin.com/?p=1739)