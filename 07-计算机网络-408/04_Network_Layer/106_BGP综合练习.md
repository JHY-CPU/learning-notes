# BGP综合练习

## 核心概念
- BGP（Border Gateway Protocol）：边界网关协议，用于自治系统（AS）之间的路由
- BGP是路径向量协议（Path Vector）
- BGP使用TCP端口179建立连接
- BGP交换的是到达各网络的完整路径（AS-PATH）

### BGP基本概念
- **AS（Autonomous System）**：自治系统，由同一机构管理的网络
- **AS号（ASN）**：16位或32位自治系统编号
- **eBGP**：不同AS之间的BGP连接
- **iBGP**：同一AS内部的BGP连接

### BGP四种报文
| 类型 | 名称 | 功能 |
|------|------|------|
| 1 | OPEN | 建立BGP连接 |
| 2 | UPDATE | 通告/撤销路由 |
| 3 | NOTIFICATION | 错误通知 |
| 4 | KEEPALIVE | 保活（60秒） |

### BGP路径属性
| 属性 | 说明 |
|------|------|
| AS-PATH | 经过的AS列表（防环） |
| NEXT-HOP | 下一跳地址 |
| LOCAL_PREF | 本地优先级（iBGP） |
| MED | 多出口鉴别符（eBGP） |

## 原理分析

### BGP选路规则（简化）
1. 最高LOCAL_PREF
2. 最短AS-PATH
3. 最低ORIGIN类型
4. 最低MED
5. eBGP优于iBGP
6. 最近的IGP下一跳
7. 最低Router ID

### BGP防环机制
- AS-PATH属性：收到包含自己AS号的UPDATE，拒绝（检测到环路）
- 水平分割：iBGP不转发从iBGP学到的路由（需RR或全网状连接）

## 直观理解
- BGP像"国际快递"——每个国家（AS）告诉邻居国家"到X国可以经过我这里的路线"
- AS-PATH像"快递经过的城市列表"——经过的城市不能重复（防环）

## 协议关联
- BGP是外部网关协议（EGP），OSPF/RIP是内部网关协议（IGP）
- BGP承载在TCP之上（端口179），保证可靠传输
- BGP决定互联网的全局路由（全球路由表）
