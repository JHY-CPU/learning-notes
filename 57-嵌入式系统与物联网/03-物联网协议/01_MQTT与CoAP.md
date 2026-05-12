# MQTT与CoAP协议


## MQTT与CoAP协议

一、物联网协议概述

## 一、物联网通信协议概述


物联网设备通常资源受限、网络不稳定，需要轻量级、低功耗的通信协议。MQTT和CoAP是两种主流的物联网消息协议。


| 协议 | 模式 | 传输层 | 特点 |
| --- | --- | --- | --- |
| MQTT | 发布/订阅 | TCP | 轻量、可靠、支持QoS |
| CoAP | 请求/响应 | UDP | RESTful、极轻量 |
| HTTP | 请求/响应 | TCP | 通用、重量级 |

二、MQTT

## 二、MQTT（消息队列遥测传输）


MQTT是IBM开发的轻量级消息协议，采用发布/订阅模式，专为低带宽、高延迟或不可靠网络设计。


### 2.1 核心概念


| 概念 | 说明 |
| --- | --- |
| Broker | 消息代理服务器，负责消息路由 |
| Client | 发布者或订阅者 |
| Topic | 消息主题，支持层级和通配符 |
| QoS | 服务质量等级，保证消息送达 |
| Session | 客户端与Broker之间的会话状态 |
| Retain | 保留消息，新订阅者立即收到 |
| Will | 遗嘱消息，客户端异常断开时发布 |


### 2.2 QoS等级


| QoS | 名称 | 保证 | 开销 |
| --- | --- | --- | --- |
| 0 | 最多一次 | 不保证送达 | 最低 |
| 1 | 至少一次 | 保证送达，可能重复 | 中等 |
| 2 | 恰好一次 | 保证送达且不重复 | 最高 |


### 2.3 Topic通配符


- `+`
   ：单层通配符，匹配一个层级


   示例：
   `sensor/+/temperature`
   匹配
   `sensor/room1/temperature`
- `#`
   ：多层通配符，匹配零或多层


   示例：
   `sensor/#`
   匹配
   `sensor/room1/temperature`


### 2.4 MQTT控制报文


| 报文类型 | 方向 | 说明 |
| --- | --- | --- |
| CONNECT | Client → Broker | 建立连接 |
| CONNACK | Broker → Client | 连接确认 |
| PUBLISH | 双向 | 发布消息 |
| SUBSCRIBE | Client → Broker | 订阅主题 |
| PINGREQ/PINGRESP | 双向 | 心跳保活 |
| DISCONNECT | Client → Broker | 正常断开 |

**MQTT报文结构：**
固定头（1-5字节）+ 可变头 + 载荷。固定头最小2字节，非常适合受限设备。
三、CoAP

## 三、CoAP（受限应用协议）


CoAP是IETF为受限设备设计的Web传输协议，基于REST架构，使用UDP传输。


### 3.1 核心特性


- **RESTful：**
   使用GET/POST/PUT/DELETE方法
- **基于UDP：**
   减少连接开销
- **确认机制：**
   通过ACK实现可靠传输
- **观察机制：**
   客户端可订阅资源变化（Observe）
- **块传输：**
   支持大数据分块传输
- **DTLS：**
   基于UDP的安全传输


### 3.2 请求/响应模式


| 模式 | 说明 | 延迟 |
| --- | --- | --- |
| Confirmable (CON) | 需要ACK确认 | 需要等待ACK |
| Non-confirmable (NON) | 不需要确认 | 最低延迟 |


### 3.3 CoAP消息类型


| 类型 | 代码 | 说明 |
| --- | --- | --- |
| CON | 0 | 可靠消息，需ACK |
| NON | 1 | 不可靠消息，无需ACK |
| ACK | 2 | 确认消息 |
| RST | 3 | 重置消息 |


### 3.4 Observe模式


CoAP的Observe扩展允许客户端"观察"服务器资源的变化，类似于MQTT的订阅。


> **Note:** 客户端发送带Observe选项的GET请求，服务器在资源变化时主动推送通知，无需客户端轮询。

四、MQTT vs CoAP vs HTTP

## 四、协议对比


| 维度 | MQTT | CoAP | HTTP |
| --- | --- | --- | --- |
| 通信模式 | 发布/订阅 | 请求/响应 | 请求/响应 |
| 传输层 | TCP | UDP | TCP |
| 报文大小 | 最小2字节 | 最小4字节 | 数百字节 |
| 开销 | 低 | 极低 | 高 |
| 安全性 | TLS/SSL | DTLS | TLS/SSL |
| 适合场景 | 低带宽、不可靠网络 | 受限设备、LAN | 通用Web服务 |
| 发现机制 | 无 | CoRE Link Format | URL |
| 代理支持 | 原生Broker | HTTP代理 | 原生代理 |

**选择建议：**


- 需要发布/订阅模式 → MQTT


- 资源极度受限、需要RESTful接口 → CoAP


- 需要与现有Web基础设施集成 → HTTP


- 网络不可靠但需可靠传输 → MQTT QoS 1/2
========================================
  文件总结
========================================
  主题：MQTT与CoAP协议
  内容概要：
    1. MQTT - 发布/订阅模式，QoS 0/1/2，Topic通配符(+/#)
    2. CoAP - RESTful协议，基于UDP，CON/NON消息类型，Observe模式
    3. 协议对比 - MQTT/CoAP/HTTP在通信模式、开销、适用场景的差异
  重点知识：
    - MQTT QoS等级：0最多一次、1至少一次、2恰好一次
    - MQTT最小报文2字节，非常轻量
    - CoAP基于UDP，使用DTLS安全传输
    - CoAP Observe类似MQTT的订阅功能
    - 选择：发布订阅选MQTT，RESTful选CoAP，通用选HTTP
========================================


<!-- Converted from: 01_MQTT与CoAP.html -->
