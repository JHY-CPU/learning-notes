# IT运维面试问题总结-数据库、监控、网络管理 - YP小站

URL: https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/

## [数据库](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E6%95%B0%E6%8D%AE%E5%BA%93 "数据库") 数据库

### [简述NoSQL是什么？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0NoSQL%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F "简述NoSQL是什么？") 简述NoSQL是什么？

NoSQL，指的是非关系型的数据库。NoSQL 有时也称作 Not Only SQL（意即”不仅仅是SQL”） 的缩写，其显著特点是不使用SQL作为查询语言，数据存储不需要特定的表格模式。

### [简述NoSQL（非关系型）数据库和SQL（关系型）数据库的区别？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0NoSQL%EF%BC%88%E9%9D%9E%E5%85%B3%E7%B3%BB%E5%9E%8B%EF%BC%89%E6%95%B0%E6%8D%AE%E5%BA%93%E5%92%8CSQL%EF%BC%88%E5%85%B3%E7%B3%BB%E5%9E%8B%EF%BC%89%E6%95%B0%E6%8D%AE%E5%BA%93%E7%9A%84%E5%8C%BA%E5%88%AB%EF%BC%9F "简述NoSQL（非关系型）数据库和SQL（关系型）数据库的区别？") 简述NoSQL（非关系型）数据库和SQL（关系型）数据库的区别？

NoSQL和SQL的主要区别有如下区别：

- 存储方式：
  - 关系型数据库是表格式的，因此存储在表的行和列中。他们之间很容易关联协作存储，提取数据很方便。
  - NoSQL数据库则与其相反，它是大块的组合在一起。通常存储在数据集中，就像文档、键值对或者图结构。
- 存储结构
  - 关系型数据库对应的是结构化数据，数据表都预先定义了结构（列的定义），结构描述了数据的形式和内容。预定义结构带来了可靠性和稳定性，但是修改这些数据比较困难。
  - NoSQL数据库基于动态结构，使用与非结构化数据。由于NoSQL数据库是动态结构，可以很容易适应数据类型和结构的变化。
- 存储规范
  - 关系型数据库的数据存储为了更高的规范性，把数据分割为最小的关系表以避免重复，获得精简的空间利用。
  - NoSQL数据存储在平面数据集中，数据经常可能会重复。单个数据库很少被分隔开，而是存储成了一个整体，这样整块数据更加便于读写 。
- 存储扩展
  - 关系型数据库数据存储在关系表中，操作的性能瓶颈可能涉及到多个表，需要通过提升计算机性能来克服，因此更多是采用纵向扩展
  - NoSQL数据库是横向扩展的，它的存储天然就是分布式的，可以通过给资源池添加更多的普通数据库服务器来分担负载。
- 查询方式
  - 关系型数据库通过结构化查询语言来操作数据库（即通常说的SQL）。SQL支持数据库CURD操作的功能非常强大，是业界的标准用法。
  - NoSQL查询以块为单元操作数据，使用的是非结构化查询语言（UnQl），它是没有标准的。
  - 关系型数据库表中主键的概念对应NoSQL中存储文档的ID。
  - 关系型数据库使用预定义优化方式（比如索引）来加快查询操作，而NoSQL更简单更精确的数据访问模式。
- 事务
  - 关系型数据库遵循ACID规则（原子性(Atomicity)、一致性(Consistency)、隔离性(Isolation)、持久性(Durability)）。
  - NoSQL数据库遵循BASE原则（基本可用（Basically Availble）、软/柔性事务（Soft-state ）、最终一致性（Eventual Consistency））。
  - 由于关系型数据库的数据强一致性，所以对事务的支持很好。关系型数据库支持对事务原子性细粒度控制，并且易于回滚事务。
  - NoSQL数据库是在CAP（一致性、可用性、分区容忍度）中任选两项，因为基于节点的分布式系统中，不可能同时全部满足，所以对事务的支持不是很好。

### [简述NoSQL（非关系型）数据库和SQL（关系型）数据库的各自主要代表？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0NoSQL%EF%BC%88%E9%9D%9E%E5%85%B3%E7%B3%BB%E5%9E%8B%EF%BC%89%E6%95%B0%E6%8D%AE%E5%BA%93%E5%92%8CSQL%EF%BC%88%E5%85%B3%E7%B3%BB%E5%9E%8B%EF%BC%89%E6%95%B0%E6%8D%AE%E5%BA%93%E7%9A%84%E5%90%84%E8%87%AA%E4%B8%BB%E8%A6%81%E4%BB%A3%E8%A1%A8%EF%BC%9F "简述NoSQL（非关系型）数据库和SQL（关系型）数据库的各自主要代表？") 简述NoSQL（非关系型）数据库和SQL（关系型）数据库的各自主要代表？

SQL：MariaDB、MySQL、SQLite、SQLServer、Oracle、PostgreSQL。

NoSQL代表：Redis、MongoDB、Memcache、HBASE。

### [简述MongoDB及其特点？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%8F%8A%E5%85%B6%E7%89%B9%E7%82%B9%EF%BC%9F "简述MongoDB及其特点？") 简述MongoDB及其特点？

MongoDB是一个开源的、基于分布式的、面向文档存储的非关系型数据库。是非关系型数据库当中功能最丰富、最像关系数据库的。其主要特点如下：

- `查询丰富`：MongoDB最大的特点是支持的查询语言非常强大，其语法有点类似于面向对象的查询语言，几乎可以实现类似关系数据库单表查询的绝大部分功能，而且还支持对数据建立索引。
- `面向文档`：文档就是存储在MongoDB中的一条记录,是一个由键值对组成的数据结构。
- `模式自由`：MongoDB每一个Document都包含了元数据信息，每个文档之间不强迫要求使用相同的格式，同时他们也支持各种索引。
- `高可用性`：MongoDB支持在复制集(Replica Set)通过异步复制达到故障转移，自动恢复，集群中主服务器崩溃停止服务和丢失数据，备份服务器通过选举获得大多数投票成为主节点，以此来实现高可用。
- `水平拓展`：MongoDB支持分片技术，它能够支持并行处理和水平扩展。
- `支持丰富`：MongoDB另外还提供了丰富的BSON数据类型，还有MongoDB的官方不同语言的driver支持(C/C++、C#、Java、Node.js、Perl、PHP、Python、Ruby、Scala)。

### [简述MongoDB的优势有哪些？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E7%9A%84%E4%BC%98%E5%8A%BF%E6%9C%89%E5%93%AA%E4%BA%9B%EF%BC%9F "简述MongoDB的优势有哪些？") 简述MongoDB的优势有哪些？

- 面向文档的存储：以 JSON 格式的文档保存数据。
- 任何属性都可以建立索引。
- 复制以及高可扩展性。
- 自动分片。
- 丰富的查询功能。
- 快速的即时更新。

### [简述MongoDB适应的场景和不适用的场景？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E9%80%82%E5%BA%94%E7%9A%84%E5%9C%BA%E6%99%AF%E5%92%8C%E4%B8%8D%E9%80%82%E7%94%A8%E7%9A%84%E5%9C%BA%E6%99%AF%EF%BC%9F "简述MongoDB适应的场景和不适用的场景？") 简述MongoDB适应的场景和不适用的场景？

MongoDB属于典型的非关系型数据库。

- 主要适应场景
  - 网站实时数据：MongoDB 非常适合实时的插入，更新与查询，并具备网站实时数据存储所需的复制及高度伸缩性。
  - 数据缓存：由于性能很高，MongoDB 也适合作为信息基础设施的缓存层。在系统重启之后，由 MongoDB 搭建的持久化缓存层可以避免下层的数据源过载。
  - 高伸缩性场景：MongoDB 非常适合由数十或数百台服务器组成的数据库。
  - 对象或 JSON 数据存储：MongoDB 的 BSON 数据格式非常适合文档化格式的存储及查询。
- 不适应场景
  - 高度事务性系统：例如银行或会计系统。传统的关系型数据库目前还是更适用于需要大量原子性复杂事务的应用程序。
  - 传统的商业智能应用：针对特定问题的 BI 数据库会对产生高度优化的查询方式。对于此类应用，数据仓库可能是更合适的选择。
  - 需要复杂 SQL 查询的场景。

### [简述MongoDB中的库、集合、文档？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E4%B8%AD%E7%9A%84%E5%BA%93%E3%80%81%E9%9B%86%E5%90%88%E3%80%81%E6%96%87%E6%A1%A3%EF%BC%9F "简述MongoDB中的库、集合、文档？") 简述MongoDB中的库、集合、文档？

- `库`：MongoDB可以建立多个数据库，MongoDB默认数据库为”db”。MongoDB的单个实例可以容纳多个独立的数据库，每一个都有自己的集合和权限，不同的数据库也放置在不同的文件中。
- `集合`：MongoDB集合就是 MongoDB 文档组，类似于 RDBMS （关系数据库中的表格）。集合存在于数据库中，集合没有固定的结构。
- `文档`：MongoDB的Document是一组键值(key-value)对(即 BSON)，相当于关系型数据库的行。且不需要设置相同的字段，并且相同的字段不需要相同的数据类型。

### [简述MongoDB支持的常见数据类型？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E6%94%AF%E6%8C%81%E7%9A%84%E5%B8%B8%E8%A7%81%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B%EF%BC%9F "简述MongoDB支持的常见数据类型？") 简述MongoDB支持的常见数据类型？

MongoDB支持丰富的数据类型，常见的有：

- String：字符串。存储数据常用的数据类型。
- Integer：整型数值。用于存储数值。
- Boolean：布尔值。用于存储布尔值（真/假）。
- Array：用于将数组或列表或多个值存储为一个键。
- Date：日期时间。用 UNIX 时间格式来存储当前日期或时间。
- Binary Data：二进制数据。用于存储二进制数据。
- Code：代码类型。用于在文档中存储 JavaScript 代码。
- Regular expression：正则表达式类型。用于存储正则表达式。

### [简述MongoDB索引及其作用？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E7%B4%A2%E5%BC%95%E5%8F%8A%E5%85%B6%E4%BD%9C%E7%94%A8%EF%BC%9F "简述MongoDB索引及其作用？") 简述MongoDB索引及其作用？

索引通常能够极大的提高查询的效率，如果没有索引，MongoDB在读取数据时必须扫描集合中的每个文件并选取那些符合查询条件的记录。

这种扫描全集合的查询效率是非常低的，特别在处理大量的数据时，查询可能要花费几十秒甚至几分钟，这对网站的性能是非常致命的。

索引是特殊的数据结构，索引存储在一个易于遍历读取的数据集合中，索引是对数据库表中一列或多列的值进行排序的一种结构。

### [简述MongoDB常见的索引有哪些？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%B8%B8%E8%A7%81%E7%9A%84%E7%B4%A2%E5%BC%95%E6%9C%89%E5%93%AA%E4%BA%9B%EF%BC%9F "简述MongoDB常见的索引有哪些？") 简述MongoDB常见的索引有哪些？

MongoDB常见的索引有：

- 单字段索引（Single Field Indexes）
- 符合索引（Compound Indexes）
- 多键索引（Multikey Indexes）
- 全文索引（Text Indexes）
- Hash索引（Hash Indexes）
- 通配符索引（Wildcard Indexes）

### [简述MongoDB复制（本）集原理？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%A4%8D%E5%88%B6%EF%BC%88%E6%9C%AC%EF%BC%89%E9%9B%86%E5%8E%9F%E7%90%86%EF%BC%9F "简述MongoDB复制（本）集原理？") 简述MongoDB复制（本）集原理？

mongodb的复制至少需要两个节点。其中一个是主节点，负责处理客户端请求，其余的都是从节点，负责复制主节点上的数据。

mongodb各个节点常见的搭配方式为：一主一从、一主多从。

主节点记录在其上的所有操作oplog，从节点定期轮询主节点获取这些操作，然后对自己的数据副本执行这些操作，从而保证从节点的数据与主节点一致。

### [简述MongoDB的复制过程？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E7%9A%84%E5%A4%8D%E5%88%B6%E8%BF%87%E7%A8%8B%EF%BC%9F "简述MongoDB的复制过程？") 简述MongoDB的复制过程？

Primary节点写入数据，Secondary通过读取Primary的oplog（即Primary的oplog.rs表）得到复制信息，开始复制数据并且将复制信息写入到自己的oplog。如果某个操作失败，则备份节点停止从当前数据源复制数据。如果某个备份节点由于某些原因挂掉了，当重新启动后，就会自动从oplog的最后一个操作开始同步。同步完成后，将信息写入自己的oplog，由于复制操作是先复制数据，复制完成后再写入oplog，有可能相同的操作会同步两份，MongoDB设定将oplog的同一个操作执行多次，与执行一次的效果是一样的。

当Primary节点完成数据操作后，Secondary的数据同步过程如下：

- 1、检查自己local库的oplog.rs集合找出最近的时间戳。
- 2、检查Primary节点local库oplog.rs集合，找出大于此时间戳的记录。
- 3、将找到的记录插入到自己的oplog.rs集合中，并执行这些操作。

### [简述MongoDB副本集及其特点？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%89%AF%E6%9C%AC%E9%9B%86%E5%8F%8A%E5%85%B6%E7%89%B9%E7%82%B9%EF%BC%9F "简述MongoDB副本集及其特点？") 简述MongoDB副本集及其特点？

MongoDB副本集是一组Mongod维护相同数据集的实例，副本集可以包含多个数据承载点和多个仲裁点。在承载数据的节点中，仅有一个节点被视为主节点，其他节点称为次节点。

主要特点：

- N 个节点的集群，任何节点可作为主节点，由选举产生；
- 最小构成是：primary，secondary，arbiter，一般部署是：primary，2 secondary。
- 所有写入操作都在主节点上，同时具有自动故障转移，自动恢复；
- 成员数应该为奇数，如果为偶数的情况下添加arbiter，arbiter不保存数据，只投票。

### [简述MongoDB有哪些特殊成员？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E6%9C%89%E5%93%AA%E4%BA%9B%E7%89%B9%E6%AE%8A%E6%88%90%E5%91%98%EF%BC%9F "简述MongoDB有哪些特殊成员？") 简述MongoDB有哪些特殊成员？

MongoDB中Secondary角色存在一些特殊的成员类型：

- Priority 0（优先级0型）：不能升为主，可以用于多数据中心场景；
- Hidden（隐藏型）：对客户端来说是不可见的，一般用作备份或统计报告用；
- Delayed（延迟型）：数据比副集晚，一般用作 rolling backup 或历史快照。
- Vote（投票型）：仅参与投票。

### [简述MongoDB分片集群？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%88%86%E7%89%87%E9%9B%86%E7%BE%A4%EF%BC%9F "简述MongoDB分片集群？") 简述MongoDB分片集群？

MongoDB分片集群（Sharded Cluster）：主要利用分片技术，使数据分散存储到多个分片（Shard）上，来实现高可扩展性。

分片是将数据水平切分到不同的物理节点。当数据量越来越大时，单台机器有可能无法存储数据或读取写入吞吐量有所降低，利用分片技术可以添加更多的机器来应对数据量增加以及读写操作的要求。

### [简述MongoDB分片集群相对副本集的优势？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%88%86%E7%89%87%E9%9B%86%E7%BE%A4%E7%9B%B8%E5%AF%B9%E5%89%AF%E6%9C%AC%E9%9B%86%E7%9A%84%E4%BC%98%E5%8A%BF%EF%BC%9F "简述MongoDB分片集群相对副本集的优势？") 简述MongoDB分片集群相对副本集的优势？

MongoDB分片集群主要可以解决副本集如下的不足：

- 副本集所有的写入操作都位于主节点；
- 延迟的敏感数据会在主节点查询；
- 单个副本集限制在12个节点；
- 当请求量巨大时会出现内存不足；
- 本地磁盘不足；
- 垂直扩展价格昂贵。

### [简述MongoDB分片集群的优势？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%88%86%E7%89%87%E9%9B%86%E7%BE%A4%E7%9A%84%E4%BC%98%E5%8A%BF%EF%BC%9F "简述MongoDB分片集群的优势？") 简述MongoDB分片集群的优势？

MongoDB分片集群主要有如下优势：

- 使用分片减少了每个分片需要处理的请求数：通过水平扩展，群集可以提高自己的存储容量。比如，当插入一条数据时，应用只需要访问存储这条数据的分片。
- 使用分片减少了每个分片存储的数据：分片的优势在于提供类似线性增长的架构，提高数据可用性，提高大型数据库查询服务器的性能。当MongoDB单点数据库服务器存储成为瓶颈、单点数据库服务器的性能成为瓶颈或需要部署大型应用以充分利用内存时，可以使用分片技术。

### [简述MongoDB分片集群的架构组件？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%88%86%E7%89%87%E9%9B%86%E7%BE%A4%E7%9A%84%E6%9E%B6%E6%9E%84%E7%BB%84%E4%BB%B6%EF%BC%9F "简述MongoDB分片集群的架构组件？") 简述MongoDB分片集群的架构组件？

MongoDB架构组件主要有：

- Shard：用于存储实际的数据块，实际生产环境中一个shard server角色可由几台机器组成一个replica set承担，防止主机单点故障。
- Config Server：mongod实例，存储了整个 ClusterMetadata，其中包括 chunk信息。
- Query Routers：前端路由，客户端由此接入，且让整个集群看上去像单一数据库，前端应用可以透明使用。

### [简述MongoDB分片集群和副本集群的区别？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%88%86%E7%89%87%E9%9B%86%E7%BE%A4%E5%92%8C%E5%89%AF%E6%9C%AC%E9%9B%86%E7%BE%A4%E7%9A%84%E5%8C%BA%E5%88%AB%EF%BC%9F "简述MongoDB分片集群和副本集群的区别？") 简述MongoDB分片集群和副本集群的区别？

副本集不是为了提高读性能存在的，在进行oplog的时候，读操作是被阻塞的；

提高读取性能应该使用分片和索引，它的存在更多是作为数据冗余，备份。

### [简述MongoDB的几种分片策略及其相互之间的差异？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E7%9A%84%E5%87%A0%E7%A7%8D%E5%88%86%E7%89%87%E7%AD%96%E7%95%A5%E5%8F%8A%E5%85%B6%E7%9B%B8%E4%BA%92%E4%B9%8B%E9%97%B4%E7%9A%84%E5%B7%AE%E5%BC%82%EF%BC%9F "简述MongoDB的几种分片策略及其相互之间的差异？") 简述MongoDB的几种分片策略及其相互之间的差异？

MongoDB的数据划分是基于集合级别为标准，通过shard key来划分集合数据。主要分片策略有如下三种：

- 范围划分：通过shard key值将数据集划分到不同的范围就称为基于范围划分。对于数值型的shard key：可以虚构一条从负无穷到正无穷的直线（理解为x轴），每个shard key 值都落在这条直线的某个点上，然后MongoDB把这条线划分为许多更小的没有重复的范围成为块（chunks），一个chunk就是某些最小值到最大值的范围。
- 散列划分：MongoDB计算每个字段的hash值，然后用这些hash值建立chunks。基于散列值的数据分布有助于更均匀的数据分布，尤其是在shard key单调变化的数据集中。
- 自定义标签划分：MongoDB支持通过自定义标签标记分片的方式直接平衡数据分布策略，可以创建标签并且将它们与shard key值的范围进行关联，然后分配这些标签到各个分片上，最终平衡器转移带有标签标记的数据到对应的分片上，确保集群总是按标签描述的那样进行数据分布。标签是控制平衡器行为及集群中块分布的主要方法。

`差异`：

- 基于范围划分对于范围查询比较高效。假设在shard key上进行范围查询，查询路由很容易能够知道哪些块与这个范围重叠，然后把相关查询按照这个路线发送到仅仅包含这些chunks的分片。
- 基于范围划分很容易导致数据不均匀分布，这样会削弱分片集群的功能。
- 基于散列划分是以牺牲高效范围查询为代价，它能够均匀的分布数据，散列值能够保证数据随机分布到各个分片上。

### [简述MongoDB分片集群采取什么方式确保数据分布的平衡？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%88%86%E7%89%87%E9%9B%86%E7%BE%A4%E9%87%87%E5%8F%96%E4%BB%80%E4%B9%88%E6%96%B9%E5%BC%8F%E7%A1%AE%E4%BF%9D%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83%E7%9A%84%E5%B9%B3%E8%A1%A1%EF%BC%9F "简述MongoDB分片集群采取什么方式确保数据分布的平衡？") 简述MongoDB分片集群采取什么方式确保数据分布的平衡？

新加入的数据及服务器都会导致集群数据分布不平衡，MongoDB采用两种方式确保数据分布的平衡：

- 拆分

拆分是一个后台进程，防止块变得太大。当一个块增长到指定块大小的时候，拆分进程就会块一分为二，整个拆分过程是高效的。不会涉及到数据的迁移等操作。

- 平衡

平衡器是一个后台进程，管理块的迁移。平衡器能够运行在集群任何的mongd实例上。当集群中数据分布不均匀时，平衡器就会将某个分片中比较多的块迁移到拥有块较少的分片中，直到数据分片平衡为止。

分片采用后台操作的方式管理着源分片和目标分片之间块的迁移。在迁移的过程中，源分片中的块会将所有文档发送到目标分片中，然后目标分片会获取并应用这些变化。最后，更新配置服务器上关于块位置元数据。


### [简述MongoDB备份及恢复方式？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E5%A4%87%E4%BB%BD%E5%8F%8A%E6%81%A2%E5%A4%8D%E6%96%B9%E5%BC%8F%EF%BC%9F "简述MongoDB备份及恢复方式？") 简述MongoDB备份及恢复方式？

mongodb备份恢复方式通常有以下三种：

- `文件快照方式`：此方式相对简单，需要系统文件支持快照和mongod必须启用journal。可以在任何时刻创建快照。恢复时，确保没有运行mongod，执行快照恢复操作命令，然后启动mongod进程，mongod将重放journal日志。
- `复制数据文件方式`：直接拷贝数据目录下的一切文件，但是在拷贝过程中必须阻止数据文件发生更改。因此需要对数据库加锁，以防止数据写入。恢复时，确保mongod没有运行，清空数据目录，将备份的数据拷贝到数据目录下，然后启动mongod。
- `使用mongodump和mongorestore方式`：在Mongodb中我们使用mongodump命令来备份MongoDB数据。该命令可以导出所有数据到指定目录中。恢复时，使用mongorestore命令来恢复MongoDB数据。该命令可以从指定目录恢复相应数据。

### [简述MongoDB的聚合操作？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E7%9A%84%E8%81%9A%E5%90%88%E6%93%8D%E4%BD%9C%EF%BC%9F "简述MongoDB的聚合操作？") 简述MongoDB的聚合操作？

聚合操作能够处理数据记录并返回计算结果。聚合操作能将多个文档中的值组合起来，对成组数据执行各种操作，返回单一的结果。它相当于 SQL 中的 count(\*) 组合 group by。对于 MongoDB 中的聚合操作，应该使用aggregate()方法。

### [简述MongoDB中的GridFS机制？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E4%B8%AD%E7%9A%84GridFS%E6%9C%BA%E5%88%B6%EF%BC%9F "简述MongoDB中的GridFS机制？") 简述MongoDB中的GridFS机制？

GridFS是一种将大型文件存储在MongoDB中的文件规范。使用GridFS可以将大文件分隔成多个小文档存放，这样我们能够有效的保存大文档，而且解决了BSON对象有限制的问题。

### [简述MongoDB针对查询优化的措施？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E9%92%88%E5%AF%B9%E6%9F%A5%E8%AF%A2%E4%BC%98%E5%8C%96%E7%9A%84%E6%8E%AA%E6%96%BD%EF%BC%9F "简述MongoDB针对查询优化的措施？") 简述MongoDB针对查询优化的措施？

MongoDB查询优化大致可能从如下步骤着手：

- 第一步：找出慢速查询

如下方式开启内置的查询分析器,记录读写操作效率：

db.setProfilingLevel(n,{m}),n的取值可选0,1,2；

  - 0：默认值，表示不记录；
  - 1：表示记录慢速操作，如果值为1，m必须赋值单位为ms，用于定义慢速查询时间的阈值；
  - 2：表示记录所有的读写操作。

    查询监控结果：监控结果保存在一个特殊的集合system.profile里。
- 第二步：分析慢速查询

找出慢速查询的原因，通常可能的原因有：应用程序设计不合理、不正确的数据模型、硬件配置问题、缺少索引等

- 第三部：根据不同的分析结果进行优化，如建立索引。


### [简述MongoDB的更新操作是否会立刻fsync到磁盘？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MongoDB%E7%9A%84%E6%9B%B4%E6%96%B0%E6%93%8D%E4%BD%9C%E6%98%AF%E5%90%A6%E4%BC%9A%E7%AB%8B%E5%88%BBfsync%E5%88%B0%E7%A3%81%E7%9B%98%EF%BC%9F "简述MongoDB的更新操作是否会立刻fsync到磁盘？") 简述MongoDB的更新操作是否会立刻fsync到磁盘？

不会，磁盘写操作默认是延时执行的，写操作可能在两三秒（默认在60秒内）后到达磁盘，可通过syncPeriodSecs参数进行配置。

### [简述MySQL索引及其作用？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E7%B4%A2%E5%BC%95%E5%8F%8A%E5%85%B6%E4%BD%9C%E7%94%A8%EF%BC%9F "简述MySQL索引及其作用？") 简述MySQL索引及其作用？

是数据库管理系统中一个排序的数据结构，根据不同的存储引擎索引分为Hash索引、B+树索引等。常见的InnoDB存储引擎的默认索引实现为：B+树索引。

索引可以协助快速查询、更新数据库表中数据。

### [简述MySQL中什么是事务？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E4%B8%AD%E4%BB%80%E4%B9%88%E6%98%AF%E4%BA%8B%E5%8A%A1%EF%BC%9F "简述MySQL中什么是事务？") 简述MySQL中什么是事务？

事务是一系列的操作，需要要符合ACID特性，即：事务中的操作要么全部成功，要么全部失败。

### [简述MySQL事务之间的隔离？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E4%BA%8B%E5%8A%A1%E4%B9%8B%E9%97%B4%E7%9A%84%E9%9A%94%E7%A6%BB%EF%BC%9F "简述MySQL事务之间的隔离？") 简述MySQL事务之间的隔离？

MySQL事务支持如下四种隔离：

- `未提交读`(Read Uncommitted)：允许脏读，其他事务只要修改了数据，即使未提交，本事务也能看到修改后的数据值。也就是可能读取到其他会话中未提交事务修改的数据。
- `提交读`(Read Committed)：只能读取到已经提交的数据。Oracle等多数数据库默认都是该级别 (不重复读)。
- `可重复读`(Repeated Read)：可重复读。无论其他事务是否修改并提交了数据，在这个事务中看到的数据值始终不受其他事务影响。
- `串行读`(Serializable)：完全串行化的读，每次读都需要获得表级共享锁，读写相互都会阻塞。

### [简述MySQL锁及其作用？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E9%94%81%E5%8F%8A%E5%85%B6%E4%BD%9C%E7%94%A8%EF%BC%9F "简述MySQL锁及其作用？") 简述MySQL锁及其作用？

锁机制是为了避免，在数据库有并发事务的时候，可能会产生数据的不一致而诞生的的一个机制。锁从类别上分为：

- `共享锁`：又叫做读锁，当用户要进行数据的读取时，对数据加上共享锁，共享锁可以同时加上多个。
- `排他锁`：又叫做写锁，当用户要进行数据的写入时，对数据加上排他锁，排他锁只可以加一个，他和其他的排他锁,共享锁都相斥。

### [简述MySQL表中为什么建议添加主键？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E8%A1%A8%E4%B8%AD%E4%B8%BA%E4%BB%80%E4%B9%88%E5%BB%BA%E8%AE%AE%E6%B7%BB%E5%8A%A0%E4%B8%BB%E9%94%AE%EF%BC%9F "简述MySQL表中为什么建议添加主键？") 简述MySQL表中为什么建议添加主键？

主键是数据库确保数据行在整张表唯一性的保障，即使数据库中表没有主键，也建议添加一个自增长的ID列作为主键，设定了主键之后，在后续的删改查的时候可能更加快速以及确保操作数据范围安全。

### [简述MySQL所支持的存储引擎？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E6%89%80%E6%94%AF%E6%8C%81%E7%9A%84%E5%AD%98%E5%82%A8%E5%BC%95%E6%93%8E%EF%BC%9F "简述MySQL所支持的存储引擎？") 简述MySQL所支持的存储引擎？

MySQL支持多种存储引擎，常见的有InnoDB、MyISAM、Memory、Archive等。通常使用InnoDB引擎都是最合适的，InnoDB也是MySQL的默认存储引擎。

### [简述MySQL InnoDB引擎和MyISAM引擎的差异？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL-InnoDB%E5%BC%95%E6%93%8E%E5%92%8CMyISAM%E5%BC%95%E6%93%8E%E7%9A%84%E5%B7%AE%E5%BC%82%EF%BC%9F "简述MySQL InnoDB引擎和MyISAM引擎的差异？") 简述MySQL InnoDB引擎和MyISAM引擎的差异？

- InnoDB支持事物，而MyISAM不支持事物。
- InnoDB支持行级锁，而MyISAM支持表级锁。
- InnoDB支持MVCC, 而MyISAM不支持。
- InnoDB支持外键，而MyISAM不支持。
- InnoDB不支持全文索引，而MyISAM支持。

### [简述MySQL主从复制过程？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E4%B8%BB%E4%BB%8E%E5%A4%8D%E5%88%B6%E8%BF%87%E7%A8%8B%EF%BC%9F "简述MySQL主从复制过程？") 简述MySQL主从复制过程？

- 1、Slave上面的IO线程连接上Master，并请求从指定日志文件的指定位置（或者从最开始的日志）之后的日志内容；
- 2、Master接收到来自Slave的IO线程的请求后，通过负责复制的IO线程根据请求信息读取指定日志指定位置之后的日志信息，返回给Slave端的IO线程。返回信息中除了日志所包含的信息之外，还包括本次返回的信息在Master端binary log文件的名称以及在Binary log中的位置；
- 3、Slave的IO线程收到信息后，将接收到的日志内容依次写入到Slave端的RelayLog文件（mysql-relay-lin.xxxxx）的最末端，并将读取到的Master端的bin-log的文件名和位置记录到master-info文件中，以便在下一次读取的时候能够明确知道从什么位置开始读取日志；
- 4、Slave的SQL线程检测到Relay Log中新增加了内容后，会马上解析该Log文件中的内容成为在Master端真实执行时候的那些可执行的查询或操作语句，并在自身执行那些查询或操作语句，这样，实际上就是在master端和Slave端执行了同样的查询或操作语句，所以两端的数据是完全一样的。

### [简述MySQL常见的读写分离方案？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E5%B8%B8%E8%A7%81%E7%9A%84%E8%AF%BB%E5%86%99%E5%88%86%E7%A6%BB%E6%96%B9%E6%A1%88%EF%BC%9F "简述MySQL常见的读写分离方案？") 简述MySQL常见的读写分离方案？

MySQL+Amoeba读写分离方案：Amoeba(变形虫)项目，这个工具致力于MySQL的分布式数据库前端代理层，它主要在应用层访问MySQL的时候充当SQL路由功能。具有负载均衡、高可用性、SQL 过滤、读写分离、可路由相关的到目标数据库、可并发请求多台数据库合并结果。通过Amoeba你能够完成多数据源的高可用、负载均衡、数据切片、读写分离的功能。

MySQL+MMM读写分离方案：MMM即Multi-Master Replication Manager for MySQL，mysql多主复制管理器是关于mysql主主复制配置的监控、故障转移和管理的一套可伸缩的脚本套件(在任何时候只有一个节点可以被写入)。MMM也能对从服务器进行读负载均衡，通过MMM方案能实现服务器的故障转移，从而实现mysql的高可用。MMM不仅能提供浮动IP的功能，如果当前的主服务器挂掉后，会将你后端的从服务器自动转向新的主服务器进行同步复制，不用手工更改同步配置。

### [简述MySQL常见的高可用方案？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E5%B8%B8%E8%A7%81%E7%9A%84%E9%AB%98%E5%8F%AF%E7%94%A8%E6%96%B9%E6%A1%88%EF%BC%9F "简述MySQL常见的高可用方案？") 简述MySQL常见的高可用方案？

- `MySQL主从复制`：Mysql内建的复制功能是构建大型，高性能应用程序的基础。将Mysql的数据分布在多个节点（slaves）之上，复制过程中一个服务器充当主服务器，而一个或多个其它服务器充当从服务器。主服务器将更新写入二进制日志文件，并维护文件的一个索引以跟踪日志循环。这些日志可以记录发送到从服务器的更新。
- `MySQL双主`：参考MySQL主从复制。
- `MySQL双主多从`：参考MySQL主从复制。
- `MySQL复制+Keepalived高可用`：MySQL自身的复制，对外基于Keepalived技术，暴露一个VIP，从而实现高可用。
- `Heartbeat + MySQL 实现MySQL的高可用`：通过Heartbeat的心跳检测和资源接管、集群中服务的监测、失效切换等功能，结合MySQL来实现高可用性。

### [简述MySQL常见的优化措施？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E5%B8%B8%E8%A7%81%E7%9A%84%E4%BC%98%E5%8C%96%E6%8E%AA%E6%96%BD%EF%BC%9F "简述MySQL常见的优化措施？") 简述MySQL常见的优化措施？

MySQL可通过如下方式优化：

- 1、开启查询缓存，优化查询。
- 2、使用explain判断select查询，从而分析查询语句或是表结构的性能瓶颈，然后有针对性的进行优化。
- 3、为搜索字段建索引
- 4、对于有限定范围取值的字段，推荐使用 ENUM 而不是 VARCHAR。
- 5、垂直分表。
- 6、选择正确的存储引擎。

### [简述MySQL常见备份方式和工具？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0MySQL%E5%B8%B8%E8%A7%81%E5%A4%87%E4%BB%BD%E6%96%B9%E5%BC%8F%E5%92%8C%E5%B7%A5%E5%85%B7%EF%BC%9F "简述MySQL常见备份方式和工具？") 简述MySQL常见备份方式和工具？

- MySQL自带

mysqldump：mysqldump支持基于innodb的热备份，使用mysqldump完全备份+二进制日志可以实现基于时间点的恢复，通常适合备份数据比较小的场景 。

- 系统层面

tar备份：可以使用tar之类的系统命令对整个数据库目录进行打包备份。

lvm快照备份：可基于文件系统的LVM制作快照，进行对整个数据库目录所在的逻辑卷备份。

- 第三方备份工具

可使用其他第三方工具进行备份，如xtrabackup工具，该工具支持innodb的物理热备份，支持完全备份、增量备份，而且速度非常快，支持innodb存储引起的数据在不同数据库之间迁移，支持复制模式下的从机备份恢复备份恢复。


## [监控](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%9B%91%E6%8E%A7 "监控") 监控

### [简述常见的监控软件？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0%E5%B8%B8%E8%A7%81%E7%9A%84%E7%9B%91%E6%8E%A7%E8%BD%AF%E4%BB%B6%EF%BC%9F "简述常见的监控软件？") 简述常见的监控软件？

常见的监控软件有：

- `Cacti`：是一套基于PHP、MySQL、SNMP及RRDTool开发的网络流量监测图形分析工具。
- `Zabbix`：Zabbix是一个企业级的高度集成开源监控软件，提供分布式监控解决方案。可以用来监控设备、服务等可用性和性能。
- `Open-falcon`：open-falcon是一款用golang和python写的监控系统，由小米启动这个项目。
- `Prometheus`：Prometheus是由SoundCloud开发的开源监控报警系统和时序列数据库(TSDB)。Prometheus使用Go语言开发，是Google BorgMon监控系统的开源版本。

### [简述Prometheus及其主要特性？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Prometheus%E5%8F%8A%E5%85%B6%E4%B8%BB%E8%A6%81%E7%89%B9%E6%80%A7%EF%BC%9F "简述Prometheus及其主要特性？") 简述Prometheus及其主要特性？

Prometheus是一个已加入CNCF的开源监控报警系统和时序列数据库项目，通过不同的组件完成数据的采集，数据的存储和告警。

Prometheus主要特性：

- 多维数据模型
  - 时间序列数据通过 metric 名和键值对来区分。
  - 所有的 metrics 都可以设置任意的多维标签。
  - 数据模型更随意，不需要刻意设置为以点分隔的字符串。
  - 可以对数据模型进行聚合，切割和切片操作。
  - 支持双精度浮点类型，标签可以设为全 unicode。
- 灵活的查询语句（PromQL），可以利用多维数据完成复杂的查询
- Prometheus server 是一个单独的二进制文件，不依赖（任何分布式）存储，支持 local 和 remote 不同模型
- 采用 http 协议，使用 pull 模式，拉取数据，或者通过中间网关推送方式采集数据
- 监控目标，可以采用服务发现或静态配置的方式
- 支持多种统计数据模型，图形化友好
- 高效：一个 Prometheus server 可以处理数百万的 metrics
- 适用于以机器为中心的监控以及高度动态面向服务架构的监控

### [简述Prometheus主要组件及其功能？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Prometheus%E4%B8%BB%E8%A6%81%E7%BB%84%E4%BB%B6%E5%8F%8A%E5%85%B6%E5%8A%9F%E8%83%BD%EF%BC%9F "简述Prometheus主要组件及其功能？") 简述Prometheus主要组件及其功能？

Prometheus 的主要模块包含：prometheus server, exporters, push gateway, PromQL, Alertmanager, WebUI 等。

- 1、`prometheus server`：定期从静态配置的 targets 或者服务发现（主要是DNS、consul、k8s、mesos等）的 targets 拉取数据，用于收集和存储时间序列数据。
- 2、`exporters`：负责向prometheus server做数据汇报，暴露一个http服务的接口给Prometheus server定时抓取。而不同的数据汇报由不同的exporters实现，比如监控主机有node-exporters，mysql有MySQL server exporter。
- 3、`push gateway`：主要使用场景为，当Prometheus 采用 pull 模式，可能由于不在一个子网或者防火墙原因，导致 Prometheus 无法直接拉取各个 target 数据。此时需要push gateway接入，以便于在监控业务数据的时候，将不同数据汇总, 由 Prometheus 统一收集。实现机制类似于zabbix-proxy功能。
- 4、`Alertmanager`：从 Prometheus server 端接收到 alerts 后，会进行去除重复数据，分组，并路由到对收的接受方式，发出报警，即主要实现prometheus的告警功能。AlertManager的整体工作流程如下图所示:
- 5、`webui`：Prometheus内置一个简单的Web控制台，可以查询指标，查看配置信息或者Service Discovery等，实践中通常结合Grafana，Prometheus仅作为Grafana的数据源。

### [简述Prometheus的机制？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Prometheus%E7%9A%84%E6%9C%BA%E5%88%B6%EF%BC%9F "简述Prometheus的机制？") 简述Prometheus的机制？

Prometheus简单机制如下：

- Prometheus以其Server为核心，用于收集和存储时间序列数据。Prometheus Server 从监控目标中拉取数据，或通过push gateway间接的把监控目标的监控数据存储到本地HDD/SSD中。
- 用户接口界面通过各种UI使用PromQL查询语言从Server获取数据。
- 一旦Server检测到异常，会推送告警到AlertManager，由告警管理负责去通知相关方。

### [简述Prometheus中什么是时序数据？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Prometheus%E4%B8%AD%E4%BB%80%E4%B9%88%E6%98%AF%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE%EF%BC%9F "简述Prometheus中什么是时序数据？") 简述Prometheus中什么是时序数据？

Prometheus 存储的是时序数据,，时序数据是指按照相同时序(相同的名字和标签)，以时间维度存储连续的数据的集合。时序(time series) 是由名字(Metric)，以及一组 key/value 标签定义的，具有相同的名字以及标签属于相同时序。

### [简述Prometheus时序数据有哪些类型？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Prometheus%E6%97%B6%E5%BA%8F%E6%95%B0%E6%8D%AE%E6%9C%89%E5%93%AA%E4%BA%9B%E7%B1%BB%E5%9E%8B%EF%BC%9F "简述Prometheus时序数据有哪些类型？") 简述Prometheus时序数据有哪些类型？

Prometheus 时序数据分为 Counter, Gauge, Histogram, Summary 四种类型。

- `Counter`：计数器表示收集的数据是按照某个趋势（增加／减少）一直变化的，通常用它记录服务请求总量，错误总数等。
- `Gauge`：计量器表示搜集的数据是一个瞬时的，与时间没有关系，可以任意变高变低，往往可以用来记录内存使用率、磁盘使用率等。
- `Histogram`：直方图 Histogram 主要用于对一段时间范围内的数据进行采样，（通常是请求持续时间或响应大小），并能够对其指定区间以及总数进行统计，通常我们用它计算分位数的直方图。
- `Summary`：汇总Summary 和 直方图Histogram 类似，主要用于表示一段时间内数据采样结果，（通常是请求持续时间或响应大小），它直接存储了 quantile 数据，而不是根据统计区间计算出来的。

### [简述Zabbix及其优势？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Zabbix%E5%8F%8A%E5%85%B6%E4%BC%98%E5%8A%BF%EF%BC%9F "简述Zabbix及其优势？") 简述Zabbix及其优势？

Zabbix是一个企业级的高度集成开源监控软件，提供分布式监控解决方案。可以用来监控设备、服务等可用性和性能。其主要优势有：

- 自由开放源代码产品，可以对其进行任意修改和二次开发，采用GPL协议；
- 安装和配置简单；
- 搭建环境简单，基于开源软件构建平台；
- 完全支持Linux、Unix、Windows、AIX、BSD等平台，采用C语言编码，系统占用小，数据采集性能和速度非常快；
- 数据采集持久存储到数据库，便于对监控数据的二次分析；
- 非常丰富的扩展能力，轻松实现自定义监控项和实现数据采集。

### [简述Zabbix体系架构？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Zabbix%E4%BD%93%E7%B3%BB%E6%9E%B6%E6%9E%84%EF%BC%9F "简述Zabbix体系架构？") 简述Zabbix体系架构？

Zabbix体系相对清晰，其主要组件有：

- `Zabbix Server`：负责接收agent发送的报告信息的核心组件，所有配置、统计数据及操作数据均由其组织进行。
- `Database Storage`：专用于存储所有配置信息，以及有zabbix收集的数据。
- `Web interface`（frontend）：zabbix的GUI接口，通常与server运行在同一台机器上。
- `Proxy`：可选组件，常用于分布式监控环境中，代理Server收集部分被监控数据并统一发往Server端。
- `Agent`：部署在被监控主机上，负责收集本地数据并发往Server端或者Proxy端。

### [简述Zabbix所支持的监控方式？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Zabbix%E6%89%80%E6%94%AF%E6%8C%81%E7%9A%84%E7%9B%91%E6%8E%A7%E6%96%B9%E5%BC%8F%EF%BC%9F "简述Zabbix所支持的监控方式？") 简述Zabbix所支持的监控方式？

目前由zabbix提供包括但不限于以下事项类型的支持：

- `Zabbix agent checks`：这些客户端来进行数据采集，又分为Zabbix agent（被动模式：客户端等着服务器端来要数据），Zabbix agent (active)（主动模式：客户端主动发送数据到服务器端）
- `SNMP agent checks`：SNMP方式，如果要监控打印机网络设备等支持SNMP设备的话，但是又不能安装agent的设备。
- `SNMP traps` ：
- `IPMI checks`：IPMI即智能平台管理接口，现在是业界通过的标准。用户可以利用IPMI监视服务器的物理特性，如温度、电压、电扇工作状态、电源供应以及机箱入侵等。

#### [简述Zabbix分布式及其适应场景？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0Zabbix%E5%88%86%E5%B8%83%E5%BC%8F%E5%8F%8A%E5%85%B6%E9%80%82%E5%BA%94%E5%9C%BA%E6%99%AF%EF%BC%9F "简述Zabbix分布式及其适应场景？") 简述Zabbix分布式及其适应场景？

zabbix proxy 可以代替 zabbix server 收集性能和可用性数据,然后把数据汇报给 zabbix server，并且在一定程度上分担了zabbix server 的压力。

此外，当所有agents和proxy报告给一个Zabbix server并且所有数据都集中收集时，使用proxy是实现集中式和分布式监控的最简单方法。

zabbix proxy 使用场景:

- 监控远程区域设备
- 监控本地网络不稳定区域
- 当 zabbix 监控上千设备时,使用它来减轻 server 的压力
- 简化分布式监控的维护

## [网络管理](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86 "网络管理") 网络管理

### [简述什么是CDN？](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/\#%E7%AE%80%E8%BF%B0%E4%BB%80%E4%B9%88%E6%98%AFCDN%EF%BC%9F "简述什么是CDN？") 简述什么是CDN？

CDN即内容分发网络，是在现有网络中增加一层新的网络架构，从而实现将源站内容发布和传送到最靠近用户的边缘地区，使用户可以就近访问想要的内容，提高用户访问的响应速度。

> - 作者：木二
> - 链接： [https://www.yuque.com/docs/share/d3dd1e8e-6828-4da7-9e30-6a4f45c6fa8e](https://www.yuque.com/docs/share/d3dd1e8e-6828-4da7-9e30-6a4f45c6fa8e)

 ---本文结束感谢您的阅读。微信扫描二维码，关注我的公众号---


![](https://www.yp14.cn/img/ypxz-2.png)

**本文作者：**

Peng Yang



**本文链接：** [https://www.yp14.cn/2020/08/11/IT运维面试问题总结-数据库-监控-网络管理/](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/)

**版权声明：**
本作品采用
[知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议](https://creativecommons.org/licenses/by-nc-sa/4.0/)
进行许可。转载请注明出处！



[![知识共享许可协议](https://cdm.yp14.cn/img/by-nc-sa-img-88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

赏

谢谢您的打赏

![](https://www.yp14.cn/img/zanshang.jpeg)微信

- Interview

扫一扫，分享到微信

![微信分享二维码](https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=https://wwww.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/)

- 全文搜索
- 友情链接

缺失模块。

1、请确保node版本大于6.2

2、在博客根目录（注意不是yilia根目录）执行以下命令：

npm i hexo-generator-json-content --save

3、在根目录\_config.yml里添加配置：

```
  jsonContent:
    meta: false
    pages: false
    posts:
      title: true
      date: true
      path: true
      text: false
      raw: false
      content: false
      slug: false
      updated: false
      comments: false
      link: false
      permalink: false
      excerpt: false
      categories: false
      tags: true
```

- [ETCD存储满了如何处理?](https://www.yp14.cn/2022/10/30/ETCD%E5%AD%98%E5%82%A8%E6%BB%A1%E4%BA%86%E5%A6%82%E4%BD%95%E5%A4%84%E7%90%86/)
2022-10-30

#kubernetes

- [业务日志告警如何做?](https://www.yp14.cn/2022/10/23/%E4%B8%9A%E5%8A%A1%E6%97%A5%E5%BF%97%E5%91%8A%E8%AD%A6%E5%A6%82%E4%BD%95%E5%81%9A/)
2022-10-23

#kubernetes

- [详解Nginx proxy\_pass 使用](https://www.yp14.cn/2022/03/27/%E8%AF%A6%E8%A7%A3Nginx-proxy-pass-%E4%BD%BF%E7%94%A8/)
2022-03-27

#OPS

- [Docker与Containerd使用区别](https://www.yp14.cn/2022/03/20/Docker%E4%B8%8EContainerd%E4%BD%BF%E7%94%A8%E5%8C%BA%E5%88%AB/)
2022-03-20

#Docker

- [docker exec 失败问题排查之旅](https://www.yp14.cn/2022/01/09/docker-exec-%E5%A4%B1%E8%B4%A5%E9%97%AE%E9%A2%98%E6%8E%92%E6%9F%A5%E4%B9%8B%E6%97%85/)
2022-01-08

#docker

- [阿里云ACK多个Service绑定单个SLB实践](https://www.yp14.cn/2021/12/08/%E9%98%BF%E9%87%8C%E4%BA%91ACK%E5%A4%9A%E4%B8%AAService%E7%BB%91%E5%AE%9A%E5%8D%95%E4%B8%AASLB%E5%AE%9E%E8%B7%B5/)
2021-12-07

#kubernetes

- [K8S部署分布式调度任务Airflow](https://www.yp14.cn/2021/11/28/K8S%E9%83%A8%E7%BD%B2%E5%88%86%E5%B8%83%E5%BC%8F%E8%B0%83%E5%BA%A6%E4%BB%BB%E5%8A%A1Airflow/)
2021-11-28

#kubernetes

- [Ingress Nginx传递用户真实IP问题](https://www.yp14.cn/2021/10/30/Ingress-Nginx%E4%BC%A0%E9%80%92%E7%94%A8%E6%88%B7%E7%9C%9F%E5%AE%9EIP%E9%97%AE%E9%A2%98/)
2021-10-30

#kubernetes

- [Kubectl 高亮输出](https://www.yp14.cn/2021/10/13/Kubectl-%E9%AB%98%E4%BA%AE%E8%BE%93%E5%87%BA/)
2021-10-13

#kubernetes

- [聊聊TPS、QPS、CPS概念和区别.md](https://www.yp14.cn/2021/07/29/%E8%81%8A%E8%81%8ATPS%E3%80%81QPS%E3%80%81CPS%E6%A6%82%E5%BF%B5%E5%92%8C%E5%8C%BA%E5%88%AB-md/)
2021-07-28

#OPS

- [K8S Configmap和Secret热更新之Reloader](https://www.yp14.cn/2021/07/24/K8S-Configmap%E5%92%8CSecret%E7%83%AD%E6%9B%B4%E6%96%B0%E4%B9%8BReloader/)
2021-07-24

#kubernetes

- [Mysqldump导入备份数据到阿里云RDS会报错吗](https://www.yp14.cn/2021/06/20/Mysqldump%E5%AF%BC%E5%85%A5%E5%A4%87%E4%BB%BD%E6%95%B0%E6%8D%AE%E5%88%B0%E9%98%BF%E9%87%8C%E4%BA%91RDS%E4%BC%9A%E6%8A%A5%E9%94%99%E5%90%97/)
2021-06-20

#OPS

- [K8S集群内Pod如何与本地网络打通实现debug](https://www.yp14.cn/2021/06/06/K8S%E9%9B%86%E7%BE%A4%E5%86%85Pod%E5%A6%82%E4%BD%95%E4%B8%8E%E6%9C%AC%E5%9C%B0%E7%BD%91%E7%BB%9C%E6%89%93%E9%80%9A%E5%AE%9E%E7%8E%B0debug/)
2021-06-05

#kubernetes

- [Harbor多实例高可用共享存储搭建](https://www.yp14.cn/2021/05/16/Harbor%E5%A4%9A%E5%AE%9E%E4%BE%8B%E9%AB%98%E5%8F%AF%E7%94%A8%E5%85%B1%E4%BA%AB%E5%AD%98%E5%82%A8%E6%90%AD%E5%BB%BA/)
2021-05-16

#OPS

- [聊聊Harbor架构](https://www.yp14.cn/2021/05/09/%E8%81%8A%E8%81%8AHarbor%E6%9E%B6%E6%9E%84/)
2021-05-09

#OPS

- [K8S Cluster Autoscaler 集群自动伸缩](https://www.yp14.cn/2021/04/21/K8S-Cluster-Autoscaler-%E9%9B%86%E7%BE%A4%E8%87%AA%E5%8A%A8%E4%BC%B8%E7%BC%A9/)
2021-04-21

#kubernetes

- [十道Kubernetes面试题](https://www.yp14.cn/2021/04/11/%E5%8D%81%E9%81%93Kubernetes%E9%9D%A2%E8%AF%95%E9%A2%98/)
2021-04-11

#kubernetes

- [Redis如何删除数量过万以上Key而不影响业务](https://www.yp14.cn/2021/03/21/Redis%E5%A6%82%E4%BD%95%E5%88%A0%E9%99%A4%E6%95%B0%E9%87%8F%E8%BF%87%E4%B8%87%E4%BB%A5%E4%B8%8AKey%E8%80%8C%E4%B8%8D%E5%BD%B1%E5%93%8D%E4%B8%9A%E5%8A%A1/)
2021-03-20

#OPS

- [Nginx 配置可视化管理](https://www.yp14.cn/2021/03/14/Nginx-%E9%85%8D%E7%BD%AE%E5%8F%AF%E8%A7%86%E5%8C%96%E7%AE%A1%E7%90%86/)
2021-03-14

#OPS

- [Kubernetes(k8s)那些套路之日志收集](https://www.yp14.cn/2021/03/04/Kubernetes-k8s-%E9%82%A3%E4%BA%9B%E5%A5%97%E8%B7%AF%E4%B9%8B%E6%97%A5%E5%BF%97%E6%94%B6%E9%9B%86/)
2021-03-04

#kubernetes

- [Grafana展示精美的nginx访问日志图表](https://www.yp14.cn/2021/02/28/Grafana%E5%B1%95%E7%A4%BA%E7%B2%BE%E7%BE%8E%E7%9A%84nginx%E8%AE%BF%E9%97%AE%E6%97%A5%E5%BF%97%E5%9B%BE%E8%A1%A8/)
2021-02-28

#kubernetes

- [Kubernetes Pod应用性能分析工具 Kubectl Flame](https://www.yp14.cn/2021/02/21/Kubernetes-pod%E5%BA%94%E7%94%A8%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E5%B7%A5%E5%85%B7-kubectl-flame/)
2021-02-21

#kubernetes

- [Redis数据同步和数据迁移如何做？](https://www.yp14.cn/2021/01/24/Redis%E6%95%B0%E6%8D%AE%E5%90%8C%E6%AD%A5%E5%92%8C%E6%95%B0%E6%8D%AE%E8%BF%81%E7%A7%BB%E5%A6%82%E4%BD%95%E5%81%9A/)
2021-01-24

#OPS

- [Zabbix简单监控es实践](https://www.yp14.cn/2021/01/17/Zabbix%E7%AE%80%E5%8D%95%E7%9B%91%E6%8E%A7es%E5%AE%9E%E8%B7%B5/)
2021-01-17

#OPS

- [容器部署ELK7.10-适用于生产](https://www.yp14.cn/2021/01/07/%E5%AE%B9%E5%99%A8%E9%83%A8%E7%BD%B2ELK7-10-%E9%80%82%E7%94%A8%E4%BA%8E%E7%94%9F%E4%BA%A7/)
2021-01-07

#OPS

- [kubernetes pod为什么需要pause容器？](https://www.yp14.cn/2020/12/27/kubernetes-pod%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81pause%E5%AE%B9%E5%99%A8/)
2020-12-27

#kubernetes

- [Kubernetes 实用技巧](https://www.yp14.cn/2020/12/20/Kubernetes-%E5%AE%9E%E7%94%A8%E6%8A%80%E5%B7%A7/)
2020-12-20

#kubernetes

- [Kubernetes 1.20版本开始不推荐使用Docker，你知道吗？](https://www.yp14.cn/2020/12/02/Kubernetes-1-20%E7%89%88%E6%9C%AC%E5%BC%80%E5%A7%8B%E4%B8%8D%E6%8E%A8%E8%8D%90%E4%BD%BF%E7%94%A8Docker%EF%BC%8C%E4%BD%A0%E7%9F%A5%E9%81%93%E5%90%97%EF%BC%9F/)
2020-12-02

#kubernetes

- [Kind：一个容器创建K8S开发集群](https://www.yp14.cn/2020/11/01/Kind%EF%BC%9A%E4%B8%80%E4%B8%AA%E5%AE%B9%E5%99%A8%E5%88%9B%E5%BB%BAK8S%E5%BC%80%E5%8F%91%E9%9B%86%E7%BE%A4/)
2020-11-01

#kubernetes

- [K8S 问题排查：cgroup 内存泄露问题](https://www.yp14.cn/2020/10/22/K8S-%E9%97%AE%E9%A2%98%E6%8E%92%E6%9F%A5%EF%BC%9Acgroup-%E5%86%85%E5%AD%98%E6%B3%84%E9%9C%B2%E9%97%AE%E9%A2%98/)
2020-10-22

#kubernetes

- [使用 Nextcloud 3分钟搭建个人网盘](https://www.yp14.cn/2020/10/22/%E4%BD%BF%E7%94%A8-Nextcloud-3%E5%88%86%E9%92%9F%E6%90%AD%E5%BB%BA%E4%B8%AA%E4%BA%BA%E7%BD%91%E7%9B%98/)
2020-10-22

#OPS

- [kubelet 原理解析六： 垃圾回收](https://www.yp14.cn/2020/10/16/kubelet-%E5%8E%9F%E7%90%86%E8%A7%A3%E6%9E%90%E5%85%AD%EF%BC%9A-%E5%9E%83%E5%9C%BE%E5%9B%9E%E6%94%B6/)
2020-10-16

#kubernetes

- [Grafana Tanka：比K8S YAML声明更简洁](https://www.yp14.cn/2020/10/14/Grafana-Tanka%EF%BC%9A%E6%AF%94K8S-YAML%E5%A3%B0%E6%98%8E%E6%9B%B4%E7%AE%80%E6%B4%81/)
2020-10-14

#kubernetes

- [推荐两个Docker配置检查与启动异常修复方法脚本](https://www.yp14.cn/2020/10/11/%E6%8E%A8%E8%8D%90%E4%B8%A4%E4%B8%AADocker%E9%85%8D%E7%BD%AE%E6%A3%80%E6%9F%A5%E4%B8%8E%E5%90%AF%E5%8A%A8%E5%BC%82%E5%B8%B8%E4%BF%AE%E5%A4%8D%E6%96%B9%E6%B3%95%E8%84%9A%E6%9C%AC/)
2020-10-11

#Docker

- [监控域名HTTPS证书过期时间](https://www.yp14.cn/2020/09/29/%E7%9B%91%E6%8E%A7%E5%9F%9F%E5%90%8DHTTPS%E8%AF%81%E4%B9%A6%E8%BF%87%E6%9C%9F%E6%97%B6%E9%97%B4/)
2020-09-29

#OPS

- [5个维度对 Kubernetes 集群优化](https://www.yp14.cn/2020/09/24/5%E4%B8%AA%E7%BB%B4%E5%BA%A6%E5%AF%B9-Kubernetes-%E9%9B%86%E7%BE%A4%E4%BC%98%E5%8C%96/)
2020-09-24

#kubernetes

- [解密 Docker 挂载文件，宿主机修改后容器里文件没有修改](https://www.yp14.cn/2020/09/23/%E8%A7%A3%E5%AF%86-Docker-%E6%8C%82%E8%BD%BD%E6%96%87%E4%BB%B6%EF%BC%8C%E5%AE%BF%E4%B8%BB%E6%9C%BA%E4%BF%AE%E6%94%B9%E5%90%8E%E5%AE%B9%E5%99%A8%E9%87%8C%E6%96%87%E4%BB%B6%E6%B2%A1%E6%9C%89%E4%BF%AE%E6%94%B9/)
2020-09-23

#Docker

- [分享Linux内存占用几个案例](https://www.yp14.cn/2020/09/20/%E5%88%86%E4%BA%ABLinux%E5%86%85%E5%AD%98%E5%8D%A0%E7%94%A8%E5%87%A0%E4%B8%AA%E6%A1%88%E4%BE%8B/)
2020-09-20

#OPS

- [如何实现rsync多并发同步？](https://www.yp14.cn/2020/09/16/%E5%A6%82%E4%BD%95%E5%AE%9E%E7%8E%B0rsync%E5%A4%9A%E5%B9%B6%E5%8F%91%E5%90%8C%E6%AD%A5/)
2020-09-16

#OPS

- [Kubernetes YAML 生成器](https://www.yp14.cn/2020/09/11/Kubernetes-YAML-%E7%94%9F%E6%88%90%E5%99%A8/)
2020-09-11

#kubernetes

- [Kubernetes 故障解决心得（一）](https://www.yp14.cn/2020/09/10/kubernetes-%E6%95%85%E9%9A%9C%E8%A7%A3%E5%86%B3%E5%BF%83%E5%BE%97%E4%B8%80/)
2020-09-10

#kubernetes

- [Kubernetes 临时存储需要限制吗？](https://www.yp14.cn/2020/09/08/Kubernetes-%E4%B8%B4%E6%97%B6%E5%AD%98%E5%82%A8%E9%9C%80%E8%A6%81%E9%99%90%E5%88%B6%E5%90%97/)
2020-09-08

#kubernetes

- [Linux Used内存到底哪里去了？](https://www.yp14.cn/2020/09/01/Linux-Used%E5%86%85%E5%AD%98%E5%88%B0%E5%BA%95%E5%93%AA%E9%87%8C%E5%8E%BB%E4%BA%86/)
2020-09-01

#OPS

- [Kubernetes v1.19.0 正式发布！](https://www.yp14.cn/2020/08/27/Kubernetes-v1-19-0-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83/)
2020-08-26

#kubernetes

- [Kubectl 备忘录](https://www.yp14.cn/2020/08/21/Kubectl-%E5%A4%87%E5%BF%98%E5%BD%95/)
2020-08-21

#kubernetes

- [Harbor v2.0 镜像回收那些事](https://www.yp14.cn/2020/08/14/Harbor-v2-0-%E9%95%9C%E5%83%8F%E5%9B%9E%E6%94%B6%E9%82%A3%E4%BA%9B%E4%BA%8B/)
2020-08-14

#OPS

- [IT运维面试问题总结-简述Etcd、Kubernetes、Lvs、HAProxy等](https://www.yp14.cn/2020/08/12/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E7%AE%80%E8%BF%B0etcd-kubernetes-lvs-haproxy/)
2020-08-11

#Interview

- [IT运维面试问题总结-数据库、监控、网络管理](https://www.yp14.cn/2020/08/11/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E6%95%B0%E6%8D%AE%E5%BA%93-%E7%9B%91%E6%8E%A7-%E7%BD%91%E7%BB%9C%E7%AE%A1%E7%90%86/)
2020-08-10

#Interview

- [IT运维面试问题总结-运维工具、开源应用(Ceph、Docker、Apache、Nginx等)](https://www.yp14.cn/2020/08/10/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E8%BF%90%E7%BB%B4%E5%B7%A5%E5%85%B7-%E5%BC%80%E6%BA%90%E5%BA%94%E7%94%A8ceph-docker-apache-nginx%E7%AD%89/)
2020-08-10

#Interview

- [IT运维面试问题总结-基础服务、磁盘管理、虚拟平台和系统管理](https://www.yp14.cn/2020/08/07/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-%E5%9F%BA%E7%A1%80%E6%9C%8D%E5%8A%A1-%E7%A3%81%E7%9B%98%E7%AE%A1%E7%90%86-%E8%99%9A%E6%8B%9F%E5%B9%B3%E5%8F%B0-%E7%B3%BB%E7%BB%9F%E7%AE%A1%E7%90%86/)
2020-08-07

#Interview

- [IT运维面试问题总结-Linux基础](https://www.yp14.cn/2020/08/07/IT%E8%BF%90%E7%BB%B4%E9%9D%A2%E8%AF%95%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93-Linux%E5%9F%BA%E7%A1%80/)
2020-08-07

#Interview

- [聊聊 resolv.conf 中 search 和 ndots 配置](https://www.yp14.cn/2020/07/24/%E8%81%8A%E8%81%8A-resolv-conf-%E4%B8%AD-search-%E5%92%8C-ndots-%E9%85%8D%E7%BD%AE/)
2020-07-24

#kubernetes

- [磁盘空间分析神器 \- ncdu](https://www.yp14.cn/2020/07/21/%E7%A3%81%E7%9B%98%E7%A9%BA%E9%97%B4%E5%88%86%E6%9E%90%E7%A5%9E%E5%99%A8-ncdu/)
2020-07-21

#OPS

- [Kubernetes v1.15.3 升级到 v1.18.5 心得](https://www.yp14.cn/2020/07/19/Kubernetes-v1-15-3-%E5%8D%87%E7%BA%A7%E5%88%B0-v1-18-5-%E5%BF%83%E5%BE%97/)
2020-07-19

#kubernetes

- [Kubernetes 升级填坑指南（一）](https://www.yp14.cn/2020/07/19/Kubernetes-%E5%8D%87%E7%BA%A7%E5%A1%AB%E5%9D%91%E6%8C%87%E5%8D%97%E4%B8%80/)
2020-07-19

#kubernetes

- [根据 PID 获取 K8S Pod名称 - 反之 POD名称 获取 PID](https://www.yp14.cn/2020/07/16/%E6%A0%B9%E6%8D%AE-PID-%E8%8E%B7%E5%8F%96-K8S-Pod%E5%90%8D%E7%A7%B0-%E5%8F%8D%E4%B9%8B-POD%E5%90%8D%E7%A7%B0-%E8%8E%B7%E5%8F%96-PID/)
2020-07-16

#kubernetes

- [Docker 网络配置那些事](https://www.yp14.cn/2020/07/13/Docker-%E7%BD%91%E7%BB%9C%E9%85%8D%E7%BD%AE%E9%82%A3%E4%BA%9B%E4%BA%8B/)
2020-07-13

#Docker

- [前端本地缓存概况之浏览器缓存策略](https://www.yp14.cn/2020/07/10/%E5%89%8D%E7%AB%AF%E6%9C%AC%E5%9C%B0%E7%BC%93%E5%AD%98%E6%A6%82%E5%86%B5%E4%B9%8B%E6%B5%8F%E8%A7%88%E5%99%A8%E7%BC%93%E5%AD%98%E7%AD%96%E7%95%A5/)
2020-07-10

#OPS

- [Java分析神器 - Arthas](https://www.yp14.cn/2020/06/30/Java%E5%88%86%E6%9E%90%E7%A5%9E%E5%99%A8-Arthas/)
2020-06-30

#OPS

- [K8S故障排查指南-Orphaned pod found, but volume paths are still present on disk](https://www.yp14.cn/2020/06/28/K8S%E6%95%85%E9%9A%9C%E6%8E%92%E9%99%A4%E6%8C%87%E5%8D%97-Orphaned-pod-found-but-volume-paths-are-still-present-on-disk/)
2020-06-28

#kubernetes

- [K8S备份、恢复、迁移神器 Velero](https://www.yp14.cn/2020/06/23/K8S%E5%A4%87%E4%BB%BD-%E6%81%A2%E5%A4%8D-%E8%BF%81%E7%A7%BB%E7%A5%9E%E5%99%A8-Velero/)
2020-06-23

#kubernetes

- [Kubernetes故障排查指南-分析容器退出状态码](https://www.yp14.cn/2020/06/22/Kubernetes%E6%95%85%E9%9A%9C%E6%8E%92%E9%99%A4%E6%8C%87%E5%8D%97-%E5%88%86%E6%9E%90%E5%AE%B9%E5%99%A8%E9%80%80%E5%87%BA%E7%8A%B6%E6%80%81%E7%A0%81/)
2020-06-22

#kubernetes

- [Kubeconfig文件自动合并-实现K8S多集群切换](https://www.yp14.cn/2020/06/21/Kubeconfig%E6%96%87%E4%BB%B6%E8%87%AA%E5%8A%A8%E5%90%88%E5%B9%B6-%E5%AE%9E%E7%8E%B0K8S%E5%A4%9A%E9%9B%86%E7%BE%A4%E5%88%87%E6%8D%A2/)
2020-06-21

#kubernetes

- [比官方K8S Dashboard好用的桌面客户端：Lens](https://www.yp14.cn/2020/06/17/%E6%AF%94%E5%AE%98%E6%96%B9K8S-Dashboard%E5%A5%BD%E7%94%A8%E7%9A%84%E6%A1%8C%E9%9D%A2%E5%AE%A2%E6%88%B7%E7%AB%AFLens/)
2020-06-17

#kubernetes

- [生产环境中helm v2升级v3版本遇到的疑难杂症](https://www.yp14.cn/2020/06/16/%E7%94%9F%E4%BA%A7%E7%8E%AF%E5%A2%83%E4%B8%ADhelm-v2%E5%8D%87%E7%BA%A7v3%E7%89%88%E6%9C%AC%E9%81%87%E5%88%B0%E7%9A%84%E7%96%91%E9%9A%BE%E6%9D%82%E7%97%87/)
2020-06-16

#kubernetes

- [Linux整个系统权限玩坏了怎么办？](https://www.yp14.cn/2020/06/14/Linux%E6%95%B4%E4%B8%AA%E7%B3%BB%E7%BB%9F%E6%9D%83%E9%99%90%E7%8E%A9%E5%9D%8F%E4%BA%86%E6%80%8E%E4%B9%88%E5%8A%9E/)
2020-06-13

#OPS

- [Kubernetes 中利用 LXCFS 控制容器资源可见性](https://www.yp14.cn/2020/06/10/Kubernetes-%E4%B8%AD%E5%88%A9%E7%94%A8-LXCFS-%E6%8E%A7%E5%88%B6%E5%AE%B9%E5%99%A8%E8%B5%84%E6%BA%90%E5%8F%AF%E8%A7%81%E6%80%A7/)
2020-06-10

#kubernetes

- [Kubernetes Node节点主机名 修改](https://www.yp14.cn/2020/06/08/Kubernetes-Node%E8%8A%82%E7%82%B9%E4%B8%BB%E6%9C%BA%E5%90%8D-%E4%BF%AE%E6%94%B9/)
2020-06-08

#kubernetes

- [Kubernetes 是否值得学习吗？](https://www.yp14.cn/2020/06/06/Kubernetes-%E6%98%AF%E5%90%A6%E5%80%BC%E5%BE%97%E5%AD%A6%E4%B9%A0%E5%90%97/)
2020-06-06

#kubernetes

- [Kubernetes 私有集群 LoadBalancer 解决方案](https://www.yp14.cn/2020/06/04/Kubernetes-%E7%A7%81%E6%9C%89%E9%9B%86%E7%BE%A4-LoadBalancer-%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/)
2020-06-04

#kubernetes

- [Nginx 之 realip模块 使用详解](https://www.yp14.cn/2020/06/03/Nginx-%E4%B9%8B-realip%E6%A8%A1%E5%9D%97-%E4%BD%BF%E7%94%A8%E8%AF%A6%E8%A7%A3/)
2020-06-03

#OPS

- [K8S Pod 内抓包快速定位网络问题](https://www.yp14.cn/2020/06/01/K8S-Pod-%E5%86%85%E6%8A%93%E5%8C%85%E5%BF%AB%E9%80%9F%E5%AE%9A%E4%BD%8D%E7%BD%91%E7%BB%9C%E9%97%AE%E9%A2%98/)
2020-06-01

#kubernetes

- [Etcd 问题 调优 监控](https://www.yp14.cn/2020/05/26/Etcd-%E9%97%AE%E9%A2%98-%E8%B0%83%E4%BC%98-%E7%9B%91%E6%8E%A7/)
2020-05-26

#kubernetes

- [Kubernetes v1.18.2 二进制一键添加 Node节点](https://www.yp14.cn/2020/05/25/Kubernetes-v1-18-2-%E4%BA%8C%E8%BF%9B%E5%88%B6%E4%B8%80%E9%94%AE%E6%B7%BB%E5%8A%A0-Node%E8%8A%82%E7%82%B9/)
2020-05-25

#kubernetes

- [不同云厂商云主机实现内网互通解决方案](https://www.yp14.cn/2020/05/21/%E4%B8%8D%E5%90%8C%E4%BA%91%E5%8E%82%E5%95%86%E4%BA%91%E4%B8%BB%E6%9C%BA%E5%AE%9E%E7%8E%B0%E5%86%85%E7%BD%91%E4%BA%92%E9%80%9A%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/)
2020-05-21

#OPS

- [Kubernetes v1.18.2 二进制高可用部署](https://www.yp14.cn/2020/05/19/Kubernetes-v1-18-2-%E4%BA%8C%E8%BF%9B%E5%88%B6%E9%AB%98%E5%8F%AF%E7%94%A8%E9%83%A8%E7%BD%B2/)
2020-05-19

#kubernetes

- [K8S Dashboard 2.0 部署并使用 Ingress-Nginx 提供访问入口](https://www.yp14.cn/2020/05/16/K8S-Dashboard-2-0-%E9%83%A8%E7%BD%B2%E5%B9%B6%E4%BD%BF%E7%94%A8-Ingress-Nginx-%E6%8F%90%E4%BE%9B%E8%AE%BF%E9%97%AE%E5%85%A5%E5%8F%A3/)
2020-05-15

#kubernetes

- [Kubernetes 无法查看 pods 日志问题](https://www.yp14.cn/2020/05/13/Kubernetes-%E6%97%A0%E6%B3%95%E6%9F%A5%E7%9C%8B-pods-%E6%97%A5%E5%BF%97%E9%97%AE%E9%A2%98/)
2020-05-13

#kubernetes

- [Gitlab CI/CD 部署应用到 K8S 演示](https://www.yp14.cn/2020/05/11/Gitlab-CI-CD-%E9%83%A8%E7%BD%B2%E5%BA%94%E7%94%A8%E5%88%B0-K8S-%E6%BC%94%E7%A4%BA/)
2020-05-11

#kubernetes

- [Calico 介绍、原理与使用](https://www.yp14.cn/2020/05/07/Calico-%E4%BB%8B%E7%BB%8D%E3%80%81%E5%8E%9F%E7%90%86%E4%B8%8E%E4%BD%BF%E7%94%A8/)
2020-05-07

#kubernetes

- [K8s Pod Command 与容器镜像 Cmd 启动优先级详解](https://www.yp14.cn/2020/05/03/K8s-Pod-Command-%E4%B8%8E%E5%AE%B9%E5%99%A8%E9%95%9C%E5%83%8F-Cmd-%E5%90%AF%E5%8A%A8%E4%BC%98%E5%85%88%E7%BA%A7%E8%AF%A6%E8%A7%A3/)
2020-05-03

#kubernetes

- [Grafana 采集阿里云SLB监控信息](https://www.yp14.cn/2020/04/30/Grafana-%E9%87%87%E9%9B%86%E9%98%BF%E9%87%8C%E4%BA%91SLB%E7%9B%91%E6%8E%A7%E4%BF%A1%E6%81%AF/)
2020-04-30

#kubernetes

- [Ingress Nginx 日志配置](https://www.yp14.cn/2020/04/27/Ingress-Nginx-%E6%97%A5%E5%BF%97%E9%85%8D%E7%BD%AE/)
2020-04-27

#kubernetes

- [Ingress Nginx 故障排除](https://www.yp14.cn/2020/04/21/Ingress-Nginx-%E6%95%85%E9%9A%9C%E6%8E%92%E9%99%A4/)
2020-04-21

#kubernetes

- [Nginx Ingress Controller 工作原理](https://www.yp14.cn/2020/04/20/Nginx-Ingress-Controller-%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86/)
2020-04-20

#kubernetes

- [图文了解 Kubernetes](https://www.yp14.cn/2020/04/19/%E5%9B%BE%E6%96%87%E4%BA%86%E8%A7%A3-Kubernetes/)
2020-04-19

#kubernetes

- [高可用 Prometheus：问题集锦](https://www.yp14.cn/2020/04/17/%E9%AB%98%E5%8F%AF%E7%94%A8-Prometheus%E9%97%AE%E9%A2%98%E9%9B%86%E9%94%A6/)
2020-04-17

#kubernetes

- [Kubernetes Pod 钩子](https://www.yp14.cn/2020/04/15/Kubernetes-Pod-%E9%92%A9%E5%AD%90/)
2020-04-15

#kubernetes

- [CKA 真题](https://www.yp14.cn/2020/04/12/CKA%E7%9C%9F%E9%A2%98/)
2020-04-12

#kubernetes

- [构建高大上的MySQL监控平台](https://www.yp14.cn/2020/04/07/%E6%9E%84%E5%BB%BA%E9%AB%98%E5%A4%A7%E4%B8%8A%E7%9A%84MySQL%E7%9B%91%E6%8E%A7%E5%B9%B3%E5%8F%B0/)
2020-04-07

#OPS

- [白话 kubernetes 网络组件 Flannel](https://www.yp14.cn/2020/04/02/%E7%99%BD%E8%AF%9D-kubernetes-%E7%BD%91%E7%BB%9C%E7%BB%84%E4%BB%B6-Flannel/)
2020-04-02

#kubernetes

- [基于 Kubernetes 的 7 大 DevOps 关键实践](https://www.yp14.cn/2020/04/01/%E5%9F%BA%E4%BA%8E-Kubernetes-%E7%9A%84-7-%E5%A4%A7-DevOps-%E5%85%B3%E9%94%AE%E5%AE%9E%E8%B7%B5/)
2020-04-01

#OPS

- [从 Docker 到 Kubernetes 日志管理机制详解](https://www.yp14.cn/2020/03/31/%E4%BB%8E-Docker-%E5%88%B0-Kubernetes-%E6%97%A5%E5%BF%97%E7%AE%A1%E7%90%86%E6%9C%BA%E5%88%B6%E8%AF%A6%E8%A7%A3/)
2020-03-31

#kubernetes

- [一文入门 ETCD](https://www.yp14.cn/2020/03/30/%E4%B8%80%E6%96%87%E5%85%A5%E9%97%A8-ETCD/)
2020-03-30

#kubernetes

- [Istio 1.5部署，回归单体](https://www.yp14.cn/2020/03/27/Istio-1-5%E9%83%A8%E7%BD%B2-%E5%9B%9E%E5%BD%92%E5%8D%95%E4%BD%93/)
2020-03-27

#Istio

- [Kubernetes v1.18 正式发布之功能预览](https://www.yp14.cn/2020/03/26/Kubernetes-v1-18-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83%E4%B9%8B%E5%8A%9F%E8%83%BD%E9%A2%84%E8%A7%88/)
2020-03-26

#kubernetes

- [Kubernetes节点频繁NotReady-处理和防范](https://www.yp14.cn/2020/03/24/Kubernetes%E8%8A%82%E7%82%B9%E9%A2%91%E7%B9%81NotReady-%E5%A4%84%E7%90%86%E5%92%8C%E9%98%B2%E8%8C%83/)
2020-03-24

#kubernetes

- [Kubernetes 面试题（一）](https://www.yp14.cn/2020/03/23/Kubernetes-%E9%9D%A2%E8%AF%95%E9%A2%98%E4%B8%80/)
2020-03-23

#kubernetes

- [终于有人把 Docker 讲清楚了，万字详解！](https://www.yp14.cn/2020/03/19/%E7%BB%88%E4%BA%8E%E6%9C%89%E4%BA%BA%E6%8A%8A-Docker-%E8%AE%B2%E6%B8%85%E6%A5%9A%E4%BA%86%EF%BC%8C%E4%B8%87%E5%AD%97%E8%AF%A6%E8%A7%A3/)
2020-03-19

#Docker

- [Descheduler 实现 K8S Pod 二次调度](https://www.yp14.cn/2020/03/18/Descheduler-%E5%AE%9E%E7%8E%B0-K8S-Pod-%E4%BA%8C%E6%AC%A1%E8%B0%83%E5%BA%A6/)
2020-03-18

#kubernetes

- [K8S 可视化监控 Weave Scope 部署](https://www.yp14.cn/2020/03/17/K8S-%E5%8F%AF%E8%A7%86%E5%8C%96%E7%9B%91%E6%8E%A7-Weave-Scope-%E9%83%A8%E7%BD%B2/)
2020-03-17

#kubernetes

- [Kubernetes 管理虚拟机之 KubeVirt](https://www.yp14.cn/2020/03/16/Kubernetes-%E7%AE%A1%E7%90%86%E8%99%9A%E6%8B%9F%E6%9C%BA%E4%B9%8B-KubeVirt/)
2020-03-16

#kubernetes

- [解决k8s无法通过svc访问其他节点pod的问题](https://www.yp14.cn/2020/03/12/%E8%A7%A3%E5%86%B3k8s%E6%97%A0%E6%B3%95%E9%80%9A%E8%BF%87svc%E8%AE%BF%E9%97%AE%E5%85%B6%E4%BB%96%E8%8A%82%E7%82%B9pod%E7%9A%84%E9%97%AE%E9%A2%98/)
2020-03-12

#kubernetes

- [缓存穿透、缓存击穿、缓存雪崩](https://www.yp14.cn/2020/03/11/%E7%BC%93%E5%AD%98%E7%A9%BF%E9%80%8F%E3%80%81%E7%BC%93%E5%AD%98%E5%87%BB%E7%A9%BF%E3%80%81%E7%BC%93%E5%AD%98%E9%9B%AA%E5%B4%A9/)
2020-03-11

#OPS

- [Kubernetes 集群安全机制详解](https://www.yp14.cn/2020/03/10/Kubernetes-%E9%9B%86%E7%BE%A4%E5%AE%89%E5%85%A8%E6%9C%BA%E5%88%B6%E8%AF%A6%E8%A7%A3/)
2020-03-10

#kubernetes

- [动画版 Kubernetes 相关名词解释](https://www.yp14.cn/2020/03/09/%E5%8A%A8%E7%94%BB%E7%89%88-Kubernetes-%E7%9B%B8%E5%85%B3%E5%90%8D%E8%AF%8D%E8%A7%A3%E9%87%8A/)
2020-03-09

#kubernetes

- [大公司为什么都有API网关？聊聊API网关的作用](https://www.yp14.cn/2020/03/06/%E5%A4%A7%E5%85%AC%E5%8F%B8%E4%B8%BA%E4%BB%80%E4%B9%88%E9%83%BD%E6%9C%89API%E7%BD%91%E5%85%B3-%E8%81%8A%E8%81%8AAPI%E7%BD%91%E5%85%B3%E7%9A%84%E4%BD%9C%E7%94%A8/)
2020-03-06

#OPS

- [Kubernetes 亲和与反亲和实用示例](https://www.yp14.cn/2020/03/05/Kubernetes-%E4%BA%B2%E5%92%8C%E4%B8%8E%E5%8F%8D%E4%BA%B2%E5%92%8C%E5%AE%9E%E7%94%A8%E7%A4%BA%E4%BE%8B/)
2020-03-05

#kubernetes

- [还在担心写的一手烂SQL，送你4款工具](https://www.yp14.cn/2020/03/04/%E8%BF%98%E5%9C%A8%E6%8B%85%E5%BF%83%E5%86%99%E7%9A%84%E4%B8%80%E6%89%8B%E7%83%82SQL%EF%BC%8C%E9%80%81%E4%BD%A04%E6%AC%BE%E5%B7%A5%E5%85%B7/)
2020-03-04

#OPS

- [小白都会设置的K8S RBAC](https://www.yp14.cn/2020/03/03/%E5%B0%8F%E7%99%BD%E9%83%BD%E4%BC%9A%E8%AE%BE%E7%BD%AE%E7%9A%84K8S-RBAC/)
2020-03-03

#kubernetes

- [容器监控实践—Prometheus存储机制](https://www.yp14.cn/2020/03/02/%E5%AE%B9%E5%99%A8%E7%9B%91%E6%8E%A7%E5%AE%9E%E8%B7%B5%E2%80%94Prometheus%E5%AD%98%E5%82%A8%E6%9C%BA%E5%88%B6/)
2020-03-02

#kubernetes

- [容器监控实践—Prometheus基本架构](https://www.yp14.cn/2020/02/28/%E5%AE%B9%E5%99%A8%E7%9B%91%E6%8E%A7%E5%AE%9E%E8%B7%B5%E2%80%94Prometheus%E5%9F%BA%E6%9C%AC%E6%9E%B6%E6%9E%84/)
2020-02-28

#kubernetes

- [K8S Pod 保护之 PodDisruptionBudget](https://www.yp14.cn/2020/02/27/K8S-Pod-%E4%BF%9D%E6%8A%A4%E4%B9%8B-PodDisruptionBudget/)
2020-02-27

#kubernetes

- [Kuboard Proxy](https://www.yp14.cn/2020/02/26/Kuboard-Proxy/)
2020-02-26

#kubernetes

- [ElasticSearch（提高篇）](https://www.yp14.cn/2020/02/25/ElasticSearch-%E6%8F%90%E9%AB%98%E7%AF%87/)
2020-02-25

#OPS

- [无需特权在Kubernetes中构建镜像之 Kaniko](https://www.yp14.cn/2020/02/24/%E6%97%A0%E9%9C%80%E7%89%B9%E6%9D%83%E5%9C%A8Kubernetes%E4%B8%AD%E6%9E%84%E5%BB%BA%E9%95%9C%E5%83%8F%E4%B9%8B-Kaniko/)
2020-02-24

#kubernetes

- [不用找了，大厂在用的分库分表方案，都在这里？](https://www.yp14.cn/2020/02/21/%E4%B8%8D%E7%94%A8%E6%89%BE%E4%BA%86%EF%BC%8C%E5%A4%A7%E5%8E%82%E5%9C%A8%E7%94%A8%E7%9A%84%E5%88%86%E5%BA%93%E5%88%86%E8%A1%A8%E6%96%B9%E6%A1%88%EF%BC%8C%E9%83%BD%E5%9C%A8%E8%BF%99%E9%87%8C/)
2020-02-21

#OPS

- [kubelet 先导篇](https://www.yp14.cn/2020/02/21/kubelet-%E5%85%88%E5%AF%BC%E7%AF%87/)
2020-02-21

#kubernetes

- [常用网络协议神图](https://www.yp14.cn/2020/02/20/%E5%B8%B8%E7%94%A8%E7%BD%91%E7%BB%9C%E5%8D%8F%E8%AE%AE%E7%A5%9E%E5%9B%BE/)
2020-02-20

#OPS

- [Linux 服务器上有挖矿病毒 kdevtmpfsi 如何处理？](https://www.yp14.cn/2020/02/20/Linux-%E6%9C%8D%E5%8A%A1%E5%99%A8%E4%B8%8A%E6%9C%89%E6%8C%96%E7%9F%BF%E7%97%85%E6%AF%92-kdevtmpfsi-%E5%A6%82%E4%BD%95%E5%A4%84%E7%90%86/)
2020-02-20

#OPS

- [NGINX 上的限流（译）](https://www.yp14.cn/2020/02/18/NGINX-%E4%B8%8A%E7%9A%84%E9%99%90%E6%B5%81/)
2020-02-18

#OPS

- [PostgreSQL 常用SQL语句](https://www.yp14.cn/2020/02/17/PostgreSQL-%E5%B8%B8%E7%94%A8SQL%E8%AF%AD%E5%8F%A5/)
2020-02-17

#DataBase

- [Kubectl 常用命令大全](https://www.yp14.cn/2020/02/12/Kubectl-%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4%E5%A4%A7%E5%85%A8/)
2020-02-12

#kubernetes

- [PromQL 常用命令](https://www.yp14.cn/2020/02/10/PromQL-%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4/)
2020-02-10

#kubernetes

- [kubernetes service 介绍](https://www.yp14.cn/2020/02/09/kubernetes-service-%E4%BB%8B%E7%BB%8D/)
2020-02-08

#kubernetes

- [Elasticsearch 可视化管理工具](https://www.yp14.cn/2020/02/06/Elasticsearch-%E5%8F%AF%E8%A7%86%E5%8C%96%E7%AE%A1%E7%90%86%E5%B7%A5%E5%85%B7/)
2020-02-05

#OPS

- [Kubernetes 中部署 Zabbix](https://www.yp14.cn/2020/02/05/Kubernetes-%E4%B8%AD%E9%83%A8%E7%BD%B2-Zabbix/)
2020-02-05

#OPS

- [互联网中台技术简介](https://www.yp14.cn/2020/02/04/%E4%BA%92%E8%81%94%E7%BD%91%E4%B8%AD%E5%8F%B0%E6%8A%80%E6%9C%AF%E7%AE%80%E4%BB%8B/)
2020-02-03

#OPS

- [Kubernetes Authenticate 安装向导](https://www.yp14.cn/2020/02/03/Kubernetes-Authenticate-%E5%AE%89%E8%A3%85%E5%90%91%E5%AF%BC/)
2020-02-03

#kubernetes

- [Prometheus BlackBox简单监控](https://www.yp14.cn/2020/01/16/Prometheus-BlackBox%E7%AE%80%E5%8D%95%E7%9B%91%E6%8E%A7/)
2020-01-16

#kubernetes

- [Ingress Nginx 常用规则使用](https://www.yp14.cn/2020/01/14/Ingress-Nginx-%E5%B8%B8%E7%94%A8%E8%A7%84%E5%88%99%E4%BD%BF%E7%94%A8/)
2020-01-14

#kubernetes

- [Kubectl创建pod过程中发生些什么事情?](https://www.yp14.cn/2020/01/14/Kubectl%E5%88%9B%E5%BB%BApod%E8%BF%87%E7%A8%8B%E4%B8%AD%E5%8F%91%E7%94%9F%E4%BA%9B%E4%BB%80%E4%B9%88%E4%BA%8B%E6%83%85/)
2020-01-14

#kubernetes

- [Kubernetes Pod 生命周期](https://www.yp14.cn/2020/01/13/Kubernetes-Pod-%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F/)
2020-01-13

#kubernetes

- [Kubernetes之容器数据写满磁盘解决方法](https://www.yp14.cn/2020/01/12/Kubernetes%E4%B9%8B%E5%AE%B9%E5%99%A8%E6%95%B0%E6%8D%AE%E5%86%99%E6%BB%A1%E7%A3%81%E7%9B%98%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95/)
2020-01-11

#kubernetes

- [OpenVpn 服务端与客户端部署](https://www.yp14.cn/2020/01/11/OpeVpn-%E6%9C%8D%E5%8A%A1%E7%AB%AF%E4%B8%8E%E5%AE%A2%E6%88%B7%E7%AB%AF%E9%83%A8%E7%BD%B2/)
2020-01-11

#OPS

- [谈谈K8S Pod Eviction 机制](https://www.yp14.cn/2020/01/10/%E8%B0%88%E8%B0%88K8S-Pod-Eviction-%E6%9C%BA%E5%88%B6/)
2020-01-10

#kubernetes

- [Kubernetes Node资源预留](https://www.yp14.cn/2020/01/09/Kubernetes-Node%E8%B5%84%E6%BA%90%E9%A2%84%E7%95%99/)
2020-01-09

#kubernetes

- [PrometheusAlert 多渠道告警通知神器](https://www.yp14.cn/2020/01/08/PrometheusAlert-%E5%A4%9A%E6%B8%A0%E9%81%93%E5%91%8A%E8%AD%A6%E9%80%9A%E7%9F%A5%E7%A5%9E%E5%99%A8/)
2020-01-08

#kubernetes

- [Mysql存储微信Emoji表情问题](https://www.yp14.cn/2020/01/06/Mysql%E5%AD%98%E5%82%A8%E5%BE%AE%E4%BF%A1Emoji%E8%A1%A8%E6%83%85%E9%97%AE%E9%A2%98/)
2020-01-06

#OPS

- [HTTP缓存机制详解](https://www.yp14.cn/2020/01/05/HTTP%E7%BC%93%E5%AD%98%E6%9C%BA%E5%88%B6%E8%AF%A6%E8%A7%A3/)
2020-01-05

#OPS

- [白话 Kubernetes 基础概念](https://www.yp14.cn/2020/01/04/%E7%99%BD%E8%AF%9D-Kubernetes-%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5/)
2020-01-04

#kubernetes

- [创建本地 LocalHost SSL 证书](https://www.yp14.cn/2020/01/02/%E5%88%9B%E5%BB%BA%E6%9C%AC%E5%9C%B0-LocalHost-SSL-%E8%AF%81%E4%B9%A6/)
2020-01-02

#OPS

- [Kubernetes Pod 故障归类与排查方法](https://www.yp14.cn/2020/01/01/Kubernetes-Pod-%E6%95%85%E9%9A%9C%E5%BD%92%E7%B1%BB%E4%B8%8E%E6%8E%92%E6%9F%A5%E6%96%B9%E6%B3%95/)
2020-01-01

#kubernetes

- [K8s Deployment YAML 名词解释](https://www.yp14.cn/2019/12/30/K8s-Deployment-YAML-%E5%90%8D%E8%AF%8D%E8%A7%A3%E9%87%8A/)
2019-12-30

#kubernetes

- [Redis 内存分析神器](https://www.yp14.cn/2019/12/28/Redis-%E5%86%85%E5%AD%98%E5%88%86%E6%9E%90%E7%A5%9E%E5%99%A8/)
2019-12-28

#OPS

- [Linux系统日志报Possible SYN flooding处理方法](https://www.yp14.cn/2019/12/27/Linux%E7%B3%BB%E7%BB%9F%E6%97%A5%E5%BF%97%E6%8A%A5Possible-SYN-flooding%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95/)
2019-12-27

#OPS

- [适用于CI容器漏洞扫描神器](https://www.yp14.cn/2019/12/25/%E9%80%82%E7%94%A8%E4%BA%8ECI%E5%AE%B9%E5%99%A8%E6%BC%8F%E6%B4%9E%E6%89%AB%E6%8F%8F%E7%A5%9E%E5%99%A8/)
2019-12-25

#Docker

- [实时Web日志分析器](https://www.yp14.cn/2019/12/25/%E5%AE%9E%E6%97%B6Web%E6%97%A5%E5%BF%97%E5%88%86%E6%9E%90%E5%99%A8/)
2019-12-25

#OPS

- [在 Kubernetes 中配置 Container Capabilities](https://www.yp14.cn/2019/12/25/%E5%9C%A8-Kubernetes-%E4%B8%AD%E9%85%8D%E7%BD%AE-Container-Capabilities/)
2019-12-25

#kubernetes

- [Kubernetes 之 Cronjob](https://www.yp14.cn/2019/12/24/Kubernetes-%E4%B9%8B-Cronjob/)
2019-12-24

#kubernetes

- [IBM 开源图形终端Kui框架](https://www.yp14.cn/2019/12/23/IBM-%E5%BC%80%E6%BA%90%E5%9B%BE%E5%BD%A2%E7%BB%88%E7%AB%AFKui%E6%A1%86%E6%9E%B6/)
2019-12-23

#kubernetes

- [谈谈kubernetes Runtime](https://www.yp14.cn/2019/12/20/%E8%B0%88%E8%B0%88kubernetes-Runtime/)
2019-12-20

#kubernetes

- [Kubeadm 证书说明](https://www.yp14.cn/2019/12/20/Kubeadm-%E8%AF%81%E4%B9%A6%E8%AF%B4%E6%98%8E/)
2019-12-20

#kubernetes

- [Calico 问题排障](https://www.yp14.cn/2019/12/20/Calico-%E9%97%AE%E9%A2%98%E6%8E%92%E9%9A%9C/)
2019-12-20

#kubernetes

- [Kubelet 状态更新机制](https://www.yp14.cn/2019/12/20/Kubelet-%E7%8A%B6%E6%80%81%E6%9B%B4%E6%96%B0%E6%9C%BA%E5%88%B6/)
2019-12-20

#kubernetes

- [一份快速实用的 tcpdump 命令参考手册](https://www.yp14.cn/2019/12/19/%E4%B8%80%E4%BB%BD%E5%BF%AB%E9%80%9F%E5%AE%9E%E7%94%A8%E7%9A%84-tcpdump-%E5%91%BD%E4%BB%A4%E5%8F%82%E8%80%83%E6%89%8B%E5%86%8C/)
2019-12-19

#OPS

- [nginx配置location与rewrite规则教程](https://www.yp14.cn/2019/12/18/nginx%E9%85%8D%E7%BD%AElocation%E4%B8%8Erewrite%E8%A7%84%E5%88%99%E6%95%99%E7%A8%8B/)
2019-12-18

#OPS

- [阿里开源 k8s 事件通知服务](https://www.yp14.cn/2019/12/17/%E9%98%BF%E9%87%8C%E5%BC%80%E6%BA%90-k8s-%E4%BA%8B%E4%BB%B6%E9%80%9A%E7%9F%A5%E6%9C%8D%E5%8A%A1/)
2019-12-17

#kubernetes

- [容器化配置生成神器](https://www.yp14.cn/2019/12/16/%E5%AE%B9%E5%99%A8%E5%8C%96%E9%85%8D%E7%BD%AE%E7%94%9F%E6%88%90%E7%A5%9E%E5%99%A8/)
2019-12-16

#Docker

- [Docker-compose 部署 ELK](https://www.yp14.cn/2019/12/16/Docker-compose-%E9%83%A8%E7%BD%B2-ELK/)
2019-12-16

#OPS

- [如何修改容器时间而不改变宿主机时间?](https://www.yp14.cn/2019/12/15/%E5%A6%82%E4%BD%95%E4%BF%AE%E6%94%B9%E5%AE%B9%E5%99%A8%E6%97%B6%E9%97%B4%E8%80%8C%E4%B8%8D%E6%94%B9%E5%8F%98%E5%AE%BF%E4%B8%BB%E6%9C%BA%E6%97%B6%E9%97%B4/)
2019-12-15

#Docker

- [K8S 滚动更新如何优雅停止 Pod](https://www.yp14.cn/2019/12/13/K8S-%E6%BB%9A%E5%8A%A8%E6%9B%B4%E6%96%B0%E5%A6%82%E4%BD%95%E4%BC%98%E9%9B%85%E5%81%9C%E6%AD%A2-Pod/)
2019-12-13

#kubernetes

- [Istio自动注入sidecar不成功解决方案](https://www.yp14.cn/2019/12/12/Istio%E8%87%AA%E5%8A%A8%E6%B3%A8%E5%85%A5sidecar%E5%87%BA%E9%94%99%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88/)
2019-12-12

#Istio

- [小米开源Istio-dashboard-Naftis服务](https://www.yp14.cn/2019/12/12/%E5%B0%8F%E7%B1%B3%E5%BC%80%E6%BA%90Istio-dashboard-Naftis%E6%9C%8D%E5%8A%A1/)
2019-12-12

#Istio

- [Nginx 服务指标监测](https://www.yp14.cn/2019/12/11/Nginx-%E6%9C%8D%E5%8A%A1%E6%8C%87%E6%A0%87%E7%9B%91%E6%B5%8B/)
2019-12-11

#OPS

- [Podman 会取代 Docker 吗](https://www.yp14.cn/2019/12/11/Podman-%E4%BC%9A%E5%8F%96%E4%BB%A3-Docker-%E5%90%97/)
2019-12-11

#Podman

- [Kubernetes v1.17.0 正式发布](https://www.yp14.cn/2019/12/10/Kubernetes-v1-17-0-%E6%AD%A3%E5%BC%8F%E5%8F%91%E5%B8%83/)
2019-12-10

#kubernetes

- [3分钟部署生产级k8s集群](https://www.yp14.cn/2019/12/10/3%E5%88%86%E9%92%9F%E9%83%A8%E7%BD%B2%E7%94%9F%E4%BA%A7%E7%BA%A7k8s%E9%9B%86%E7%BE%A4/)
2019-12-10

#kubernetes

- [Kubernetes 终端管理神器](https://www.yp14.cn/2019/12/09/Kubernetes-%E7%BB%88%E7%AB%AF%E7%AE%A1%E7%90%86%E7%A5%9E%E5%99%A8/)
2019-12-09

#kubernetes

- [Kubernetes 必须掌握技能之 RBAC](https://www.yp14.cn/2019/12/09/Kubernetes-%E5%BF%85%E9%A1%BB%E6%8E%8C%E6%8F%A1%E6%8A%80%E8%83%BD%E4%B9%8B-RBAC/)
2019-12-09

#kubernetes

- [Kubernetes deployments 故障排除流程图](https://www.yp14.cn/2019/12/08/Kubernetes-deployments-%E6%95%85%E9%9A%9C%E6%8E%92%E9%99%A4%E6%B5%81%E7%A8%8B%E5%9B%BE/)
2019-12-07

#kubernetes

- [Nginx必须知道哪些事](https://www.yp14.cn/2019/12/07/Nginx%E5%BF%85%E9%A1%BB%E7%9F%A5%E9%81%93%E5%93%AA%E4%BA%9B%E4%BA%8B/)
2019-12-06

#OPS

- [Linux IO分析小神器](https://www.yp14.cn/2019/12/06/Linux-IO%E5%88%86%E6%9E%90%E5%B0%8F%E7%A5%9E%E5%99%A8/)
2019-12-06

#OPS

- [比 docker stats 命令好用工具 ctop](https://www.yp14.cn/2019/12/06/%E6%AF%94-docker-stats-%E5%91%BD%E4%BB%A4%E5%A5%BD%E7%94%A8%E5%B7%A5%E5%85%B7-ctop/)
2019-12-06

#Docker

- [升级到 Kubernetes v1.16 须知API问题总结](https://www.yp14.cn/2019/12/05/%E5%8D%87%E7%BA%A7%E5%88%B0-Kubernetes-v1-16-%E9%A1%BB%E7%9F%A5API%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93/)
2019-12-05

#kubernetes

- [Docker最简单管理方法之Portainer](https://www.yp14.cn/2019/12/05/Docker%E6%9C%80%E7%AE%80%E5%8D%95%E7%AE%A1%E7%90%86%E6%96%B9%E6%B3%95%E4%B9%8BPortainer/)
2019-12-05

#Docker

- [Docker 镜像分析之 dive](https://www.yp14.cn/2019/12/04/Docker-%E9%95%9C%E5%83%8F%E5%88%86%E6%9E%90%E4%B9%8B-dive/)
2019-12-04

#Docker

- [Kubernetes 北极星指标](https://www.yp14.cn/2019/12/03/Kubernetes-%E5%8C%97%E6%9E%81%E6%98%9F%E6%8C%87%E6%A0%87/)
2019-12-03

#kubernetes

- [浅谈 K8S QoS(服务质量等级)](https://www.yp14.cn/2019/12/03/%E6%B5%85%E8%B0%88-K8S-QoS-%E6%9C%8D%E5%8A%A1%E8%B4%A8%E9%87%8F%E7%AD%89%E7%BA%A7/)
2019-12-03

#kubernetes

- [提高阅读代码效率神器 Sourcetrail](https://www.yp14.cn/2019/12/02/%E6%8F%90%E9%AB%98%E9%98%85%E8%AF%BB%E4%BB%A3%E7%A0%81%E6%95%88%E7%8E%87%E7%A5%9E%E5%99%A8-Sourcetrail/)
2019-12-02

#OPS

- [K8S node NotReady 后如何保证服务可用](https://www.yp14.cn/2019/12/02/K8S-node-NotReady-%E5%90%8E%E5%A6%82%E4%BD%95%E4%BF%9D%E8%AF%81%E6%9C%8D%E5%8A%A1%E5%8F%AF%E7%94%A8/)
2019-12-02

#kubernetes

- [Docker 必修课程 Dockerfile](https://www.yp14.cn/2019/12/01/Docker-%E5%BF%85%E4%BF%AE%E8%AF%BE%E7%A8%8B-Dockerfile/)
2019-12-01

#Docker

- [RedHat 开源企业镜像项目 Quay](https://www.yp14.cn/2019/11/30/RedHat-%E5%BC%80%E6%BA%90%E4%BC%81%E4%B8%9A%E9%95%9C%E5%83%8F%E9%A1%B9%E7%9B%AE-Quay/)
2019-11-30

#Docker

- [一次构建多平台docker镜像](https://www.yp14.cn/2019/11/29/%E4%B8%80%E6%AC%A1%E6%9E%84%E5%BB%BA%E5%A4%9A%E5%B9%B3%E5%8F%B0docker%E9%95%9C%E5%83%8F/)
2019-11-29

#Docker

- [K8S 之 Headless 浅谈](https://www.yp14.cn/2019/11/28/K8S-%E4%B9%8B-Headless-%E6%B5%85%E8%B0%88/)
2019-11-28

#kubernetes

- [K8S Dashboard V2.0.0 Beta6 部署](https://www.yp14.cn/2019/11/27/K8S-Dashboard-V2-0-0-Beta6-%E9%83%A8%E7%BD%B2/)
2019-11-27

#kubernetes

- [Nginx 基于客户端IP分析](https://www.yp14.cn/2019/11/26/Nginx-%E5%9F%BA%E4%BA%8E%E5%AE%A2%E6%88%B7%E7%AB%AFIP%E5%88%86%E6%9E%90/)
2019-11-26

#OPS

- [Elasticsearch RESTful API 常用操作](https://www.yp14.cn/2019/11/25/Elasticsearch-RESTful-API-%E5%B8%B8%E7%94%A8%E6%93%8D%E4%BD%9C/)
2019-11-25

#OPS

- [K8S 之 kubeadm 安装](https://www.yp14.cn/2019/11/24/K8S-%E4%B9%8B-kubeadm-%E5%AE%89%E8%A3%85/)
2019-11-24

#kubernetes

- [Nginx 流量统计分析](https://www.yp14.cn/2019/11/23/Nginx-%E6%B5%81%E9%87%8F%E7%BB%9F%E8%AE%A1%E5%88%86%E6%9E%90/)
2019-11-23

#OPS

- [网页主体格式转换神器](https://www.yp14.cn/2019/11/22/%E7%BD%91%E9%A1%B5%E4%B8%BB%E4%BD%93%E6%A0%BC%E5%BC%8F%E8%BD%AC%E6%8D%A2%E7%A5%9E%E5%99%A8/)
2019-11-22

#OPS

- [K8S 金丝雀部署之 Istio](https://www.yp14.cn/2019/11/21/K8S-%E9%87%91%E4%B8%9D%E9%9B%80%E9%83%A8%E7%BD%B2%E4%B9%8B-Istio/)
2019-11-21

#Istio

- [k8s 蓝绿部署之 Service Label](https://www.yp14.cn/2019/11/20/k8s-%E8%93%9D%E7%BB%BF%E9%83%A8%E7%BD%B2%E4%B9%8B-Service-Label/)
2019-11-20

#kubernetes

- [K8s Ingress Nginx 支持 Socket.io](https://www.yp14.cn/2019/11/19/K8s-Ingress-Nginx-%E6%94%AF%E6%8C%81-Socket-io/)
2019-11-19

#kubernetes

- [Helm v3 新的功能](https://www.yp14.cn/2019/11/18/Helm-v3-%E6%96%B0%E7%9A%84%E5%8A%9F%E8%83%BD/)
2019-11-18

#kubernetes

- [Harbor v1.7.0自动镜像回收](https://www.yp14.cn/2019/11/17/Harbor-v1-7-0%E8%87%AA%E5%8A%A8%E9%95%9C%E5%83%8F%E5%9B%9E%E6%94%B6/)
2019-11-16

#OPS

- [Asciinema：Linux操作命令录制神器](https://www.yp14.cn/2019/11/16/Asciinema%EF%BC%9ALinux%E6%93%8D%E4%BD%9C%E5%91%BD%E4%BB%A4%E5%BD%95%E5%88%B6%E7%A5%9E%E5%99%A8/)
2019-11-15

#OPS

- [Kubelet 证书自动续期](https://www.yp14.cn/2019/11/14/Kubelet-%E8%AF%81%E4%B9%A6%E8%87%AA%E5%8A%A8%E7%BB%AD%E6%9C%9F/)
2019-11-14

#kubernetes

- [Prometheus 如何自动发现 Kubernetes Metrics 接口](https://www.yp14.cn/2019/11/13/Prometheus-%E5%A6%82%E4%BD%95%E8%87%AA%E5%8A%A8%E5%8F%91%E7%8E%B0-Kubernetes-Metrics-%E6%8E%A5%E5%8F%A3/)
2019-11-13

#kubernetes

- [AlertManager 钉钉报警](https://www.yp14.cn/2019/11/12/AlertManager-%E9%92%89%E9%92%89%E6%8A%A5%E8%AD%A6/)
2019-11-12

#kubernetes

- [批量创建阿里云ECS并初始化](https://www.yp14.cn/2019/11/11/%E6%89%B9%E9%87%8F%E5%88%9B%E5%BB%BA%E9%98%BF%E9%87%8C%E4%BA%91ECS%E5%B9%B6%E5%88%9D%E5%A7%8B%E5%8C%96/)
2019-11-11

#OPS

- [监视Kubernetes事件并通过钉钉机器人通知](https://www.yp14.cn/2019/11/10/%E7%9B%91%E8%A7%86Kubernetes%E4%BA%8B%E4%BB%B6%E5%B9%B6%E9%80%9A%E8%BF%87%E9%92%89%E9%92%89%E6%9C%BA%E5%99%A8%E4%BA%BA%E9%80%9A%E7%9F%A5/)
2019-11-10

#kubernetes

- [Gitlab CI + Helm + Kubernetes 构建CI/CD](https://www.yp14.cn/2019/11/10/Gitlab-CI-Helm-Kubernetes-%E6%9E%84%E5%BB%BACI-CD/)
2019-11-10

#OPS

- [Dockerfile-最佳实践](https://www.yp14.cn/2019/11/09/Dockerfile-%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5/)
2019-11-09

#Docker

- [Gitlab CI 搭建持续集成环境](https://www.yp14.cn/2019/11/07/Gitlab-CI-%E6%90%AD%E5%BB%BA%E6%8C%81%E7%BB%AD%E9%9B%86%E6%88%90%E7%8E%AF%E5%A2%83/)
2019-11-07

#OPS

- [sentry9.1.2部署](https://www.yp14.cn/2019/11/05/sentry9-1-2%E9%83%A8%E7%BD%B2/)
2019-11-05

#OPS

- [sentry历史数据清理](https://www.yp14.cn/2019/10/21/sentry%E5%8E%86%E5%8F%B2%E6%95%B0%E6%8D%AE%E6%B8%85%E7%90%86/)
2019-10-21

#OPS

- [Istio Helm 1.2.5 版本安装](https://www.yp14.cn/2019/08/29/Istio-Helm-%E5%AE%89%E8%A3%85/)
2019-08-29

#Istio

- [Gitlab Docker Compose 启动配置](https://www.yp14.cn/2019/08/29/Gitlab-Docker-Compose-%E5%90%AF%E5%8A%A8%E9%85%8D%E7%BD%AE/)
2019-08-29

#OPS

- [centos7内核升级](https://www.yp14.cn/2019/08/29/centos7%E5%86%85%E6%A0%B8%E5%8D%87%E7%BA%A7/)
2019-08-29

#OPS

- [prometheus-operator手动部署](https://www.yp14.cn/2019/08/29/prometheus-operator%E6%89%8B%E5%8A%A8%E9%83%A8%E7%BD%B2/)
2019-08-29

#kubernetes

- [postgresql基本sql语句用法](https://www.yp14.cn/2019/08/29/postgresql%E5%9F%BA%E6%9C%ACsql%E8%AF%AD%E5%8F%A5%E7%94%A8%E6%B3%95/)
2019-08-29

#DataBase

- [docker 容器日志清理方案](https://www.yp14.cn/2019/08/29/docker-%E5%AE%B9%E5%99%A8%E6%97%A5%E5%BF%97%E6%B8%85%E7%90%86%E6%96%B9%E6%A1%88/)
2019-08-29

#Docker

- [Etcd使用命令](https://www.yp14.cn/2019/08/29/Etcd%E4%BD%BF%E7%94%A8%E5%91%BD%E4%BB%A4/)
2019-08-29

#kubernetes

- [Etcd v3备份与恢复](https://www.yp14.cn/2019/08/29/Etcd-v3%E5%A4%87%E4%BB%BD%E4%B8%8E%E6%81%A2%E5%A4%8D/)
2019-08-29

#kubernetes

- [es6自定义索引模板](https://www.yp14.cn/2019/08/29/es6%E8%87%AA%E5%AE%9A%E4%B9%89%E7%B4%A2%E5%BC%95%E6%A8%A1%E6%9D%BF/)
2019-08-29

#OPS

- [Elasticsearch查询](https://www.yp14.cn/2019/08/29/Elasticsearch%E6%9F%A5%E8%AF%A2/)
2019-08-29

#OPS

- [Metrics-Server v0.3.2版本安装](https://www.yp14.cn/2019/08/29/Metrics-Server-v0-3-2%E7%89%88%E6%9C%AC%E5%AE%89%E8%A3%85/)
2019-08-29

#kubernetes

- [Kubernetes v1.12.0 HA搭建](https://www.yp14.cn/2018/09/30/Kubernetes-v1-12-0-HA%E6%90%AD%E5%BB%BA/)
2018-09-30

#kubernetes#Docker

- [http思维导向图](https://www.yp14.cn/2017/12/20/http%E6%80%9D%E7%BB%B4%E5%AF%BC%E5%90%91%E5%9B%BE/)
2017-12-19

#OPS#Linux

- [harbor部署](https://www.yp14.cn/2017/03/02/harbor%E9%83%A8%E7%BD%B2/)
2017-03-02

#Linux

- [Gitlab\_ce\_mysql\_to\_postgresql](https://www.yp14.cn/2017/02/10/Gitlab-ce-mysql-to-postgresql/)
2017-02-10

#Linux

- [Centos7下LVM对基于xfs文件系统进行在线扩容方法](https://www.yp14.cn/2016/12/27/Centos7%E4%B8%8BLVM%E5%AF%B9%E5%9F%BA%E4%BA%8Exfs%E6%96%87%E4%BB%B6%E7%B3%BB%E7%BB%9F%E8%BF%9B%E8%A1%8C%E5%9C%A8%E7%BA%BF%E6%89%A9%E5%AE%B9%E6%96%B9%E6%B3%95/)
2016-12-27

#OPS

- [IP归属地查询](https://www.yp14.cn/2016/12/20/IP%E5%BD%92%E5%B1%9E%E5%9C%B0%E6%9F%A5%E8%AF%A2/)
2016-12-19

#OPS#Linux#Python

- [Centos7.2安装Ambari2.4.2+HDP2.5.3搭建Hadoop集群](https://www.yp14.cn/2016/12/09/Centos7-2%E5%AE%89%E8%A3%85Ambari2-4-2-HDP2-5-3%E6%90%AD%E5%BB%BAHadoop%E9%9B%86%E7%BE%A4/)
2016-12-09

#OPS#DataBase

- [kubernetes+keepalived高可用部署](https://www.yp14.cn/2016/11/28/kubernetes-keepalived%E9%AB%98%E5%8F%AF%E7%94%A8%E9%83%A8%E7%BD%B2/)
2016-11-28

#kubernetes#Docker

- [部署分布式kubernetes(v1.2.0)-centos7](https://www.yp14.cn/2016/09/29/%E9%83%A8%E7%BD%B2%E5%88%86%E5%B8%83%E5%BC%8Fkubernetes-v1-2-0-centos7/)
2016-09-29

#kubernetes#Docker

- [vi/vim按键盘布局](https://www.yp14.cn/2016/08/15/vi-vim%E6%8C%89%E9%94%AE%E7%9B%98%E5%B8%83%E5%B1%80/)
2016-08-14

#OPS#Linux

- [apache\_analysis\_log3](https://www.yp14.cn/2016/08/09/apache-analysis-log3/)
2016-08-09

#OPS#Linux#Python

- [nginx\_analysis\_log3](https://www.yp14.cn/2016/08/09/nginx-analysis-log3/)
2016-08-08

#OPS#Linux#Python

- [varnish\_analysis\_log3](https://www.yp14.cn/2016/08/09/varnish-analysis-log3/)
2016-08-08

#OPS#Linux#Python

- [Linux扫描进程使用情况](https://www.yp14.cn/2016/07/21/Linux%E6%89%AB%E6%8F%8F%E8%BF%9B%E7%A8%8B%E4%BD%BF%E7%94%A8%E6%83%85%E5%86%B5/)
2016-07-21

#OPS#Linux#Python

- [Python3.x标准模块库目录](https://www.yp14.cn/2016/07/14/Python3-x%E6%A0%87%E5%87%86%E6%A8%A1%E5%9D%97%E5%BA%93%E7%9B%AE%E5%BD%95/)
2016-07-14

#Python

- [Markdown11种基本语法](https://www.yp14.cn/2016/07/11/Markdown11%E7%A7%8D%E5%9F%BA%E6%9C%AC%E8%AF%AD%E6%B3%95/)
2016-07-11

#InformalEssay

- [Gitlab备份升级与恢复](https://www.yp14.cn/2016/07/11/Gitlab%E5%A4%87%E4%BB%BD%E5%8D%87%E7%BA%A7%E4%B8%8E%E6%81%A2%E5%A4%8D/)
2016-07-11

#Linux

- [Python3异常处理](https://www.yp14.cn/2016/07/06/Python%E5%BC%82%E5%B8%B8%E5%A4%84%E7%90%86/)
2016-07-05

#Python

- [Mysql常用命令](https://www.yp14.cn/2016/07/06/Mysql%E5%B8%B8%E7%94%A8%E5%91%BD%E4%BB%A4/)
2016-07-05

#InformalEssay#DataBase

- [Ubuntu设置开机启动方法](https://www.yp14.cn/2016/07/06/Ubuntu%E8%AE%BE%E7%BD%AE%E5%BC%80%E6%9C%BA%E5%90%AF%E5%8A%A8%E6%96%B9%E6%B3%95/)
2016-07-05

#OPS

- [GitLab社区稳定版8-9部署方法](https://www.yp14.cn/2016/07/05/GitLab/)
2016-07-04

#Linux

- [openshift](https://www.yp14.cn/2016/06/23/openshift-1/)
2016-06-23

#Linux


- [Happiness"Blog](https://blog.k8s.fit/)
- [Zhik8s](https://www.zhik8s.com/)