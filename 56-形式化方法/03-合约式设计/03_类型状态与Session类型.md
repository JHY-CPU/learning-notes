# 类型状态与Session类型


## 类型状态与Session类型

一、类型状态

## 一、类型状态 (Typestate)


类型状态是一种编程范式，将对象的状态编码在类型系统中，使得编译器可以在编译时检查状态相关的错误。只有在对象处于正确状态时，才能调用特定的方法。


### 1.1 核心思想


| 传统方式 | Typestate方式 |
| --- | --- |
| 运行时检查状态 | 编译时检查状态 |
| if (file.is_open()) { file.read(); } | 只有OpenFile类型才能调用read() |
| 运行时异常 | 编译错误 |
| 状态隐含在字段值中 | 状态显式编码在类型中 |


### 1.2 Rust中的类型状态模式


> **Note:** **Rust类型状态示例（文件句柄）：**
>
>
>
>
> // 状态标记类型（零大小）
>
>
> struct Closed;
>
>
> struct Open;
>
>
>
>
> // 泛型文件结构体，状态编码在类型参数中
>
>
> struct File<State> {
>
>
> path: String,
>
>
> _state: std::marker::PhantomData<State>,
>
>
> }
>
>
>
>
> // Closed状态的方法
>
>
> impl File<Closed> {
>
>
> fn new(path: &str) -> Self {
>
>
> File { path: path.to_string(), _state: PhantomData }
>
>
> }
>
>
>
>
> fn open(self) -> Result<File<Open>, io::Error> {
>
>
> // 打开文件逻辑
>
>
> Ok(File { path: self.path, _state: PhantomData })
>
>
> }
>
>
> }
>
>
>
>
> // Open状态的方法
>
>
> impl File<Open> {
>
>
> fn read(&self) -> Result<Vec<u8>, io::Error> {
>
>
> // 读取文件逻辑
>
>
> }
>
>
>
>
> fn close(self) -> File<Closed> {
>
>
> File { path: self.path, _state: PhantomData }
>
>
> }
>
>
> }
>
>
>
>
> // 使用
>
>
> let file = File::new("data.txt");     // File<Closed>
>
>
> let file = file.open().unwrap();       // File<Open>
>
>
> let data = file.read().unwrap();       // OK: File<Open>有read方法
>
>
> let file = file.close();               // File<Closed>
>
>
> // file.read(); // 编译错误！File<Closed>没有read方法


### 1.3 类型状态的状态机建模


> **Note:** **连接器状态机：**
>
>
>
>
> Disconnected --connect()--> Connecting
>
>
> Connecting --handshake()--> Connected
>
>
> Connected --send()--> Connected (自环)
>
>
> Connected --disconnect()--> Disconnected
>
>
>
>
> 非法转换（编译时捕获）：
>
>
> Disconnected --send()--> 编译错误！
>
>
> Connected --connect()--> 编译错误！

二、Session类型

## 二、Session类型 (Session Types)


Session类型是对通信协议的形式化描述，将通信协议编码在类型系统中，使得编译器可以检查通信双方是否遵循了协议。


### 2.1 核心概念


| 概念 | 符号 | 含义 |
| --- | --- | --- |
| 发送 | !T.P | 发送类型T的值，继续协议P |
| 接收 | ?T.P | 接收类型T的值，继续协议P |
| 选择（外部选择） | &{l1:P1, l2:P2} | 对方选择分支l1或l2 |
| 提供（内部选择） | +{l1:P1, l2:P2} | 自己选择分支l1或l2 |
| 结束 | end | 通信结束 |
| 递归 | μX.P | 递归协议，X是递归变量 |


### 2.2 Session类型示例


> **Note:** **简单请求-响应协议：**
>
>
>
>
> Server: ?String.!Response.end
>
>
> Client: !String.?Response.end
>
>
>
>
> 含义：服务器接收一个String，发送一个Response，然后结束。
>
>
> 客户端发送一个String，接收一个Response，然后结束。
>
>
>
>
> **多次交互协议：**
>
>
>
>
> Server: μX.?Command.&{ok:!Result.X, quit:end}
>
>
> Client: μX.!Command.+{ok:?Result.X, quit:end}
>
>
>
>
> 含义：服务器接收命令，然后选择：如果是ok则发送结果并递归继续，如果是quit则结束。


### 2.3 对偶性 (Duality)


Session类型的核心性质是"对偶性"：如果一个端点的类型是S，那么另一个端点的类型必须是S的对偶类型dual(S)。对偶类型确保双方的发送和接收完全匹配。


| 类型 | 对偶类型 |
| --- | --- |
| !T.P（发送T） | ?T.dual(P)（接收T） |
| ?T.P（接收T） | !T.dual(P)（发送T） |
| +{l1:P1, l2:P2}（选择） | &{l1:dual(P1), l2:dual(P2)}（提供） |
| end（结束） | end（结束） |

三、Rust所有权与协议验证

## 三、Rust所有权作为协议验证


Rust的所有权系统和借用检查器天然地提供了协议验证能力。所有权转移可以建模状态转换，生命周期可以建模会话时序。


### 3.1 所有权 = 能力 (Capability)


- 拥有某个值 = 有权对该值执行操作
- 所有权转移 = 能力转移（类似Session类型的通道传递）
- 借用 = 临时能力（有生命周期约束）
- 移动语义确保每个操作在编译时精确执行一次


### 3.2 协议验证示例


> **Note:** **Rust通道实现Session类型：**
>
>
>
>
> // 协议：先发送用户名(String)，再发送密码(String)，然后接收Token(String)
>
>
> // Client: !String.!String.?String.end
>
>
> // Server: ?String.?String.!String.end
>
>
>
>
> struct AuthClient { channel: mpsc::Sender<String> }
>
>
> struct AuthServer { channel: mpsc::Receiver<String> }
>
>
>
>
> impl AuthClient {
>
>
> fn send_username(self, username: String) -> AuthClientStep2 {
>
>
> self.channel.send(username).unwrap();
>
>
> AuthClientStep2 { channel: self.channel }
>
>
> }
>
>
> }
>
>
>
>
> struct AuthClientStep2 { channel: mpsc::Sender<String> }
>
>
>
>
> impl AuthClientStep2 {
>
>
> fn send_password(self, password: String) -> AuthClientStep3 {
>
>
> self.channel.send(password).unwrap();
>
>
> AuthClientStep3 { receiver: ... }
>
>
> }
>
>
> }
>
>
>
>
> struct AuthClientStep3 { receiver: mpsc::Receiver<String> }
>
>
>
>
> impl AuthClientStep3 {
>
>
> fn receive_token(self) -> String {
>
>
> self.receiver.recv().unwrap()
>
>
> }
>
>
> }
>
>
>
>
> // 编译时保证：不能跳过步骤直接receive_token()
>         // client.receive_token(); // 编译错误！AuthClient没有此方法


### 3.3 实际应用


| 应用 | 如何使用类型状态/Session类型 |
| --- | --- |
| HTTP客户端 | Builder模式 → Configured → Connected → Sent → Received |
| 数据库连接 | Unconnected → Connected → InTransaction → Committed/RolledBack |
| TLS握手 | ClientHello → ServerHello → Certificate → Established |
| 状态机驱动的解析器 | Start → InProgress → Complete，每步类型不同 |
| 硬件寄存器访问 | 初始化序列的严格顺序保证 |

四、实现方式

## 四、其他语言的实现


### 4.1 Java Sealed Types + Pattern Matching


> **Note:** **Java 17+ 类型状态模拟：**
>
>
>
>
> sealed interface ConnectionState permits Disconnected, Connected { }
>
>
> record Disconnected() implements ConnectionState { }
>
>
> record Connected(Socket socket) implements ConnectionState { }
>
>
>
>
> // 只有Connected有send方法
>
>
> void send(Connected conn, byte[] data) {
>
>
> conn.socket().getOutputStream().write(data);
>
>
> }
>
>
>
>
> // Pattern matching保证类型安全
>
>
> String receive(ConnectionState state) {
>
>
> return switch (state) {
>
>
> case Disconnected d -> throw new IllegalStateException();
>
>
> case Connected c -> readFrom(c.socket());
>
>
> }
>
>
> }


### 4.2 TypeScript 条件类型


> **Note:** **TypeScript类型状态：**
>
>
>
>
> type Closed = { state: "closed" };
>
>
> type Open = { state: "open"; handle: number };
>
>
>
>
> type File<S extends Closed | Open> = S & { path: string };
>
>
>
>
> function open(f: File<Closed>): File<Open> { ... }
>
>
> function read(f: File<Open>): Buffer { ... }
>
>
> function close(f: File<Open>): File<Closed> { ... }
>
>
>
>
> const f: File<Closed> = { path: "data.txt", state: "closed" };
>
>
> const opened = open(f);  // File<Open>
>
>
> const data = read(opened);  // OK
>
>
> // read(f); // Type error!

**类型状态和Session类型的价值：**
它们将运行时才能发现的协议违规错误提前到编译时。Rust的所有权系统天然支持这种范式，使协议验证成为零成本抽象。在安全关键系统（TLS、数据库、硬件驱动）中，这种编译时保证极具价值。
========================================
  文件总结
========================================
  主题：类型状态与Session类型
  内容概要：
    1. Typestate - 将对象状态编码在类型中，编译时检查状态相关错误
    2. Session类型 - 通信协议的形式化描述，检查双方是否遵循协议
    3. 对偶性 - Session类型的核心性质，确保发送和接收匹配
    4. Rust所有权 - 天然支持协议验证，所有权转移=能力转移
  重点知识：
    - Typestate的核心：只有正确状态的类型才有对应方法
    - Session类型六种构造：发送/接收/选择/提供/结束/递归
    - 对偶性保证通信安全：!T的对偶是?T
    - Rust的所有权和移动语义天然实现Session类型的通道语义
========================================


<!-- Converted from: 03_类型状态与Session类型.html -->
