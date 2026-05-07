# 非虚接口NVI

## 一、概念说明

**NVI**（Non-Virtual Interface）模式将公共接口声明为非虚函数，内部调用受保护的虚函数。这允许基类在虚函数调用前后添加不变量检查、日志、锁等公共逻辑。

## 二、具体用法

### 2.1 NVI模式实现

```cpp
#include <iostream>

class Connection {
public:
    // 非虚公共接口
    bool connect(const std::string& host) {
        std::cout << "连接前: 检查状态..." << std::endl;
        bool result = doConnect(host);  // 调用虚函数
        std::cout << "连接后: 记录日志..." << std::endl;
        return result;
    }

    void disconnect() {
        std::cout << "断开连接..." << std::endl;
        doDisconnect();
    }

    virtual ~Connection() = default;

protected:
    // 受保护虚函数：派生类实现
    virtual bool doConnect(const std::string& host) = 0;
    virtual void doDisconnect() = 0;
};

class TcpConnection : public Connection {
protected:
    bool doConnect(const std::string& host) override {
        std::cout << "TCP连接到: " << host << std::endl;
        return true;
    }
    void doDisconnect() override {
        std::cout << "TCP断开" << std::endl;
    }
};

int main() {
    TcpConnection tcp;
    tcp.connect("192.168.1.1");
    tcp.disconnect();
    return 0;
}
```

**输出：**
```
连接前: 检查状态...
TCP连接到: 192.168.1.1
连接后: 记录日志...
断开连接...
TCP断开
```

## 三、注意事项与常见陷阱

- 公共接口提供不变量保证
- 派生类只需实现`doXxx()`方法
- 可以在公共接口中添加线程安全逻辑
- NVI比直接暴露虚函数更灵活
