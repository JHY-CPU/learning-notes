# 非虚接口NVI

## 一、概念说明

**NVI**（Non-Virtual Interface，C++标准 §10.3）模式将公共接口声明为非虚函数，内部调用受保护的虚函数。这允许基类在虚函数调用前后添加不变量检查、日志、锁等公共逻辑，而派生类只需关注核心实现。

### 1.1 NVI模式结构

```
public:  void action() {          ← 非虚公共接口
             前置处理();            ← 基类控制
             doAction();           ← 调用虚函数
             后置处理();            ← 基类控制
         }
protected: virtual void doAction() = 0;  ← 派生类实现
```

## 二、具体用法

### 2.1 NVI模式实现

```cpp
#include <iostream>

class Connection {
public:
    // 非虚公共接口
    bool connect(const std::string& host) {
        if (connected) {
            std::cout << "已连接，无需重复连接" << std::endl;
            return false;
        }
        std::cout << "连接前: 检查状态..." << std::endl;
        bool result = doConnect(host);
        if (result) {
            connected = true;
            std::cout << "连接后: 记录日志..." << std::endl;
        }
        return result;
    }

    void disconnect() {
        if (!connected) return;
        std::cout << "断开连接..." << std::endl;
        doDisconnect();
        connected = false;
    }

    virtual ~Connection() = default;

protected:
    // 受保护虚函数：派生类实现
    virtual bool doConnect(const std::string& host) = 0;
    virtual void doDisconnect() = 0;

private:
    bool connected = false;
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

class SshConnection : public Connection {
protected:
    bool doConnect(const std::string& host) override {
        std::cout << "SSH握手: " << host << std::endl;
        return true;
    }
    void doDisconnect() override {
        std::cout << "SSH关闭" << std::endl;
    }
};

int main() {
    TcpConnection tcp;
    tcp.connect("192.168.1.1");
    tcp.connect("重复连接");  // 被NVI拦截
    tcp.disconnect();

    SshConnection ssh;
    ssh.connect("server.example.com");
    ssh.disconnect();

    return 0;
}
```

### 2.2 NVI与线程安全

```cpp
#include <iostream>
#include <mutex>

class ThreadSafeProcessor {
public:
    void process(int data) {
        std::lock_guard<std::mutex> lock(mtx);  // 公共锁
        std::cout << "处理数据: " << data << std::endl;
        doProcess(data);  // 派生类实现
        std::cout << "处理完成" << std::endl;
    }
    virtual ~ThreadSafeProcessor() = default;

protected:
    virtual void doProcess(int data) = 0;

private:
    std::mutex mtx;
};

class ConcreteProcessor : public ThreadSafeProcessor {
protected:
    void doProcess(int data) override {
        std::cout << "  具体处理: " << data * 2 << std::endl;
    }
};

int main() {
    ConcreteProcessor p;
    p.process(42);
    return 0;
}
```

### 2.3 NVI与不变量

```cpp
#include <iostream>
#include <stdexcept>

class Validator {
public:
    void validate(const std::string& input) {
        // 前置条件检查
        if (input.empty())
            throw std::invalid_argument("输入不能为空");

        std::cout << "验证输入: " << input << std::endl;
        bool valid = doValidate(input);

        // 后置条件处理
        if (valid)
            std::cout << "验证通过" << std::endl;
        else
            std::cout << "验证失败" << std::endl;
    }
    virtual ~Validator() = default;

protected:
    virtual bool doValidate(const std::string& input) = 0;
};

class EmailValidator : public Validator {
protected:
    bool doValidate(const std::string& input) override {
        return input.find('@') != std::string::npos;
    }
};

class PhoneValidator : public Validator {
protected:
    bool doValidate(const std::string& input) override {
        return input.size() >= 10;
    }
};

int main() {
    EmailValidator ev;
    ev.validate("test@example.com");  // 通过
    ev.validate("invalid");           // 失败

    PhoneValidator pv;
    pv.validate("1234567890");  // 通过
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **公共接口提供不变量保证**：前置检查、后置处理、日志、锁
2. **派生类只需实现`doXxx()`方法**：不需要关心公共逻辑
3. **可以在公共接口中添加线程安全逻辑**：NVI是天然的锁封装点
4. **NVI比直接暴露虚函数更灵活**：基类可以在不改变接口的情况下添加逻辑
5. **虚函数设为protected**：避免外部直接调用跳过公共逻辑
6. **NVI与模板方法模式密切相关**：NVI是模板方法的简化版
