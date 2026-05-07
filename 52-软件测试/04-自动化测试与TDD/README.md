# 自动化测试与TDD

## 一、自动化测试的价值与成本

### 1.1 价值

- **提高效率**：重复性测试由机器执行，释放人力
- **提高准确性**：消除人工执行的疏忽和疲劳
- **快速反馈**：每次代码变更后立即验证
- **支持持续集成**：是 CI/CD 流水线的核心环节
- **可重复执行**：回归测试成本趋近于零
- **覆盖率可控**：可以执行大量用例而不增加人力成本

### 1.2 成本

- **初始投入大**：脚本开发、框架搭建需要时间和技术
- **维护成本高**：UI变更导致脚本频繁失败
- **不适合所有场景**：探索性测试、可用性测试仍需人工
- **工具学习曲线**：团队需要掌握自动化工具和技术

### 1.3 自动化适用场景

- 需要频繁回归测试的功能
- 数据驱动的测试
- 性能测试和负载测试
- 构建验证测试（冒烟测试）
- 不适合：一次性测试、界面频繁变更、探索性测试

---

## 二、自动化测试框架

### 2.1 JUnit（Java）

```java
import org.junit.jupiter.api.*;

public class CalculatorTest {

    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }

    @Test
    @DisplayName("除法测试")
    void testDivide() {
        Calculator calc = new Calculator();
        assertEquals(2.5, calc.divide(5, 2));
    }

    @Test
    void testDivideByZero() {
        Calculator calc = new Calculator();
        assertThrows(ArithmeticException.class, () -> calc.divide(1, 0));
    }
}
```

- JUnit 5 支持参数化测试、嵌套测试、动态测试
- 生命周期注解：`@BeforeAll`、`@BeforeEach`、`@AfterEach`、`@AfterAll`

### 2.2 TestNG（Java）

```java
@Test
public class LoginTest {
    @DataProvider(name = "loginData")
    public Object[][] data() {
        return new Object[][] {
            {"admin", "123456", true},
            {"admin", "wrong", false}
        };
    }

    @Test(dataProvider = "loginData")
    public void testLogin(String user, String pass, boolean expected) {
        assertEquals(LoginService.login(user, pass), expected);
    }
}
```

- 支持依赖测试、分组、并行执行、数据驱动
- 配置文件：`testng.xml`

### 2.3 pytest（Python）

```python
import pytest

def add(a, b):
    return a + b

class TestAdd:
    def test_positive(self):
        assert add(2, 3) == 5

    def test_negative(self):
        assert add(-1, -1) == -2

    @pytest.mark.parametrize("a,b,expected", [
        (1, 2, 3), (0, 0, 0), (-1, 1, 0)
    ])
    def test_parametrized(self, a, b, expected):
        assert add(a, b) == expected
```

- Fixture 机制：`@pytest.fixture`
- 插件丰富：pytest-cov、pytest-xdist、pytest-html

---

## 三、Mock 与 Stub 技术

### 3.1 概念区分

| 类型 | 作用 |
|------|------|
| **Stub** | 替代真实对象，返回预设数据 |
| **Mock** | 替代真实对象，验证方法是否被正确调用 |
| **Spy** | 包装真实对象，可以部分模拟 |
| **Fake** | 轻量级实现（如内存数据库） |

### 3.2 Mockito（Java）

```java
@ExtendWith(MockitoExtension.class)
class OrderServiceTest {

    @Mock
    PaymentService paymentService;

    @InjectMocks
    OrderService orderService;

    @Test
    void testPlaceOrder() {
        // 设置 Mock 行为
        when(paymentService.charge(anyDouble())).thenReturn(true);

        boolean result = orderService.placeOrder(100.0);

        assertTrue(result);
        // 验证调用
        verify(paymentService, times(1)).charge(100.0);
    }
}
```

### 3.3 WireMock（HTTP Mock）

```java
@Rule
public WireMockRule wireMockRule = new WireMockRule(8080);

@Test
void testExternalApi() {
    stubFor(get(urlEqualTo("/api/users"))
        .willReturn(aResponse()
            .withHeader("Content-Type", "application/json")
            .withBody("{\"name\": \"test\"}")));

    // 调用被测服务，它会访问 mock 的 /api/users
    User user = userService.getUser();
    assertEquals("test", user.getName());
}
```

### 3.4 Python mock

```python
from unittest.mock import MagicMock, patch

def test_send_email():
    with patch('myapp.email.send') as mock_send:
        mock_send.return_value = True
        result = notify_user("user@example.com")
        assert result is True
        mock_send.assert_called_once_with("user@example.com")
```

---

## 四、测试覆盖率工具

### 4.1 覆盖率指标

- **行覆盖率**：执行了多少行代码
- **分支覆盖率**：覆盖了多少分支
- **方法/函数覆盖率**：调用了多少方法
- **条件覆盖率**：子条件的真/假是否都覆盖

### 4.2 JaCoCo（Java）

```xml
<!-- Maven 配置 -->
<plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <executions>
        <execution>
            <goals><goal>prepare-agent</goal></goals>
        </execution>
        <execution>
            <id>report</id>
            <phase>test</phase>
            <goals><goal>report</goal></goals>
        </execution>
    </executions>
</plugin>
```

生成 HTML 报告到 `target/site/jacoco/index.html`。

### 4.3 gcov（C/C++）

```bash
gcc -fprofile-arcs -ftest-coverage -o myapp myapp.c
./myapp
gcov myapp.c
# 生成 .gcov 文件显示每行的执行次数
```

### 4.4 pytest-cov（Python）

```bash
pytest --cov=myapp --cov-report=html tests/
```

### 4.5 覆盖率目标

- 行覆盖率：80%+ 为良好
- 分支覆盖率：70%+ 为良好
- 不要追求100%覆盖率，关注**关键业务逻辑**的覆盖

---

## 五、UI 自动化测试

### 5.1 Selenium

```java
WebDriver driver = new ChromeDriver();
driver.get("https://example.com");

// 元素定位
WebElement searchBox = driver.findElement(By.id("search"));
searchBox.sendKeys("test query");
searchBox.submit();

// 等待
WebDriverWait wait = new WebDriverWait(driver, Duration.ofSeconds(10));
WebElement result = wait.until(
    ExpectedConditions.visibilityOfElementLocated(By.className("result"))
);

assertEquals("Expected Title", result.getText());
driver.quit();
```

- 支持多种语言：Java、Python、C#、JavaScript
- 支持多种浏览器：Chrome、Firefox、Safari、Edge
- 配合 Page Object Model（POM）设计模式

### 5.2 Playwright（现代替代方案）

```javascript
const { test, expect } = require('@playwright/test');

test('login test', async ({ page }) => {
    await page.goto('https://example.com/login');
    await page.fill('#username', 'admin');
    await page.fill('#password', 'password');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('.welcome')).toContainText('Welcome');
});
```

- 自动等待机制
- 支持多浏览器并行
- 内置截图和视频录制
- 网络拦截和 Mock

### 5.3 Cypress

```javascript
describe('Login', () => {
    it('should login successfully', () => {
        cy.visit('/login');
        cy.get('#username').type('admin');
        cy.get('#password').type('password');
        cy.get('button[type="submit"]').click();
        cy.url().should('include', '/dashboard');
    });
});
```

- 仅支持 JavaScript/TypeScript
- 实时重载和时间旅行调试
- 网络请求控制

---

## 六、API 自动化测试

### 6.1 Postman / Newman

```json
{
    "name": "User API Tests",
    "item": [{
        "name": "GET Users",
        "request": {
            "method": "GET",
            "url": "{{baseUrl}}/api/users"
        },
        "event": [{
            "listen": "test",
            "script": {
                "exec": [
                    "pm.test('Status is 200', () => { pm.response.to.have.status(200); });",
                    "pm.test('Response is array', () => { pm.expect(pm.response.json()).to.be.an('array'); });"
                ]
            }
        }]
    }]
}
```

Newman 命令行执行：`newman run collection.json -e environment.json`

### 6.2 REST Assured（Java）

```java
given()
    .baseUri("https://api.example.com")
    .header("Authorization", "Bearer " + token)
    .queryParam("page", 1)
.when()
    .get("/users")
.then()
    .statusCode(200)
    .body("size()", greaterThan(0))
    .body("[0].name", equalTo("John"));
```

---

## 七、性能测试工具

### 7.1 Apache JMeter

- 基于 Java 的开源负载测试工具
- 支持 HTTP、FTP、JDBC、JMS 等协议
- GUI 和命令行两种模式
- 分布式测试支持

```bash
# 命令行执行
jmeter -n -t test_plan.jmx -l results.jtl -e -o ./report
```

### 7.2 Locust（Python）

```python
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)

    @task(3)
    def index(self):
        self.client.get("/")

    @task(1)
    def api_users(self):
        self.client.get("/api/users", headers={"Authorization": "Bearer token"})
```

- 使用 Python 编写测试场景
- 支持分布式运行
- 实时 Web UI 监控

### 7.3 wrk

```bash
# 高性能 HTTP 基准测试
wrk -t12 -c400 -d30s http://localhost:8080/api/endpoint
```

- C 语言编写，性能极高
- 支持 Lua 脚本自定义请求
- 适合简单的基准测试

### 7.4 关键指标

- **吞吐量（RPS/TPS）**：每秒请求数/事务数
- **响应时间**：平均、P50、P95、P99
- **错误率**：失败请求的比例
- **并发数**：同时在线用户数

---

## 八、TDD（测试驱动开发）

### 8.1 红-绿-重构循环

```
1. 红（Red）  ：编写一个失败的测试
2. 绿（Green）：编写最少的代码使测试通过
3. 重构（Refactor）：优化代码，消除重复
```

### 8.2 TDD 三法则

1. 除非为了让一个失败的单元测试通过，否则不允许编写任何产品代码
2. 只允许编写刚好能够导致失败或编译不通过的单元测试
3. 只允许编写刚好能够使一个失败的单元测试通过的产品代码

### 8.3 TDD 示例

```python
# 第1步：写测试（红）
def test_fizzbuzz():
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(5) == "Buzz"
    assert fizzbuzz(15) == "FizzBuzz"
    assert fizzbuzz(7) == "7"

# 第2步：写实现（绿）
def fizzbuzz(n):
    if n % 15 == 0: return "FizzBuzz"
    if n % 3 == 0: return "Fizz"
    if n % 5 == 0: return "Buzz"
    return str(n)

# 第3步：重构（如需要）
```

### 8.4 TDD 的优势

- 代码自带测试套件
- 设计更简洁、耦合度更低
- 重构有安全保障
- 活的文档（测试即用例）

---

## 九、BDD（行为驱动开发）

### 9.1 定义

BDD 使用**自然语言**描述系统行为，让开发人员、测试人员和业务人员使用统一的语言沟通。

### 9.2 Gherkin 语法

```gherkin
Feature: 用户登录

  Scenario: 成功登录
    Given 用户在登录页面
    When 输入用户名 "admin" 和密码 "123456"
    And 点击登录按钮
    Then 跳转到首页
    And 显示欢迎信息 "Welcome, admin"

  Scenario: 密码错误
    Given 用户在登录页面
    When 输入用户名 "admin" 和密码 "wrong"
    And 点击登录按钮
    Then 显示错误信息 "密码错误"
```

### 9.3 Cucumber

```java
@Given("用户在登录页面")
public void navigateToLogin() {
    driver.get("http://localhost/login");
}

@When("输入用户名 {string} 和密码 {string}")
public void enterCredentials(String user, String pass) {
    driver.findElement(By.id("username")).sendKeys(user);
    driver.findElement(By.id("password")).sendKeys(pass);
    driver.findElement(By.id("loginBtn")).click();
}

@Then("跳转到首页")
public void verifyHomePage() {
    assertEquals("/home", driver.getCurrentUrl());
}
```

---

## 十、持续测试与 CI/CD 中的测试策略

### 10.1 CI/CD 流水线中的测试阶段

```
代码提交 → 静态分析 → 单元测试 → 集成测试 → 构建 → 部署到测试环境 → E2E测试 → 性能测试 → 部署到生产
```

### 10.2 测试分层策略

| 阶段 | 测试类型 | 工具 | 执行时间 |
|------|---------|------|---------|
| 提交时 | Lint + 单元测试 | ESLint、JUnit | 秒级 |
| 构建时 | 集成测试 | Testcontainers | 分钟级 |
| 部署后 | E2E测试 | Playwright | 分钟级 |
| 定期 | 性能测试 | JMeter | 小时级 |

### 10.3 测试左移与右移

- **测试左移（Shift Left）**：尽早测试，在需求和设计阶段就引入质量活动
- **测试右移（Shift Right）**：生产环境中持续监控和测试（A/B测试、金丝雀发布、混沌工程）

### 10.4 质量门禁（Quality Gate）

- 测试通过率 >= 100%
- 代码覆盖率 >= 80%
- 无阻塞性缺陷
- 静态分析无严重问题

---

## 十一、回归测试策略

### 11.1 回归测试定义

在代码变更后重新执行测试，确保变更未引入新的缺陷。

### 11.2 策略

- **完全回归**：重新执行全部测试用例（成本高、耗时长）
- **选择性回归**：仅执行受影响模块的测试
- **基于风险的回归**：优先执行高风险模块的测试
- **基于变更的回归**：分析代码变更的依赖关系

### 11.3 回归测试用例选择

- 修改模块的直接和间接测试用例
- 与修改模块有接口依赖的测试用例
- 历史上经常出错的测试用例
- 核心业务流程的端到端测试

### 11.4 自动化回归

- 维护稳定的自动化回归测试套件
- 每次构建自动触发
- 失败的用例及时修复和维护
- 定期清理过时或冗余的测试用例
