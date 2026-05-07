# 测试替身与 Mock

## 一、测试替身类型

```
测试替身:
├── Dummy - 仅占位，不使用
├── Stub - 返回预设值
├── Spy - 记录调用信息
├── Mock - 验证交互行为
└── Fake - 简化实现（如内存数据库）
```

## 二、Mockito 用法

```java
class MockExamples {

    @Mock
    private PaymentClient paymentClient;

    @Spy
    private InventoryService inventoryService;

    @InjectMocks
    private OrderService orderService;

    // Stub - 预设返回值
    @Test
    void stubExample() {
        Mockito.when(paymentClient.deduct(any()))
               .thenReturn(new PaymentResult(true));
    }

    // Mock - 验证交互
    @Test
    void mockExample() {
        orderService.createOrder(new OrderRequest("P001", 2));
        Mockito.verify(paymentClient, Mockito.times(1))
               .deduct(any());
        Mockito.verify(paymentClient, Mockito.never())
               .refund(any());
    }

    // Spy - 部分 Mock
    @Test
    void spyExample() {
        Mockito.doReturn(true)
               .when(inventoryService).checkStock(any(), anyInt());
        // 其他方法保持真实逻辑
    }

    // 异常 Mock
    @Test
    void exceptionMock() {
        Mockito.when(paymentClient.deduct(any()))
               .thenThrow(new PaymentException("余额不足"));
    }
}
```

## 三、WireMock HTTP Stub

```java
@SpringBootTest
@AutoConfigureWireMock(port = 8089)
class WireMockTest {

    @Test
    void shouldCallExternalApi() {
        stubFor(get(urlEqualTo("/external/api/data"))
            .willReturn(aResponse()
                .withHeader("Content-Type", "application/json")
                .withBody("{\"result\":\"success\"}")
                .withStatus(200)));

        String result = externalClient.getData();
        assertThat(result).isEqualTo("success");
    }
}
```

## 四、注意事项

1. **优先使用接口做 Mock**，而非具体类
2. **Mock 要尽量少**，只 Mock 直接依赖
3. **不要过度验证内部实现**
4. **@Spy 要谨慎使用**，可能导致真实调用
5. **WireMock 适合外部服务依赖的测试**
