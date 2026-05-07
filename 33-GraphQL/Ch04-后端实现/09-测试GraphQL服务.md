# 测试 GraphQL 服务

## 一、Schema 测试

```java
@SpringBootTest
class SchemaTest {

    @Autowired
    private GraphQLSchema schema;

    @Test
    void shouldHaveAllQueryFields() {
        GraphQLObjectType queryType = schema.getQueryType();
        assertThat(queryType.getFieldDefinition("user")).isNotNull();
        assertThat(queryType.getFieldDefinition("users")).isNotNull();
    }

    @Test
    void shouldHaveAllMutationFields() {
        GraphQLObjectType mutationType = schema.getMutationType();
        assertThat(mutationType.getFieldDefinition("createUser")).isNotNull();
    }
}
```

## 二、Resolver 单元测试

```java
@SpringBootTest
class UserControllerTest {

    @MockBean
    private UserService userService;

    @Autowired
    private GraphQLTester graphQLTester;

    @Test
    void shouldReturnUserById() {
        User user = new User("1", "张三", "zhangsan@test.com");
        when(userService.findById("1")).thenReturn(user);

        graphQLTester.document("""
            query {
              user(id: "1") {
                id
                name
                email
              }
            }
            """)
            .execute()
            .path("user.name")
            .entity(String.class)
            .isEqualTo("张三");
    }

    @Test
    void shouldCreateUser() {
        CreateUserInput input = new CreateUserInput("李四", "lisi@test.com");
        when(userService.create(any())).thenReturn(
            new User("2", "李四", "lisi@test.com"));

        graphQLTester.document("""
            mutation($input: CreateUserInput!) {
              createUser(input: $input) {
                id
                name
              }
            }
            """)
            .variable("input", Map.of("name", "李四", "email", "lisi@test.com"))
            .execute()
            .path("createUser.name")
            .entity(String.class)
            .isEqualTo("李四");
    }
}
```

## 三、集成测试

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class GraphQLIntegrationTest {

    @Autowired
    private TestRestTemplate restTemplate;

    @Test
    void shouldExecuteQueryViaHttp() {
        Map<String, Object> request = Map.of(
            "query", "{ user(id: \"1\") { name } }"
        );

        ResponseEntity<Map> response = restTemplate.postForEntity(
            "/graphql", request, Map.class);

        assertThat(response.getStatusCode().is2xxSuccessful()).isTrue();
        Map<String, Object> body = response.getBody();
        Map<String, Object> data = (Map<String, Object>) body.get("data");
        assertThat(data.get("user")).isNotNull();
    }
}
```

## 四、注意事项

1. **GraphQLTester 是 Spring GraphQL 的测试工具**
2. **Mock 数据源测试 Resolver 逻辑**
3. **集成测试验证完整请求链路**
4. **测试错误场景**
5. **CI 中自动运行测试**
