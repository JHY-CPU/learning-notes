# E2E 测试 Cypress

## 一、概念说明

Cypress 是一个前端 E2E（端到端）测试框架，在真实浏览器中运行测试，模拟完整用户操作流程。

```bash
# 安装
npm install -D cypress
npx cypress open  # 打开测试界面
```

```js
// cypress/e2e/todo.cy.js
describe('Todo 应用', () => {
  beforeEach(() => {
    cy.visit('http://localhost:5173')
  })

  it('添加一个新任务', () => {
    cy.get('[data-testid="input"]').type('学习 Cypress')
    cy.get('[data-testid="add-btn"]').click()
    cy.get('[data-testid="todo-list"] li').should('have.length', 1)
    cy.get('[data-testid="todo-list"] li').first().should('contain', '学习 Cypress')
  })

  it('完成任务', () => {
    cy.get('[data-testid="input"]').type('学习测试')
    cy.get('[data-testid="add-btn"]').click()
    cy.get('[data-testid="todo-list"] li input[type="checkbox"]').click()
    cy.get('[data-testid="todo-list"] li').should('have.class', 'completed')
  })

  it('删除任务', () => {
    cy.get('[data-testid="input"]').type('待删除')
    cy.get('[data-testid="add-btn"]').click()
    cy.get('[data-testid="todo-list"] li button.delete').click()
    cy.get('[data-testid="todo-list"] li').should('have.length', 0)
  })
})
```

## 二、具体用法

### 2.1 常用命令

```js
cy.visit('/path')                    // 访问页面
cy.get('.selector')                  // 获取元素
cy.contains('文本')                   // 按文本查找
cy.get('input').type('hello')        // 输入
cy.get('button').click()             // 点击
cy.get('.item').should('be.visible') // 断言
cy.wait('@apiRoute')                 // 等待 API
```

### 2.2 API 拦截

```js
// 拦截 API 请求
cy.intercept('GET', '/api/users', { fixture: 'users.json' }).as('getUsers')
cy.visit('/')
cy.wait('@getUsers')
```

### 2.3 配置文件

```js
// cypress.config.js
import { defineConfig } from 'cypress'

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:5173',
    supportFile: 'cypress/support/e2e.js',
    specPattern: 'cypress/e2e/**/*.cy.js'
  }
})
```

## 三、注意事项与常见陷阱

- Cypress 运行在浏览器内，不能访问 Node.js API
- 使用 `data-testid` 选择元素更稳定
- 每个测试应该独立，不依赖其他测试的状态
- CI 环境需要无头模式：`cypress run --headless`
