# E2E 测试 Playwright

## 一、概念说明

Playwright 是微软开源的 E2E 测试框架，支持 Chromium、Firefox、WebKit 三种浏览器引擎，API 简洁，自动等待机制优秀。

```bash
npm init playwright@latest
```

```js
// tests/todo.spec.js
import { test, expect } from '@playwright/test'

test.describe('Todo 应用', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:5173')
  })

  test('添加新任务', async ({ page }) => {
    await page.getByTestId('input').fill('学习 Playwright')
    await page.getByTestId('add-btn').click()
    await expect(page.getByTestId('todo-list').locator('li')).toHaveCount(1)
  })

  test('完成任务', async ({ page }) => {
    await page.getByTestId('input').fill('测试任务')
    await page.getByTestId('add-btn').click()
    await page.getByRole('checkbox').click()
    await expect(page.locator('li')).toHaveClass(/completed/)
  })

  test('截图对比', async ({ page }) => {
    await expect(page).toHaveScreenshot('todo-page.png')
  })
})
```

## 二、具体用法

### 2.1 定位器

```js
page.getByRole('button', { name: '提交' })  // 按角色和文本
page.getByText('欢迎')                        // 按文本
page.getByTestId('input')                     // 按 testid
page.locator('.class-name')                   // CSS 选择器
```

### 2.2 多浏览器测试

```js
// playwright.config.js
export default defineConfig({
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'mobile', use: { ...devices['iPhone 13'] } }
  ]
})
```

### 2.3 API Mock

```js
test('mock API 响应', async ({ page }) => {
  await page.route('/api/users', route => {
    route.fulfill({
      status: 200,
      body: JSON.stringify([{ id: 1, name: 'Mock 用户' }])
    })
  })
  await page.goto('/')
  await expect(page.getByText('Mock 用户')).toBeVisible()
})
```

### 2.4 生成代码

```bash
npx playwright codegen http://localhost:5173
# 打开录制工具，操作后自动生成测试代码
```

## 三、注意事项与常见陷阱

- Playwright 自动等待元素可交互，不需要手动 `wait`
- 使用 `getByRole` 等语义化选择器比 CSS 选择器更稳定
- 首次运行需要下载浏览器：`npx playwright install`
