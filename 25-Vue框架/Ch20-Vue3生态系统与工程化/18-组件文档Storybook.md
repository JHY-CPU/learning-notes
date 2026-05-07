# 组件文档Storybook

## 一、概念说明

Storybook 是组件开发和文档工具，允许在隔离环境中独立开发、测试和展示 Vue 组件。每个"Story"是一个组件的状态变体，支持交互测试、可访问性检查、视口切换等功能。

## 二、具体用法

### 安装

```bash
npx storybook@latest init
# 自动检测 Vue 3 项目并配置
```

### 编写 Story

```ts
// src/components/Button.stories.ts
import type { Meta, StoryObj } from '@storybook/vue3'
import MyButton from './MyButton.vue'

const meta: Meta<typeof MyButton> = {
  title: 'Components/MyButton',
  component: MyButton,
  tags: ['autodocs'],
  argTypes: {
    type: {
      control: 'select',
      options: ['primary', 'secondary', 'danger']
    },
    size: {
      control: 'radio',
      options: ['small', 'medium', 'large']
    },
    onClick: { action: 'clicked' }
  }
}

export default meta
type Story = StoryObj<typeof meta>

// 基础 Story
export const Primary: Story = {
  args: {
    label: '主要按钮',
    type: 'primary'
  }
}

export const Secondary: Story = {
  args: {
    label: '次要按钮',
    type: 'secondary'
  }
}

export const Large: Story = {
  args: {
    label: '大按钮',
    type: 'primary',
    size: 'large'
  }
}

// 带交互的 Story
export const ClickInteraction: Story = {
  args: {
    label: '点击测试'
  },
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement)
    const button = canvas.getByRole('button')
    await userEvent.click(button)
    // 在 Storybook 的 Interactions 面板中可以看到点击被触发
  }
}
```

### 复合组件 Story

```ts
// src/components/Card.stories.ts
import Card from './Card.vue'

export default {
  title: 'Components/Card',
  component: Card
}

export const WithSlot = {
  render: () => ({
    components: { Card },
    template: `
      <Card title="用户信息">
        <template #default>
          <p>姓名：张三</p>
          <p>邮箱：zhang@example.com</p>
        </template>
        <template #footer>
          <button>编辑</button>
        </template>
      </Card>
    `
  })
}
```

### Storybook 配置

```ts
// .storybook/main.ts
import type { StorybookConfig } from '@storybook/vue3-vite'

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(ts|tsx)'],
  addons: [
    '@storybook/addon-essentials',
    '@storybook/addon-interactions',
    '@storybook/addon-a11y'     // 可访问性检查
  ],
  framework: {
    name: '@storybook/vue3-vite',
    options: {}
  }
}

export default config
```

## 三、注意事项与常见陷阱

1. **Story 文件与组件放在一起**：`Button.vue` 和 `Button.stories.ts` 同目录
2. **args 控制组件 props**：在 Storybook UI 中可以实时修改 props
3. **autodocs 自动生成文档**：`tags: ['autodocs']` 自动生成组件 API 文档
4. **play 函数需要 @storybook/addon-interactions**：否则交互测试不生效
5. **样式可能与实际应用不同**：全局样式需要在 preview 中导入
