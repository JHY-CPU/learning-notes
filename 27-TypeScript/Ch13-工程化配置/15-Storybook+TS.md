# Storybook+TS

## 一、概念说明

Storybook 是组件文档和开发工具，对 TypeScript 有完善支持。可以为 React/Vue 组件编写带有类型信息的故事文件。

## 二、具体用法

### 2.1 安装

```bash
npx storybook@latest init
# 自动检测 TS 项目并配置
```

### 2.2 React 故事文件

```tsx
// Button.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'danger'],
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Primary: Story = {
  args: {
    variant: 'primary',
    children: '主要按钮',
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: '次要按钮',
  },
};
```

### 2.3 Vue 故事文件

```typescript
// Button.stories.ts
import type { Meta, StoryObj } from '@storybook/vue3';
import Button from './Button.vue';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Primary: Story = {
  args: {
    variant: 'primary',
  },
  render: (args) => ({
    components: { Button },
    setup: () => ({ args }),
    template: '<Button v-bind="args">按钮文本</Button>',
  }),
};
```

### 2.4 TypeScript 配置

```json
// tsconfig.json 中包含 stories
{
  "include": ["src/**/*.ts", "src/**/*.tsx", "src/**/*.stories.tsx"]
}
```

## 三、注意事项与常见陷阱

1. **`Meta` 和 `StoryObj` 类型确保故事配置正确**
2. **`argTypes` 的类型与组件 Props 一致**
3. **`autodocs` 自动生成文档页面**
4. **故事文件放在 `src/` 内**：确保被 tsconfig 覆盖
5. **组件类型要完整导出**：Storybook 需要读取 Props 类型
