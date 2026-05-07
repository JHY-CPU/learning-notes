# Vue+TS测试

## 一、概念说明

Vue 组件测试使用 Vitest + Vue Test Utils，两者都提供了完整的 TypeScript 支持。测试中需要正确配置组件挂载、模拟函数和断言的类型。

## 二、具体用法

### 2.1 测试配置

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import vue from '@vitejs/plugin-vue';

export default defineConfig({
  plugins: [vue()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
});

// src/test/setup.ts
import { config } from '@vue/test-utils';
import { vi } from 'vitest';

// 全局配置
config.global.mocks = {
  $t: (key: string) => key, // i18n mock
};
```

### 2.2 基本组件测试

```typescript
import { mount } from '@vue/test-utils';
import { describe, it, expect } from 'vitest';
import MyButton from '../components/MyButton.vue';

describe('MyButton', () => {
  it('应该渲染文本', () => {
    const wrapper = mount(MyButton, {
      props: {
        variant: 'primary',
        disabled: false,
      },
      slots: {
        default: '点击我',
      },
    });

    expect(wrapper.text()).toContain('点击我');
    expect(wrapper.classes()).toContain('btn-primary');
  });

  it('应该触发点击事件', async () => {
    const wrapper = mount(MyButton, {
      props: { variant: 'primary' },
    });

    await wrapper.trigger('click');

    expect(wrapper.emitted('click')).toHaveLength(1);
  });

  it('disabled 状态不应该触发事件', async () => {
    const wrapper = mount(MyButton, {
      props: { variant: 'primary', disabled: true },
    });

    await wrapper.trigger('click');

    expect(wrapper.emitted('click')).toBeUndefined();
  });
});
```

### 2.3 测试 emits 类型

```typescript
import { mount } from '@vue/test-utils';
import FormInput from '../components/FormInput.vue';

it('应该 emit 正确类型的值', async () => {
  const wrapper = mount(FormInput, {
    props: {
      modelValue: '',
      label: '用户名',
    },
  });

  await wrapper.find('input').setValue('张三');

  const emitted = wrapper.emitted();
  expect(emitted['update:modelValue']).toHaveLength(1);
  expect(emitted['update:modelValue'][0]).toEqual(['张三']);
});
```

### 2.4 测试 Pinia Store

```typescript
import { setActivePinia, createPinia } from 'pinia';
import { useCounterStore } from '../stores/counter';

describe('Counter Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia());
  });

  it('应该正确增加计数', () => {
    const store = useCounterStore();

    expect(store.count).toBe(0);
    store.increment();
    expect(store.count).toBe(1);
    expect(store.doubleCount).toBe(2);
  });

  it('应该正确重置', () => {
    const store = useCounterStore();
    store.increment();
    store.increment();
    store.reset();
    expect(store.count).toBe(0);
  });
});
```

### 2.5 测试异步组件

```typescript
import { flushPromises, mount } from '@vue/test-utils';
import UserProfile from '../components/UserProfile.vue';

it('应该加载并显示用户数据', async () => {
  const wrapper = mount(UserProfile, {
    props: { userId: 1 },
  });

  // 初始状态：加载中
  expect(wrapper.find('.loading').exists()).toBe(true);

  // 等待异步操作完成
  await flushPromises();

  // 数据加载完成
  expect(wrapper.find('.loading').exists()).toBe(false);
  expect(wrapper.find('.user-name').text()).toBe('张三');
});
```

### 2.6 自定义 Wrapper 类型

```typescript
import { mount, VueWrapper } from '@vue/test-utils';

function mountWithProviders(component: any, options = {}): VueWrapper {
  return mount(component, {
    global: {
      plugins: [createPinia()],
      stubs: {
        RouterLink: true,
        RouterView: true,
      },
    },
    ...options,
  });
}
```

## 三、注意事项与常见陷阱

1. **`wrapper.emitted()` 返回值是数组的数组**：每个事件触发对应一个数组
2. **`flushPromises()` 等待所有微任务完成**：异步测试必须使用
3. **`mount` 的 `props` 类型与组件一致**：类型不匹配会报错
4. **全局 mock 在 setup 文件中配置**：避免每个测试重复设置
5. **使用 `vi.mocked()` 处理 mock 类型**：确保 mock 的类型安全
