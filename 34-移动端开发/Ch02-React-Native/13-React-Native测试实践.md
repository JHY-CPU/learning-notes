# React Native 测试实践

## 一、概念说明

测试是保证 React Native 应用质量的关键环节。完整的测试策略包括单元测试、组件测试、集成测试和端到端测试。

```javascript
// Jest 配置 (jest.config.js)
module.exports = {
  preset: 'react-native',
  setupFilesAfterSetup: ['@testing-library/jest-native/extend-expect'],
  transformIgnorePatterns: [
    'node_modules/(?!(react-native|@react-native|@react-navigation)/)',
  ],
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/index.js',
  ],
  coverageThreshold: {
    global: { branches: 70, functions: 70, lines: 70, statements: 70 },
  },
};
```

## 二、单元测试

### 2.1 工具函数测试

```javascript
// utils/format.js
export const formatPrice = (price) => {
  if (typeof price !== 'number') return '¥0.00';
  return `¥${price.toFixed(2)}`;
};

export const formatDate = (date) => {
  const d = new Date(date);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

export const truncateText = (text, maxLength = 50) => {
  if (!text) return '';
  return text.length > maxLength ? text.slice(0, maxLength) + '...' : text;
};

// utils/__tests__/format.test.js
import { formatPrice, formatDate, truncateText } from '../format';

describe('formatPrice', () => {
  it('格式化正数', () => {
    expect(formatPrice(100)).toBe('¥100.00');
    expect(formatPrice(99.9)).toBe('¥99.90');
  });

  it('处理零值', () => {
    expect(formatPrice(0)).toBe('¥0.00');
  });

  it('处理无效输入', () => {
    expect(formatPrice(null)).toBe('¥0.00');
    expect(formatPrice(undefined)).toBe('¥0.00');
    expect(formatPrice('abc')).toBe('¥0.00');
  });
});

describe('truncateText', () => {
  it('短文本不截断', () => {
    expect(truncateText('你好')).toBe('你好');
  });

  it('长文本截断并添加省略号', () => {
    const longText = 'a'.repeat(60);
    expect(truncateText(longText, 50)).toBe('a'.repeat(50) + '...');
  });

  it('处理空值', () => {
    expect(truncateText(null)).toBe('');
    expect(truncateText(undefined)).toBe('');
  });
});
```

### 2.2 Redux 测试

```javascript
// store/__tests__/userSlice.test.js
import userReducer, { setUser, logout, fetchUser } from '../userSlice';

describe('userSlice', () => {
  const initialState = { data: null, loading: false, error: null };

  it('应该返回初始状态', () => {
    expect(userReducer(undefined, { type: 'unknown' })).toEqual(initialState);
  });

  it('应该设置用户', () => {
    const user = { id: 1, name: '张三' };
    const actual = userReducer(initialState, setUser(user));
    expect(actual.data).toEqual(user);
  });

  it('应该登出', () => {
    const state = { data: { id: 1 }, loading: false, error: null };
    const actual = userReducer(state, logout());
    expect(actual.data).toBeNull();
  });

  it('应该处理 fetchUser.pending', () => {
    const actual = userReducer(initialState, fetchUser.pending('1'));
    expect(actual.loading).toBe(true);
    expect(actual.error).toBeNull();
  });

  it('应该处理 fetchUser.fulfilled', () => {
    const user = { id: 1, name: '张三' };
    const actual = userReducer(initialState, fetchUser.fulfilled(user, '1'));
    expect(actual.loading).toBe(false);
    expect(actual.data).toEqual(user);
  });

  it('应该处理 fetchUser.rejected', () => {
    const error = '请求失败';
    const actual = userReducer(initialState, fetchUser.rejected(null, '1', null, error));
    expect(actual.loading).toBe(false);
    expect(actual.error).toBe(error);
  });
});
```

## 三、组件测试

```javascript
// components/__tests__/Button.test.js
import { render, fireEvent } from '@testing-library/react-native';
import Button from '../Button';

describe('Button', () => {
  it('应该渲染按钮文本', () => {
    const { getByText } = render(<Button title="提交" onPress={() => {}} />);
    expect(getByText('提交')).toBeTruthy();
  });

  it('应该响应点击事件', () => {
    const onPress = jest.fn();
    const { getByText } = render(<Button title="提交" onPress={onPress} />);

    fireEvent.press(getByText('提交'));
    expect(onPress).toHaveBeenCalledTimes(1);
  });

  it('禁用状态不应该响应点击', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <Button title="提交" onPress={onPress} disabled={true} />
    );

    fireEvent.press(getByText('提交'));
    expect(onPress).not.toHaveBeenCalled();
  });

  it('应该显示加载状态', () => {
    const { getByTestId } = render(
      <Button title="提交" onPress={() => {}} loading={true} />
    );
    expect(getByTestId('loading-indicator')).toBeTruthy();
  });
});
```

```javascript
// screens/__tests__/LoginScreen.test.js
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import LoginScreen from '../LoginScreen';
import { AuthProvider } from '../../contexts/AuthContext';

const renderWithProvider = (component) => {
  return render(<AuthProvider>{component}</AuthProvider>);
};

describe('LoginScreen', () => {
  it('应该渲染登录表单', () => {
    const { getByPlaceholderText, getByText } = renderWithProvider(<LoginScreen />);
    expect(getByPlaceholderText('邮箱')).toBeTruthy();
    expect(getByPlaceholderText('密码')).toBeTruthy();
    expect(getByText('登录')).toBeTruthy();
  });

  it('空表单提交应该显示错误', async () => {
    const { getByText, findByText } = renderWithProvider(<LoginScreen />);
    fireEvent.press(getByText('登录'));

    await waitFor(async () => {
      expect(await findByText('请输入邮箱')).toBeTruthy();
    });
  });

  it('有效表单应该调用登录 API', async () => {
    const mockLogin = jest.fn().mockResolvedValue({ success: true });
    const { getByPlaceholderText, getByText } = renderWithProvider(
      <LoginScreen onLogin={mockLogin} />
    );

    fireEvent.changeText(getByPlaceholderText('邮箱'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('密码'), 'password123');
    fireEvent.press(getByText('登录'));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });
});
```

## 四、端到端测试

```javascript
// e2e/login.e2e.js (Detox)
describe('登录流程', () => {
  beforeAll(async () => {
    await device.launchApp({ newInstance: true });
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('应该成功登录', async () => {
    await element(by.id('email-input')).typeText('test@example.com');
    await element(by.id('password-input')).typeText('password123');
    await element(by.id('login-button')).tap();

    await expect(element(by.id('home-screen'))).toBeVisible();
  });

  it('应该显示登录错误', async () => {
    await element(by.id('email-input')).typeText('wrong@email.com');
    await element(by.id('password-input')).typeText('wrong');
    await element(by.id('login-button')).tap();

    await expect(element(by.text('登录失败'))).toBeVisible();
  });
});
```

## 五、注意事项与常见陷阱

1. **测试隔离**：每个测试应该独立，不依赖其他测试的状态
2. **Mock 适度**：只 Mock 外部依赖，不要 Mock 被测试对象本身
3. **异步测试**：正确使用 waitFor 和 findBy 处理异步操作
4. **快照测试谨慎使用**：快照测试对 UI 变更敏感，维护成本高
5. **真机 E2E 测试**：端到端测试应在真机上运行，模拟器无法完全模拟真实环境
