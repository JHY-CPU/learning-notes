# Reusable Workflows

## 一、概念说明

可复用工作流允许在多个工作流中共享通用的CI/CD逻辑，减少重复代码。

## 二、创建可复用工作流

```yaml
# .github/workflows/reusable-test.yml
name: Reusable Test
on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
    secrets:
      NPM_TOKEN:
        required: true

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
      - run: npm install
      - run: npm test
```

## 三、调用可复用工作流

```yaml
# .github/workflows/ci.yml
name: CI
on: [push]

jobs:
  test:
    uses: ./.github/workflows/reusable-test.yml
    with:
      node-version: '20'
    secrets:
      NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## 四、跨仓库复用

```yaml
jobs:
  test:
    uses: my-org/shared-workflows/.github/workflows/test.yml@main
    with:
      node-version: '20'
```

## 五、注意事项

1. **最多嵌套4层**：可复用工作流最多嵌套4层
2. **不能调用自己**：不能递归调用
3. **Secrets传递**：需要显式传递Secrets
4. **版本管理**：使用Git标签管理版本
