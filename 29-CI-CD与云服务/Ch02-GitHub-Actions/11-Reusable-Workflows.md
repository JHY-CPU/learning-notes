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

## 六、完整的可复用工作流设计

```yaml
# .github/workflows/shared-ci.yml
name: Shared CI
on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      run-e2e:
        type: boolean
        default: false
      deploy-environment:
        type: string
        default: ''
    secrets:
      NPM_TOKEN:
        required: false
      DEPLOY_KEY:
        required: false
    outputs:
      version:
        description: 'Package version'
        value: ${{ jobs.build.outputs.version }}
      artifact-url:
        description: 'Build artifact URL'
        value: ${{ jobs.build.outputs.artifact-url }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run lint

  build:
    needs: lint
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
      artifact-url: ${{ steps.upload.outputs.artifact-url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - id: version
        run: echo "version=$(node -p 'require(\"./package.json\").version')" >> $GITHUB_OUTPUT
      - id: upload
        uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/

  e2e:
    needs: build
    if: inputs.run-e2e
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/
      - run: npm run test:e2e

  deploy:
    needs: build
    if: inputs.deploy-environment != ''
    runs-on: ubuntu-latest
    environment: ${{ inputs.deploy-environment }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/
      - run: ./deploy.sh ${{ inputs.deploy-environment }}
```

## 七、组织级共享工作流

```yaml
# my-org/.github/workflows/reusable-deploy.yml (组织仓库)
name: Reusable Deploy
on:
  workflow_call:
    inputs:
      environment:
        type: string
        required: true
      image:
        type: string
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - name: Deploy
        run: echo "Deploying ${{ inputs.image }} to ${{ inputs.environment }}"
```

```yaml
# 调用组织共享工作流
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy-staging:
    uses: my-org/.github/.github/workflows/reusable-deploy.yml@main
    with:
      environment: staging
      image: ghcr.io/my-org/app:${{ github.sha }}
    secrets: inherit
```

## 八、模板库与Action组合

```yaml
# 组合多个可复用工作流
name: Complete Pipeline
on: [push]

jobs:
  ci:
    uses: ./.github/workflows/reusable-ci.yml
    with:
      node-version: '20'
      run-e2e: true

  security:
    uses: ./.github/workflows/reusable-security.yml

  deploy:
    needs: [ci, security]
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: staging
    secrets: inherit
```
