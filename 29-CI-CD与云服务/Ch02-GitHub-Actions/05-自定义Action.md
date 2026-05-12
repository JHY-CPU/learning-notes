# 自定义Action

## 一、概念说明

自定义Action允许封装可复用的逻辑，支持JavaScript、Docker和Composite三种类型。

## 二、JavaScript Action

```yaml
# action.yml
name: 'Hello Action'
description: 'Greet someone'
inputs:
  who-to-greet:
    description: 'Who to greet'
    required: true
    default: 'World'
outputs:
  time:
    description: 'The time we greeted you'
runs:
  using: 'node20'
  main: 'index.js'
```

```javascript
// index.js
const core = require('@actions/core');

const whoToGreet = core.getInput('who-to-greet');
console.log(`Hello ${whoToGreet}!`);

const time = new Date().toTimeString();
core.setOutput('time', time);
```

## 三、Docker Action

```yaml
# action.yml
name: 'Docker Action'
runs:
  using: 'docker'
  image: 'Dockerfile'
  args:
    - ${{ inputs.name }}
```

```dockerfile
FROM alpine:3.18
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

## 四、Composite Action

```yaml
# action.yml
name: 'Composite Action'
runs:
  using: 'composite'
  steps:
    - run: echo "Step 1"
      shell: bash
    - run: echo "Step 2"
      shell: bash
```

## 五、使用自定义Action

```yaml
- uses: ./path/to/action
  with:
    who-to-greet: 'GitHub Actions'
```

## 六、注意事项

1. **版本发布**：使用Git标签发布版本
2. **文档完善**：编写清晰的README
3. **输入验证**：验证所有输入参数
4. **测试充分**：充分测试Action功能

## 七、JavaScript Action完整示例

```javascript
// index.js
const core = require('@actions/core');
const github = require('@actions/github');
const exec = require('@actions/exec');

async function run() {
  try {
    // 获取输入
    const name = core.getInput('name', { required: true });
    const greeting = core.getInput('greeting') || 'Hello';
    const debug = core.getBooleanInput('debug');

    // 执行命令
    let output = '';
    await exec.exec('date', [], {
      listeners: {
        stdout: (data) => { output += data.toString(); }
      }
    });

    // 设置输出
    core.setOutput('time', output.trim());
    core.setOutput('greeting', `${greeting}, ${name}!`);

    // 设置环境变量
    core.exportVariable('MY_VAR', 'value');

    // 添加到路径
    core.addPath('/custom/path');

    // 日志
    core.info(`Greeting: ${greeting}, ${name}`);
    core.debug('Debug message');
    core.warning('Warning message');
    core.notice('Notice message');

    // 上传工件等高级操作
    if (debug) {
      core.startGroup('Debug Info');
      core.info(JSON.stringify(github.context, null, 2));
      core.endGroup();
    }

  } catch (error) {
    core.setFailed(error.message);
  }
}

run();
```

```yaml
# action.yml
name: 'Greeting Action'
description: 'A comprehensive example action'
author: 'Your Name'

inputs:
  name:
    description: 'Name to greet'
    required: true
  greeting:
    description: 'Greeting word'
    required: false
    default: 'Hello'
  debug:
    description: 'Enable debug mode'
    required: false
    default: 'false'

outputs:
  greeting:
    description: 'The full greeting message'
  time:
    description: 'Current time'

runs:
  using: 'node20'
  main: 'dist/index.js'  # 使用打包后的文件

branding:
  icon: 'smile'
  color: 'blue'
```

## 八、Composite Action示例

```yaml
# action.yml
name: 'Setup and Test'
description: 'Setup environment and run tests'

inputs:
  node-version:
    description: 'Node.js version'
    required: false
    default: '20'
  test-command:
    description: 'Test command to run'
    required: false
    default: 'npm test'

outputs:
  test-result:
    description: 'Test result'
    value: ${{ steps.test.outputs.result }}

runs:
  using: 'composite'
  steps:
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
        cache: 'npm'

    - name: Install dependencies
      shell: bash
      run: npm ci

    - name: Run tests
      id: test
      shell: bash
      run: |
        ${{ inputs.test-command }}
        echo "result=success" >> $GITHUB_OUTPUT
```

## 九、Docker Action示例

```yaml
# action.yml
name: 'Docker Action'
description: 'Action running in Docker'
inputs:
  filename:
    description: 'File to process'
    required: true
outputs:
  result:
    description: 'Processing result'
runs:
  using: 'docker'
  image: 'Dockerfile'
  args:
    - ${{ inputs.filename }}
```

```dockerfile
# Dockerfile
FROM alpine:3.18
RUN apk add --no-cache curl jq
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

```bash
#!/bin/sh
# entrypoint.sh
FILENAME="$1"
echo "Processing: $FILENAME"
RESULT=$(cat "$FILENAME" | jq length)
echo "result=$RESULT" >> $GITHUB_OUTPUT
```

## 十、Action测试与发布

```yaml
# .github/workflows/test-action.yml
name: Test Action
on:
  push:
    paths:
      - 'action.yml'
      - 'index.js'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./
        with:
          name: 'World'
        id: greeting
      - run: echo "${{ steps.greeting.outputs.greeting }}"

  release:
    needs: test
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4
      - uses: softprops/action-gh-release@v1
```
