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
