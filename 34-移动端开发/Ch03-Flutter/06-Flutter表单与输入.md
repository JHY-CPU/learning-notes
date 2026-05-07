# Flutter 表单与输入

## 一、概念说明

Flutter 提供了丰富的表单组件和验证机制，支持文本输入、选择器、开关等常用表单元素。Form 和 TextFormField 组合提供了完整的表单验证能力。

```dart
// 表单基础
class FormBasics extends StatefulWidget {
  const FormBasics({super.key});

  @override
  State<FormBasics> createState() => _FormBasicsState();
}

class _FormBasicsState extends State<FormBasics> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    super.dispose();
  }

  void _submit() {
    if (_formKey.currentState!.validate()) {
      // 表单验证通过
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('提交成功')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          TextFormField(
            controller: _nameController,
            decoration: const InputDecoration(
              labelText: '姓名',
              hintText: '请输入姓名',
              prefixIcon: Icon(Icons.person),
              border: OutlineInputBorder(),
            ),
            validator: (value) {
              if (value == null || value.isEmpty) return '姓名不能为空';
              if (value.length < 2) return '姓名至少2个字符';
              return null;
            },
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _emailController,
            decoration: const InputDecoration(
              labelText: '邮箱',
              prefixIcon: Icon(Icons.email),
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.emailAddress,
            validator: (value) {
              if (value == null || value.isEmpty) return '请输入邮箱';
              if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$').hasMatch(value)) {
                return '邮箱格式不正确';
              }
              return null;
            },
          ),
          const SizedBox(height: 24),
          ElevatedButton(onPressed: _submit, child: const Text('提交')),
        ],
      ),
    );
  }
}
```

## 二、输入组件详解

### 2.1 TextField 详解

```dart
class TextFieldDemo extends StatefulWidget {
  const TextFieldDemo({super.key});

  @override
  State<TextFieldDemo> createState() => _TextFieldDemoState();
}

class _TextFieldDemoState extends State<TextFieldDemo> {
  final _controller = TextEditingController();
  bool _obscure = true;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // 普通输入
        TextField(
          controller: _controller,
          decoration: InputDecoration(
            labelText: '用户名',
            hintText: '请输入用户名',
            helperText: '3-20个字符',
            prefixIcon: const Icon(Icons.person),
            suffixIcon: IconButton(
              icon: const Icon(Icons.clear),
              onPressed: () => _controller.clear(),
            ),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
          ),
          onChanged: (value) => print('输入: $value'),
          onSubmitted: (value) => print('提交: $value'),
        ),

        const SizedBox(height: 16),

        // 密码输入
        TextField(
          obscureText: _obscure,
          decoration: InputDecoration(
            labelText: '密码',
            prefixIcon: const Icon(Icons.lock),
            suffixIcon: IconButton(
              icon: Icon(_obscure ? Icons.visibility_off : Icons.visibility),
              onPressed: () => setState(() => _obscure = !_obscure),
            ),
            border: const OutlineInputBorder(),
          ),
        ),

        const SizedBox(height: 16),

        // 多行输入
        const TextField(
          maxLines: 5,
          maxLength: 500,
          decoration: InputDecoration(
            labelText: '详细描述',
            alignLabelWithHint: true,
            border: OutlineInputBorder(),
          ),
        ),
      ],
    );
  }
}
```

### 2.2 选择器

```dart
class PickerDemo extends StatefulWidget {
  const PickerDemo({super.key});

  @override
  State<PickerDemo> createState() => _PickerDemoState();
}

class _PickerDemoState extends State<PickerDemo> {
  DateTime? _selectedDate;
  TimeOfDay? _selectedTime;
  String? _selectedValue;
  final _items = ['选项一', '选项二', '选项三', '选项四'];

  Future<void> _pickDate() async {
    final date = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(2020),
      lastDate: DateTime(2030),
    );
    if (date != null) setState(() => _selectedDate = date);
  }

  Future<void> _pickTime() async {
    final time = await showTimePicker(
      context: context,
      initialTime: TimeOfDay.now(),
    );
    if (time != null) setState(() => _selectedTime = time);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // 日期选择
        ListTile(
          leading: const Icon(Icons.calendar_today),
          title: Text(_selectedDate?.toString().split(' ')[0] ?? '选择日期'),
          onTap: _pickDate,
        ),

        // 时间选择
        ListTile(
          leading: const Icon(Icons.access_time),
          title: Text(_selectedTime?.format(context) ?? '选择时间'),
          onTap: _pickTime,
        ),

        // 下拉选择
        DropdownButtonFormField<String>(
          value: _selectedValue,
          decoration: const InputDecoration(
            labelText: '选择选项',
            border: OutlineInputBorder(),
          ),
          items: _items.map((item) => DropdownMenuItem(
            value: item,
            child: Text(item),
          )).toList(),
          onChanged: (value) => setState(() => _selectedValue = value),
          validator: (value) => value == null ? '请选择' : null,
        ),
      ],
    );
  }
}
```

### 2.3 开关和复选框

```dart
class ToggleDemo extends StatefulWidget {
  const ToggleDemo({super.key});

  @override
  State<ToggleDemo> createState() => _ToggleDemoState();
}

class _ToggleDemoState extends State<ToggleDemo> {
  bool _switchValue = false;
  bool _checkboxValue = false;
  int _radioValue = 0;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // Switch 开关
        SwitchListTile(
          title: const Text('深色模式'),
          subtitle: const Text('切换应用主题'),
          secondary: const Icon(Icons.dark_mode),
          value: _switchValue,
          onChanged: (value) => setState(() => _switchValue = value),
        ),

        // Checkbox 复选框
        CheckboxListTile(
          title: const Text('同意服务条款'),
          value: _checkboxValue,
          onChanged: (value) => setState(() => _checkboxValue = value ?? false),
        ),

        // Radio 单选按钮
        ...List.generate(3, (index) {
          return RadioListTile<int>(
            title: Text('选项 ${index + 1}'),
            value: index,
            groupValue: _radioValue,
            onChanged: (value) => setState(() => _radioValue = value!),
          );
        }),
      ],
    );
  }
}
```

## 三、自定义表单验证

```dart
// 自定义验证器
class Validators {
  static String? required(String? value, [String? fieldName]) {
    if (value == null || value.trim().isEmpty) {
      return '${fieldName ?? '此字段'}不能为空';
    }
    return null;
  }

  static String? Function(String?) minLength(int min, [String? fieldName]) {
    return (value) {
      if (value != null && value.length < min) {
        return '${fieldName ?? '输入'}至少 $min 个字符';
      }
      return null;
    };
  }

  static String? Function(String?) combine(List<String? Function(String?)> validators) {
    return (value) {
      for (final validator in validators) {
        final error = validator(value);
        if (error != null) return error;
      }
      return null;
    };
  }
}

// 使用
TextFormField(
  validator: Validators.combine([
    (v) => Validators.required(v, '用户名'),
    Validators.minLength(3, '用户名'),
  ]),
);
```

## 四、注意事项与常见陷阱

1. **TextEditingController 释放**：在 dispose 中调用 dispose()，避免内存泄漏
2. **表单验证时机**：在提交时统一验证，而非每次输入都验证
3. **键盘类型选择**：根据输入内容选择合适的 keyboardType
4. **输入法兼容**：中文输入法的合成阶段会产生中间文本，onChanged 可能触发多次
5. **焦点管理**：使用 FocusNode 和 FocusScope 管理输入焦点
