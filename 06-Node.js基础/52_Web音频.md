# Web音频


## Web 音频


AudioContext、OscillatorNode、GainNode、音频可视化、录音。


## Web Audio API


```
// ========== AudioContext ==========
const ctx = new AudioContext();          // 创建音频上下文
const ctx2 = new AudioContext({          // 可选配置
    sampleRate: 44100,                   // 采样率 (22050/44100/48000)
    latencyHint: 'interactive'           // 'balanced' | 'interactive' | 'playback'
});

// ========== 音频节点 ==========
// OscillatorNode — 音调发生器
// GainNode — 音量控制
// AnalyserNode — 频谱/波形分析
// BiquadFilterNode — 滤波器 (lowpass/highpass/bandpass)
// DelayNode — 延迟效果
// ConvolverNode — 卷积混响
// AudioBufferSourceNode — 播放音频缓冲
// MediaStreamSourceNode — 麦克风输入

// ========== 音频节点连接 ==========
// source.connect(destination);
// source.connect(gain).connect(destination);
// source.connect(analyser); analyser.connect(destination);

// ========== 录音 ==========
// navigator.mediaDevices.getUserMedia({ audio: true })
// MediaRecorder — 录音控制
```


## 演示：Web 音频

点击按钮查看


<!-- Converted from: 52_Web音频.html -->
