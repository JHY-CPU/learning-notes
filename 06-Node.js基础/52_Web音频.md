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


## 播放音频与可视化

```javascript
// ========== 播放音频文件 ==========
async function playAudio(url) {
    const ctx = new AudioContext();
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start();

    return { source, context: ctx };
}

// ========== 音量控制 ==========
function createGainNode(ctx, volume = 1) {
    const gain = ctx.createGain();
    gain.gain.value = volume;
    return gain;
}

// 淡入淡出
function fadeIn(gainNode, duration = 1) {
    gainNode.gain.setValueAtTime(0, gainNode.context.currentTime);
    gainNode.gain.linearRampToValueAtTime(1, gainNode.context.currentTime + duration);
}

function fadeOut(gainNode, duration = 1) {
    gainNode.gain.setValueAtTime(1, gainNode.context.currentTime);
    gainNode.gain.linearRampToValueAtTime(0, gainNode.context.currentTime + duration);
}

// ========== 音频可视化 ==========
function visualize(audioCtx, source, canvas) {
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    analyser.connect(audioCtx.destination);

    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = canvas.width / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (dataArray[i] / 255) * canvas.height;
            const hue = (i / bufferLength) * 360;
            ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
            ctx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
            x += barWidth;
        }
    }
    draw();
}

// ========== 录音 ==========
async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const chunks = [];

    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
    mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        // 可以创建 <audio> 播放或下载
        const audio = new Audio(url);
        audio.play();
    };

    mediaRecorder.start();
    return {
        stop: () => {
            mediaRecorder.stop();
            stream.getTracks().forEach(t => t.stop());
        }
    };
}
```

## 音效处理

```javascript
// ========== 滤波器 ==========
function createLowPassFilter(ctx, frequency = 1000) {
    const filter = ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = frequency;
    filter.Q.value = 1;
    return filter;
}

function createHighPassFilter(ctx, frequency = 1000) {
    const filter = ctx.createBiquadFilter();
    filter.type = 'highpass';
    filter.frequency.value = frequency;
    return filter;
}

// ========== 混响效果 ==========
async function createReverb(ctx, duration = 2, decay = 2) {
    const sampleRate = ctx.sampleRate;
    const length = sampleRate * duration;
    const impulse = ctx.createBuffer(2, length, sampleRate);

    for (let channel = 0; channel < 2; channel++) {
        const channelData = impulse.getChannelData(channel);
        for (let i = 0; i < length; i++) {
            channelData[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
        }
    }

    const convolver = ctx.createConvolver();
    convolver.buffer = impulse;
    return convolver;
}

// 使用:
// source.connect(reverb).connect(ctx.destination);

// ========== 播放器架构 ==========
// source → gain(音量) → filter(滤波) → analyser(分析) → destination(输出)
```

<!-- Converted from: 52_Web音频.html -->
