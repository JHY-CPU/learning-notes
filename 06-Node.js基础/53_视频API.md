# 视频API


## 视频 API


HTML5 Video 事件、play/pause/seek、字幕 track、自定义控件。


## HTML5 Video API


```
// ========== 视频元素 ==========
<video src="video.mp4" controls></video>

// ========== 控制方法 ==========
video.play();              // 播放
video.pause();             // 暂停
video.load();              // 重新加载

// ========== 属性 ==========
video.currentTime          // 当前播放位置 (秒)
video.duration             // 总时长 (秒)
video.volume               // 音量 (0-1)
video.playbackRate         // 播放速度 (0.5-16)
video.paused               // 是否暂停
video.ended                // 是否播放完毕
video.muted                // 是否静音

// ========== 事件 ==========
// loadstart/durationchange/loadedmetadata/loadeddata
// progress/canplay/canplaythrough
// play/pause/playing/ended/waiting/seeking/seeked
// ratechange/volumechange/timeupdate

// ========== 字幕 ==========
<track kind="subtitles" src="sub.vtt" srclang="zh" label="中文">
```


## 演示：视频 API

点击按钮查看


## 自定义视频播放器

```javascript
// ========== 自定义视频播放器 ==========
class VideoPlayer {
    constructor(container, src) {
        this.container = container;
        this.video = document.createElement('video');
        this.video.src = src;
        this.video.preload = 'metadata';
        container.appendChild(this.video);

        this.createControls();
        this.bindEvents();
    }

    createControls() {
        this.controls = document.createElement('div');
        this.controls.className = 'video-controls';
        this.controls.innerHTML = `
            <button class="play-btn">播放</button>
            <input type="range" class="progress" min="0" max="100" value="0">
            <span class="time">0:00 / 0:00</span>
            <input type="range" class="volume" min="0" max="1" step="0.1" value="1">
            <select class="speed">
                <option value="0.5">0.5x</option>
                <option value="1" selected>1x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2x</option>
            </select>
            <button class="fullscreen-btn">全屏</button>
        `;
        this.container.appendChild(this.controls);
    }

    bindEvents() {
        const playBtn = this.controls.querySelector('.play-btn');
        const progress = this.controls.querySelector('.progress');
        const volume = this.controls.querySelector('.volume');
        const speed = this.controls.querySelector('.speed');
        const time = this.controls.querySelector('.time');
        const fullBtn = this.controls.querySelector('.fullscreen-btn');

        playBtn.onclick = () => {
            if (this.video.paused) { this.video.play(); playBtn.textContent = '暂停'; }
            else { this.video.pause(); playBtn.textContent = '播放'; }
        };

        this.video.ontimeupdate = () => {
            const pct = (this.video.currentTime / this.video.duration) * 100;
            progress.value = pct;
            time.textContent = `${this.formatTime(this.video.currentTime)} / ${this.formatTime(this.video.duration)}`;
        };

        progress.oninput = () => {
            this.video.currentTime = (progress.value / 100) * this.video.duration;
        };

        volume.oninput = () => { this.video.volume = volume.value; };
        speed.onchange = () => { this.video.playbackRate = parseFloat(speed.value); };
        fullBtn.onclick = () => { this.container.requestFullscreen(); };
    }

    formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    }
}
```

## 视频截图与画中画

```javascript
// ========== 视频截图 ==========
function captureFrame(video, time) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');

        video.currentTime = time;
        video.onseeked = () => {
            ctx.drawImage(video, 0, 0);
            canvas.toBlob((blob) => resolve(blob), 'image/png');
        };
    });
}

// ========== 画中画 ==========
async function togglePiP(video) {
    if (document.pictureInPictureElement) {
        await document.exitPictureInPicture();
    } else {
        await video.requestPictureInPicture();
    }
}

// ========== 视频流录制 ==========
async function recordVideo(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');

    const stream = canvas.captureStream(30); // 30fps
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks = [];

    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'recording.webm';
        a.click();
    };

    recorder.start();

    // 持续绘制视频帧
    function drawFrame() {
        if (recorder.state !== 'recording') return;
        ctx.drawImage(videoElement, 0, 0);
        requestAnimationFrame(drawFrame);
    }
    drawFrame();

    return { stop: () => recorder.stop() };
}
```

<!-- Converted from: 53_视频API.html -->
