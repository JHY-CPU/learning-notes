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


<!-- Converted from: 53_视频API.html -->
