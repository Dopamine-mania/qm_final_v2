#!/usr/bin/env python3
"""
06 - 音乐生成工作坊
展示AI音乐生成在睡眠治疗中的应用
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import json
import wave
import struct
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MusicTheoryEngine:
    """音乐理论引擎"""
    
    def __init__(self):
        # 音符频率表 (A4 = 440Hz)
        self.note_frequencies = {
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
            'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
            'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
            'G5': 783.99, 'A5': 880.00, 'B5': 987.77
        }
        
        # 和弦定义
        self.chord_patterns = {
            'major': [0, 4, 7],      # 大三和弦
            'minor': [0, 3, 7],      # 小三和弦
            'dim': [0, 3, 6],        # 减三和弦
            'sus4': [0, 5, 7],       # 挂四和弦
            'maj7': [0, 4, 7, 11],   # 大七和弦
            'min7': [0, 3, 7, 10]    # 小七和弦
        }
        
        # 睡眠音乐和弦进行
        self.sleep_progressions = [
            ['C', 'Am', 'F', 'G'],           # I-vi-IV-V
            ['Am', 'F', 'C', 'G'],           # vi-IV-I-V
            ['C', 'G', 'Am', 'F'],           # I-V-vi-IV
            ['F', 'G', 'C', 'Am'],           # IV-V-I-vi
            ['Dm', 'G', 'C', 'F'],           # ii-V-I-IV
            ['C', 'Em', 'F', 'C']            # I-iii-IV-I
        ]
    
    def get_scale_notes(self, root='C', scale_type='major'):
        """获取音阶音符"""
        # 简化版本：只实现大调和自然小调
        intervals = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10]
        }
        
        # 获取根音索引
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_idx = note_names.index(root)
        
        # 生成音阶
        scale = []
        for interval in intervals[scale_type]:
            note_idx = (root_idx + interval) % 12
            scale.append(note_names[note_idx])
        
        return scale
    
    def generate_melody(self, scale_notes, num_bars=8, notes_per_bar=4):
        """生成旋律"""
        melody = []
        
        # 使用简单的规则生成舒缓的旋律
        for bar in range(num_bars):
            for beat in range(notes_per_bar):
                if beat == 0:  # 强拍
                    # 倾向于使用根音、三音、五音
                    note_idx = np.random.choice([0, 2, 4], p=[0.5, 0.3, 0.2])
                else:
                    # 弱拍使用临近音符
                    if melody:
                        last_idx = scale_notes.index(melody[-1]['note']) if melody[-1]['note'] in scale_notes else 0
                        # 上下级进
                        step = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.3, 0.2, 0.3, 0.1])
                        note_idx = (last_idx + step) % len(scale_notes)
                    else:
                        note_idx = np.random.choice(range(len(scale_notes)))
                
                # 添加音符
                melody.append({
                    'note': scale_notes[note_idx],
                    'duration': 0.5 if beat % 2 == 0 else 0.25,  # 节奏变化
                    'velocity': 0.7 - bar * 0.05  # 渐弱
                })
        
        return melody

class SleepMusicGenerator:
    """睡眠音乐生成器"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.theory_engine = MusicTheoryEngine()
    
    def generate_sine_wave(self, frequency, duration, amplitude=0.5):
        """生成正弦波"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def apply_envelope(self, signal, attack=0.01, decay=0.1, sustain=0.7, release=0.2):
        """应用ADSR包络"""
        total_length = len(signal)
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        sustain_samples = total_length - attack_samples - decay_samples - release_samples
        
        envelope = np.ones_like(signal)
        
        # Attack
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay
        decay_start = attack_samples
        decay_end = decay_start + decay_samples
        envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_samples)
        
        # Sustain
        sustain_start = decay_end
        sustain_end = sustain_start + sustain_samples
        envelope[sustain_start:sustain_end] = sustain
        
        # Release
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
        
        return signal * envelope
    
    def add_reverb(self, signal, delay=0.02, decay=0.5):
        """添加简单混响效果"""
        delay_samples = int(delay * self.sample_rate)
        reverb = np.zeros_like(signal)
        
        # 添加多个延迟回声
        for i in range(3):
            delay_time = delay_samples * (i + 1)
            decay_factor = decay ** (i + 1)
            if delay_time < len(signal):
                reverb[delay_time:] += signal[:-delay_time] * decay_factor
        
        return signal + reverb * 0.3
    
    def synthesize_note(self, note_name, duration, octave=4):
        """合成单个音符"""
        # 获取基础频率
        base_note = note_name + str(octave)
        if base_note in self.theory_engine.note_frequencies:
            frequency = self.theory_engine.note_frequencies[base_note]
        else:
            # 计算频率
            frequency = 440.0  # 默认A4
        
        # 生成基础波形
        signal = self.generate_sine_wave(frequency, duration)
        
        # 添加泛音
        signal += self.generate_sine_wave(frequency * 2, duration, 0.2)  # 第二泛音
        signal += self.generate_sine_wave(frequency * 3, duration, 0.1)  # 第三泛音
        
        # 应用包络
        signal = self.apply_envelope(signal)
        
        return signal
    
    def generate_sleep_track(self, duration_minutes=2, bpm=60, key='C', mode='major'):
        """生成睡眠音乐轨道"""
        print(f"\n🎵 生成睡眠音乐")
        print(f"  时长: {duration_minutes}分钟")
        print(f"  速度: {bpm} BPM")
        print(f"  调性: {key} {mode}")
        
        # 计算参数
        beat_duration = 60.0 / bpm
        total_beats = int(duration_minutes * 60 / beat_duration)
        
        # 获取音阶
        scale_notes = self.theory_engine.get_scale_notes(key, mode)
        
        # 生成和弦进行
        progression = self.theory_engine.sleep_progressions[0]  # 使用第一个进行
        
        # 初始化音轨
        track_length = int(duration_minutes * 60 * self.sample_rate)
        track = np.zeros(track_length)
        
        # 生成旋律
        melody = self.theory_engine.generate_melody(scale_notes, num_bars=total_beats//4)
        
        # 合成音符
        current_pos = 0
        for i, note_info in enumerate(melody):
            if current_pos >= track_length:
                break
            
            # 合成音符
            note_duration = note_info['duration'] * beat_duration
            note_signal = self.synthesize_note(
                note_info['note'], 
                note_duration,
                octave=4 + (i % 2)  # 音高变化
            )
            
            # 调整音量
            note_signal *= note_info['velocity'] * 0.3
            
            # 添加到音轨
            end_pos = min(current_pos + len(note_signal), track_length)
            track[current_pos:end_pos] += note_signal[:end_pos-current_pos]
            
            current_pos += int(note_duration * self.sample_rate * 0.9)  # 轻微重叠
        
        # 添加和弦垫底
        chord_track = self.generate_chord_pad(progression, duration_minutes, bpm, key)
        track += chord_track * 0.2
        
        # 后处理
        track = self.add_reverb(track)
        track = self.apply_fade(track, fade_in=5, fade_out=10)
        
        # 归一化
        track = track / (np.max(np.abs(track)) + 1e-6) * 0.8
        
        return track
    
    def generate_chord_pad(self, progression, duration_minutes, bpm, key):
        """生成和弦垫底"""
        track_length = int(duration_minutes * 60 * self.sample_rate)
        pad = np.zeros(track_length)
        
        # 每个和弦的持续时间
        chord_duration = 60.0 / bpm * 4  # 4拍一个和弦
        samples_per_chord = int(chord_duration * self.sample_rate)
        
        current_pos = 0
        chord_idx = 0
        
        while current_pos < track_length:
            # 获取当前和弦
            chord_name = progression[chord_idx % len(progression)]
            
            # 生成和弦音符（简化版本）
            if 'm' in chord_name:
                chord_type = 'minor'
                root = chord_name.replace('m', '')
            else:
                chord_type = 'major'
                root = chord_name
            
            # 生成和弦
            chord_signal = np.zeros(samples_per_chord)
            
            # 根音
            root_freq = self.theory_engine.note_frequencies.get(root + '3', 261.63)
            chord_signal += self.generate_sine_wave(root_freq, chord_duration, 0.5)
            
            # 三音
            third_freq = root_freq * (1.25 if chord_type == 'major' else 1.2)
            chord_signal += self.generate_sine_wave(third_freq, chord_duration, 0.3)
            
            # 五音
            fifth_freq = root_freq * 1.5
            chord_signal += self.generate_sine_wave(fifth_freq, chord_duration, 0.3)
            
            # 应用包络
            chord_signal = self.apply_envelope(chord_signal, attack=0.5, release=0.5)
            
            # 添加到垫底轨道
            end_pos = min(current_pos + len(chord_signal), track_length)
            pad[current_pos:end_pos] += chord_signal[:end_pos-current_pos]
            
            current_pos += samples_per_chord
            chord_idx += 1
        
        return pad
    
    def apply_fade(self, signal, fade_in=2, fade_out=5):
        """应用淡入淡出"""
        fade_in_samples = int(fade_in * self.sample_rate)
        fade_out_samples = int(fade_out * self.sample_rate)
        
        # 淡入
        if fade_in_samples > 0:
            signal[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples)
        
        # 淡出
        if fade_out_samples > 0:
            signal[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples)
        
        return signal
    
    def save_audio(self, signal, filename):
        """保存音频文件"""
        # 转换为16位整数
        signal_int = np.int16(signal * 32767)
        
        # 写入WAV文件
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(signal_int.tobytes())
    
    def visualize_waveform(self, signal, title="Waveform", output_path=None):
        """可视化波形"""
        plt.figure(figsize=(12, 4))
        
        # 降采样以便显示
        downsample = max(1, len(signal) // 10000)
        time_axis = np.arange(0, len(signal), downsample) / self.sample_rate
        signal_display = signal[::downsample]
        
        plt.plot(time_axis, signal_display, linewidth=0.5)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
        plt.close()

def run_music_workshop():
    """运行音乐生成工作坊"""
    print("《心境流转》音乐生成工作坊")
    print("=" * 50)
    print(f"开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    # 创建生成器
    generator = SleepMusicGenerator()
    
    # 创建输出目录
    output_dir = Path("outputs/music")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试不同参数的音乐生成
    test_configs = [
        {
            "name": "深度放松",
            "duration": 1,  # 1分钟演示
            "bpm": 50,
            "key": "C",
            "mode": "major",
            "description": "极慢速度，C大调，适合深度放松"
        },
        {
            "name": "温柔入眠",
            "duration": 1,
            "bpm": 60,
            "key": "F",
            "mode": "major",
            "description": "缓慢节奏，F大调，营造温暖氛围"
        },
        {
            "name": "宁静冥想",
            "duration": 1,
            "bpm": 55,
            "key": "A",
            "mode": "minor",
            "description": "中慢速度，A小调，适合冥想"
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*40}")
        print(f"🎼 生成: {config['name']}")
        print(f"说明: {config['description']}")
        
        # 生成音乐
        track = generator.generate_sleep_track(
            duration_minutes=config['duration'],
            bpm=config['bpm'],
            key=config['key'],
            mode=config['mode']
        )
        
        # 保存音频
        audio_file = output_dir / f"{config['name'].replace(' ', '_')}.wav"
        generator.save_audio(track, str(audio_file))
        print(f"✅ 音频已保存: {audio_file}")
        
        # 生成波形图
        waveform_file = output_dir / f"{config['name'].replace(' ', '_')}_waveform.png"
        generator.visualize_waveform(
            track[:int(10 * generator.sample_rate)],  # 前10秒
            title=f"{config['name']} - Waveform",
            output_path=str(waveform_file)
        )
        print(f"✅ 波形图已保存: {waveform_file}")
        
        # 分析音频特征
        features = analyze_audio_features(track, generator.sample_rate)
        
        # 记录结果
        results.append({
            "name": config['name'],
            "config": config,
            "audio_file": str(audio_file),
            "waveform_file": str(waveform_file),
            "features": features
        })
    
    # 显示分析结果
    print_audio_analysis(results)
    
    # 保存工作坊结果
    save_workshop_results(results)
    
    return results

def analyze_audio_features(signal, sample_rate):
    """分析音频特征"""
    # 计算RMS能量
    rms = np.sqrt(np.mean(signal**2))
    
    # 计算频谱质心
    fft = np.fft.fft(signal)
    magnitude = np.abs(fft[:len(fft)//2])
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(magnitude)]
    spectral_centroid = np.sum(frequencies * magnitude) / np.sum(magnitude)
    
    # 计算动态范围
    dynamic_range = 20 * np.log10(np.max(np.abs(signal)) / (rms + 1e-6))
    
    return {
        "rms_energy": float(rms),
        "spectral_centroid": float(spectral_centroid),
        "dynamic_range_db": float(dynamic_range),
        "duration_seconds": len(signal) / sample_rate
    }

def print_audio_analysis(results):
    """打印音频分析结果"""
    print("\n📊 音频特征分析")
    print("=" * 50)
    
    for result in results:
        print(f"\n{result['name']}:")
        features = result['features']
        print(f"  RMS能量: {features['rms_energy']:.4f}")
        print(f"  频谱质心: {features['spectral_centroid']:.1f} Hz")
        print(f"  动态范围: {features['dynamic_range_db']:.1f} dB")
        print(f"  时长: {features['duration_seconds']:.1f} 秒")

def save_workshop_results(results):
    """保存工作坊结果"""
    output_file = Path("outputs/music/workshop_results.json")
    
    # 准备数据
    workshop_data = {
        "timestamp": datetime.now().isoformat(),
        "total_tracks": len(results),
        "tracks": results,
        "summary": {
            "formats": ["WAV 16-bit 44.1kHz"],
            "visualizations": ["Waveform plots"],
            "features_analyzed": ["RMS Energy", "Spectral Centroid", "Dynamic Range"]
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(workshop_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 工作坊结果已保存: {output_file}")

def main():
    """主函数"""
    try:
        # 运行工作坊
        results = run_music_workshop()
        
        # 生成建议
        print("\n💡 音乐治疗建议")
        print("-" * 40)
        print("1. 根据用户情绪状态动态调整BPM")
        print("2. 使用自然音色增强放松效果")
        print("3. 加入双耳节拍技术促进睡眠")
        print("4. 结合用户音乐偏好个性化生成")
        
        print("\n🎵 音频处理技巧")
        print("-" * 40)
        print("1. 使用更复杂的合成技术提升音质")
        print("2. 添加环境音效（雨声、海浪等）")
        print("3. 实现实时参数调整")
        print("4. 支持多声道空间音频")
        
        print("\n" + "=" * 50)
        print("音乐生成工作坊完成")
        print("=" * 50)
        print(f"完成时间: {datetime.now().strftime('%H:%M:%S')}")
        print("\n🚀 下一步: 运行 07_video_generation_workshop.py")
        
    except Exception as e:
        print(f"\n❌ 工作坊出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()