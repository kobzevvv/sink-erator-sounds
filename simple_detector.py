#!/usr/bin/env python3
"""
Simple Audio Erator Detector
Automatically detects erator ON/OFF states without prior knowledge
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

def detect_erator_automatic(audio_file, threshold=70):
    """Automatically detect erator sounds without prior knowledge."""
    
    # Load audio
    print("Loading audio...")
    audio, sr = librosa.load(audio_file, sr=22050)
    print(f"Loaded: {len(audio)/sr:.2f} seconds at {sr} Hz")
    
    # Analyze frequency content
    hop_length = 512
    stft = librosa.stft(audio, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Define frequency bands
    low_freq_mask = (freqs >= 50) & (freqs <= 200)      # Water/background
    mid_freq_mask = (freqs >= 200) & (freqs <= 1000)   # Mixed sounds  
    high_freq_mask = (freqs >= 1000) & (freqs <= 5000) # Erator mechanical noise
    
    # Calculate energy in each band
    low_energy = np.sum(magnitude[low_freq_mask, :], axis=0)
    mid_energy = np.sum(magnitude[mid_freq_mask, :], axis=0)
    high_energy = np.sum(magnitude[high_freq_mask, :], axis=0)
    total_energy = low_energy + mid_energy + high_energy
    
    # Calculate erator signature ratio
    erator_ratio = (mid_energy + high_energy) / (total_energy + 1e-10)
    
    # Detection criteria
    ratio_threshold = np.percentile(erator_ratio, threshold)
    energy_threshold = np.percentile(total_energy, 20)
    high_energy_threshold = np.percentile(high_energy, threshold)
    
    # Detect erator
    erator_mask = (
        (erator_ratio > ratio_threshold) & 
        (total_energy > energy_threshold) & 
        (high_energy > high_energy_threshold)
    )
    
    # Find ON/OFF transitions
    frame_times = librosa.frames_to_time(np.arange(len(erator_mask)), sr=sr, hop_length=hop_length)
    diff = np.diff(np.concatenate([[False], erator_mask, [False]]).astype(int))
    on_times = frame_times[np.where(diff == 1)[0]]
    off_times = frame_times[np.where(diff == -1)[0]]
    
    # Print results
    print(f"\n=== AUTOMATIC ERATOR DETECTION ===")
    print(f"Detection threshold: {threshold}%")
    print(f"Total ON periods: {len(on_times)}")
    
    if len(on_times) > 0:
        print(f"\nDetected erator periods:")
        for i, (on_time, off_time) in enumerate(zip(on_times, off_times)):
            duration = off_time - on_time
            print(f"  Period {i+1}: {on_time:.2f}s - {off_time:.2f}s (duration: {duration:.2f}s)")
        
        total_on_time = np.sum([off - on for on, off in zip(on_times, off_times)])
        print(f"\nTotal erator ON time: {total_on_time:.2f}s")
        print(f"Total erator OFF time: {len(audio)/sr - total_on_time:.2f}s")
    
    # Create simple visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Waveform with erator periods
    time = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(time, audio, alpha=0.7, color='blue')
    axes[0].set_title('Audio Waveform with Detected Erator Periods')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Highlight erator periods
    for on_time, off_time in zip(on_times, off_times):
        axes[0].axvspan(on_time, off_time, alpha=0.3, color='red')
    
    # Plot 2: Frequency analysis
    axes[1].plot(frame_times, low_energy, label='Low Freq (50-200 Hz)', alpha=0.8, color='blue')
    axes[1].plot(frame_times, mid_energy, label='Mid Freq (200-1000 Hz)', alpha=0.8, color='green')
    axes[1].plot(frame_times, high_energy, label='High Freq (1000-5000 Hz)', alpha=0.8, color='red')
    axes[1].set_title('Frequency Band Energy')
    axes[1].set_ylabel('Energy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Erator detection
    axes[2].plot(frame_times, erator_mask.astype(int), label='Erator ON/OFF', alpha=0.8, color='green', linewidth=2)
    axes[2].set_title('Automatic Erator Detection')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Erator State (1=ON, 0=OFF)')
    axes[2].legend()
    axes[2].grid(True)
    
    # Add transition markers
    for on_time in on_times:
        axes[2].axvline(on_time, color='green', linestyle='--', alpha=0.7)
    for off_time in off_times:
        axes[2].axvline(off_time, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('erator_detection.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to erator_detection.png")
    
    return erator_mask, frame_times, on_times, off_times

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_detector.py <audio_file> [threshold]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 70
    
    if not os.path.exists(audio_file):
        print(f"Error: File '{audio_file}' not found")
        sys.exit(1)
    
    detect_erator_automatic(audio_file, threshold)
