#!/usr/bin/env python3
"""
Audio Source Separation Script for Erator Sounds
Separates erator sounds from water sounds and other background noise.
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AudioSeparator:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def load_audio(self, file_path):
        """Load audio file and return audio data and sample rate."""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            print(f"Loaded audio: {len(audio)/sr:.2f} seconds at {sr} Hz")
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_features(self, audio, frame_length=2048, hop_length=512):
        """Extract audio features for analysis."""
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, 
                                                              hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, 
                                                           hop_length=hop_length)[0]
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13, 
                                    hop_length=hop_length)
        
        # Zero crossing rate (good for detecting erator sounds)
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Combine features
        features = np.vstack([
            spectral_centroids,
            spectral_rolloff,
            zcr,
            rms,
            mfccs
        ]).T
        
        return features
    
    def detect_erator_sounds(self, audio, features, threshold_percentile=75):
        """Detect erator sounds based on characteristic features."""
        # Erator sounds typically have:
        # - High zero crossing rate (mechanical noise)
        # - Specific frequency characteristics
        # - Higher energy in certain frequency bands
        
        # Calculate zero crossing rate threshold
        zcr_threshold = np.percentile(features[:, 2], threshold_percentile)
        
        # Calculate energy threshold
        energy_threshold = np.percentile(features[:, 3], threshold_percentile)
        
        # Detect erator segments
        erator_mask = (features[:, 2] > zcr_threshold) & (features[:, 3] > energy_threshold)
        
        return erator_mask
    
    def filter_water_sounds(self, audio, low_freq=100, high_freq=2000):
        """Filter out water sounds using bandpass filtering."""
        # Water sounds are typically in lower frequency ranges
        # Apply bandpass filter to remove water sounds
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def separate_sources(self, audio, features, erator_mask):
        """Separate audio into different source types."""
        hop_length = 512
        frame_length = 2048
        
        # Create time frames
        n_frames = len(features)
        frame_times = librosa.frames_to_time(np.arange(n_frames), 
                                           sr=self.sample_rate, 
                                           hop_length=hop_length)
        
        # Create masks for different sound types
        erator_frames = erator_mask
        water_frames = ~erator_mask  # Assume non-erator is water/background
        
        # Convert frame masks to sample masks
        erator_samples = np.zeros(len(audio), dtype=bool)
        water_samples = np.zeros(len(audio), dtype=bool)
        
        for i, (is_erator, is_water) in enumerate(zip(erator_frames, water_frames)):
            start_sample = int(i * hop_length)
            end_sample = min(start_sample + frame_length, len(audio))
            
            if is_erator:
                erator_samples[start_sample:end_sample] = True
            if is_water:
                water_samples[start_sample:end_sample] = True
        
        # Extract separated audio
        erator_audio = audio.copy()
        erator_audio[~erator_samples] = 0
        
        water_audio = audio.copy()
        water_audio[~water_samples] = 0
        
        return erator_audio, water_audio, erator_samples, water_samples
    
    def enhance_erator_sounds(self, audio, erator_mask):
        """Enhance erator sounds by applying noise reduction and filtering."""
        # Apply spectral gating to reduce background noise
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Create mask for erator frequencies
        # Erator sounds typically have energy in 1-8 kHz range
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        erator_freq_mask = (freqs >= 1000) & (freqs <= 8000)
        
        # Apply frequency mask
        enhanced_magnitude = magnitude.copy()
        enhanced_magnitude[~erator_freq_mask, :] *= 0.1  # Reduce non-erator frequencies
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def save_results(self, original_audio, erator_audio, water_audio, enhanced_audio, 
                    output_dir="output"):
        """Save separated audio files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original
        sf.write(os.path.join(output_dir, "original.wav"), original_audio, self.sample_rate)
        
        # Save separated sources
        sf.write(os.path.join(output_dir, "erator_sounds.wav"), erator_audio, self.sample_rate)
        sf.write(os.path.join(output_dir, "water_sounds.wav"), water_audio, self.sample_rate)
        sf.write(os.path.join(output_dir, "enhanced_erator.wav"), enhanced_audio, self.sample_rate)
        
        print(f"Results saved to {output_dir}/")
        print("- original.wav: Original audio")
        print("- erator_sounds.wav: Isolated erator sounds")
        print("- water_sounds.wav: Isolated water/background sounds")
        print("- enhanced_erator.wav: Enhanced erator sounds with noise reduction")
    
    def visualize_separation(self, audio, features, erator_mask, output_dir="output"):
        """Create visualization of the separation process."""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Original waveform
        time = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        axes[0].plot(time, audio, alpha=0.7)
        axes[0].set_title('Original Audio Waveform')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Plot 2: Feature analysis
        frame_times = librosa.frames_to_time(np.arange(len(features)), 
                                           sr=self.sample_rate, 
                                           hop_length=512)
        axes[1].plot(frame_times, features[:, 2], label='Zero Crossing Rate', alpha=0.7)
        axes[1].plot(frame_times, features[:, 3], label='RMS Energy', alpha=0.7)
        axes[1].set_title('Audio Features')
        axes[1].set_ylabel('Feature Value')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Erator detection
        axes[2].plot(frame_times, erator_mask.astype(int), label='Erator Detection', alpha=0.8)
        axes[2].set_title('Erator Sound Detection')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Erator Present (1=Yes, 0=No)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'separation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {output_dir}/separation_analysis.png")

def main():
    parser = argparse.ArgumentParser(description='Separate erator sounds from water sounds')
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('-t', '--threshold', type=float, default=75, 
                       help='Detection threshold percentile (0-100)')
    parser.add_argument('-v', '--visualize', action='store_true', 
                       help='Create visualization plots')
    parser.add_argument('--sample-rate', type=int, default=22050, 
                       help='Sample rate for processing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Initialize separator
    separator = AudioSeparator(sample_rate=args.sample_rate)
    
    # Load audio
    print("Loading audio...")
    audio, sr = separator.load_audio(args.input_file)
    if audio is None:
        sys.exit(1)
    
    # Extract features
    print("Extracting features...")
    features = separator.extract_features(audio)
    
    # Detect erator sounds
    print("Detecting erator sounds...")
    erator_mask = separator.detect_erator_sounds(audio, features, args.threshold)
    
    # Separate sources
    print("Separating audio sources...")
    erator_audio, water_audio, erator_samples, water_samples = separator.separate_sources(
        audio, features, erator_mask)
    
    # Enhance erator sounds
    print("Enhancing erator sounds...")
    enhanced_audio = separator.enhance_erator_sounds(audio, erator_mask)
    
    # Save results
    print("Saving results...")
    separator.save_results(audio, erator_audio, water_audio, enhanced_audio, args.output)
    
    # Create visualization if requested
    if args.visualize:
        print("Creating visualization...")
        separator.visualize_separation(audio, features, erator_mask, args.output)
    
    print("Separation complete!")

if __name__ == "__main__":
    main()
