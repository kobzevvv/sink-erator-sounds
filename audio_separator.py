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
        """Automatically detect erator ON/OFF states based on audio characteristics."""
        # Analyze frequency content to detect erator states
        hop_length = 512
        stft = librosa.stft(audio, hop_length=hop_length)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        # Define frequency bands for different sound types
        low_freq_mask = (freqs >= 50) & (freqs <= 200)      # Water/background
        mid_freq_mask = (freqs >= 200) & (freqs <= 1000)    # Mixed sounds
        high_freq_mask = (freqs >= 1000) & (freqs <= 5000)  # Erator mechanical noise
        
        # Calculate energy in each frequency band over time
        low_energy = np.sum(magnitude[low_freq_mask, :], axis=0)
        mid_energy = np.sum(magnitude[mid_freq_mask, :], axis=0)
        high_energy = np.sum(magnitude[high_freq_mask, :], axis=0)
        
        # Calculate energy ratios to identify erator characteristics
        # Erator typically has high mid+high frequency energy relative to low frequency
        total_energy = low_energy + mid_energy + high_energy
        erator_ratio = (mid_energy + high_energy) / (total_energy + 1e-10)  # Avoid division by zero
        
        # Use multiple criteria for erator detection
        # 1. High erator ratio (mechanical noise vs water)
        ratio_threshold = np.percentile(erator_ratio, threshold_percentile)
        
        # 2. Sufficient overall energy (not silence)
        energy_threshold = np.percentile(total_energy, 20)  # Lower threshold to catch quiet erator
        
        # 3. High frequency energy (mechanical noise signature)
        high_energy_threshold = np.percentile(high_energy, threshold_percentile)
        
        # Combine criteria for erator detection
        erator_mask = (
            (erator_ratio > ratio_threshold) & 
            (total_energy > energy_threshold) & 
            (high_energy > high_energy_threshold)
        )
        
        # Apply temporal smoothing to avoid flickering detection
        erator_mask = self._smooth_detection(erator_mask, min_duration_frames=5)
        
        return erator_mask, {
            'erator_ratio': erator_ratio,
            'low_energy': low_energy,
            'mid_energy': mid_energy,
            'high_energy': high_energy,
            'total_energy': total_energy
        }
    
    def _smooth_detection(self, mask, min_duration_frames=5):
        """Apply temporal smoothing to detection to avoid flickering."""
        # Remove very short detections
        smoothed = mask.copy()
        
        # Find continuous segments
        diff = np.diff(np.concatenate([[False], smoothed, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Remove segments shorter than min_duration_frames
        for start, end in zip(starts, ends):
            if end - start < min_duration_frames:
                smoothed[start:end] = False
        
        return smoothed
    
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
        
        # Create mask for erator frequencies (low frequencies for this case)
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        erator_freq_mask = (freqs >= 50) & (freqs <= 500)  # Low frequency erator
        
        # Apply frequency mask - enhance low frequencies, reduce others
        enhanced_magnitude = magnitude.copy()
        enhanced_magnitude[erator_freq_mask, :] *= 1.5  # Boost erator frequencies
        enhanced_magnitude[~erator_freq_mask, :] *= 0.3  # Reduce non-erator frequencies
        
        # Reconstruct audio
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft)
        
        return enhanced_audio
    
    def analyze_erator_states(self, erator_mask, sample_rate):
        """Analyze erator ON/OFF states and transitions."""
        hop_length = 512
        frame_times = librosa.frames_to_time(np.arange(len(erator_mask)), 
                                           sr=sample_rate, 
                                           hop_length=hop_length)
        
        # Find state transitions
        diff = np.diff(np.concatenate([[False], erator_mask, [False]]).astype(int))
        on_times = frame_times[np.where(diff == 1)[0]]
        off_times = frame_times[np.where(diff == -1)[0]]
        
        # Calculate durations
        durations = []
        for i in range(len(on_times)):
            if i < len(off_times):
                duration = off_times[i] - on_times[i]
                durations.append(duration)
        
        # Print analysis results
        print(f"\n=== ERATOR STATE ANALYSIS ===")
        print(f"Total ON periods: {len(on_times)}")
        print(f"Total OFF periods: {len(off_times)}")
        
        if len(on_times) > 0:
            print(f"\nErator ON periods:")
            for i, (on_time, off_time) in enumerate(zip(on_times, off_times)):
                print(f"  Period {i+1}: {on_time:.2f}s - {off_time:.2f}s (duration: {off_time-on_time:.2f}s)")
        
        if len(durations) > 0:
            print(f"\nDuration statistics:")
            print(f"  Average ON duration: {np.mean(durations):.2f}s")
            print(f"  Longest ON duration: {np.max(durations):.2f}s")
            print(f"  Shortest ON duration: {np.min(durations):.2f}s")
        
        return {
            'on_times': on_times,
            'off_times': off_times,
            'durations': durations,
            'total_on_time': np.sum(durations) if durations else 0,
            'total_off_time': frame_times[-1] - (np.sum(durations) if durations else 0)
        }
    
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
    
    def visualize_separation(self, audio, features, erator_mask, analysis_data, output_dir="output"):
        """Create visualization of the automatic separation process."""
        fig, axes = plt.subplots(5, 1, figsize=(15, 14))
        
        # Plot 1: Original waveform with erator periods highlighted
        time = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        axes[0].plot(time, audio, alpha=0.7, color='blue')
        axes[0].set_title('Original Audio Waveform with Detected Erator Periods')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Highlight detected erator periods
        hop_length = 512
        frame_times = librosa.frames_to_time(np.arange(len(erator_mask)), 
                                           sr=self.sample_rate, 
                                           hop_length=hop_length)
        
        # Find continuous erator periods
        diff = np.diff(np.concatenate([[False], erator_mask, [False]]).astype(int))
        on_times = frame_times[np.where(diff == 1)[0]]
        off_times = frame_times[np.where(diff == -1)[0]]
        
        for on_time, off_time in zip(on_times, off_times):
            axes[0].axvspan(on_time, off_time, alpha=0.3, color='red', label='Detected Erator' if on_time == on_times[0] else "")
        if len(on_times) > 0:
            axes[0].legend()
        
        # Plot 2: Frequency band analysis
        axes[1].plot(frame_times, analysis_data['low_energy'], label='Low Freq (50-200 Hz)', alpha=0.8, color='blue')
        axes[1].plot(frame_times, analysis_data['mid_energy'], label='Mid Freq (200-1000 Hz)', alpha=0.8, color='green')
        axes[1].plot(frame_times, analysis_data['high_energy'], label='High Freq (1000-5000 Hz)', alpha=0.8, color='red')
        axes[1].set_title('Frequency Band Energy Analysis')
        axes[1].set_ylabel('Energy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: Erator ratio (mechanical vs water signature)
        axes[2].plot(frame_times, analysis_data['erator_ratio'], label='Erator Ratio (Mid+High)/(Total)', alpha=0.8, color='purple')
        axes[2].set_title('Erator Signature Ratio (Higher = More Mechanical)')
        axes[2].set_ylabel('Ratio')
        axes[2].legend()
        axes[2].grid(True)
        
        # Plot 4: Audio features
        frame_times = librosa.frames_to_time(np.arange(len(features)), 
                                           sr=self.sample_rate, 
                                           hop_length=512)
        axes[3].plot(frame_times, features[:, 2], label='Zero Crossing Rate', alpha=0.7)
        axes[3].plot(frame_times, features[:, 3], label='RMS Energy', alpha=0.7)
        axes[3].set_title('Audio Features')
        axes[3].set_ylabel('Feature Value')
        axes[3].legend()
        axes[3].grid(True)
        
        # Plot 5: Final erator detection
        axes[4].plot(frame_times, erator_mask.astype(int), label='Erator ON/OFF', alpha=0.8, color='green', linewidth=2)
        axes[4].set_title('Automatic Erator Detection (ON/OFF States)')
        axes[4].set_xlabel('Time (seconds)')
        axes[4].set_ylabel('Erator State (1=ON, 0=OFF)')
        axes[4].legend()
        axes[4].grid(True)
        
        # Add state transition markers
        for on_time in on_times:
            axes[4].axvline(on_time, color='green', linestyle='--', alpha=0.7, label='ON' if on_time == on_times[0] else "")
        for off_time in off_times:
            axes[4].axvline(off_time, color='red', linestyle='--', alpha=0.7, label='OFF' if off_time == off_times[0] else "")
        
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
    
    # Automatically detect erator sounds (no prior knowledge of timing)
    print("Automatically detecting erator ON/OFF states...")
    erator_mask, analysis_data = separator.detect_erator_sounds(audio, features, args.threshold)
    
    # Analyze erator states and transitions
    state_analysis = separator.analyze_erator_states(erator_mask, args.sample_rate)
    
    # Print detection statistics
    erator_frames = np.sum(erator_mask)
    total_frames = len(erator_mask)
    print(f"\nOverall detection: {erator_frames}/{total_frames} frames ({erator_frames/total_frames*100:.1f}%)")
    print(f"Total erator ON time: {state_analysis['total_on_time']:.2f}s")
    print(f"Total erator OFF time: {state_analysis['total_off_time']:.2f}s")
    
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
        separator.visualize_separation(audio, features, erator_mask, analysis_data, args.output)
    
    print("Separation complete!")

if __name__ == "__main__":
    main()
