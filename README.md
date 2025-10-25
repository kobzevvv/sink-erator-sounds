# Audio Source Separation for Erator Sounds

This script helps separate erator sounds from water sounds and other background noise in audio recordings.

## Features

- **Automatic Detection**: Uses machine learning techniques to identify erator sounds based on audio characteristics
- **Source Separation**: Separates audio into erator sounds, water sounds, and background noise
- **Noise Reduction**: Enhances erator sounds by reducing background noise
- **Visualization**: Creates plots showing the separation process
- **Multiple Output Formats**: Saves separated audio in different formats

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python audio_separator.py sounds-examples/2025-10-25\ 13.54.36.ogg
```

### Advanced Usage
```bash
# With custom output directory and visualization
python audio_separator.py sounds-examples/2025-10-25\ 13.54.36.ogg -o my_output -v

# With custom detection threshold (higher = more selective)
python audio_separator.py sounds-examples/2025-10-25\ 13.54.36.ogg -t 80 -v
```

### Command Line Options

- `input_file`: Path to the input audio file
- `-o, --output`: Output directory (default: "output")
- `-t, --threshold`: Detection threshold percentile 0-100 (default: 75)
- `-v, --visualize`: Create visualization plots
- `--sample-rate`: Sample rate for processing (default: 22050)

## Output Files

The script generates the following files in the output directory:

- `original.wav`: Original audio file
- `erator_sounds.wav`: Isolated erator sounds
- `water_sounds.wav`: Isolated water/background sounds  
- `enhanced_erator.wav`: Enhanced erator sounds with noise reduction
- `separation_analysis.png`: Visualization of the separation process (if -v flag used)

## How It Works

1. **Feature Extraction**: Analyzes audio characteristics including:
   - Spectral centroid and rolloff
   - Zero crossing rate (important for mechanical sounds)
   - RMS energy
   - MFCC coefficients

2. **Erator Detection**: Identifies erator sounds based on:
   - High zero crossing rate (mechanical noise characteristic)
   - Energy patterns in specific frequency bands
   - Temporal characteristics

3. **Source Separation**: Uses the detected features to separate:
   - Erator sounds (mechanical noise)
   - Water sounds (lower frequency, flowing characteristics)
   - Background noise

4. **Enhancement**: Applies noise reduction and frequency filtering to improve erator sound quality

## Technical Details

- **Sample Rate**: Default 22050 Hz (can be customized)
- **Frame Analysis**: 2048 sample frames with 512 sample hop length
- **Frequency Analysis**: Focuses on 1-8 kHz range for erator detection
- **Filtering**: Bandpass filtering to remove water sounds (100-2000 Hz)

## Troubleshooting

- **Low Detection**: Try lowering the threshold (-t 60)
- **Too Much Noise**: Try raising the threshold (-t 85)
- **Poor Quality**: Ensure input audio is clear and not too compressed
- **Memory Issues**: Reduce sample rate or use shorter audio clips

## Example Results

The script will show:
- Audio duration and sample rate
- Number of erator segments detected
- Processing progress
- Output file locations
