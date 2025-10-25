#!/bin/bash

# Example script to run audio separation on the provided audio file
# Make sure to install dependencies first: pip install -r requirements.txt

echo "Starting audio separation process..."
echo "Input file: sounds-examples/2025-10-25 13.54.36.ogg"
echo ""

# Run the audio separator with visualization
python audio_separator.py "sounds-examples/2025-10-25 13.54.36.ogg" -v -o output

echo ""
echo "Process complete! Check the 'output' directory for results."
echo ""
echo "Generated files:"
echo "- original.wav: Original audio"
echo "- erator_sounds.wav: Isolated erator sounds"
echo "- water_sounds.wav: Isolated water/background sounds"
echo "- enhanced_erator.wav: Enhanced erator sounds"
echo "- separation_analysis.png: Visualization of the process"
