#!/bin/bash
# scripts/generate_icon.sh
# Genera icon.png dall'SVG per il tray e la taskbar Ubuntu
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$SCRIPT_DIR/../ui/assets"
SVG="$ASSETS/icon.svg"
PNG="$ASSETS/icon.png"

if command -v inkscape &>/dev/null; then
    inkscape "$SVG" --export-png="$PNG" --export-width=256 --export-height=256
    echo "✓ icon.png generata con Inkscape"
elif command -v rsvg-convert &>/dev/null; then
    rsvg-convert -w 256 -h 256 "$SVG" -o "$PNG"
    echo "✓ icon.png generata con rsvg-convert"
elif venv-runtime/bin/python -c "import cairosvg" 2>/dev/null; then
    venv-runtime/bin/python -c "
import cairosvg
cairosvg.svg2png(url='$SVG', write_to='$PNG', output_width=256, output_height=256)
print('✓ icon.png generata con cairosvg')
"
else
    echo "⚠ Nessun convertitore SVG trovato. Installa uno tra:"
    echo "   sudo apt install inkscape"
    echo "   sudo apt install librsvg2-bin"
    echo "   pip install cairosvg"
    echo "Il file SVG è comunque usabile come icona."
fi
