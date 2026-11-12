#!/bin/bash
set -euo pipefail

app_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
desktop="$(xdg-user-dir DESKTOP)"
autostart="\${XDG_CONFIG_HOME:-$HOME/.config}/autostart"

if [[ ! -x "$app_dir/bin/rpc_cap4" || ! -f "$app_dir/models/bnn/feature.bolt" ]]; then
    printf '%s\n' '请先解压完整的 Pi 5 离线交接包。' >&2
    exit 1
fi

install -d "$desktop" "$autostart"
for directory in "$desktop" "$autostart"; do
    shortcut="$directory/binparam-camera.desktop"
    exec_path="\${app_dir//\\/\\\\}"
    exec_path="\${exec_path//\"/\\\"}"
    exec_path="\${exec_path//\$/\\\$}"
    exec_path="\${exec_path//%/%%}"
    cat > "$shortcut" <<EOF
[Desktop Entry]
Type=Application
Name=BinParam 3D Camera
Comment=Capture and reconstruct a 3D human mesh on the Pi 5
Exec="$exec_path/start-camera.sh"
Icon=camera-video
Terminal=false
Categories=Graphics;
StartupNotify=false
EOF
    if [[ "$directory" == "$desktop" ]]; then
        chmod +x "$shortcut"
    fi
    if command -v desktop-file-validate >/dev/null; then
        desktop-file-validate "$shortcut"
    fi
done
