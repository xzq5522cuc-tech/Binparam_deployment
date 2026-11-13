#!/bin/bash
set -eu
systemctl --user import-environment DISPLAY
if [ -n "\${XAUTHORITY:-}" ]; then systemctl --user import-environment XAUTHORITY; fi
if [ -n "\${WAYLAND_DISPLAY:-}" ]; then systemctl --user import-environment WAYLAND_DISPLAY; fi
systemctl --user reset-failed pymaf-camera.service 2>/dev/null || true
exec systemctl --user start pymaf-camera.service
