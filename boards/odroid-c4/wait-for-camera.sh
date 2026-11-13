#!/bin/bash
set -eu
app="$(cd -- "$(dirname -- "$0")" && pwd)"
while [ ! -c /dev/video0 ]; do sleep 3; done
exec "$app/start-camera.sh"
