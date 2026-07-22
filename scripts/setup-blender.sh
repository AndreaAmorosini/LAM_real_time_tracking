#!/usr/bin/env bash
set -euo pipefail

BLENDER_VERSION="4.0.2"
BLENDER_DIR="thirdparties/blender"
BLENDER_ARCHIVE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender4.0/${BLENDER_ARCHIVE}"
BLENDER_EXE="${BLENDER_DIR}/blender-${BLENDER_VERSION}-linux-x64/blender"

mkdir -p "${BLENDER_DIR}"

if [ -x "${BLENDER_EXE}" ]; then
  echo "Blender already installed: ${BLENDER_EXE}"
  "${BLENDER_EXE}" --version
  exit 0
fi

echo "Downloading Blender ${BLENDER_VERSION}..."
wget -O "${BLENDER_DIR}/${BLENDER_ARCHIVE}" "${BLENDER_URL}"

echo "Extracting Blender..."
tar -xf "${BLENDER_DIR}/${BLENDER_ARCHIVE}" -C "${BLENDER_DIR}"

rm "${BLENDER_DIR}/${BLENDER_ARCHIVE}"

echo "Blender installed:"
"${BLENDER_EXE}" --version

echo ""
echo "Use:"
echo "export LAM_BLENDER_PATH=${PWD}/${BLENDER_EXE}"
