#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# CONFIG (updated for your repo)
# -------------------------
REPO="HarshPatel0404/DL_CapstoneProject"
TAG="v1.0-large-assets"
RELEASE_NAME="Large Assets Upload"
RELEASE_NOTES="Upload of legalbert_out (split) and models archive."
SPLIT_SIZE="1000M"
WORKDIR="$(pwd)/upload_temp"
BRANCH="main"
# -------------------------

FOLDERS_SMALL=("figures" "logs")
FOLDER_MODELS="models"
FOLDER_LEGAL="legalbert_out"

REQUIRED=("tar" "split" "git" "zip" "gh" "cat")

echo "Checking prerequisites..."
for cmd in "${REQUIRED[@]}"; do
  if ! command -v "$cmd" > /dev/null 2>&1; then
    echo "ERROR: Required command '$cmd' not found."
    echo "Install GitHub CLI: https://cli.github.com/"
    exit 1
  fi
done

echo "Preparing workdir: $WORKDIR"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

# -------------------------
# Step 1: archive & split legalbert_out (BIG FOLDER)
# -------------------------
if [ ! -d "$FOLDER_LEGAL" ]; then
  echo "ERROR: Folder '$FOLDER_LEGAL' not found."
  exit 1
fi

echo "Creating archive for $FOLDER_LEGAL ..."
tar -czf "$WORKDIR/legalbert_out.tar.gz" "$FOLDER_LEGAL"

echo "Splitting into 1GB parts..."
cd "$WORKDIR"
split -b "$SPLIT_SIZE" -d --additional-suffix=.part legalbert_out.tar.gz legalbert_out.tar.gz.part.
echo "Parts created:"
ls -lh legalbert_out.tar.gz.part.*

# -------------------------
# Step 2: archive models (400MB)
# -------------------------
cd "$WORKDIR"
echo "Creating archive for $FOLDER_MODELS ..."
tar -czf "$WORKDIR/models.tar.gz" -C .. "$FOLDER_MODELS"
ls -lh models.tar.gz

# -------------------------
# Step 3: commit & push small folders (figures, logs)
# -------------------------
cd ..
echo "Adding small folders to Git: ${FOLDERS_SMALL[*]}"
for folder in "${FOLDERS_SMALL[@]}"; do
  if [ -d "$folder" ]; then
    git add "$folder"
  else
    echo "WARNING: Small folder '$folder' missing — skipping."
  fi
done

if git status --porcelain | grep .; then
  git commit -m "Add small folders (figures, logs)"
  git push origin "$BRANCH"
else
  echo "No changes to commit in small folders."
fi

# -------------------------
# Step 4: GitHub Release upload (large files)
# -------------------------
cd "$WORKDIR"

echo "Creating or updating release '$TAG' in $REPO ..."

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "Release exists — will upload assets to existing release."
else
  echo "Creating new release..."
  gh release create "$TAG" --repo "$REPO" \
    --title "$RELEASE_NAME" \
    --notes "$RELEASE_NOTES"
fi

ASSETS=("models.tar.gz")
for p in legalbert_out.tar.gz.part.*; do
  ASSETS+=("$p")
done

echo "Uploading ${#ASSETS[@]} assets..."
gh release upload "$TAG" --repo "$REPO" "${ASSETS[@]}" --clobber

echo ""
echo "=== DONE ==="
echo "Uploaded large files to GitHub Release: $TAG"
echo "Repo: $REPO"
echo ""
echo "Reassembly Instructions:"
echo "1) Download all legalbert_out.tar.gz.part.* files"
echo "2) Run:"
echo "     cat legalbert_out.tar.gz.part.* > legalbert_out.tar.gz"
echo "     tar -xzf legalbert_out.tar.gz"
echo ""
echo "Temp archives are stored in: $WORKDIR"
echo "Remove them with:"
echo "     rm -rf \"$WORKDIR\""

exit 0
