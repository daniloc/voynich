#!/bin/zsh
set -euo pipefail

ROOT=${0:A:h:h:h}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
CACHE=${PUBLIC_OBJECT_CACHE:-"$ROOT/.cache/public_object_attack"}

SURYA_REVISION=0aee81d5fd9275c0582e545bf3a56944b1e75679
SURYA_CONFIG_SHA256=bd2db9bcc338841fc4496b10a69f07ccc544b0c80e94d94482295197b23d5605
SURYA_WEIGHTS_SHA256=e01b79f858778cdad8a1384e644ac2b35f9c095fbfd102a34942e23f2f179fe7
HORAE_ARCHIVE_MD5=2e896696a8bb490183191e232c7e8eba
HORAE_WEIGHTS_SHA256=210d36d8505812847ee44d0190429fc9d1f0c114c19610e07af13e6decd287ba
SAM_WEIGHTS_SHA256=7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
DINO_REVISION=7764ea0f912e53c92e82eb78a2a1631e92725fc8
DINO_WEIGHTS_SHA256=b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9

SURYA_DIR="$CACHE/surya_layout2"
HORAE_DIR="$CACHE/horae_yolo12s"
SAM_DIR="$CACHE/sam2"
DINO_DIR="$CACHE/dinov2"
HORAE_ARCHIVE="$CACHE/HORAE_A_finetuned_B_yolo12s_i640_e120_b8_w0.zip"

for command_name in curl md5 shasum unzip; do
  command -v "$command_name" >/dev/null || {
    print -u2 "missing required command: $command_name"
    exit 1
  }
done
[[ -x "$PYTHON" ]] || {
  print -u2 "missing project Python: $PYTHON"
  exit 1
}

mkdir -p "$SURYA_DIR" "$HORAE_DIR" "$SAM_DIR" "$DINO_DIR"

download() {
  local url=$1
  local target=$2
  if [[ -s "$target" ]]; then
    print "cached: ${target#$ROOT/}"
    return
  fi
  print "download: $url"
  curl --fail --location --retry 3 --retry-delay 2 --continue-at - \
    --output "$target.part" "$url"
  mv "$target.part" "$target"
}

verify_sha256() {
  local expected=$1
  local target=$2
  local actual
  actual=$(shasum -a 256 "$target" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    print -u2 "SHA-256 mismatch for $target"
    print -u2 "expected: $expected"
    print -u2 "actual:   $actual"
    exit 1
  }
}

verify_md5() {
  local expected=$1
  local target=$2
  local actual
  actual=$(md5 -q "$target")
  [[ "$actual" == "$expected" ]] || {
    print -u2 "MD5 mismatch for $target"
    print -u2 "expected: $expected"
    print -u2 "actual:   $actual"
    exit 1
  }
}

print "install pinned inference runtimes"
"$PYTHON" -m pip install "ultralytics==8.3.173"
"$PYTHON" -m pip install "transformers==5.14.1"
"$PYTHON" -m pip install --no-deps "surya-ocr==0.22.1"

SURYA_BASE="https://huggingface.co/datalab-to/surya_layout2/resolve/$SURYA_REVISION"
download "$SURYA_BASE/config.json" "$SURYA_DIR/config.json"
download "$SURYA_BASE/rfdetr_layout.pth" "$SURYA_DIR/rfdetr_layout.pth"
download "$SURYA_BASE/LICENSE" "$SURYA_DIR/LICENSE"
verify_sha256 "$SURYA_CONFIG_SHA256" "$SURYA_DIR/config.json"
verify_sha256 "$SURYA_WEIGHTS_SHA256" "$SURYA_DIR/rfdetr_layout.pth"

HORAE_BASE="https://zenodo.org/api/records/17279775/files"
if [[ -s "$HORAE_ARCHIVE" ]] && \
  [[ "$(md5 -q "$HORAE_ARCHIVE")" != "$HORAE_ARCHIVE_MD5" ]]; then
  print "quarantine: corrupt HORAE archive"
  mv "$HORAE_ARCHIVE" "$HORAE_ARCHIVE.invalid"
fi
download \
  "$HORAE_BASE/HORAE_A_finetuned_B_yolo12s_i640_e120_b8_w0.zip/content" \
  "$HORAE_ARCHIVE"
download "$HORAE_BASE/README.md/content" "$HORAE_DIR/README.md"
verify_md5 "$HORAE_ARCHIVE_MD5" "$HORAE_ARCHIVE"
if [[ ! -s "$HORAE_DIR/best.pt" ]]; then
  unzip -j -o "$HORAE_ARCHIVE" "weights/best.pt" -d "$HORAE_DIR"
fi
verify_sha256 "$HORAE_WEIGHTS_SHA256" "$HORAE_DIR/best.pt"

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
if [[ -s "$SAM_DIR/sam2.1_hiera_tiny.pt" ]] && \
  [[ ! -s "$SAM_DIR/sam2.1_t.pt" ]]; then
  mv "$SAM_DIR/sam2.1_hiera_tiny.pt" "$SAM_DIR/sam2.1_t.pt"
fi
download "$SAM_URL" "$SAM_DIR/sam2.1_t.pt"
verify_sha256 "$SAM_WEIGHTS_SHA256" "$SAM_DIR/sam2.1_t.pt"

if [[ ! -d "$DINO_DIR/repository/.git" ]]; then
  git clone --filter=blob:none \
    https://github.com/facebookresearch/dinov2.git \
    "$DINO_DIR/repository"
fi
git -C "$DINO_DIR/repository" checkout --detach "$DINO_REVISION"
DINO_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
download "$DINO_URL" "$DINO_DIR/dinov2_vits14_pretrain.pth"
verify_sha256 \
  "$DINO_WEIGHTS_SHA256" \
  "$DINO_DIR/dinov2_vits14_pretrain.pth"

print "smoke-test imports and checkpoint loading"
YOLO_CONFIG_DIR="$CACHE/ultralytics" "$PYTHON" -c \
  'from surya.common.rfdetr_torch import RfDetrTorch; from ultralytics import YOLO, SAM; print("imports ok")'
YOLO_CONFIG_DIR="$CACHE/ultralytics" "$PYTHON" -c \
  "from surya.common.rfdetr_torch import RfDetrTorch; RfDetrTorch('$SURYA_DIR', device='cpu'); print('Surya weights ok')"
YOLO_CONFIG_DIR="$CACHE/ultralytics" "$PYTHON" -c \
  "from ultralytics import YOLO, SAM; YOLO('$HORAE_DIR/best.pt'); SAM('$SAM_DIR/sam2.1_t.pt'); print('HORAE and SAM weights ok')"
YOLO_CONFIG_DIR="$CACHE/ultralytics" "$PYTHON" -c \
  "import torch; torch.hub.load('$DINO_DIR/repository', 'dinov2_vits14', source='local', pretrained=True, weights='$DINO_DIR/dinov2_vits14_pretrain.pth'); print('DINOv2 weights ok')"

(
  cd "$CACHE"
  shasum -a 256 \
    surya_layout2/config.json \
    surya_layout2/rfdetr_layout.pth \
    horae_yolo12s/best.pt \
    sam2/sam2.1_t.pt \
    dinov2/dinov2_vits14_pretrain.pth \
    > artifact_manifest.sha256
)

print
print "public object attack resources ready"
print "cache: ${CACHE#$ROOT/}"
cat "$CACHE/artifact_manifest.sha256"
