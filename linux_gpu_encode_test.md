# Runpod GPU Test Guide for Melder

## Overview

Test melder's `gpu-encode` (CUDA) path on a Runpod GPU instance. The test data lives at `test-data/gpu-test/` (20 A × 20 B records). Clone main — no branch needed.

## Step 1: Sign up for Runpod

1. Go to **https://console.runpod.io/signup**
2. Create account with email, verify, enable 2FA
3. Add a payment method — buy ~$5 credits (plenty for this test)
4. Go to **Account Settings > SSH Keys** and add your public key (`~/.ssh/id_ed25519.pub`)
   - Paste the entire contents of `~/.ssh/id_ed25519.pub` (the long string + email)

## Step 2: Deploy a GPU Pod

1. Go to **Pods > Deploy**
2. Choose **Community Cloud** (cheaper, ~$0.22/hr for RTX 3090)
3. Select **RTX 3090** (24GB VRAM — more than enough)
4. Template: search for **`runpod/pytorch`** and pick a `devel` image with CUDA 12.x (e.g. `2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
   - Must be CUDA 12.x — ONNX Runtime 1.23.x requires it
5. Disk: **10GB** is fine
6. Click **Deploy**

## Step 3: SSH In

Once the pod is running, you'll see two SSH options. Use **SSH over exposed TCP** (direct SSH):

```bash
ssh root@<PUBLIC_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

If it asks for a password, your SSH key wasn't injected. Use the **Web Terminal** to add it manually:

```bash
mkdir -p ~/.ssh
echo "PASTE_ENTIRE_id_ed25519.pub_CONTENTS" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Install `tmux` so your session survives disconnects:

```bash
apt update && apt install -y tmux
tmux new -s gpu-test
```

## Step 4: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

## Step 5: Verify CUDA

```bash
nvidia-smi
nvcc --version
```

Both should show CUDA 12.x and a functioning GPU.

## Step 6: Install System Dependencies

```bash
apt update && apt install -y pkg-config libssl-dev
```

## Step 7: Install CUDA-enabled ONNX Runtime

Melder's `ort` crate requires **ONNX Runtime >= 1.23.x**. Download the GPU build matching your CUDA version:

```bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.1/onnxruntime-linux-x64-gpu-1.23.1.tgz
tar xzf onnxruntime-linux-x64-gpu-1.23.1.tgz
rm -f /usr/local/lib/libonnxruntime.so*
cp onnxruntime-linux-x64-gpu-1.23.1/lib/libonnxruntime.so* /usr/local/lib/
ln -sf /usr/local/lib/libonnxruntime.so.1.23.1 /usr/local/lib/libonnxruntime.so
ldconfig

# Verify
ls -la /usr/local/lib/libonnxruntime.so
ldconfig -p | grep onnxruntime
```

If the URL changes, check https://github.com/microsoft/onnxruntime/releases for a `linux-x64-gpu` release matching your CUDA version.

**Do NOT use older versions (e.g. 1.17.x)** — the `ort` crate will reject them with a version mismatch error.

## Step 8: Clone and Build

```bash
cd /workspace
git clone https://github.com/<YOU>/melder.git
cd melder
```

**Build with `--no-default-features`** to skip usearch. The default PyTorch image ships with GCC 11.x which cannot compile the simsimd AVX-512 FP16 ("sapphire") intrinsics bundled with usearch. The flat vector backend is sufficient for a GPU encoding smoke test:

```bash
cargo build --release --no-default-features --features gpu-encode
```

This will take several minutes. The ONNX model downloads on first run.

**If you need usearch** (HNSW indexing), you must upgrade GCC first:

```bash
apt install -y software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt update
apt install -y gcc-13 g++-13
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

cargo build --release --features gpu-encode
```

## Step 9: Run the GPU Test

```bash
./target/release/meld run --config test-data/gpu-test/config.yaml
```

You should see:
```
GPU encoding enabled: CUDA
auto-detected onnxruntime path="/usr/local/lib/libonnxruntime.so"
```

If you see `encoding failed: onnxruntime not found`, set:
```bash
export ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
```
and retry.

Expected output: 14 auto-matched, 6 review.

## Step 10: Compare with CPU Run

```bash
sed -i 's/encoder_device: gpu/encoder_device: cpu/' test-data/gpu-test/config.yaml
rm -rf test-data/gpu-test/cache test-data/gpu-test/output test-data/gpu-test/crossmap.csv
./target/release/meld run --config test-data/gpu-test/config.yaml
```

Must produce identical results: 14 auto-matched, 6 review. The 20-record dataset is too small for meaningful timing differences — this is a correctness smoke test, not a benchmark.

## Cost Estimate

- RTX 3090 Community Cloud: ~$0.22/hr
- Rust build: ~15-20 min
- Test run: ~2-5 min
- **Total: under $0.15** if you stop/terminate promptly

## Cleanup

1. **Stop** the pod (keeps disk, costs $0.10/GB/mo while idle)
2. **Terminate** the pod when done (deletes everything)
