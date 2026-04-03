# Linux GPU Encoding Test Plan (AWS)

Test melder's `gpu-encode` feature (CUDA path) on a Linux instance with an
NVIDIA GPU. The code uses `ort` v2.0.0-rc.11 which wraps **ONNX Runtime 1.23**,
and selects `CUDAExecutionProvider` with default settings on Linux.

---

## Step 1 — Launch AWS Instance

- **Instance type**: `g4dn.xlarge` (cheapest NVIDIA GPU option)
  - 4 vCPUs, 16 GB RAM, 1x Tesla T4 (16 GB VRAM)
  - ~$0.526/hr on-demand
- **AMI**: Ubuntu 24.04 LTS (standard Canonical AMI — not a Deep Learning AMI,
  so we test the real setup path a user would follow)
- **Storage**: 50 GB gp3 (Rust toolchain + ONNX Runtime + CUDA + model)
- **Security group**: SSH only (port 22)

> Alternative: use the AWS Deep Learning AMI (CUDA pre-installed) and skip
> Step 3. Faster, but doesn't test the documented setup path.

---

## Step 2 — System Prerequisites

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential pkg-config libssl-dev git curl
```

---

## Step 3 — CUDA Toolkit + cuDNN

ONNX Runtime 1.23 requires **CUDA 12.x** and **cuDNN 9.x**.

```bash
# NVIDIA driver
sudo apt install -y nvidia-driver-550
sudo reboot

# Verify after reboot
nvidia-smi   # Should show Tesla T4

# CUDA 12.6 toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# cuDNN 9
sudo apt install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12

# PATH setup
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

Skip this step entirely if using the Deep Learning AMI — just verify
`nvidia-smi` and `nvcc --version` work.

---

## Step 4 — Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

---

## Step 5 — ONNX Runtime 1.23 (CUDA build)

```bash
cd ~
wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-linux-x64-gpu-1.23.0.tgz
tar xzf onnxruntime-linux-x64-gpu-1.23.0.tgz

export ORT_DYLIB_PATH=$HOME/onnxruntime-linux-x64-gpu-1.23.0/lib/libonnxruntime.so
export LD_LIBRARY_PATH=$HOME/onnxruntime-linux-x64-gpu-1.23.0/lib:$LD_LIBRARY_PATH

# Persist
echo "export ORT_DYLIB_PATH=\$HOME/onnxruntime-linux-x64-gpu-1.23.0/lib/libonnxruntime.so" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$HOME/onnxruntime-linux-x64-gpu-1.23.0/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```

---

## Step 6 — Clone and Build

The Linux `ensure_ort_dylib()` fix and test data are already pushed to `main`.

```bash
git clone https://github.com/judepayne/melder.git
cd melder

cargo build --release --features usearch,gpu-encode
```

If this succeeds, dynamic linking is wired correctly.

---

## Step 7 — Tests

```bash
# All tests with gpu-encode compiled in
cargo test --features usearch,gpu-encode,parquet-format,simd

# Full feature matrix
cargo test --all-features
```

This exercises the `ort-load-dynamic` codepath — `ensure_ort_dylib()` runs for
ALL sessions when the feature is compiled in, even CPU ones.

---

## Step 8 — Functional GPU Test (batch mode)

Test data is in the repo at `test-data/gpu-test/` (cloned in Step 6).

```bash
cd ~/melder
./target/release/meld run \
  --config test-data/gpu-test/config.yaml \
  --file-a test-data/gpu-test/a.csv \
  --file-b test-data/gpu-test/b.csv \
  --id-field-a id \
  --id-field-b id \
  --output /tmp/gpu-test-output.csv
```

**What to look for in the logs:**

- `GPU encoding enabled: CUDA` — CUDA EP was selected
- No errors about missing `libonnxruntime.so` or CUDA providers
- Match output in `/tmp/gpu-test-output.csv`

---

## Step 9 — GPU vs CPU Comparison (optional)

Edit `config.yaml` to set `encoder_device: cpu`, re-run, compare encoding times.
Use a larger dataset (100+ records) for meaningful numbers.

---

## Step 10 — Error / Fallback Path Verification

1. **Unset `ORT_DYLIB_PATH` and run** — should get a clear error about finding
   `libonnxruntime.so` (melder's own message, not a cryptic dlopen error)
2. **Download the CPU-only ORT build, point `ORT_DYLIB_PATH` at it, set
   `encoder_device: gpu`** — should produce a clear CUDA provider error or
   graceful CPU fallback

---

## Step 11 — Teardown

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
```

---

## Troubleshooting

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| `libonnxruntime.so: cannot open shared object` | `LD_LIBRARY_PATH` missing ORT lib dir | Add ORT lib dir to `LD_LIBRARY_PATH` |
| CUDA provider fails to initialize | CUDA/cuDNN version mismatch with ORT 1.23 | Check [ORT 1.23 release notes](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0) |
| `ensure_ort_dylib()` succeeds but GPU not used | CUDA EP registration failed silently | Check for `warn!` logs; ORT may fall back to CPU |
| Build fails on `ort-sys` | Missing system deps | Ensure `build-essential` and `pkg-config` installed |

## Time / Cost Estimate

- Instance setup + CUDA: ~15-20 min (or ~5 min with Deep Learning AMI)
- Rust + build: ~10-15 min (first release build)
- Tests + functional run: ~10 min
- **Total: ~30-45 min, ~$0.50 AWS cost**
