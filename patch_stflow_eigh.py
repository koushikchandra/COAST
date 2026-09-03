#!/usr/bin/env python
"""Patch a local STFlow install for the torch-2.10 cusolver eigh regression.

STFlow's equivariant-frame builder calls torch.linalg.eigh, which crashes on
torch 2.10 + some GPUs (CUSOLVER_STATUS_INVALID_VALUE). This adds a CPU fallback
(numerically identical; matrices are tiny). Run ONCE after installing STFlow:

    pip install -e STFlow/
    python patch_stflow_eigh.py [STFLOW_DIR]     # default: ./STFlow

Idempotent. Also set the MAGMA linalg backend in your training entrypoint via
torch.backends.cuda.preferred_linalg_library("magma") (train_cross_organ.py does).
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "STFlow"
fa = os.path.join(root, "stflow", "model", "fa.py")
src = open(fa).read()

if "cusolver batched-eigh" in src:
    print("already patched:", fa)
    sys.exit(0)

orig = "        _, eigenvectors = torch.linalg.eigh(C, UPLO='U')  # [B,dim,dim]"
patched = (
    "        # torch 2.10 cusolver batched-eigh regresses on some GPUs\n"
    "        # (CUSOLVER_STATUS_INVALID_VALUE); C is only [B,dim,dim] (dim=2/3)\n"
    "        # so a CPU fallback is cheap and numerically identical.\n"
    "        try:\n"
    "            _, eigenvectors = torch.linalg.eigh(C, UPLO='U')  # [B,dim,dim]\n"
    "        except torch._C._LinAlgError:\n"
    "            _, eig_cpu = torch.linalg.eigh(C.detach().cpu(), UPLO='U')\n"
    "            eigenvectors = eig_cpu.to(C.device)"
)
if orig not in src:
    print("ERROR: expected eigh line not found in", fa)
    sys.exit(1)
open(fa, "w").write(src.replace(orig, patched))
print("patched:", fa)
