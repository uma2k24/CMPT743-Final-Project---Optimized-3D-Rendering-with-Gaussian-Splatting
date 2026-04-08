# TripoSR Setup

This document is for the `SAM -> TripoSR -> mesh -> Gaussian splats` part of the project.


## What this pipeline does

```text
input image
-> optional SAM foreground crop
-> TripoSR single-image reconstruction
-> mesh.obj
-> Gaussian splatting baseline in this repo
```

The code entry point for this is:

- `sam_triposr_pipeline.py`

## Recommended environment split

Keep these in separate environments:

1. `SAM + this repo`
2. `TripoSR`

That keeps package conflicts lower and makes debugging easier.

## 1. SAM + this repo

You already have:

- `scripts/install_sam_env.ps1`

Run:

```powershell
.\scripts\install_sam_env.ps1
```

That should create either:

- `.venv`
- or `.conda/sam`

and download:

- `checkpoints/sam_vit_h_4b8939.pth`

## 2. TripoSR environment

Clone the official repo in a separate location:

```powershell
git clone https://github.com/VAST-AI-Research/TripoSR.git
```

Create a dedicated Conda environment:

```powershell
conda create -y -n triposr python=3.10
conda activate triposr
```

Install TripoSR dependencies from the official repo:

```powershell
cd C:\path\to\TripoSR
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- TripoSR will download the pretrained model on first run unless you point it to a local model path.
- TripoSR can run its own background removal. If you are already using SAM, you will usually want `--triposr-no-remove-bg`.

## First-run sanity check in the TripoSR repo

Before using this repo's wrapper, make sure TripoSR itself works directly:

```powershell
cd C:\path\to\TripoSR
python .\run.py C:\path\to\image.png --output-dir C:\temp\triposr_test --model-save-format obj
```

Expected output:

- `C:\temp\triposr_test\0\mesh.obj`

If that file is not created, fix TripoSR first before debugging the wrapper pipeline in this repo.

## Preflight check from this repo

Use the validation script before running the full pipeline:

```powershell
python .\scripts\validate_triposr_pipeline.py `
  --triposr-workdir C:\path\to\TripoSR `
  --triposr-python C:\path\to\miniconda3\envs\triposr\python.exe `
  --input-image .\data\input.png `
  --sam-checkpoint .\checkpoints\sam_vit_h_4b8939.pth
```

This checks:

- the TripoSR repo path exists
- `run.py` exists
- the TripoSR Python executable exists
- the input image exists
- the SAM checkpoint exists

## Test only the TripoSR mesh stage

If you want to test only mesh generation and stop before Gaussian splatting:

```powershell
python .\sam_triposr_pipeline.py `
  --input-image .\data\input.png `
  --prompt "ceramic vase" `
  --sam-checkpoint .\checkpoints\sam_vit_h_4b8939.pth `
  --triposr-python C:\path\to\miniconda3\envs\triposr\python.exe `
  --triposr-workdir C:\path\to\TripoSR `
  --triposr-no-remove-bg `
  --skip-gs `
  --out-dir .\outputs\sam_triposr_test
```

Expected outputs:

- `outputs/sam_triposr_test/sam/`
- `outputs/sam_triposr_test/triposr/`
- `outputs/sam_triposr_test/triposr/triposr_mesh.obj`
- `outputs/sam_triposr_test/pipeline_manifest.json`

## Verify output after a run

Use the verification mode:

```powershell
python .\scripts\validate_triposr_pipeline.py `
  --verify-output .\outputs\sam_triposr_test
```

This checks:

- the manifest exists
- the TripoSR OBJ exists
- the OBJ appears to contain vertex and face lines

## Local Testing

1. Run TripoSR directly in its own repo.
2. Run `validate_triposr_pipeline.py` preflight.
3. Run `sam_triposr_pipeline.py --skip-gs`.
4. Run `validate_triposr_pipeline.py --verify-output ...`.
5. Only then hand the mesh output and wrapper script to the Gaussian splatting branch.

## Files needed for Gaussian Splatting

The main files they need are:

- `sam_triposr_pipeline.py`
- `sam_stage.py`
- `TRIPOSR_SETUP.md`
- `scripts/validate_triposr_pipeline.py`


