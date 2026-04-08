$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CheckpointDir = Join-Path $RepoRoot "checkpoints"
$CheckpointPath = Join-Path $CheckpointDir "sam_vit_h_4b8939.pth"
$VenvPath = Join-Path $RepoRoot ".venv"
$CondaEnvPath = Join-Path $RepoRoot ".conda\sam"

$PyLauncherList = ""
try {
    $PyLauncherList = py -0p 2>$null | Out-String
}
catch {
    $PyLauncherList = ""
}

$HasPy310 = $PyLauncherList -match "3\.10"

if ($HasPy310) {
    if (-not (Test-Path $VenvPath)) {
        py -3.10 -m venv $VenvPath
    }
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
}
else {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "Python 3.10 was not found and Conda is unavailable. Install Python 3.10 or Conda first."
    }
    if (-not (Test-Path $CondaEnvPath)) {
        conda create -y -p $CondaEnvPath python=3.10
    }
    $PythonExe = Join-Path $CondaEnvPath "python.exe"
}

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r (Join-Path $RepoRoot "requirements.txt")
& $PythonExe -m pip install git+https://github.com/facebookresearch/segment-anything.git

if (-not (Test-Path $CheckpointDir)) {
    New-Item -ItemType Directory -Path $CheckpointDir | Out-Null
}

if (-not (Test-Path $CheckpointPath)) {
    Invoke-WebRequest `
        -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" `
        -OutFile $CheckpointPath
}

Write-Host "SAM environment ready at $PythonExe"
Write-Host "Checkpoint: $CheckpointPath"
