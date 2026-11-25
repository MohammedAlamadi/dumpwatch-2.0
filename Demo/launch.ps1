$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    foreach ($tool in @('ffmpeg','ffprobe')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool must be on PATH." }
    }
    python -B -c "import streamlit, torch, torchvision, numpy, PIL"
    if ($LASTEXITCODE -ne 0) { throw 'Install requirements.txt in the active environment.' }
    if (-not (Test-Path -LiteralPath 'models/resnet18-f37072fd.pth')) { throw 'Run python tools/fetch_encoder.py from the repository root first.' }
    python -B -m streamlit run app.py @args
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit exited with an error.' }
} finally { Pop-Location }
