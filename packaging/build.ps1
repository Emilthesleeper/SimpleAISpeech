$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildVenv = Join-Path $projectRoot ".venv-build"
$buildPython = Join-Path $buildVenv "Scripts\python.exe"
$highModel = Join-Path $projectRoot "models\de_DE-thorsten-high.onnx"
$highModelConfig = Join-Path $projectRoot "models\de_DE-thorsten-high.onnx.json"
$payloadDirectory = Join-Path $projectRoot "build\package\SimpleAISpeech"
$outputDirectory = Join-Path $projectRoot "dist-installer"

foreach ($requiredFile in @($highModel, $highModelConfig)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required Piper asset is missing: $requiredFile"
    }
}

if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python is required on the build PC only. It is not needed on the target PC."
    }
    & $pythonCommand.Source -m venv $buildVenv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the packaging virtual environment."
    }
}

& $buildPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not prepare pip in the build-only virtual environment."
}

& $buildPython -m pip install -r (Join-Path $PSScriptRoot "requirements-build.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the build dependencies."
}

if (Test-Path -LiteralPath $payloadDirectory) {
    Remove-Item -LiteralPath $payloadDirectory -Recurse -Force
}

New-Item -ItemType Directory -Force -Path (Split-Path $payloadDirectory) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "build\pyinstaller") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "build\spec") | Out-Null
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", "SimpleAISpeech",
    "--distpath", (Split-Path $payloadDirectory),
    "--workpath", (Join-Path $projectRoot "build\pyinstaller"),
    "--specpath", (Join-Path $projectRoot "build\spec"),
    "--collect-all", "faster_whisper",
    "--collect-all", "huggingface_hub",
    "--collect-all", "tokenizers",
    "--collect-all", "ctranslate2",
    "--collect-all", "av",
    "--collect-all", "piper",
    "--collect-all", "piper_phonemize",
    "--collect-all", "onnxruntime",
    "--collect-all", "sounddevice",
    "--collect-all", "ollama",
    "--copy-metadata", "faster-whisper",
    "--copy-metadata", "piper-tts",
    "--add-data", "$highModel;models",
    "--add-data", "$highModelConfig;models",
    (Join-Path $projectRoot "main.py")
)

& $buildPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed. See the build output above."
}

$innoCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
)
$isccCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($isccCommand) {
    $isccPath = $isccCommand.Source
}
else {
    $isccPath = $innoCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}

if (-not $isccPath) {
    throw "The app folder was built at $payloadDirectory. Install Inno Setup 6 on the build PC, then rerun this script to create the installer."
}

& $isccPath (Join-Path $PSScriptRoot "SimpleAISpeech.iss")
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed. See the build output above."
}

Write-Host "Installer created at: $(Join-Path $outputDirectory 'SimpleAISpeech-Setup.exe')"
