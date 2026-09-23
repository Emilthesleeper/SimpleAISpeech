$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildVenv = Join-Path $projectRoot ".venv-build"
$buildPython = Join-Path $buildVenv "Scripts\python.exe"
$highModel = Join-Path $projectRoot "models\de_DE-thorsten-high.onnx"
$highModelConfig = Join-Path $projectRoot "models\de_DE-thorsten-high.onnx.json"
$outputDirectory = Join-Path $projectRoot "dist-release"
$workDirectory = Join-Path $projectRoot "build\pyinstaller-onefile"
$specDirectory = Join-Path $projectRoot "build\spec-onefile"

if ([Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
    throw "Build the Windows release executable on Windows."
}

foreach ($requiredFile in @($highModel, $highModelConfig)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required Piper asset is missing: $requiredFile"
    }
}

if (-not (Test-Path -LiteralPath $buildPython -PathType Leaf)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python is required on the build PC only. Release users do not need Python."
    }

    & $pythonCommand.Source -m venv $buildVenv
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the build-only virtual environment."
    }
}

$is64BitPython = & $buildPython -c "import struct; print(int(struct.calcsize('P') == 8))"
if ($LASTEXITCODE -ne 0 -or "$is64BitPython".Trim() -ne "1") {
    throw "A working 64-bit Python installation is required to build the Windows x64 release."
}

& $buildPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Could not prepare pip in the build-only virtual environment."
}

& $buildPython -m pip install -r (Join-Path $PSScriptRoot "requirements-build.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the release build dependencies."
}

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $workDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $specDirectory | Out-Null

$arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", "SimpleAISpeech",
    "--distpath", $outputDirectory,
    "--workpath", $workDirectory,
    "--specpath", $specDirectory,
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

$releaseExecutable = Join-Path $outputDirectory "SimpleAISpeech.exe"
if (-not (Test-Path -LiteralPath $releaseExecutable -PathType Leaf)) {
    throw "PyInstaller completed without producing $releaseExecutable."
}

Write-Host "Standalone release executable created: $releaseExecutable"
