$ErrorActionPreference = "Stop"

function Find-Ollama {
    $command = Get-Command ollama.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"),
        (Join-Path $env:ProgramFiles "Ollama\ollama.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return $null
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$env:Path = "$machinePath;$userPath;$env:Path"
$ollamaPath = Find-Ollama

if (-not $ollamaPath) {
    Write-Host "Installing Ollama using the official installer..."
    irm https://ollama.com/install.ps1 | iex

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:Path = "$machinePath;$userPath;$env:Path"
    $ollamaPath = Find-Ollama
}

if (-not $ollamaPath) {
    throw "Ollama's installer finished, but ollama.exe could not be found."
}

function Test-OllamaReady {
    try {
        $null = Invoke-RestMethod `
            -Uri "http://127.0.0.1:11434/api/tags" `
            -TimeoutSec 3
        return $true
    }
    catch {
        return $false
    }
}

if (-not (Test-OllamaReady)) {
    Write-Host "Starting the Ollama service..."
    Start-Process -FilePath $ollamaPath -ArgumentList "serve" -WindowStyle Hidden
}

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if (Test-OllamaReady) {
        $ready = $true
        break
    }
    Start-Sleep -Seconds 2
}

if (-not $ready) {
    throw "Ollama did not start listening on http://127.0.0.1:11434."
}

Write-Host "Downloading the required Ollama model llama3.2:3b..."
& $ollamaPath pull "llama3.2:3b"
if ($LASTEXITCODE -ne 0) {
    throw "ollama pull llama3.2:3b failed with exit code $LASTEXITCODE."
}

Write-Host "Ollama and llama3.2:3b are ready."
