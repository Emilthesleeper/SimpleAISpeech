# Windows distribution

The installer creates a per-user Windows x64 installation under
`%LOCALAPPDATA%\Programs\SimpleAISpeech`. The application and its Python
runtime/dependencies are bundled with PyInstaller; the target PC does not need
Python or pip.

The setup then:

1. Runs the official Ollama Windows installer command if Ollama is not already
   installed.
2. Starts Ollama if needed and runs `ollama pull llama3.2:3b`.
3. Downloads the `faster-whisper` base model into
   `%LOCALAPPDATA%\SimpleAISpeech\whisper-cache`.
4. Includes the Piper Thorsten high model in the application payload.

Ollama itself and its model files use Ollama's standard per-user install and
model locations. The Whisper cache lives in the current user's Local AppData;
the Piper voice lives with the application. Installation requires an internet
connection.

Before distributing the installer, confirm that the licenses and attribution
requirements for the Piper voice, Ollama model, and bundled Python dependencies
permit the intended distribution.

## Build

On the build PC, install Python and Inno Setup 6. Python and pip are only needed
for building; neither is installed on the target PC. Ensure these ignored model
files exist locally:

- `models/de_DE-thorsten-high.onnx`
- `models/de_DE-thorsten-high.onnx.json`

From the repository root, run:

```powershell
.\packaging\build.ps1
```

The script creates the self-contained PyInstaller folder under
`build/package/SimpleAISpeech` and the setup executable at
`dist-installer/SimpleAISpeech-Setup.exe`.

## Single-file release executable

To build one self-extracting executable for a GitHub release, run:

```powershell
.\packaging\build-release.ps1
```

The output is `dist-release/SimpleAISpeech.exe`. It bundles Python and the
application's Python dependencies, plus the Piper voice. Ollama is still
installed separately; the `llama3.2:3b` model must be available to it.
