# SimpleAISpeech

A local German voice assistant for Windows. It captures microphone audio,
transcribes speech with faster-whisper, generates responses with Ollama, and
reads them aloud with Piper.

## Features

- Modern browser-based voice studio with live transcription and read-along
  captions.
- Browser microphone streaming, a tactile mute control, and voice interruption
  during response generation and playback.
- Animated page-edge colors show speech detection, recognition, AI processing,
  speech playback, interruption, and combined search/speech activity.
- Live web-search progress and results are shown alongside the conversation.
- Voice interruption during response generation and playback.
- Optional web research through DuckDuckGo, with Bing RSS as a fallback when
  DuckDuckGo is blocked or returns no usable results; no search API key is
  required.
- Extracts short visible-text excerpts from up to five result pages per search
  query. Common navigation, scripts, cookie notices, and ad-related elements
  are filtered where identifiable. Pages rendered only by JavaScript may not
  provide extractable text.

## Windows installer

The installer is per-user and installs the application under:

```text
%LOCALAPPDATA%\Programs\SimpleAISpeech
```

It bundles the application, Python runtime, required Python libraries, and
Piper Thorsten High voice files. It also:

1. Installs Ollama with its official Windows PowerShell installer if Ollama is
   not already present.
2. Starts Ollama if needed and downloads `llama3.2:3b`.
3. Downloads the faster-whisper `small` model into
   `%LOCALAPPDATA%\SimpleAISpeech\whisper-cache`.

The target PC does **not** need Python or pip. Installation requires an
internet connection. The `small` Whisper model is larger and slower than
`base`, but improves transcription accuracy. Ollama and its model data use
Ollama's standard per-user locations.

## Standalone release executable

To create one self-extracting Windows executable for the GitHub Releases tab,
install Python with pip on the build PC and make sure the two Piper voice files
listed above are present. Then run:

```powershell
.\packaging\build-release.ps1
```

Attach `dist-release/SimpleAISpeech.exe` to the release. The executable
bundles the app, Python runtime, Python libraries, and Piper voice; users do
not need Python or pip. Ollama remains a separate dependency and must be
installed and running with `llama3.2:3b` available. The Whisper `small` model
downloads on first launch into `%LOCALAPPDATA%\SimpleAISpeech\whisper-cache`.
An internet connection is needed for that first download and for web searches.

### Build the installer

On the build PC, install Python with pip and Inno Setup 6. The Piper voice
files are excluded from Git, so these two files must be present locally:

- `models/de_DE-thorsten-high.onnx`
- `models/de_DE-thorsten-high.onnx.json`

Run this from the repository root in PowerShell:

```powershell
.\packaging\build.ps1
```

The build script creates a build-only virtual environment and packages the
application with PyInstaller. It writes:

- Bundled application: `build/package/SimpleAISpeech`
- Installer: `dist-installer/SimpleAISpeech-Setup.exe`

Python and pip are used only on the build PC. See
[packaging/README.md](packaging/README.md) for packaging details.

## Run from source

For development, install the packages listed in
[`packaging/requirements-build.txt`](packaging/requirements-build.txt), make
sure Ollama is running with `llama3.2:3b` available, and ensure the Piper voice
files above are present. Then run:

```powershell
python main.py
```

The app opens its local web interface at `http://127.0.0.1:8765`. Allow
microphone access in the browser and click the microphone button to start.
Audio is streamed from the browser to the local app; the interface is in
German.

## Notes

- Search providers or individual sites may block automated requests. If a
  result page refuses access, its search-result snippet is retained; pages
  rendered only by JavaScript may not provide extractable text.
- Before redistributing the installer, check the license and attribution
  requirements for the Piper voice, Ollama model, and bundled dependencies.
