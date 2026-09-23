#define AppName "SimpleAISpeech"
#define AppVersion "1.0.0"
#define AppPublisher "SimpleAISpeech"

[Setup]
AppId={{5E6529A2-652E-4D6E-899C-C23447C24916}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
CloseApplications=yes
RestartApplications=no
OutputDir=..\dist-installer
OutputBaseFilename=SimpleAISpeech-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\SimpleAISpeech.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\build\package\SimpleAISpeech\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "Install-Dependencies.ps1"; DestDir: "{app}\setup"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\SimpleAISpeech"; Filename: "{app}\SimpleAISpeech.exe"
Name: "{autodesktop}\SimpleAISpeech"; Filename: "{app}\SimpleAISpeech.exe"; Tasks: desktopicon

[Run]
Filename: "{sys}\WindowsPowerShell\v1.0\powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\setup\Install-Dependencies.ps1"""; StatusMsg: "Installing Ollama and downloading llama3.2:3b..."; Flags: waituntilterminated runasoriginaluser
Filename: "{app}\SimpleAISpeech.exe"; Parameters: "--prepare-whisper-model"; StatusMsg: "Downloading the Whisper model into the application folder..."; Flags: waituntilterminated runasoriginaluser
Filename: "{app}\SimpleAISpeech.exe"; Description: "Launch SimpleAISpeech"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\whisper-cache"
