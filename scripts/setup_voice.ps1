param([ValidateSet('tiny','base','small','medium','large-v3','large-v3-turbo')][string]$Model = 'small')
$ErrorActionPreference = 'Stop'
$voiceRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $voiceRoot
if (-not (Test-Path -LiteralPath '.venv-voice/Scripts/python.exe')) {
    py -3.10 -m venv .venv-voice
    if ($LASTEXITCODE -ne 0) { throw 'Création du runtime vocal impossible.' }
}
& .venv-voice/Scripts/python.exe -m pip install -r scripts/voice-requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Installation vocale impossible.' }
& .venv-voice/Scripts/python.exe scripts/voice_worker.py --model $Model --prepare
if ($LASTEXITCODE -ne 0) { throw 'Préparation du modèle impossible.' }
