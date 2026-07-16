$ErrorActionPreference = "Stop"

$SourceRoot = Split-Path -Parent $PSScriptRoot
$Destination = "C:\Users\xiele\Documents\RingFusion\firmware-esp"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null
Copy-Item -Path (Join-Path $SourceRoot "*") -Destination $Destination -Recurse -Force

Write-Host "Project files copied to:"
Write-Host "  $Destination"
Write-Host ""
Write-Host "Run this next:"
Write-Host "  & `"$Destination\tools\install_official_tmf8829_driver.ps1`""
