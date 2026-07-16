$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ComponentDir = Join-Path $ProjectRoot "components\tmf8829"
$BaseUrl = "https://raw.githubusercontent.com/ams-OSRAM/tmf8829_driver_arduino/main"

New-Item -ItemType Directory -Force -Path $ComponentDir | Out-Null

$Files = @(
    @{ Url = "$BaseUrl/tmf8829/tmf8829.c";       Name = "tmf8829.c" },
    @{ Url = "$BaseUrl/tmf8829/tmf8829.h";       Name = "tmf8829.h" },
    @{ Url = "$BaseUrl/tmf8829/tmf8829_image.c"; Name = "tmf8829_image.c" },
    @{ Url = "$BaseUrl/tmf8829/tmf8829_image.h"; Name = "tmf8829_image.h" },
    @{ Url = "$BaseUrl/LICENSES-MIT.TXT";         Name = "LICENSES-MIT.TXT" }
)

foreach ($File in $Files) {
    $Destination = Join-Path $ComponentDir $File.Name
    Write-Host "Downloading $($File.Name)..."
    Invoke-WebRequest -Uri $File.Url -OutFile $Destination
}

Write-Host "Official ams OSRAM TMF8829 driver installed in:"
Write-Host "  $ComponentDir"
Write-Host ""
Write-Host "Next commands from the ESP-IDF PowerShell:"
Write-Host "  cd `"$ProjectRoot`""
Write-Host "  idf.py set-target esp32c6"
Write-Host "  idf.py build"
