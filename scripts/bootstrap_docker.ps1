param(
    [string]$Version = "27.0.3",
    [string]$ComposeVersion = "2.29.2"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$toolDir = Join-Path $repoRoot '.tools/docker'
$cliDir = Join-Path $toolDir 'cli'
$pluginDir = Join-Path $HOME '.docker/cli-plugins'
New-Item -ItemType Directory -Force -Path $cliDir, $pluginDir | Out-Null

$dockerZip = Join-Path $toolDir "docker-$Version.zip"
$dockerExe = Join-Path $cliDir 'docker.exe'
$composeExe = Join-Path $pluginDir 'docker-compose.exe'

if (-not (Test-Path $dockerExe)) {
    Write-Host "Downloading Docker CLI $Version ..."
    Invoke-WebRequest -Uri "https://download.docker.com/win/static/stable/x86_64/docker-$Version.zip" -OutFile $dockerZip
    Expand-Archive -Path $dockerZip -DestinationPath $cliDir -Force
    if (-not (Test-Path $dockerExe)) {
        $candidate = Get-ChildItem -Path $cliDir -Recurse -Filter 'docker.exe' | Select-Object -First 1 -ExpandProperty FullName
        if ($candidate) {
            Copy-Item $candidate $dockerExe -Force
        }
    }
}

if (-not (Test-Path $composeExe)) {
    Write-Host "Downloading Docker Compose $ComposeVersion ..."
    Invoke-WebRequest -Uri "https://github.com/docker/compose/releases/download/v$ComposeVersion/docker-compose-windows-x86_64.exe" -OutFile $composeExe
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$paths = $userPath -split ';' | Where-Object { $_ -and $_ -ne $cliDir }
if ($paths -notcontains $cliDir) {
    $paths += $cliDir
    [Environment]::SetEnvironmentVariable('Path', ($paths -join ';'), 'User')
    Write-Host "Added $cliDir to the user PATH."
}

$env:Path = "$cliDir;$env:Path"

Write-Host 'Docker CLI available at:' $dockerExe
Write-Host 'Docker Compose plugin available at:' $composeExe

& $dockerExe version | Out-Host
Write-Host '---'
& $dockerExe compose version | Out-Host
