[CmdletBinding()]
param(
    [ValidateSet('instinct3amoled45mm', 'instinct3amoled50mm')]
    [string[]] $Device = @('instinct3amoled45mm', 'instinct3amoled50mm'),
    [string] $SdkPath = $env:GARMIN_SDK_HOME,
    [string] $KeyPath = $env:GARMIN_DEVELOPER_KEY,
    [switch] $Tests
)

$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($SdkPath) -and $env:APPDATA) {
    $sdkConfig = Join-Path $env:APPDATA 'Garmin\ConnectIQ\current-sdk.cfg'
    if (Test-Path -LiteralPath $sdkConfig) {
        $SdkPath = (Get-Content -LiteralPath $sdkConfig -Raw).Trim()
    }
}
if ([string]::IsNullOrWhiteSpace($SdkPath)) {
    throw 'Set GARMIN_SDK_HOME or pass -SdkPath with the Connect IQ SDK directory.'
}
if ([string]::IsNullOrWhiteSpace($KeyPath) -and $env:USERPROFILE) {
    $KeyPath = Join-Path $env:USERPROFILE 'Downloads\developer_key'
}
if ([string]::IsNullOrWhiteSpace($KeyPath)) {
    throw 'Set GARMIN_DEVELOPER_KEY or pass -KeyPath with your signing key.'
}
$compiler = Join-Path $SdkPath 'bin\monkeyc.bat'
if (!(Test-Path -LiteralPath $compiler)) { throw "Compiler not found: $compiler" }
if (!(Test-Path -LiteralPath $KeyPath)) { throw "Signing key not found: $KeyPath" }
$key = (Resolve-Path -LiteralPath $KeyPath).Path
$kind = if ($Tests) { 'tests' } else { 'v1.1.1' }
$destination = Join-Path $project "bin\$kind"
$staging = Join-Path $project ('bin\build-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $destination, $staging -Force | Out-Null

Push-Location $project
try {
    foreach ($target in $Device) {
        $name = "OvernightAccelBbi-$target.prg"
        $output = Join-Path $staging $name
        $jungles = if ($Tests) { 'monkey.jungle;tests.jungle' } else { 'monkey.jungle' }
        $arguments = @('-f', $jungles, '-d', $target, '-o', $output, '-w', '-y', $key)
        if ($Tests) { $arguments += '-t' } else { $arguments += '-r' }
        & $compiler @arguments
        if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $output)) {
            throw "Build failed for $target. No replacement binary was published."
        }
        Copy-Item -LiteralPath $output -Destination (Join-Path $destination $name) -Force
        if ($Tests) {
            Copy-Item -LiteralPath "$output.debug.xml" -Destination "$destination\$name.debug.xml" -Force
        }
        Write-Host "Built: $destination\$name"
    }
} finally {
    Pop-Location
}
