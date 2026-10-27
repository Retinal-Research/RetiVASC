[CmdletBinding()]
param(
    [ValidateSet("cpu", "cu118")]
    [string]$Backend = "cpu",

    [string]$PythonExecutable = "",

    [switch]$WithReport,

    [switch]$WithTest
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

$isWindows = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
if (-not $isWindows) {
    throw "bootstrap.ps1 is the Windows installer. Use scripts/bootstrap.sh on Linux or macOS."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvDir = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

function Test-CompatiblePython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$Arguments = @()
    )
    if (-not (Get-Command $Executable -ErrorAction SilentlyContinue)) {
        return $false
    }
    & $Executable @Arguments -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

if ($PythonExecutable) {
    $hostPython = $PythonExecutable
    $hostPythonArgs = @()
}
else {
    # Python 3.10 is the release-certified and compatibility-first default.
    $candidates = @(
        @{ Executable = "py"; Arguments = @("-3.10") },
        @{ Executable = "python3.10"; Arguments = @() },
        @{ Executable = "py"; Arguments = @("-3.11") },
        @{ Executable = "python3.11"; Arguments = @() },
        @{ Executable = "py"; Arguments = @("-3.12") },
        @{ Executable = "python3.12"; Arguments = @() },
        @{ Executable = "python"; Arguments = @() }
    )
    $selected = $null
    foreach ($candidate in $candidates) {
        if (Test-CompatiblePython -Executable $candidate.Executable -Arguments $candidate.Arguments) {
            $selected = $candidate
            break
        }
    }
    if ($null -eq $selected) {
        throw "Compatible Python was not found. Install Python 3.10 (recommended), 3.11, or 3.12; or pass -PythonExecutable PATH."
    }
    $hostPython = $selected.Executable
    $hostPythonArgs = $selected.Arguments
}

$versionCheck = @"
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(f'AutoMorph v2 requires Python >=3.10,<3.13; found {sys.version.split()[0]}')
print(f'Host Python: {sys.version.split()[0]}')
"@
Invoke-Checked -Executable $hostPython -Arguments ($hostPythonArgs + @("-c", $versionCheck))

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating isolated environment at $venvDir"
    Invoke-Checked -Executable $hostPython -Arguments ($hostPythonArgs + @("-m", "venv", $venvDir))
}
else {
    Write-Host "Reusing existing environment at $venvDir"
}

Invoke-Checked -Executable $venvPython -Arguments @("-c", $versionCheck)
Invoke-Checked -Executable $venvPython -Arguments @(
    "-m", "pip", "install", "--upgrade",
    "pip==24.2", "setuptools==80.9.0", "wheel==0.44.0"
)

$backendFile = Join-Path $projectRoot "requirements\torch-$Backend.txt"
Invoke-Checked -Executable $venvPython -Arguments @("-m", "pip", "install", "-r", $backendFile)
Invoke-Checked -Executable $venvPython -Arguments @(
    "-m", "pip", "install", "-r", (Join-Path $projectRoot "requirements\runtime.txt")
)
Invoke-Checked -Executable $venvPython -Arguments @(
    "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", $projectRoot
)

if ($WithReport -or $WithTest) {
    Invoke-Checked -Executable $venvPython -Arguments @(
        "-m", "pip", "install", "-r", (Join-Path $projectRoot "requirements\report.txt")
    )
}
if ($WithTest) {
    Invoke-Checked -Executable $venvPython -Arguments @(
        "-m", "pip", "install", "-r", (Join-Path $projectRoot "requirements\test.txt")
    )
}

Invoke-Checked -Executable $venvPython -Arguments @(
    (Join-Path $projectRoot "scripts\doctor.py"),
    "--root", $projectRoot,
    "--backend", $Backend,
    "--write-root", $projectRoot
)

Write-Host ""
Write-Host "AutoMorph v2 environment is ready."
Write-Host "Python: $venvPython"
Write-Host "Run:    $venvPython -m automorph_v2.cli --help"
