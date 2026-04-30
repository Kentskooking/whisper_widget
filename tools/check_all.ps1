$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Missing venv Python at $pythonExe"
}

function Invoke-NativeStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repoRoot
try {
    $pythonFiles = @(git ls-files --cached --others --exclude-standard -- *.py tools/*.py)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list repo Python files."
    }
    if (-not $pythonFiles) {
        throw "No repo Python files found."
    }

    Invoke-NativeStep "syntax check" { & $pythonExe ".\tools\check_syntax.py" }
    Invoke-NativeStep "ruff check" { & $pythonExe -m ruff check --no-cache -- @pythonFiles }
    Invoke-NativeStep "git diff --check" { git diff --check }
}
finally {
    Pop-Location
}
