param(
    [string]$WorktreeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$WorktreeRoot = (Resolve-Path $WorktreeRoot).Path

$mainRoot = $null
$candidateRoot = $null
foreach ($line in (& git -C $WorktreeRoot worktree list --porcelain)) {
    if ($line.StartsWith("worktree ")) {
        $candidateRoot = $line.Substring(9)
    }
    elseif ($line -eq "branch refs/heads/main") {
        $mainRoot = $candidateRoot
        break
    }
}

if (-not $mainRoot) {
    throw "Could not locate the worktree that has the main branch checked out."
}

$mainRoot = (Resolve-Path $mainRoot).Path
if ($WorktreeRoot -eq $mainRoot) {
    Write-Host "Main worktree detected; shared dependencies already live here."
    exit 0
}

function Add-DependencyJunction {
    param(
        [string]$RelativePath
    )

    $linkPath = Join-Path $WorktreeRoot $RelativePath
    $targetPath = Join-Path $mainRoot $RelativePath

    if (Test-Path -LiteralPath $linkPath) {
        Write-Host "Keeping existing $RelativePath"
        return
    }
    if (-not (Test-Path -LiteralPath $targetPath)) {
        throw "Missing shared dependency at $targetPath. Prepare dependencies in the main worktree first."
    }

    $parent = Split-Path -Parent $linkPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    New-Item -ItemType Junction -Path $linkPath -Target $targetPath | Out-Null
    Write-Host "Linked $RelativePath -> $targetPath"
}

Add-DependencyJunction ".venv"
Add-DependencyJunction "frontend\node_modules"

Write-Host "Worktree dependency setup complete."
Write-Host "Use Webpack for frontend commands with the shared node_modules junction: npm run build -- --webpack"
Write-Host "Do not install packages from this worktree; update shared dependencies from main."
