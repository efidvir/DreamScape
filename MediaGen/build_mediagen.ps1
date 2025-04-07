# build_mediagen.ps1

Write-Host "Building MediaGen container..." -ForegroundColor Green

# Ensure directories exist (Windows PowerShell syntax)
if (-not (Test-Path -Path ".\generated_media")) {
    New-Item -Path ".\generated_media" -ItemType Directory
    Write-Host "Created generated_media directory" -ForegroundColor Yellow
}

if (-not (Test-Path -Path ".\model_cache")) {
    New-Item -Path ".\model_cache" -ItemType Directory
    Write-Host "Created model_cache directory" -ForegroundColor Yellow
}

# Windows doesn't use chmod, but we can try to remove any restrictions
try {
    Get-Acl ".\generated_media" | Set-Acl ".\generated_media"
    Get-Acl ".\model_cache" | Set-Acl ".\model_cache"
    Write-Host "Set permissions on directories" -ForegroundColor Yellow
} catch {
    Write-Host "Could not set permissions. May need administrator privileges." -ForegroundColor Red
}

# Enable BuildKit for better caching (Windows syntax)
$env:DOCKER_BUILDKIT = 1

# Build with special focus on the mediagen service
Write-Host "Building MediaGen service..." -ForegroundColor Cyan
docker compose build mediagen

# If successful, start the service
if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful, starting services..." -ForegroundColor Green
    docker compose up -d mediagen
} else {
    Write-Host "Build failed. See error messages above." -ForegroundColor Red
    exit 1
}