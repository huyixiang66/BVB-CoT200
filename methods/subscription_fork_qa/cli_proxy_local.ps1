param(
    [Parameter(Mandatory=$true)][ValidateSet('init','login','serve')][string]$Action,
    [Parameter(Mandatory=$true)][string]$Executable,
    [string]$StateDir = (Join-Path $env:USERPROFILE '.bvb-cliproxy'),
    [int]$Port = 18318
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw 'CLIProxyAPI executable not found.' }
$exe = (Resolve-Path -LiteralPath $Executable).Path
$config = Join-Path $StateDir 'config.yaml'
if ($Action -eq 'init') {
    if (Test-Path -LiteralPath $config) { throw 'Configuration already exists; refusing to overwrite it.' }
    if (-not $env:BVB_CLI_PROXY_KEY -or $env:BVB_CLI_PROXY_KEY.Length -lt 24) { throw 'Set BVB_CLI_PROXY_KEY to a random value of at least 24 characters.' }
    New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
    $authDir = Join-Path $StateDir 'auth'
    $keyJson = ConvertTo-Json -Compress -InputObject $env:BVB_CLI_PROXY_KEY
    $authJson = ConvertTo-Json -Compress -InputObject $authDir
    $yaml = "config-version: 8`nserver:`n  host: 127.0.0.1`n  port: $Port`naccess:`n  api-keys:`n    - $keyJson`noauth:`n  auth-dir: $authJson`nrequests:`n  proxy-url: ''`n"
    [System.IO.File]::WriteAllText($config, $yaml, [System.Text.UTF8Encoding]::new($false))
    Write-Host 'Local configuration created. Keep it private and do not commit it.'
    exit 0
}
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) { throw 'Run init first.' }
if ($Action -eq 'login') { & $exe -config $config -codex-login }
if ($Action -eq 'serve') { & $exe -config $config }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
