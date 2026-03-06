param(
    [Parameter(Mandatory = $true)]
    [string]$Script,

    [Parameter(Mandatory = $true)]
    [string]$Config
)

$config = Get-Content $Config -Raw | ConvertFrom-Json
$args = @()

foreach ($prop in $config.PSObject.Properties) {
    if ($null -eq $prop.Value) {
        continue
    }

    $args += "--$($prop.Name)"

    if ($prop.Value -is [System.Collections.IEnumerable] -and -not ($prop.Value -is [string])) {
        foreach ($item in $prop.Value) {
            $args += "$item"
        }
    } else {
        $args += "$($prop.Value)"
    }
}

& uv run $Script @args
