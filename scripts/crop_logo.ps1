$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$path = "c:\Users\Burda\Documents\New project\frontend\src\assets\wowpumpLOGO.png"
$backup = "c:\Users\Burda\Documents\New project\frontend\src\assets\wowpumpLOGO.original.png"

if (-not (Test-Path $backup)) {
  Copy-Item $path $backup
}

$bmp = [System.Drawing.Bitmap]::FromFile($path)
try {
  $bg = $bmp.GetPixel(0, 0)
  $threshold = 18
  $minX = [int]$bmp.Width
  $minY = [int]$bmp.Height
  $maxX = -1
  $maxY = -1

  for ($y = 0; $y -lt $bmp.Height; $y++) {
    for ($x = 0; $x -lt $bmp.Width; $x++) {
      $p = $bmp.GetPixel($x, $y)
      $diff =
        [Math]::Abs([int]$p.R - [int]$bg.R) +
        [Math]::Abs([int]$p.G - [int]$bg.G) +
        [Math]::Abs([int]$p.B - [int]$bg.B)

      if ($diff -gt $threshold) {
        if ($x -lt $minX) { $minX = [int]$x }
        if ($y -lt $minY) { $minY = [int]$y }
        if ($x -gt $maxX) { $maxX = [int]$x }
        if ($y -gt $maxY) { $maxY = [int]$y }
      }
    }
  }

  if ($maxX -lt 0 -or $maxY -lt 0) {
    throw "Could not detect non-background content."
  }

  $padding = 8
  $minX = [Math]::Max(0, $minX - $padding)
  $minY = [Math]::Max(0, $minY - $padding)
  $maxX = [Math]::Min($bmp.Width - 1, $maxX + $padding)
  $maxY = [Math]::Min($bmp.Height - 1, $maxY + $padding)

  $width = [int](($maxX - $minX) + 1)
  $height = [int](($maxY - $minY) + 1)
  $rect = New-Object System.Drawing.Rectangle($minX, $minY, $width, $height)
  $cropped = $bmp.Clone($rect, $bmp.PixelFormat)
  try {
    $tempPath = [System.IO.Path]::ChangeExtension($path, ".cropped.png")
    $cropped.Save($tempPath, [System.Drawing.Imaging.ImageFormat]::Png)
  }
  finally {
    $cropped.Dispose()
  }

  $bmp.Dispose()
  Move-Item -Force $tempPath $path

  Write-Output ("Cropped to {0}x{1}" -f $width, $height)
}
finally {
  if ($null -ne $bmp) {
    try { $bmp.Dispose() } catch {}
  }
}
