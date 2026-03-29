$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$path = "c:\Users\Burda\Documents\New project\frontend\src\assets\wowpumpLOGO.png"
$bmp = [System.Drawing.Bitmap]::FromFile($path)

try {
  $source = $bmp.GetPixel(0, 0)
  $target = [System.Drawing.Color]::FromArgb(255, 21, 24, 33)
  $threshold = 42

  for ($y = 0; $y -lt $bmp.Height; $y++) {
    for ($x = 0; $x -lt $bmp.Width; $x++) {
      $p = $bmp.GetPixel($x, $y)
      $diff =
        [Math]::Abs([int]$p.R - [int]$source.R) +
        [Math]::Abs([int]$p.G - [int]$source.G) +
        [Math]::Abs([int]$p.B - [int]$source.B)

      if ($diff -le $threshold) {
        $bmp.SetPixel($x, $y, $target)
      }
    }
  }

  $tempPath = [System.IO.Path]::ChangeExtension($path, ".tinted.png")
  $bmp.Save($tempPath, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
  $bmp.Dispose()
}

Move-Item -Force $tempPath $path
Write-Output "Logo background tinted."
