Add-Type -AssemblyName System.Drawing

$OutputDir = Join-Path (Get-Location) "test-images"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function New-ConcreteTexture {
    param(
        [string]$Path,
        [int]$Width = 960,
        [int]$Height = 640,
        [string]$Mode = "crack"
    )

    $bitmap = New-Object System.Drawing.Bitmap $Width, $Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

    $random = New-Object System.Random
    $baseColor = [System.Drawing.Color]::FromArgb(178, 178, 170)
    $graphics.Clear($baseColor)

    for ($x = 0; $x -lt $Width; $x += 4) {
        for ($y = 0; $y -lt $Height; $y += 4) {
            $noise = $random.Next(-18, 19)
            $shade = [Math]::Max(110, [Math]::Min(215, 178 + $noise))
            $brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb($shade, $shade, [Math]::Max(100, $shade - 8)))
            $graphics.FillRectangle($brush, $x, $y, 4, 4)
            $brush.Dispose()
        }
    }

    for ($i = 0; $i -lt 120; $i++) {
        $dotBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb($random.Next(70, 120), $random.Next(70, 120), $random.Next(65, 110)))
        $graphics.FillEllipse($dotBrush, $random.Next(0, $Width), $random.Next(0, $Height), $random.Next(2, 7), $random.Next(2, 7))
        $dotBrush.Dispose()
    }

    if ($Mode -eq "crack") {
        $penShadow = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(55, 55, 50)), 8
        $penMain = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(18, 18, 16)), 4
        $points = @(
            [System.Drawing.Point]::new(120, 70),
            [System.Drawing.Point]::new(185, 145),
            [System.Drawing.Point]::new(250, 188),
            [System.Drawing.Point]::new(310, 280),
            [System.Drawing.Point]::new(405, 335),
            [System.Drawing.Point]::new(490, 455),
            [System.Drawing.Point]::new(610, 575)
        )
        $graphics.DrawLines($penShadow, $points)
        $graphics.DrawLines($penMain, $points)
        $graphics.DrawLine($penMain, 310, 280, 420, 240)
        $graphics.DrawLine($penMain, 405, 335, 360, 430)
        $penShadow.Dispose()
        $penMain.Dispose()
    }

    if ($Mode -eq "spalling") {
        $darkBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(82, 78, 70))
        $edgePen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(45, 43, 38)), 5
        $graphics.FillEllipse($darkBrush, 270, 160, 390, 270)
        $graphics.DrawEllipse($edgePen, 270, 160, 390, 270)
        for ($i = 0; $i -lt 35; $i++) {
            $graphics.FillEllipse($darkBrush, $random.Next(290, 630), $random.Next(180, 410), $random.Next(8, 28), $random.Next(8, 28))
        }
        $darkBrush.Dispose()
        $edgePen.Dispose()
    }

    if ($Mode -eq "rebar") {
        $damageBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(88, 80, 70))
        $rustPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(130, 62, 30)), 18
        $darkPen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(58, 50, 44)), 4
        $graphics.FillRectangle($damageBrush, 210, 210, 560, 160)
        $graphics.DrawLine($rustPen, 235, 260, 750, 260)
        $graphics.DrawLine($rustPen, 235, 320, 750, 320)
        $graphics.DrawLine($darkPen, 235, 260, 750, 260)
        $graphics.DrawLine($darkPen, 235, 320, 750, 320)
        $damageBrush.Dispose()
        $rustPen.Dispose()
        $darkPen.Dispose()
    }

    $bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Jpeg)
    $graphics.Dispose()
    $bitmap.Dispose()
}

New-ConcreteTexture -Path (Join-Path $OutputDir "sample.jpg") -Mode "crack"
New-ConcreteTexture -Path (Join-Path $OutputDir "spalling_sample.jpg") -Mode "spalling"
New-ConcreteTexture -Path (Join-Path $OutputDir "rebar_sample.jpg") -Mode "rebar"

Write-Host "Created sample images in $OutputDir"
