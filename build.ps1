# SharpDiffusion Build and Publish Script
# Copyright (C) Gianni Rosa Gallina.
# Licensed under the Apache License, Version 2.0.

$env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = 1
$env:DOTNET_CLI_TELEMETRY_OPTOUT = 1

# Build Windows library
dotnet publish .\src\SharpDiffusion.sln -c Release -f net7.0 -r win-x64 --self-contained

echo "Package available at: .\src\SharpDiffusion\bin\Release\net7.0\win-x64\publish"