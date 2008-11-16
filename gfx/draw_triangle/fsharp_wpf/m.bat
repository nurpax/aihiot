@echo off
set FSC="c:\program files\FSharp-1.9.6.2\bin\fsc.exe"
set SQLITE_NET_DLL="c:\program files\SQLite.NET\bin\System.Data.SQLite.DLL"

REM set REFS=-I "C:\Program Files\Reference Assemblies\Microsoft\WinFx\v3.0"
set REFS=-I "C:\Program Files\Reference Assemblies\Microsoft\Framework\v3.0"
set REFS=%REFS% -r "PresentationCore.dll" -r "PresentationFramework.dll" -r "WindowsBase.dll"
%FSC% -g -r %SQLITE_NET_DLL% %REFS% main.fs
