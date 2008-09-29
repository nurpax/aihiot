@echo off
setlocal
rem ---- Configure this for your installation dirs -----
set CUDA_SDK_OPTS=-I "C:\Program Files\NVIDIA Corporation\NVIDIA CUDA SDK\common/inc" -L "C:\Program Files\NVIDIA Corporation\NVIDIA CUDA SDK\common/lib"
@set LIB_OPTS=rendercheckgl32.lib glew32.lib cutil32D.lib

echo --------------------------------------

if "%1" == "emu" goto :emu
if "%1" == "hw" goto :hw

:emu
echo Building for device emulation

set DEVICE_SWITCH=--device-emulation
goto :build
:hw
echo Building for real HW
set DEVICE_SWITCH=

:build

echo --------------------------------------
nvcc %DEVICE_SWITCH% %CUDA_SDK_OPTS% %LIB_OPTS% glutmain.cpp triangle_render.cu
