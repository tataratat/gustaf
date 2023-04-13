REM Description: Run all examples and check for errors
REM Author: ChatGPT based on the file run_all_examples.sh by clemens.fricke
@echo off

setlocal enabledelayedexpansion

REM run all examples
REM iterate over all files with .py extension
for %%f in (*.py) do (
    REM run the file
    echo Running %%f
    REM run file and check for errors
    python "%%f"
    REM check if the last command was successful
    if !errorlevel! neq 0 (
        echo Error running %%f
        REM push file name into a list
        set failed_files=!failed_files! %%f
    )
    timeout /t 5
    echo Done
)

echo The following examples had errors: %failed_files%
