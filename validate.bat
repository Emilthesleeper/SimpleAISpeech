@echo off
cd /d "C:\Users\emilw\Documents\Dev\Python\AI Neurales\SimpleAISpeech"

echo ===== Command 1: git status --short =====
git status --short
echo.

echo ===== Command 2: git diff --check =====
git diff --check
echo.

echo ===== Command 3: python -m py_compile main.py =====
python -m py_compile main.py
echo.

echo ===== Command 4: git diff -- main.py =====
git diff -- main.py
echo.
