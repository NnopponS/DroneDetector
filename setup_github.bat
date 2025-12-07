@echo off
REM Script สำหรับ setup GitHub repository (Windows)

echo ==========================================
echo GitHub Repository Setup
echo ==========================================
echo.
echo ขั้นตอนที่ 1: สร้าง repository ใหม่บน GitHub
echo   1. ไปที่ https://github.com/new
echo   2. ตั้งชื่อ repository (เช่น: drone-detection-system)
echo   3. เลือก Public หรือ Private
echo   4. อย่าเลือก 'Initialize with README' (เรามีแล้ว)
echo   5. คลิก 'Create repository'
echo.
pause

echo.
echo ขั้นตอนที่ 2: เพิ่ม remote และ push
echo.
set /p GITHUB_USER="ใส่ GitHub username: "
set /p REPO_NAME="ใส่ repository name: "

echo.
echo กำลังเพิ่ม remote...
git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git

echo.
echo กำลัง push ไปยัง GitHub...
git branch -M main
git push -u origin main

echo.
echo ✅ เสร็จสิ้น! โปรเจกต์ของคุณอยู่ที่:
echo    https://github.com/%GITHUB_USER%/%REPO_NAME%
echo.
pause
