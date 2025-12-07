# คำแนะนำการ Push โปรเจกต์ไปยัง GitHub

## ขั้นตอนที่ 1: สร้าง Repository บน GitHub

1. ไปที่ https://github.com/new
2. ตั้งชื่อ repository (แนะนำ: `drone-detection-system` หรือ `drone-detector`)
3. เลือก **Public** หรือ **Private** ตามต้องการ
4. **อย่าเลือก** "Initialize with README" (เรามี README.md แล้ว)
5. **อย่าเลือก** "Add .gitignore" (เรามีแล้ว)
6. **อย่าเลือก** "Choose a license" (ถ้าไม่ต้องการ)
7. คลิก **"Create repository"**

## ขั้นตอนที่ 2: เพิ่ม Remote และ Push

หลังจากสร้าง repository แล้ว ให้รันคำสั่งต่อไปนี้ใน terminal:

### Windows (PowerShell/CMD):

```bash
# เปลี่ยน USERNAME และ REPO_NAME ตามที่คุณสร้าง
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### หรือใช้สคริปต์ที่เตรียมไว้:

```bash
# Windows
setup_github.bat

# Linux/Mac
chmod +x setup_github.sh
./setup_github.sh
```

## ตัวอย่างคำสั่ง

สมมติว่า:
- GitHub username: `yourusername`
- Repository name: `drone-detection-system`

```bash
git remote add origin https://github.com/yourusername/drone-detection-system.git
git branch -M main
git push -u origin main
```

## ถ้าเกิด Error

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/USERNAME/REPO_NAME.git
```

### Error: "Authentication failed"
- ใช้ Personal Access Token แทน password
- สร้าง token ที่: https://github.com/settings/tokens
- เลือก scope: `repo` (full control)

### Error: "Permission denied"
- ตรวจสอบว่า username และ repository name ถูกต้อง
- ตรวจสอบว่า repository ถูกสร้างแล้วบน GitHub

## หลังจาก Push สำเร็จ

โปรเจกต์ของคุณจะอยู่ที่:
```
https://github.com/USERNAME/REPO_NAME
```

## การ Push ครั้งต่อไป

เมื่อมีการเปลี่ยนแปลงไฟล์:

```bash
git add .
git commit -m "Description of changes"
git push
```

## หมายเหตุ

- ไฟล์ที่อยู่ใน `.gitignore` จะไม่ถูก push (เช่น `.pkl`, `__pycache__/`, `.venv/`)
- ข้อมูล training (Drones/, Birds/, P2_DATA_TRAIN/) อาจมีขนาดใหญ่ ถ้าต้องการ push ต้องลบออกจาก `.gitignore`
- สำหรับข้อมูลขนาดใหญ่ แนะนำให้ใช้ Git LFS (Large File Storage)
