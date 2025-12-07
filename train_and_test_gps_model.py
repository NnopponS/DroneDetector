#!/usr/bin/env python3
"""
Script สำหรับเทรนและทดสอบ GPS Model
ใช้ข้อมูลจาก P2_DATA_TRAIN สำหรับเทรน และ P2_DATA_TEST สำหรับทดสอบ
"""

import os
import cv2
import numpy as np
from pathlib import Path
from gps_model_pytorch import GPSModelPyTorch
from drone_detector import DroneDetector, ShapeCodeLibrary

def load_training_data(train_dir):
    """โหลดข้อมูลจาก P2_DATA_TRAIN"""
    train_dir = Path(train_dir)
    samples = []
    
    print(f"กำลังโหลดข้อมูลจาก {train_dir}...")
    
    # ตรวจสอบว่ามี shape code library หรือไม่
    shape_cache = Path("shape_codes.json")
    if not shape_cache.exists():
        print("⚠️  ไม่พบ shape_codes.json กำลังสร้างใหม่...")
        shape_lib = ShapeCodeLibrary()
        shape_lib.build()
        shape_lib.save()
    else:
        shape_lib = ShapeCodeLibrary(cache_path=shape_cache)
        shape_lib.ensure_ready()
    
    detector = DroneDetector(shape_lib)
    
    image_count = 0
    detection_count = 0
    
    for img_path in sorted(train_dir.glob("*.jpg")):
        csv_path = img_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        
        # อ่าน GPS จาก CSV
        try:
            with open(csv_path, "r", encoding="utf-8") as fp:
                lines = fp.read().strip().splitlines()
            if len(lines) < 2:
                continue
            # ข้าม header และอ่านข้อมูล
            lat, lon, alt = map(float, lines[1].split(","))
        except Exception as e:
            print(f"⚠️  ไม่สามารถอ่าน {csv_path}: {e}")
            continue
        
        # อ่านภาพและตรวจจับ drone
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_count += 1
        detections, _ = detector.detect(image, debug=False, min_conf=0.12)
        
        if not detections:
            continue
        
        # ใช้ detection ที่ใหญ่ที่สุด
        det = max(detections, key=lambda d: d.area)
        samples.append((det.center[0], det.center[1], lat, lon, alt))
        detection_count += 1
        
        if image_count % 50 == 0:
            print(f"  ประมวลผลแล้ว {image_count} ภาพ, พบ {detection_count} samples")
    
    print(f"✓ โหลดข้อมูลเสร็จ: {len(samples)} samples จาก {image_count} ภาพ")
    return samples

def train_model(samples, model_path="gps_model_pytorch.pkl", epochs=100):
    """เทรนโมเดล GPS"""
    if len(samples) < 10:
        print(f"❌ ข้อมูลไม่เพียงพอ: ต้องการอย่างน้อย 10 samples แต่มี {len(samples)}")
        return None
    
    print(f"\n{'='*60}")
    print(f"เริ่มเทรน GPS Model")
    print(f"{'='*60}")
    print(f"จำนวน samples: {len(samples)}")
    print(f"Epochs: {epochs}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}\n")
    
    # สร้างโมเดล
    model = GPSModelPyTorch()
    
    # เทรนโมเดล
    success = model.train(
        data_points=samples,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2
    )
    
    if not success:
        print("❌ การเทรนล้มเหลว")
        return None
    
    # บันทึกโมเดล
    if model.save(model_path):
        print(f"\n✓ บันทึกโมเดลสำเร็จ: {model_path}")
        return model
    else:
        print("❌ ไม่สามารถบันทึกโมเดลได้")
        return None

def test_model(model, test_dir, output_file="test_results.txt"):
    """ทดสอบโมเดลด้วยข้อมูลจาก P2_DATA_TEST"""
    test_dir = Path(test_dir)
    
    if not test_dir.exists():
        print(f"❌ ไม่พบโฟลเดอร์ทดสอบ: {test_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"เริ่มทดสอบโมเดล")
    print(f"{'='*60}")
    print(f"Test directory: {test_dir}")
    print(f"{'='*60}\n")
    
    # ตรวจสอบว่ามี shape code library หรือไม่
    shape_cache = Path("shape_codes.json")
    if not shape_cache.exists():
        print("⚠️  ไม่พบ shape_codes.json กำลังสร้างใหม่...")
        shape_lib = ShapeCodeLibrary()
        shape_lib.build()
        shape_lib.save()
    else:
        shape_lib = ShapeCodeLibrary(cache_path=shape_cache)
        shape_lib.ensure_ready()
    
    detector = DroneDetector(shape_lib, geo_regressor=None)
    
    results = []
    image_count = 0
    detection_count = 0
    
    for img_path in sorted(test_dir.glob("*.jpg")):
        csv_path = img_path.with_suffix(".csv")
        
        # อ่าน GPS จริงจาก CSV (ถ้ามี)
        true_lat = true_lon = true_alt = None
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as fp:
                    lines = fp.read().strip().splitlines()
                if len(lines) >= 2:
                    true_lat, true_lon, true_alt = map(float, lines[1].split(","))
            except:
                pass
        
        # อ่านภาพและตรวจจับ drone
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_count += 1
        detections, _ = detector.detect(image, debug=False, min_conf=0.12)
        
        if not detections:
            continue
        
        detection_count += 1
        det = max(detections, key=lambda d: d.area)
        
        # ทำนาย GPS
        pred_lat, pred_lon, pred_alt = model.predict(det.center[0], det.center[1])
        
        if pred_lat is None:
            continue
        
        result = {
            'image': img_path.name,
            'pixel_x': det.center[0],
            'pixel_y': det.center[1],
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'pred_alt': pred_alt,
            'true_lat': true_lat,
            'true_lon': true_lon,
            'true_alt': true_alt
        }
        
        # คำนวณ error ถ้ามีค่า true
        if true_lat is not None:
            lat_error = abs(pred_lat - true_lat)
            lon_error = abs(pred_lon - true_lon)
            alt_error = abs(pred_alt - true_alt)
            result['lat_error'] = lat_error
            result['lon_error'] = lon_error
            result['alt_error'] = alt_error
        
        results.append(result)
        
        if image_count % 50 == 0:
            print(f"  ทดสอบแล้ว {image_count} ภาพ, พบ {detection_count} detections")
    
    # แสดงผลลัพธ์
    print(f"\n✓ ทดสอบเสร็จ: {len(results)} detections จาก {image_count} ภาพ\n")
    
    if results:
        # คำนวณสถิติ
        if any(r.get('true_lat') is not None for r in results):
            lat_errors = [r['lat_error'] for r in results if 'lat_error' in r]
            lon_errors = [r['lon_error'] for r in results if 'lon_error' in r]
            alt_errors = [r['alt_error'] for r in results if 'alt_error' in r]
            
            print("สถิติความคลาดเคลื่อน:")
            if lat_errors:
                print(f"  Latitude - Mean: {np.mean(lat_errors):.6f}, Std: {np.std(lat_errors):.6f}, Max: {np.max(lat_errors):.6f}")
            if lon_errors:
                print(f"  Longitude - Mean: {np.mean(lon_errors):.6f}, Std: {np.std(lon_errors):.6f}, Max: {np.max(lon_errors):.6f}")
            if alt_errors:
                print(f"  Altitude - Mean: {np.mean(alt_errors):.2f}, Std: {np.std(alt_errors):.2f}, Max: {np.max(alt_errors):.2f}")
        
        # บันทึกผลลัพธ์
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ผลการทดสอบ GPS Model\n")
            f.write("="*80 + "\n\n")
            for r in results:
                f.write(f"Image: {r['image']}\n")
                f.write(f"  Pixel: ({r['pixel_x']:.1f}, {r['pixel_y']:.1f})\n")
                f.write(f"  Predicted: Lat={r['pred_lat']:.6f}, Lon={r['pred_lon']:.6f}, Alt={r['pred_alt']:.2f}\n")
                if r['true_lat'] is not None:
                    f.write(f"  True: Lat={r['true_lat']:.6f}, Lon={r['true_lon']:.6f}, Alt={r['true_alt']:.2f}\n")
                    f.write(f"  Error: Lat={r['lat_error']:.6f}, Lon={r['lon_error']:.6f}, Alt={r['alt_error']:.2f}\n")
                f.write("\n")
        
        print(f"\n✓ บันทึกผลการทดสอบ: {output_file}")

def main():
    """ฟังก์ชันหลัก"""
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    base_dir = Path(os.getcwd())
    train_dir = base_dir / "P2_DATA_TRAIN"
    test_dir = base_dir / "P2_DATA_TEST"
    model_path = base_dir / "gps_model_pytorch.pkl"
    
    print("="*60, flush=True)
    print("GPS Model Training and Testing", flush=True)
    print("="*60, flush=True)
    
    # 1. โหลดข้อมูลเทรน
    samples = load_training_data(train_dir)
    
    if not samples:
        print("❌ ไม่พบข้อมูลสำหรับเทรน")
        return
    
    # 2. เทรนโมเดล
    model = train_model(samples, model_path=str(model_path), epochs=100)
    
    if not model:
        print("❌ ไม่สามารถเทรนโมเดลได้")
        return
    
    # 3. ทดสอบโมเดล
    if test_dir.exists():
        test_model(model, test_dir, output_file="test_results.txt")
    else:
        print(f"⚠️  ไม่พบโฟลเดอร์ทดสอบ: {test_dir}")
        print("   ข้ามการทดสอบ")
    
    print("\n" + "="*60)
    print("✓ เสร็จสิ้น!")
    print("="*60)

if __name__ == "__main__":
    main()
