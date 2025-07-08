import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import deque
import torch

# ==================== GPU KONFİGÜRASYONU ====================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n⚡ Kullanılan cihaz: {device.upper()}")
if device == 'cuda':
    print(f"🔍 GPU Bilgisi: {torch.cuda.get_device_name(0)}")
else:
    print("🔍 GPU Bilgisi: GPU bulunamadı")

# 📦 YOLOv8 modelini GPU'ya yükle
model = YOLO("yolov8x.pt").to(device)

# 🔢 Her kaç karede bir YOLO çalıştırılsın?
SKIP_FRAMES = 2

# Yön tespiti için gerekli parametreler
MIN_TRACK_LENGTH = 10     # Yön tespiti için minimum takip uzunluğu
DIRECTION_THRESHOLD = 15   # Yön kararı için minimum piksel hareketi
CONSISTENCY_FRAMES = 5     # Tutarlı yön için gereken frame sayısı

# GELİŞMİŞ FİLTRELEME PARAMETRELERİ - DAHA HASSAS AYARLAR
IoU_THRESHOLD = 0.6        # Çok yüksek IoU eşiği - sadece gerçekten üst üste olanlar
HEIGHT_RATIO_THRESHOLD = 0.8  # Daha esnek boyut kontrolü
SPEED_THRESHOLD = 15.0     # Daha yüksek hız eşiği (piksel/frame)
MIN_SPEED = 0.2           # Daha düşük minimum hareket hızı
MOVEMENT_SMOOTHNESS_THRESHOLD = 0.7  # Daha yüksek düzgünlük eşiği
VERTICAL_MOVEMENT_RATIO = 0.05  # Daha düşük dikey hareket eşiği
PEDESTRIAN_AREA_RATIO = 0.7  # Yayaların genellikle bulunduğu alan (alt %70)

# Video dosyası (kodla aynı klasörde)
VIDEO_PATH = "part_11.mp4"

# Yeni sayım setleri oluştur
counted_ids_A = set()  # Kameraya gelenler (A yönü)
counted_ids_B = set()  # Kameradan uzaklaşanlar (B yönü)

# 📹 Video aç
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Video açılamadı!")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Çıktı dosya isimleri
base_name = os.path.splitext(VIDEO_PATH)[0]
output_video_path = f"{base_name}_islenmis.mp4"
excel_output_path = f"{base_name}_totalCounts.xlsx"

# VideoWriter oluştur
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
results = None

# Gelişmiş takip geçmişi
track_history = {}

def calculate_iou(box1, box2):
    """İki kutu arasındaki Intersection over Union (IoU) değerini hesaplar"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_speed(positions):
    """Pozisyon geçmişinden hızı hesaplar"""
    if len(positions) < 2:
        return 0
    
    distances = []
    for i in range(1, len(positions)):
        x1, y1 = positions[i-1]
        x2, y2 = positions[i]
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        distances.append(distance)
    
    return np.mean(distances) if distances else 0

def calculate_movement_smoothness(positions):
    """Hareket düzgünlüğünü hesaplar (0-1 arası, 1 = çok düzgün)"""
    if len(positions) < 3:
        return 0
    
    # İkinci türev hesaplama (ivme değişimi)
    accelerations = []
    for i in range(2, len(positions)):
        p1, p2, p3 = positions[i-2], positions[i-1], positions[i]
        
        # Hız vektörleri
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # İvme (hız değişimi)
        acc = np.sqrt((v2[0] - v1[0])**2 + (v2[1] - v1[1])**2)
        accelerations.append(acc)
    
    if not accelerations:
        return 0
    
    # Düşük varyans = düzgün hareket
    variance = np.var(accelerations)
    smoothness = 1 / (1 + variance)  # Normalize et
    return smoothness

def analyze_walking_pattern(positions):
    """Yürüme deseni analizi"""
    if len(positions) < MIN_TRACK_LENGTH:
        return False
    
    # Dikey hareket analizi (yürürken hafif yukarı-aşağı hareket)
    y_positions = [pos[1] for pos in positions]
    y_diff = np.diff(y_positions)
    y_variance = np.var(y_diff)
    
    # Toplam hareket mesafesi
    total_movement = 0
    for i in range(1, len(positions)):
        total_movement += np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                  (positions[i][1] - positions[i-1][1])**2)
    
    if total_movement == 0:
        return False
    
    # Dikey hareketin toplam hareket içindeki oranı
    vertical_ratio = np.sqrt(y_variance) / (total_movement / len(positions))
    
    return vertical_ratio > VERTICAL_MOVEMENT_RATIO

def is_person_on_vehicle(person_box, vehicle_boxes, track_id, frame):
    """Gelişmiş sürücü tespiti - Yandan görüş için optimize edilmiş"""
    if not vehicle_boxes:
        return False
        
    px1, py1, px2, py2 = person_box
    p_center_x = (px1 + px2) / 2
    p_center_y = (py1 + py2) / 2
    p_height = py2 - py1
    p_width = px2 - px1
    
    for vbox in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vbox
        v_height = vy2 - vy1
        v_width = vx2 - vx1
        
        # 1. Gelişmiş IoU kontrolü (daha düşük eşik)
        iou = calculate_iou(person_box, vbox)
        if iou > 0.35:  # Daha düşük IoU eşiği (yandan görüş için)
            cv2.rectangle(frame, (int(vx1), int(vy1)), (int(vx2), int(vy2)), (0, 255, 255), 2)
            cv2.putText(frame, f"IoU:{iou:.2f}", (int(vx1), int(vy1) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            return True
            
        # 2. Boyut ve konum analizi (yandan görüş için optimize)
        size_ratio = p_height / v_height
        if (0.25 < size_ratio < 0.65 and  # Sürücü boyutu aracın %25-65'i arasında
            abs(p_center_x - (vx1+vx2)/2) < v_width*0.5 and  # Yatay hizalama
            p_center_y < (vy1+vy2)/2 + v_height*0.3):  # Aracın üst yarısında
            
            cv2.rectangle(frame, (int(vx1), int(vy1)), (int(vx2), int(vy2)), (0, 200, 255), 2)
            cv2.putText(frame, f"RATIO:{size_ratio:.2f}", (int(vx1), int(vy1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            return True
    
    return False


def is_pedestrian(track_id, current_position, frame_height):
    """Yaya tespiti - Sadece hız ve hareket analizine dayalı"""
    if track_id not in track_history or len(track_history[track_id]['positions']) < 8:
        return True
        
    positions = list(track_history[track_id]['positions'])
    
    # 1. Aşırı hız kontrolü (daha toleranslı)
    speed = calculate_speed(positions)
    if speed > SPEED_THRESHOLD * 1.5:  # %50 daha toleranslı
        return False
        
    return True

print(f"\n🔁 İşleniyor: {VIDEO_PATH}...")

# İşleme döngüsü
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔍 Her SKIP_FRAMES karede bir YOLO çalıştır
    if frame_count % SKIP_FRAMES == 0:
        # Daha fazla sınıf dahil et (araçlar için)
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        results = model.track(frame, persist=True, classes=[0, 1, 2, 3, 5, 7], device=device)

    current_centers = {}

    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()

        # Araç kutularını al (daha geniş sınıf listesi)
        vehicle_boxes = [box for box, cls, conf in zip(boxes, classes, confidences) 
                         if cls in [1, 2, 3, 5, 7] and conf > 0.5]
        
        # İnsan kutularını işle
        person_boxes = [(box, track_id, conf) for box, track_id, cls, conf in zip(boxes, ids, classes, confidences) 
                        if cls == 0 and conf > 0.6]  # Güven eşiğini artır

        for person_box, track_id, conf in person_boxes:
            # Araç üzerindeki insanları filtrele
            if is_person_on_vehicle(person_box, vehicle_boxes, track_id, frame):
                continue
                
            # Merkez noktasını hesapla
            x1, y1, x2, y2 = map(int, person_box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_centers[track_id] = (center_x, center_y)

            # Takip geçmişini güncelle
            if track_id not in track_history:
                track_history[track_id] = {
                    'positions': deque(maxlen=MIN_TRACK_LENGTH*2),
                    'direction_count': 0,
                    'last_direction': None,
                    'counted': False
                }
            
            track_history[track_id]['positions'].append((center_x, center_y))
            
            # Yaya kontrolü - sadece kesin araç sürücülerini filtrele
            if not is_pedestrian(track_id, (center_x, center_y), height):
                # Araç sürücüsü olarak işaretle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "RIDER", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                # Debug: Hangi kritere takıldığını göster
                if len(track_history[track_id]['positions']) >= 8:
                    speed = calculate_speed(list(track_history[track_id]['positions']))
                    cv2.putText(frame, f"S:{speed:.1f}", (x1, y1 - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                continue
            
            # Yön tespiti
            if len(track_history[track_id]['positions']) > MIN_TRACK_LENGTH:
                first_x, first_y = track_history[track_id]['positions'][0]
                last_x, last_y = track_history[track_id]['positions'][-1]
                
                y_movement = last_y - first_y
                
                current_direction = None
                if y_movement < -DIRECTION_THRESHOLD:
                    current_direction = 'A'
                elif y_movement > DIRECTION_THRESHOLD:
                    current_direction = 'B'
                
                if current_direction:
                    if track_history[track_id]['last_direction'] == current_direction:
                        track_history[track_id]['direction_count'] += 1
                    else:
                        track_history[track_id]['direction_count'] = 1
                    
                    track_history[track_id]['last_direction'] = current_direction
                    
                    if (track_history[track_id]['direction_count'] >= CONSISTENCY_FRAMES and 
                        not track_history[track_id]['counted']):
                        
                        if current_direction == 'A':
                            counted_ids_A.add(track_id)
                            color = (255, 0, 0)
                        else:
                            counted_ids_B.add(track_id)
                            color = (0, 255, 0)
                        
                        track_history[track_id]['counted'] = True
                else:
                    color = (0, 255, 255)
            else:
                color = (0, 255, 255)
            
            # Kutuyu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Hız bilgisini göster (debug için)
            if len(track_history[track_id]['positions']) >= 2:
                speed = calculate_speed(list(track_history[track_id]['positions']))
                cv2.putText(frame, f"ID:{track_id} S:{speed:.1f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Hareket yönünü göster
            if len(track_history[track_id]['positions']) > 1:
                prev_x, prev_y = track_history[track_id]['positions'][-2]
                cv2.arrowedLine(frame, (prev_x, prev_y), (center_x, center_y), color, 2)

    # Eski takipleri temizle
    lost_ids = [tid for tid in track_history if tid not in current_centers]
    for tid in lost_ids:
        if tid in track_history:
            if len(track_history[tid]['positions']) < MIN_TRACK_LENGTH/2:
                del track_history[tid]

    # Sayımları ekrana yaz
    cv2.putText(frame, f"Total A (Towards Camera): {len(counted_ids_A)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Total B (Away from Camera): {len(counted_ids_B)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Filtreleme bilgisi
    cv2.putText(frame, "Orange=Rider, Yellow=Uncertain, Blue/Green=Pedestrian", (10, height-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    out.write(frame)

    # Görüntüleme
    cv2.imshow('İşlenen Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_count % 100 == 0:
        print(f"📊 Frame: {frame_count} | A: {len(counted_ids_A)}, B: {len(counted_ids_B)}")

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()

# Excel dosyası oluştur
results_df = pd.DataFrame({
    "Direction": ["Towards Camera (A)", "Away from Camera (B)"],
    "Count": [len(counted_ids_A), len(counted_ids_B)],
    "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 2
})

results_df.to_excel(excel_output_path, index=False)

print(f"\n✅ Tamamlandı: {VIDEO_PATH}")
print(f"🎥 İşlenmiş video: {output_video_path}")
print(f"📊 Sayım sonuçları: {excel_output_path}")
print(f"[➡️ A YONU] Kameraya gelenler: {len(counted_ids_A)}")
print(f"[⬅️ B YONU] Kameradan uzaklaşanlar: {len(counted_ids_B)}\n")