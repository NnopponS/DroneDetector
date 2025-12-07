"""
Simple and Stable Drone Tracker
ระบบ tracking ที่เสถียรสำหรับโดรนสูงสุด 2 ตัว
"""

import numpy as np
import cv2

class DroneTracker:
    def __init__(self, max_drones=2, max_lost_frames=30, distance_threshold=500):
        """
        Args:
            max_drones: จำนวนโดรนสูงสุด (default: 2)
            max_lost_frames: เก็บ track ที่หายไปสูงสุดกี่เฟรม
            distance_threshold: ระยะห่างสูงสุดสำหรับ matching (pixels)
        """
        self.max_drones = max_drones
        self.max_lost_frames = max_lost_frames
        self.distance_threshold = distance_threshold
        self.max_jump_distance = 200  # ลดลงเพื่อป้องกัน jump มากเกินไป
        self.last_matched_positions = {}
        self.track_locked = {}
        
        self.tracks = {}
        self.next_id = 0
        
        self.colors = [
            (0, 255, 0),      # เขียว
            (0, 255, 255),    # เหลือง
        ]
    
    def update(self, detections, frame_number):
        """อัปเดต tracks ด้วย detections ใหม่"""
        if len(self.tracks) == 0:
            for det in detections[:self.max_drones]:
                self._create_track(det, frame_number)
            return detections[:self.max_drones]
        
        matched_detections = self._match_detections_optimal(detections, frame_number)
        
        for det in detections:
            if det.get('track_id') is None:
                if len(self.tracks) < self.max_drones:
                    too_close = False
                    for track_id, track in self.tracks.items():
                        last_pos = track.get('last_pos')
                        if last_pos:
                            dx = det['center_x'] - last_pos[0]
                            dy = det['center_y'] - last_pos[1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < 300:
                                too_close = True
                                break
                    
                    if not too_close:
                        self._create_track(det, frame_number)
        
        matched_track_ids = {det.get('track_id') for det in matched_detections if det.get('track_id') is not None}
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track['lost_frames'] += 1
            else:
                track['lost_frames'] = 0
        
        return matched_detections
    
    def _create_track(self, detection, frame_number):
        """สร้าง track ใหม่"""
        track_id = self.next_id
        self.next_id += 1
        
        center_x = detection['center_x']
        center_y = detection['center_y']
        
        self.tracks[track_id] = {
            'id': track_id,
            'history': [(center_x, center_y, frame_number)],
            'last_pos': (center_x, center_y),
            'last_frame': frame_number,
            'velocity': (0, 0),
            'lost_frames': 0,
            'color': self.colors[track_id % len(self.colors)],
            'total_frames': 1,
            'gps': None,
            'last_gps': None,
            'spatial_region': None
        }
        
        detection['track_id'] = track_id
    
    def _match_detections_optimal(self, detections, frame_number):
        """Match detections กับ tracks โดยใช้ optimal matching"""
        if len(detections) == 0 or len(self.tracks) == 0:
            return []
        
        tracks_list = list(self.tracks.items())
        
        # สร้าง cost matrix
        cost_matrix = np.full((len(detections), len(tracks_list)), float('inf'))
        
        for i, det in enumerate(detections):
            for j, (track_id, track) in enumerate(tracks_list):
                cost = self._calculate_cost(det, track, frame_number)
                cost_matrix[i, j] = cost
        
        # เรียง tracks ตาม priority
        def track_priority(item):
            idx, (tid, track) = item
            history_len = len(track.get('history', []))
            total_frames = track.get('total_frames', 0)
            lost_frames = track.get('lost_frames', 0)
            is_locked = self.track_locked.get(tid, False)
            # Locked tracks ได้ priority สูงสุด
            lock_bonus = 1000000000 if is_locked else 0
            return lock_bonus + history_len * 10000000 + total_frames * 1000000 - lost_frames * 10000
        
        sorted_tracks = sorted(
            enumerate(tracks_list),
            key=track_priority,
            reverse=True
        )
        
        matched_tracks = set()
        matched_detections = []
        sorted_detections = sorted(enumerate(detections), key=lambda x: x[1].get('area', 0), reverse=True)
        
        for det_idx, det in sorted_detections:
            best_track_idx = None
            best_cost = float('inf')
            second_best_cost = float('inf')
            
            for track_idx, (track_id, track) in sorted_tracks:
                if track_idx in matched_tracks:
                    continue
                
                cost = cost_matrix[det_idx, track_idx]
                
                history_len = len(track.get('history', []))
                total_frames = track.get('total_frames', 0)
                lost_frames = track.get('lost_frames', 0)
                last_pos = track.get('last_pos')
                velocity = track.get('velocity', (0, 0))
                is_locked = self.track_locked.get(track_id, False)
                
                # ถ้า track ถูก lock แล้ว ให้ลด cost มากมากมาก
                if is_locked:
                    cost *= 0.0000000000000001  # ลด cost 10,000,000,000,000,000 เท่า
                
                # ตรวจสอบ spatial region
                spatial_region = track.get('spatial_region')
                if spatial_region:
                    region_x, region_y, region_radius = spatial_region
                    dx_region = det['center_x'] - region_x
                    dy_region = det['center_y'] - region_y
                    dist_to_region = np.sqrt(dx_region*dx_region + dy_region*dy_region)
                    
                    if dist_to_region <= region_radius:
                        cost *= 0.0000000001  # ลด cost 10,000,000,000 เท่า
                    else:
                        cost *= 100.0
                
                # ตรวจสอบ spatial separation
                min_dist_to_other = float('inf')
                for other_track_id, other_track in self.tracks.items():
                    if other_track_id == track_id:
                        continue
                    other_last_pos = other_track.get('last_pos')
                    if other_last_pos:
                        dx_other = det['center_x'] - other_last_pos[0]
                        dy_other = det['center_y'] - other_last_pos[1]
                        dist_to_other = np.sqrt(dx_other*dx_other + dy_other*dy_other)
                        min_dist_to_other = min(min_dist_to_other, dist_to_other)
                
                if min_dist_to_other < 350:
                    penalty_multiplier = 1 + (350 - min_dist_to_other) / 8  # สูงสุด ~44 เท่า
                    cost *= penalty_multiplier
                
                # ตรวจสอบตำแหน่งและทิศทาง
                if last_pos:
                    dx = det['center_x'] - last_pos[0]
                    dy = det['center_y'] - last_pos[1]
                    distance_to_last = np.sqrt(dx*dx + dy*dy)
                    
                    if velocity[0] != 0 or velocity[1] != 0:
                        det_angle = np.arctan2(dy, dx)
                        vel_angle = np.arctan2(velocity[1], velocity[0])
                        angle_diff = abs(det_angle - vel_angle)
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        
                        if angle_diff < np.pi / 6:  # 30 degrees
                            cost *= 0.0000001  # ลด cost 10,000,000 เท่า
                        elif angle_diff < np.pi / 4:  # 45 degrees
                            cost *= 0.000001
                        elif angle_diff > np.pi / 2:  # 90 degrees
                            cost *= 2000.0
                
                # Track locking - lock track เมื่อมี history ยาว
                if history_len > 8 and total_frames > 6 and lost_frames <= 2:
                    if not is_locked:
                        self.track_locked[track_id] = True
                    cost *= 0.00000000000000001  # ลด cost 100,000,000,000,000,000 เท่า
                elif history_len > 4 and total_frames > 3:
                    cost *= 0.000000000001  # ลด cost 1,000,000,000,000 เท่า
                elif history_len > 2:
                    cost *= 0.000000001  # ลด cost 1,000,000,000 เท่า
                
                # เพิ่ม bonus สำหรับ track ที่ match กับตำแหน่งเดิม
                if track_id in self.last_matched_positions:
                    last_matched_pos = self.last_matched_positions[track_id]
                    dx = det['center_x'] - last_matched_pos[0]
                    dy = det['center_y'] - last_matched_pos[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 350:
                        bonus_multiplier = 0.00001 * (1 - dist / 350)  # ลด cost มากถึง 100,000 เท่า
                        cost *= max(bonus_multiplier, 0.0000001)
                
                # เพิ่ม penalty ถ้า lost_frames มาก
                if lost_frames > 10:
                    cost *= 2000.0
                elif lost_frames > 5:
                    cost *= 1000.0
                
                # หา best และ second best
                if cost < best_cost and cost != float('inf'):
                    second_best_cost = best_cost
                    best_cost = cost
                    best_track_idx = track_idx
                elif cost < second_best_cost and cost != float('inf'):
                    second_best_cost = cost
            
            # Match ถ้า cost ต่ำพอและมี margin ที่ดี
            if best_track_idx is not None:
                track_id, track = tracks_list[best_track_idx]
                history_len = len(track.get('history', []))
                total_frames = track.get('total_frames', 0)
                is_locked = self.track_locked.get(track_id, False)
                
                # กำหนด threshold และ margin
                if is_locked:
                    threshold = 0.0001
                    margin_ratio = 1.001
                elif history_len > 8 and total_frames > 6:
                    threshold = 0.001
                    margin_ratio = 1.01
                elif history_len > 4 and total_frames > 3:
                    threshold = 0.01
                    margin_ratio = 1.05
                else:
                    threshold = 0.1
                    margin_ratio = 1.1
                
                # ตรวจสอบ margin
                has_good_margin = (second_best_cost == float('inf') or 
                                 best_cost * margin_ratio < second_best_cost)
                
                if best_cost < threshold and has_good_margin:
                    # ตรวจสอบ jump distance อีกครั้ง - ถ้า jump มากเกินไปให้ไม่ match
                    last_pos = track.get('last_pos')
                    if last_pos:
                        dx = det['center_x'] - last_pos[0]
                        dy = det['center_y'] - last_pos[1]
                        jump_distance = np.sqrt(dx*dx + dy*dy)
                        
                        # คำนวณ max allowed jump
                        max_allowed_jump = self.max_jump_distance
                        if lost_frames > 0:
                            max_allowed_jump = self.max_jump_distance * (1 + min(lost_frames / 5, 1.5))
                        
                        # ถ้า jump มากเกินไป ให้ไม่ match
                        if jump_distance > max_allowed_jump:
                            continue
                    
                    # ตรวจสอบ spatial consistency
                    if track_id in self.last_matched_positions:
                        last_matched_pos = self.last_matched_positions[track_id]
                        dx = det['center_x'] - last_matched_pos[0]
                        dy = det['center_y'] - last_matched_pos[1]
                        distance_to_last_match = np.sqrt(dx*dx + dy*dy)
                        
                        max_allowed_jump = self.max_jump_distance * 2.0
                        if history_len > 8:
                            max_allowed_jump = self.max_jump_distance * 3.0
                        
                        if distance_to_last_match > max_allowed_jump:
                            continue
                    
                    matched_tracks.add(best_track_idx)
                    
                    self._update_track(track, det, frame_number)
                    det['track_id'] = track_id
                    
                    self.last_matched_positions[track_id] = (det['center_x'], det['center_y'])
                    
                    matched_detections.append(det)
                else:
                    det['track_id'] = None
                    matched_detections.append(det)
            else:
                det['track_id'] = None
                matched_detections.append(det)
        
        return matched_detections
    
    def _calculate_cost(self, detection, track, frame_number):
        """คำนวณ cost สำหรับ matching"""
        center_x = detection['center_x']
        center_y = detection['center_y']
        
        last_pos = track['last_pos']
        last_frame = track['last_frame']
        lost_frames = track.get('lost_frames', 0)
        
        frames_since_last = frame_number - last_frame
        
        # ตรวจสอบ jump distance ก่อน - ถ้า jump มากเกินไปให้ return infinity
        dx = center_x - last_pos[0]
        dy = center_y - last_pos[1]
        jump_distance = np.sqrt(dx*dx + dy*dy)
        
        # คำนวณ max allowed jump (ขึ้นอยู่กับ lost_frames)
        max_allowed_jump = self.max_jump_distance
        if lost_frames > 0:
            # ถ้าหายไปนาน อาจผ่านเมฆ ให้ยืดหยุ่นมากขึ้น
            max_allowed_jump = self.max_jump_distance * (1 + min(lost_frames / 5, 1.5))
        else:
            # ถ้าไม่หายไป ให้เข้มงวดมาก
            max_allowed_jump = self.max_jump_distance
        
        # ถ้า jump มากเกินไป ให้ return infinity (ไม่ match)
        if jump_distance > max_allowed_jump:
            return float('inf')
        
        velocity = track['velocity']
        if frames_since_last > 0 and (velocity[0] != 0 or velocity[1] != 0):
            predicted_x = last_pos[0] + velocity[0] * frames_since_last
            predicted_y = last_pos[1] + velocity[1] * frames_since_last
            
            pred_dx = center_x - predicted_x
            pred_dy = center_y - predicted_y
            pred_distance = np.sqrt(pred_dx*pred_dx + pred_dy*pred_dy)
            
            distance = pred_distance * 0.15
        else:
            distance = jump_distance
        
        if lost_frames > 0:
            adjusted_threshold = self.distance_threshold * (1 + min(lost_frames / 8, 2.0))
        else:
            adjusted_threshold = self.distance_threshold
        
        if lost_frames > 0:
            penalty = min(lost_frames * 1.5, 15)
        else:
            penalty = 0
        
        cost = distance + penalty
        
        history_len = len(track.get('history', []))
        total_frames = track.get('total_frames', 0)
        
        if history_len > 30 and total_frames > 20:
            cost *= 0.005
        elif history_len > 20 and total_frames > 15:
            cost *= 0.01
        elif history_len > 10 and total_frames > 8:
            cost *= 0.05
        elif history_len > 5 and total_frames > 4:
            cost *= 0.1
        elif history_len > 2:
            cost *= 0.3
        
        if distance > adjusted_threshold:
            return float('inf')
        
        return cost
    
    def _update_track(self, track, detection, frame_number):
        """อัปเดต track ด้วย detection ใหม่"""
        center_x = detection['center_x']
        center_y = detection['center_y']
        
        last_pos = track['last_pos']
        if last_pos:
            dx = center_x - last_pos[0]
            dy = center_y - last_pos[1]
            jump_distance = np.sqrt(dx*dx + dy*dy)
            
            if jump_distance > self.max_jump_distance and len(track['history']) > 0:
                velocity = track.get('velocity', (0, 0))
                frames_since_last = frame_number - track['last_frame']
                if frames_since_last > 0 and (velocity[0] != 0 or velocity[1] != 0):
                    predicted_x = last_pos[0] + velocity[0] * frames_since_last
                    predicted_y = last_pos[1] + velocity[1] * frames_since_last
                    if abs(center_x - predicted_x) < abs(center_x - last_pos[0]):
                        center_x = int(predicted_x)
                    if abs(center_y - predicted_y) < abs(center_y - last_pos[1]):
                        center_y = int(predicted_y)
        
        track['history'].append((center_x, center_y, frame_number))
        
        track['last_pos'] = (center_x, center_y)
        track['last_frame'] = frame_number
        track['lost_frames'] = 0
        track['total_frames'] += 1
        
        # อัปเดต spatial region
        if len(track['history']) >= 5:
            recent_positions = track['history'][-5:]
            xs = [x for x, y, _ in recent_positions]
            ys = [y for y, x, _ in recent_positions]
            center_x_region = sum(xs) / len(xs)
            center_y_region = sum(ys) / len(ys)
            max_dist = 0
            for x, y, _ in recent_positions:
                dist = np.sqrt((x - center_x_region)**2 + (y - center_y_region)**2)
                max_dist = max(max_dist, dist)
            track['spatial_region'] = (center_x_region, center_y_region, max_dist + 150)
        
        # คำนวณ velocity
        if len(track['history']) >= 2:
            prev_x, prev_y, prev_frame = track['history'][-2]
            frame_diff = frame_number - prev_frame
            if frame_diff > 0:
                vx = (center_x - prev_x) / frame_diff
                vy = (center_y - prev_y) / frame_diff
                
                old_velocity = track.get('velocity', (0, 0))
                if old_velocity[0] != 0 or old_velocity[1] != 0:
                    alpha = 0.65
                    vx = alpha * vx + (1 - alpha) * old_velocity[0]
                    vy = alpha * vy + (1 - alpha) * old_velocity[1]
                
                track['velocity'] = (vx, vy)
    
    def get_track(self, track_id):
        """ดึง track ตาม ID"""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self):
        """ดึง tracks ทั้งหมด"""
        return self.tracks
    
    def reset(self):
        """รีเซ็ต tracker"""
        self.tracks = {}
        self.next_id = 0
        self.last_matched_positions = {}
        self.track_locked = {}
