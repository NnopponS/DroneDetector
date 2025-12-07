import cv2 
import numpy as np
import os


class VideoProcessor:
    """Class สำหรับประมวลผลวิดีโอ"""
    
    def __init__(self, tracker, gps_model, base_lat, base_lon, base_alt):
        """
        Initialize VideoProcessor
        
        Args:
            tracker: DroneTracker instance
            gps_model: GPSModel instance
            base_lat: Base latitude
            base_lon: Base longitude
            base_alt: Base altitude
        """
        self.tracker = tracker
        self.gps_model = gps_model
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.base_alt = base_alt
        self.frame_count = 0
    
    def reset(self):
        """Reset video processor state"""
        self.frame_count = 0
        if self.tracker:
            self.tracker.reset()
    
    def process_frame(self, frame, frame_number, fps, detections, using_custom_model=False):
        """
        Process a single video frame
        
        Args:
            frame: Video frame (numpy array)
            frame_number: Current frame number
            fps: Frames per second
            detections: List of detections from _detect_drones
            using_custom_model: Whether using custom GPS model
        
        Returns:
            processed_frame: Frame with tracking and GPS info drawn
            track_info: Dictionary with tracking information
            timestamp: Timestamp string
        """
        self.frame_count = frame_number
        processed_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate timestamp
        timestamp_seconds = frame_number / fps if fps > 0 else 0
        timestamp = f"{int(timestamp_seconds // 60):02d}:{int(timestamp_seconds % 60):02d}"
        
        # Convert detections to format expected by tracker
        detection_list = []
        for det in detections:
            detection_list.append({
                'center_x': det['center_x'],
                'center_y': det['center_y'],
                'corner_x': det['corner_x'],
                'corner_y': det['corner_y'],
                'cw': det['cw'],
                'ch': det['ch'],
                'area': det['area']
            })
        
        # Update tracker
        matched_detections = self.tracker.update(detection_list, frame_number)
        
        # Get all tracks from tracker
        all_tracks = self.tracker.get_all_tracks()
        
        # Create mapping from track_id to detection for bounding box info
        track_detections = {}
        for det in matched_detections:
            if det.get('track_id') is not None:
                track_detections[det['track_id']] = det
        
        # Draw tracks and GPS info
        track_info = {}
        colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Green, Yellow, Magenta, Cyan
        
        for track_id, track in all_tracks.items():
            # Skip tracks that are lost for too long
            if track.get('lost_frames', 0) > self.tracker.max_lost_frames:
                continue
            
            # Get position from last_pos or history
            last_pos = track.get('last_pos')
            if not last_pos and track.get('history'):
                last_pos = (track['history'][-1][0], track['history'][-1][1])
            
            if not last_pos:
                continue
            
            center_x = int(last_pos[0])
            center_y = int(last_pos[1])
            
            # Get bounding box from matched detection if available, otherwise use default size
            if track_id in track_detections:
                det = track_detections[track_id]
                corner_x = int(det['corner_x'])
                corner_y = int(det['corner_y'])
                cw = int(det['cw'])
                ch = int(det['ch'])
            else:
                # Use default bounding box size if no detection matched
                corner_x = center_x - 20
                corner_y = center_y - 20
                cw = 40
                ch = 40
            
            # Use track color if available, otherwise use default
            color = track.get('color', colors[track_id % len(colors)])
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (corner_x, corner_y), 
                         (corner_x + cw, corner_y + ch), color, 2)
            cv2.circle(processed_frame, (center_x, center_y), 5, color, -1)
            
            # Calculate GPS
            center_img_x = w / 2
            center_img_y = h / 2
            pixel_to_degree = 0.00001
            offset_x = (center_x - center_img_x) * pixel_to_degree
            offset_y = (center_img_y - center_y) * pixel_to_degree
            
            if using_custom_model and self.gps_model.is_trained:
                gps_result = self.gps_model.predict(center_x, center_y)
                if gps_result:
                    lat, lon, alt = gps_result
                else:
                    lat = self.base_lat + offset_y
                    lon = self.base_lon + offset_x
                    y_ratio = center_y / h
                    alt = self.base_alt + (1 - y_ratio) * 10
            else:
                lat = self.base_lat + offset_y
                lon = self.base_lon + offset_x
                y_ratio = center_y / h
                alt = self.base_alt + (1 - y_ratio) * 10
            
            # Calculate speed using velocity from track
            speed = 0.0
            velocity = track.get('velocity', (0, 0))
            if velocity[0] != 0 or velocity[1] != 0:
                # Convert pixel velocity to m/s (rough conversion: 1 pixel = 0.1 meters)
                pixel_speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                speed = pixel_speed * 0.1 * fps  # m/s
            
            # Store track info
            track_info[track_id] = {
                'center_x': center_x,
                'center_y': center_y,
                'lat': lat,
                'lon': lon,
                'alt': alt,
                'age': track.get('total_frames', 0),
                'color': color,
                'speed': speed
            }
            
            # Draw track ID
            cv2.putText(processed_frame, f"ID:{track_id}", 
                       (corner_x, corner_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw GPS info
            info_y = 30 + (track_id * 80)
            cv2.putText(processed_frame, f"track id:{track_id}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(processed_frame, f"lat:{lat:.5f}", (10, info_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(processed_frame, f"lon:{lon:.5f}", (10, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(processed_frame, f"alt:{alt:.2f}", (10, info_y + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw path history if available
            if 'history' in track and len(track['history']) > 1:
                history = track['history']
                for i in range(1, len(history)):
                    pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                    pt2 = (int(history[i][0]), int(history[i][1]))
                    cv2.line(processed_frame, pt1, pt2, color, 2)
        
        return processed_frame, track_info, timestamp


def image_process_drone(img_path, CSV=False, visual=True, dil=False, DEBUG=False, bgr=(0,0,255), DETECT=False):
    vids_frame = []

    if not img_path:
        raise ValueError("No path")
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error + ", img_path)
        raise RuntimeError("No image")
    
    # Convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret1, th1 = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)

    th1_inv = cv2.bitwise_not(th1)

    kernel = np.ones((3,3), np.uint8)   # bigger kernel
    dilated_inv = cv2.dilate(th1_inv, kernel, iterations=3)

    dilated = dilated_inv
    if dil:
        cv2.imshow("Dilated Image", dilated)
        cv2.waitKey(0)

    # find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)

    drone_csv = []
    h, w = img.shape
    if contours:

        for c in contours:
            corner_x, corner_y, cw, ch = cv2.boundingRect(c)
            center_x = int(corner_x + (cw / 2))
            center_y = int(corner_y + (ch / 2))
            c_area = cw*ch

            label = f"x={corner_x}, y={corner_y}, w={cw}, h={ch} area={c_area}"
            if DEBUG:
                cv2.putText(image, label, (corner_x, max(15, corner_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            y_threshold = int(0.35 * h)
            if c_area > 50 and corner_y < h - y_threshold:  # must bigger than 50 and higher than 30%
                if DEBUG:
                    cv2.line(image, (0, h-y_threshold), (w, h-y_threshold), (255, 0, 255), 1, cv2.LINE_AA)

                if corner_y > 500 and ch > 150: 
                    continue

                if corner_x > 1300 and corner_y < 90:  # deals with the time label
                    continue
                
                cv2.drawContours(mask, [c], -1, 255, -1)
                # Draw red rectangle on original color image with 10% padding
                pad_w = int(0 * cw)
                pad_h = int(0 * ch)  # 5% padding
                if DETECT:
                    cv2.rectangle(image,
                                (corner_x - pad_w, corner_y - pad_h),
                                (corner_x + cw + pad_w, corner_y + ch + pad_h),
                                bgr, 2)  # BGR Red
                if DEBUG:
                    cv2.rectangle(image,
                            (center_x, center_y),
                            (center_x + cw, center_y + ch),
                            (255, 0, 0), 2) 

                output = [os.path.basename(img_path), center_x, center_y, cw, ch]
                if DEBUG:
                    print(output)         
                drone_csv.append(output)
    
    if DEBUG:
        cv2.putText(image, str(img_path), (int(w/2), int(h-(h*0.05))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    
    if visual: 
        cv2.imshow("Color Image with Rectangle", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if contours and CSV:
        text = ""
        for row in drone_csv:
            text = text + ','.join(map(str, row)) + '\n'
        if dil:
            dilated_3ch = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            return dilated_3ch, text
        else:
            return image, text
    
    return image

