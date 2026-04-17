import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- 1. GLOBAL CONFIG & MODEL ---
model = None

def _get_model():
    global model
    if model is not None:
        return model
    weights_path = "yolov8s.pt"
    try:
        model = YOLO(weights_path)
    except RuntimeError:
        if os.path.exists(weights_path):
            os.remove(weights_path)
        model = YOLO(weights_path)
    return model

# --- 2. ENTERPRISE CENTROID TRACKER ---
class CentroidTracker:
    def __init__(self, max_disappeared=4, max_distance=100):
        self.next_object_id = 0
        self.objects = {} 
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance 

    def update(self, detected_boxes):
        if len(detected_boxes) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]["disappeared"] += 1
                if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                    del self.objects[obj_id]
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(detected_boxes)):
                self.objects[self.next_object_id] = {**detected_boxes[i], "disappeared": 0}
                self.next_object_id += 1
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = []
        for oid in object_ids:
            box = self.objects[oid]["bbox"]
            cX = (box[0] + box[2]) / 2.0
            cY = (box[1] + box[3]) / 2.0
            object_centroids.append((cX, cY))

        used_rows = set()
        
        for j, det in enumerate(detected_boxes):
            det_box = det["bbox"]
            det_cX = (det_box[0] + det_box[2]) / 2.0
            det_cY = (det_box[1] + det_box[3]) / 2.0
            
            best_dist = self.max_distance
            best_obj_idx = -1
            
            for i, existing_centroid in enumerate(object_centroids):
                if i in used_rows: continue
                dist = np.sqrt((existing_centroid[0] - det_cX)**2 + (existing_centroid[1] - det_cY)**2)
                
                if dist < best_dist and self.objects[object_ids[i]]["type"] == det["type"]:
                    best_dist = dist
                    best_obj_idx = i

            if best_obj_idx != -1:
                obj_id = object_ids[best_obj_idx]
                old_b = self.objects[obj_id]["bbox"]
                new_b = det_box
                smooth_bbox = (
                    int(old_b[0]*0.6 + new_b[0]*0.4), int(old_b[1]*0.6 + new_b[1]*0.4),
                    int(old_b[2]*0.6 + new_b[2]*0.4), int(old_b[3]*0.6 + new_b[3]*0.4)
                )
                self.objects[obj_id]["bbox"] = smooth_bbox
                self.objects[obj_id]["lum"] = (self.objects[obj_id]["lum"] * 0.7) + (det["lum"] * 0.3)
                self.objects[obj_id]["disappeared"] = 0
                used_rows.add(best_obj_idx)
            else:
                self.objects[self.next_object_id] = {**det, "disappeared": 0}
                self.next_object_id += 1

        for i, obj_id in enumerate(object_ids):
            if i not in used_rows:
                self.objects[obj_id]["disappeared"] += 1
                if self.objects[obj_id]["disappeared"] > self.max_disappeared:
                    del self.objects[obj_id]

        return self.objects

# --- 3. MAIN ADAS PIPELINE ---
def process_dashcam_video(input_path: str, output_dir: str = "output"):
    model_instance = _get_model()
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"processed_{filename}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise Exception(f"Could not open {input_path}")
        
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'avc1'), fps, (width, height))
    tracker = CentroidTracker(max_disappeared=4, max_distance=120)
    
    # IPM Matrices (Focused strictly on ego-lane and immediate left/right to ignore outer barriers)
    src_pts = np.array([
        [width * 0.38, height * 0.65],
        [width * 0.62, height * 0.65],
        [width * 0.85, height * 0.95],
        [width * 0.15, height * 0.95]
    ], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    matrix_fwd = cv2.getPerspectiveTransform(src_pts, dst_pts)
    matrix_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    frame_count = 0
    print("\n--- INITIATING ADAS PRODUCTION PIPELINE ---")
    active_objects = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing Frame: {frame_count} / {total_frames} ({(frame_count/total_frames)*100:.1f}%)")

        if frame_count % 2 == 0:
            current_detections = []
            
            # CLAHE for dynamic range
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            l_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l_channel)
            frame_clahe = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
            
            hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
            hls = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HLS)
            v_channel = hsv[:,:,2]

            # YOLO Vehicle Blackout (Expanded mask to cover headlight glow)
            traffic_mask = np.zeros((height, width), dtype=np.uint8)
            results = model_instance(frame, verbose=False)
            for box in results[0].boxes:
                if int(box.cls[0]) in [0, 1, 2, 3, 5, 7]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(traffic_mask, (max(0, x1-30), max(0, y1-30)), 
                                  (min(width, x2+30), min(height, y2+30)), 255, -1)

            # --- ZONE A: SIGNS & HOARDINGS (Edge Density & Solidity) ---
            # 1. Broad Color Masking (Top 60% only)
            zone_a_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(zone_a_mask, (0, 0), (width, int(height*0.60)), 255, -1)
            
            mask_gb = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([35, 40, 40]), np.array([95, 255, 255])), # Green
                cv2.inRange(hsv, np.array([96, 40, 40]), np.array([130, 255, 255])) # Blue
            )
            mask_a = cv2.bitwise_and(mask_gb, mask_gb, mask=zone_a_mask)
            mask_a = cv2.bitwise_and(mask_a, cv2.bitwise_not(traffic_mask))
            
            # Clean noise
            mask_a = cv2.morphologyEx(mask_a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            contours_a, _ = cv2.findContours(mask_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours_a:
                area = cv2.contourArea(cnt)
                if area < 1000: continue # Must be large
                
                # Pass 1: Geometric Solidity
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0: continue
                if (area / hull_area) < 0.55: continue # Trees fail this instantly
                
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / float(h)
                if aspect_ratio < 0.5 or aspect_ratio > 8.0: continue
                
                # Pass 2: Edge Density (Verifying it has text)
                roi_l = hls[y:y+h, x:x+w, 1] # Extract Lightness channel
                edges = cv2.Canny(roi_l, 100, 200)
                edge_density = np.count_nonzero(edges) / float(w * h)
                
                # Sky is blank (< 0.02), Trees are chaotic (> 0.40). Hoardings sit in the middle.
                if edge_density < 0.02 or edge_density > 0.40: continue 
                
                lum_90 = np.percentile(v_channel[y:y+h, x:x+w], 90)
                current_detections.append({"bbox": (x, y, x+w, y+h), "lum": lum_90, "type": "SIGN"})

            # --- ZONE B: ROAD MARKINGS (Sobel-X Gradient + IPM) ---
            l_channel = hls[:,:,1]
            s_channel = hls[:,:,2]

            # 1. Sobel X Gradient (Finds sharp horizontal changes in lightness - i.e., lines on dark roads)
            sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
            abs_sobelx = np.absolute(sobelx)
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
            sxbinary = np.zeros_like(scaled_sobel)
            sxbinary[(scaled_sobel >= 25) & (scaled_sobel <= 150)] = 255

            # 2. Saturation Threshold (Captures bright white/yellow lines ignoring dark shadows)
            s_binary = np.zeros_like(s_channel)
            s_binary[(s_channel >= 100) & (s_channel <= 255)] = 255

            # 3. Combine mathematically robust features
            combined_binary = cv2.bitwise_or(sxbinary, s_binary)
            
            # 4. Warp to Bird's Eye View
            warped_bev = cv2.warpPerspective(combined_binary, matrix_fwd, (width, height))
            
            # Sieve out horizontal noise in the top-down view
            warped_bev = cv2.morphologyEx(warped_bev, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20)))
            
            lines = cv2.HoughLinesP(warped_bev, 1, np.pi/180, threshold=40, minLineLength=50, maxLineGap=30)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                    
                    if 75 < angle < 105: # True lanes are strictly vertical in Bird's Eye View
                        pts = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)
                        mapped_pts = cv2.perspectiveTransform(pts, matrix_inv)
                        
                        mx1, my1 = map(int, mapped_pts[0][0])
                        mx2, my2 = map(int, mapped_pts[1][0])
                        
                        bx1, bx2 = min(mx1, mx2) - 15, max(mx1, mx2) + 15
                        by1, by2 = min(my1, my2) - 5, max(my1, my2) + 5
                        
                        if by1 > height * 0.55 and by2 < height * 0.95:
                            roi_v = v_channel[max(0,by1):min(height,by2), max(0,bx1):min(width,bx2)]
                            if roi_v.size > 0:
                                current_detections.append({
                                    "bbox": (bx1, by1, bx2, by2), 
                                    "lum": np.percentile(roi_v, 90), 
                                    "type": "MARK"
                                })

            # Update Memory
            active_objects = tracker.update(current_detections)

        # --- 4. RENDER UI ---
        for obj_id, obj_data in active_objects.items():
            if obj_data["disappeared"] > 1: continue 
            
            x1, y1, x2, y2 = obj_data["bbox"]
            lum = obj_data["lum"]
            obj_type = obj_data["type"]
            
            if lum > 140: status, color = "PASS", (0, 255, 0)
            elif lum > 80: status, color = "DEGRAD", (0, 165, 255)
            else: status, color = "FAIL", (0, 0, 255)
                
            text = f"{status}:{obj_type} (L:{lum:.0f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + 200, y1), color, -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)
        
    cap.release()
    out.release()
    print(f"--- PROCESSING COMPLETE ---\nOutput saved to: {output_path}")
    return output_path