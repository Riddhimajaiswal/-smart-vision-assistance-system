import cv2 
import numpy as np 
import os 
import urllib.request 
import threading 
import time 
import pyttsx3 
from ultralytics import YOLO 
import easyocr  # <--- Added OCR Library 

# ======================= 
# Load Stereo Calibration 
# ======================= 
file_path = "stereo_calib2.xml" 

if not os.path.exists(file_path): 
    if os.path.exists("stereo_calib2.xml"):
        file_path = "stereo_calib2.xml"
    else:
        print(f"Error: Calibration file '{file_path}' not found in: {os.getcwd()}") 
        exit(1) 

print(f"       Successfully located calibration asset: {file_path}")
cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ) 
K1 = cv_file.getNode("K1").mat() 
D1 = cv_file.getNode("D1").mat() 
K2 = cv_file.getNode("K2").mat() 
D2 = cv_file.getNode("D2").mat() 
R = cv_file.getNode("R").mat() 
T = cv_file.getNode("T").mat() 
cv_file.release() 

h, w = 480, 640  # SGBM Mapping Matrices Resolution 

# ============================ 
# Compute Rectification Maps 
# ============================ 
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify( 
    K1, D1, K2, D2, (w, h), R, T, alpha=-1 
) 
map1L, map2L = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1) 
map1R, map2R = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1) 

# ==================== 
# Stereo Matchers 
# ==================== 
left_matcher = cv2.StereoSGBM_create( 
    minDisparity=0, 
    numDisparities=16 * 6, 
    blockSize=5, 
    P1=8 * 3 * 5 ** 2, 
    P2=32 * 3 * 5 ** 2, 
    disp12MaxDiff=1, 
    uniquenessRatio=10, 
    speckleWindowSize=50, 
    speckleRange=32, 
    preFilterCap=31, 
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY 
) 
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher) 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher) 
wls_filter.setLambda(8000) 
wls_filter.setSigmaColor(1.5) 

# ========================================== 
# BACKGROUND AUDIO ENGINE (Linux Core Driver) 
# ========================================== 
class AudioAlertWorker: 
    def __init__(self): 
        self.text_queue = None 
        self.running = True 
        self.thread = threading.Thread(target=self._speech_loop) 
        self.thread.daemon = True 
        self.thread.start() 
 
    def _speech_loop(self): 
        try: 
            engine = pyttsx3.init('espeak') 
        except Exception: 
            try: 
                engine = pyttsx3.init() 
            except Exception as e: 
                print(f"    TTS Initialization Warning: {e}") 
                return 
                 
        engine.setProperty('rate', 145)  
         
        while self.running: 
            if self.text_queue is not None: 
                text_to_say = self.text_queue 
                self.text_queue = None   
                try: 
                    engine.say(text_to_say) 
                    engine.runAndWait() 
                except Exception as e: 
                    print(f"    Voice Core Error: {e}") 
            time.sleep(0.1) 
 
    def speak(self, text): 
        self.text_queue = text 
 
    def stop(self): 
        self.running = False 

# ========================================== 
# THREADED STREAM RECEIVER (With Mirror Fix) 
# ========================================== 
class CameraStreamThread: 
    def __init__(self, url, name="Camera"): 
        self.url = url 
        self.name = name 
        self.frame = None 
        self.running = True 
        self.bytes_accumulator = b'' 
        self.connect() 
        self.thread = threading.Thread(target=self.update, args=()) 
        self.thread.daemon = True 
        self.thread.start() 
 
    def connect(self): 
        try: 
            self.stream = urllib.request.urlopen(self.url, timeout=4) 
            print(f"   Connection successful to {self.name} pipeline.") 
        except Exception as e: 
            print(f"  Connection FAILED to {self.name} via target url: {self.url}. Error: {e}") 
            self.stream = None 
 
    def update(self): 
        while self.running: 
            if self.stream is None: 
                time.sleep(2) 
                self.connect() 
                continue 
            try: 
                chunk = self.stream.read(4096) 
                if not chunk:  
                    continue 
                self.bytes_accumulator += chunk 
                a = self.bytes_accumulator.find(b'\xff\xd8') 
                b = self.bytes_accumulator.find(b'\xff\xd9') 
                if a != -1 and b != -1 and b > a: 
                    jpg_bytes = self.bytes_accumulator[a:b+2] 
                    self.bytes_accumulator = self.bytes_accumulator[b+2:] 
                    decoded_matrix = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR) 
                    if decoded_matrix is not None: 
                        # CRITICAL FIX: Flip frame horizontally to fix mirror orientation 
                        self.frame = cv2.flip(decoded_matrix, 1) 
            except Exception: 
                self.stream = None  
 
    def get_frame(self): 
        return self.frame.copy() if self.frame is not None else None 
 
    def stop(self): 
        self.running = False 

# ========================================== 
# ISOLATED ASYNC OCR WORKER THREAD 
# ========================================== 
class AsyncOCRWorker: 
    def __init__(self, audio_worker): 
        self.audio_worker = audio_worker 
        self.ocr_reader = easyocr.Reader(['en'], gpu=False) 
        self.pending_frame = None 
        self.running = True 
        self.lock = threading.Lock() 
         
        self.thread = threading.Thread(target=self._ocr_loop, daemon=True) 
        self.thread.start() 
 
    def submit_frame(self, frame): 
        with self.lock: 
            self.pending_frame = frame 
 
    def _ocr_loop(self): 
        while self.running: 
            working_frame = None 
            with self.lock: 
                if self.pending_frame is not None: 
                    working_frame = self.pending_frame.copy() 
                    self.pending_frame = None  # Clear to avoid re-processing old frames 
 
            if working_frame is not None: 
                try: 
                    gray_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY) 
                    ocr_results = self.ocr_reader.readtext(gray_frame, detail=0, paragraph=True) 
                    if ocr_results: 
                        detected_text = " ".join(ocr_results).strip() 
                        if 1 < len(detected_text) < 120: 
                            print(f"    OCR Text Detected: {detected_text}") 
                            self.audio_worker.speak(f"Text reads: {detected_text}") 
                except Exception as e: 
                    print(f"    Async OCR Execution Error: {e}") 
             
            time.sleep(0.4)  # Prevent continuous thread locking 
 
    def stop(self): 
        self.running = False 

# ===================== 
# Main Engine Instances 
# ===================== 
urlL = "http://192.168.137.66/"   
urlR = "http://192.168.137.188/"   
 
print("       Launching Asynchronous Multi-Thread Video Links...") 
cam_left = CameraStreamThread(urlL, "Left-Eye Cam") 
cam_right = CameraStreamThread(urlR, "Right-Eye Cam") 
 
print("       Spawning Asynchronous Voice Navigation Worker...") 
audio_system = AudioAlertWorker() 
 
print("       Spawning Asynchronous OCR Engine Thread...") 
ocr_system = AsyncOCRWorker(audio_system) 
 
print("       Compiling YOLO Architecture Matrices...") 
model = YOLO("yolov8n.pt")  
 
print("\n        System Core Online! Interval Detection Engine is running. Press 'q' to close windows.") 

# Named window initialization to allow separate manipulation
cv2.namedWindow("Live Object Detection & OCR", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Stereo Depth Map Visualization", cv2.WINDOW_AUTOSIZE)
 
last_detection_time = 0 
DETECTION_INTERVAL = 2.0   
active_alerts = [] 

color_depth_map = np.zeros((h, w, 3), dtype=np.uint8)

while True: 
    frameL = cam_left.get_frame() 
    frameR = cam_right.get_frame() 
 
    if frameL is None or frameR is None: 
        print("    Waiting for initial streams to settle in memory...") 
        time.sleep(0.5) 
        continue 
 
    frameL = cv2.resize(frameL, (w, h)) 
    frameR = cv2.resize(frameR, (w, h)) 
 
    current_time = time.time() 
    annotated_frame = frameL.copy() 

    # --- TIMER DRIVEN ANALYTICS BLOCK --- 
    if current_time - last_detection_time >= DETECTION_INTERVAL: 
        last_detection_time = current_time 
        active_alerts = []   
 
        # Submit fresh left image matrix to the isolated background OCR engine 
        ocr_system.submit_frame(frameL) 
 
        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY) 
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY) 
 
        rectL = cv2.remap(grayL, map1L, map2L, cv2.INTER_LINEAR) 
        rectR = cv2.remap(grayR, map1R, map2R, cv2.INTER_LINEAR) 
 
        displ = left_matcher.compute(rectL, rectR).astype(np.float32) / 16.0 
        dispr = right_matcher.compute(rectR, rectL).astype(np.float32) / 16.0 
        filteredDisparity = wls_filter.filter(displ, rectL, disparity_map_right=dispr) 

        # --- PROCESS COLOR VISUALIZATION ---
        disp_visual = cv2.normalize(filteredDisparity, None, alpha=0, beta=255, 
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_depth_map = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)
 
        points_3D = cv2.reprojectImageTo3D(filteredDisparity, Q) 
        depth_map = points_3D[:, :, 2] 
 
        results = model(frameL, verbose=False) 
        boxes = results[0].boxes if len(results) > 0 else [] 
 
        closest_object_text = None 
        min_found_depth = float('inf') 
 
        for box in boxes: 
            conf = float(box.conf[0].item()) 
            if conf < 0.60: 
                continue 
 
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
            cls = int(box.cls[0].item()) 
            label = model.names[cls] 
 
            cx1 = int(x1 + 0.25 * (x2 - x1)) 
            cy1 = int(y1 + 0.25 * (y2 - y1)) 
            cx2 = int(x2 - 0.25 * (x2 - x1)) 
            cy2 = int(y2 - 0.25 * (y2 - y1)) 
            cx1, cy1 = max(0, cx1), max(0, cy1) 
            cx2, cy2 = min(w, cx2), min(h, cy2) 
 
            depth_roi = depth_map[cy1:cy2, cx1:cx2] 
            
            # Filter out near-zero disparity anomalies by setting a 10-meter max sanity threshold 
            valid_depth = depth_roi[(depth_roi > 0.1) & (depth_roi < 10.0) & (~np.isnan(depth_roi))] 
 
            if valid_depth.size > 0: 
                obj_depth = np.mean(valid_depth) 
                active_alerts.append(((x1, y1, x2, y2), f"{label} ({obj_depth:.2f}m)")) 
                 
                if obj_depth < min_found_depth: 
                    min_found_depth = obj_depth 
                    closest_object_text = f"{label} at {obj_depth:.1f} meters" 
 
        if closest_object_text is not None: 
            audio_system.speak(closest_object_text) 
            print(f"    Dispatching Voice Alert: {closest_object_text}") 
 
    # --- UI RENDERING SECTION --- 
    # Create a clean visual instance of the depth data for current UI pass
    active_depth_frame = color_depth_map.copy()

    for box_coords, text_data in active_alerts: 
        x1, y1, x2, y2 = box_coords 
        
        # Window 1 Layout: Video Stream annotations
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.putText(annotated_frame, text_data, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 
        
        # Window 2 Layout: Overlay thin alignment boxes onto the color map
        cv2.rectangle(active_depth_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    cv2.putText(annotated_frame, "Navigation Status: Active + OCR", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) 
 
    # --- OUTPUT DISPLAY WINDOWS ---
    cv2.imshow("Live Object Detection & OCR", annotated_frame) 
    cv2.imshow("Stereo Depth Map Visualization", active_depth_frame)
     
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break 

# Destructors execution 
cam_left.stop() 
cam_right.stop() 
ocr_system.stop() 
audio_system.stop() 
cv2.destroyAllWindows() 
print("Safely disconnected system interfaces.")