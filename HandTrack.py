import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque
import time
import tkinter as tk
from tkinter import ttk
import threading
try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Please install PIL: pip install pillow")

class HandController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # State tracking
        self.tracking_enabled = False
        self.dragging = False
        self.pinch_start_pos = None
        
        # Performance settings
        self.show_preview = False
        self.running = False
        self.fps = 0
        self.last_frames = deque(maxlen=30)
        self.hand_choice = "right"
        
        # Gesture settings
        self.screen_width, self.screen_height = pyautogui.size()
        self.prev_positions = deque(maxlen=5)
        self.gesture_history = deque(maxlen=10)
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.3
        
        # Sensitivity settings
        self.click_threshold = 0.04
        self.drag_threshold = 0.03
        self.movement_smoothing = 0.15
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.setup_ui()

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("Hand Gesture Controller")
        self.root.geometry("300x450")
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Hand Gesture Controller", 
                 font=("Helvetica", 16)).pack(pady=(0,10))
        
        # Hand selection
        hand_frame = ttk.LabelFrame(main_frame, text="Hand Selection", padding="5")
        hand_frame.pack(fill=tk.X, pady=5)
        
        self.hand_var = tk.StringVar(value="right")
        ttk.Radiobutton(hand_frame, text="Right Hand", 
                       variable=self.hand_var, value="right",
                       command=self.update_hand).pack(fill=tk.X)
        ttk.Radiobutton(hand_frame, text="Left Hand", 
                       variable=self.hand_var, value="left",
                       command=self.update_hand).pack(fill=tk.X)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Status: Stopped")
        self.status_label.pack()
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack()
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_var = tk.BooleanVar(value=False)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack()
        
        ttk.Checkbutton(preview_frame, text="Show Preview", 
                       variable=self.preview_var,
                       command=self.toggle_preview).pack()
        
        # Control buttons
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Start", 
                  command=self.start_tracking).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(controls_frame, text="Stop", 
                  command=self.stop_tracking).pack(side=tk.LEFT, expand=True, padx=2)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_hand(self):
        self.hand_choice = self.hand_var.get()

    def update_preview(self, frame):
        if frame is not None and self.show_preview:
            try:
                frame = cv2.resize(frame, (300, 225))
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(image=Image.fromarray(img))
                self.preview_label.config(image=img)
                self.preview_label.image = img
            except Exception as e:
                print(f"Preview error: {e}")

    def count_raised_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        raised = 0
        
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                raised += 1
        return raised

    def detect_gestures(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        wrist = hand_landmarks.landmark[0]
        
        thumb_index_dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        thumb_middle_dist = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
        
        raised_fingers = self.count_raised_fingers(hand_landmarks)
        
        gesture_state = {
            'thumb_index_dist': thumb_index_dist,
            'thumb_middle_dist': thumb_middle_dist,
            'raised_fingers': raised_fingers,
            'index_height': index_tip.y - wrist.y
        }
        self.gesture_history.append(gesture_state)
        
        if len(self.gesture_history) < 3:
            return None
            
        if raised_fingers >= 4:
            return 'start_tracking'
        elif raised_fingers == 0:
            return 'stop_tracking'
            
        if not self.tracking_enabled:
            return None
            
        if thumb_index_dist < self.click_threshold:
            if not self.dragging and thumb_middle_dist < self.drag_threshold:
                return 'start_drag'
            elif self.dragging and thumb_middle_dist >= self.drag_threshold:
                return 'end_drag'
            else:
                return 'click'
                
        if thumb_middle_dist < self.click_threshold and thumb_index_dist > self.click_threshold:
            return 'right_click'
            
        if abs(index_tip.x - middle_tip.x) < 0.04 and raised_fingers == 2:
            avg_y = (index_tip.y + middle_tip.y) / 2
            if len(self.gesture_history) > 5:
                prev_avg_y = (self.gesture_history[-5]['thumb_index_dist'] + 
                            self.gesture_history[-5]['thumb_middle_dist']) / 2
                if abs(avg_y - prev_avg_y) > 0.02:
                    return 'scroll' if avg_y < prev_avg_y else 'scroll_down'
        
        return None

    def process_hand_gesture(self, hand_landmarks, handedness):
        if handedness.classification[0].label.lower() != self.hand_choice:
            return

        try:
            index_tip = hand_landmarks.landmark[8]
            x = int(np.interp(index_tip.x, [0, 1], [0, self.screen_width]))
            y = int(np.interp(index_tip.y, [0, 1], [0, self.screen_height]))
            x, y = self.smooth_movement(x, y)
            
            gesture = self.detect_gestures(hand_landmarks)
            current_time = time.time()
            
            if gesture and current_time - self.last_gesture_time > self.gesture_cooldown:
                self.last_gesture_time = current_time
                
                if gesture == 'start_tracking':
                    self.tracking_enabled = True
                    self.status_label.config(text="Status: Tracking Active")
                elif gesture == 'stop_tracking':
                    self.tracking_enabled = False
                    self.status_label.config(text="Status: Tracking Paused")
                elif gesture == 'click':
                    pyautogui.click(x, y)
                elif gesture == 'right_click':
                    pyautogui.rightClick(x, y)
                elif gesture == 'start_drag':
                    self.dragging = True
                    pyautogui.mouseDown(x, y)
                elif gesture == 'end_drag':
                    self.dragging = False
                    pyautogui.mouseUp(x, y)
                elif gesture == 'scroll':
                    pyautogui.scroll(2)
                elif gesture == 'scroll_down':
                    pyautogui.scroll(-2)
            
            if self.tracking_enabled:
                pyautogui.moveTo(x, y)

        except Exception as e:
            print(f"Gesture processing error: {e}")

    def process_frame(self):
        if not self.cap.isOpened():
            self.stop_tracking()
            return

        success, frame = self.cap.read()
        if not success:
            return

        try:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label.lower() == self.hand_choice:
                        self.process_hand_gesture(hand_landmarks, handedness)
                        if self.show_preview:
                            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            self.last_frames.append(time.time())
            if self.show_preview:
                self.update_preview(frame)

        except Exception as e:
            print(f"Frame processing error: {e}")

    def smooth_movement(self, x, y):
        self.prev_positions.append((x, y))
        if len(self.prev_positions) < 3:
            return x, y
        
        x = sum(p[0] for p in self.prev_positions) / len(self.prev_positions)
        y = sum(p[1] for p in self.prev_positions) / len(self.prev_positions)
        return int(x), int(y)

    def tracking_loop(self):
        while self.running:
            self.process_frame()
            time.sleep(0.01)

    def toggle_preview(self):
        self.show_preview = self.preview_var.get()
        if not self.show_preview:
            self.preview_label.config(image='')

    def start_tracking(self):
        if not self.running:
            self.running = True
            self.tracking_enabled = True
            self.status_label.config(text="Status: Running")
            threading.Thread(target=self.tracking_loop, daemon=True).start()
            self.update_fps()

    def update_fps(self):
        try:
            if len(self.last_frames) > 1:
                time_diff = self.last_frames[-1] - self.last_frames[0]
                if time_diff > 0:
                    self.fps = len(self.last_frames) / time_diff
                    self.fps_label.config(text=f"FPS: {self.fps:.1f}")
        except (IndexError, ZeroDivisionError):
            pass
        if self.running:
            self.root.after(1000, self.update_fps)

    def stop_tracking(self):
        self.running = False
        self.tracking_enabled = False
        self.status_label.config(text="Status: Stopped")
        self.preview_label.config(image='')

    def on_closing(self):
        self.stop_tracking()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    controller = HandController()
    controller.run()