import cv2
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import math

class FPSCheatDetector:
    def __init__(self):
        self.frame_count = 0
        self.crosshair_positions = []
        self.target_positions = []
        self.shots_fired = []
        self.movement_data = []
        self.detection_results = {
            'aimbot_suspicious': False,
            'wallhack_suspicious': False,
            'speed_hack_suspicious': False,
            'recoil_suspicious': False,
            'confidence_scores': {},
            'timestamps': []
        }
    
    def detect_crosshair(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect crosshair position in frame using template matching or color detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for typical crosshair patterns (bright pixels in center area)
        height, width = gray.shape
        center_region = gray[height//2-50:height//2+50, width//2-50:width//2+50]
        
        # Find brightest point in center region (likely crosshair)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(center_region)
        
        if max_val > 200:  # Bright enough to be a crosshair
            crosshair_x = max_loc[0] + width//2 - 50
            crosshair_y = max_loc[1] + height//2 - 50
            return (crosshair_x, crosshair_y)
        
        return None
    
    def detect_enemies(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect potential enemy positions using color detection and contours
        This is a simplified version - real implementation would use ML models
        """
        enemies = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for typical enemy colors (adjust based on game)
        # This example looks for red team colors
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum size for an enemy
                x, y, w, h = cv2.boundingRect(contour)
                enemies.append((x + w//2, y + h//2))
        
        return enemies
    
    def detect_muzzle_flash(self, frame: np.ndarray, prev_frame: np.ndarray) -> bool:
        """
        Detect muzzle flash by comparing frame differences
        """
        if prev_frame is None:
            return False
        
        # Calculate frame difference
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Look for sudden bright spots (muzzle flash)
        _, thresh = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY)
        
        # Count bright pixels
        bright_pixels = cv2.countNonZero(thresh)
        
        # If significant bright area detected, likely a shot
        return bright_pixels > 1000
    
    def analyze_aimbot_patterns(self) -> float:
        """
        Analyze crosshair movement patterns for aimbot detection
        """
        if len(self.crosshair_positions) < 10:
            return 0.0
        
        suspicious_score = 0.0
        
        # Check for inhuman precision and snap movements
        for i in range(1, len(self.crosshair_positions)):
            prev_pos = self.crosshair_positions[i-1]
            curr_pos = self.crosshair_positions[i]
            
            if prev_pos and curr_pos:
                # Calculate movement distance
                distance = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                
                # Suspicious if very large instant movements (snapping)
                if distance > 200:  # Adjust threshold based on game sensitivity
                    suspicious_score += 0.1
        
        # Check for perfect tracking (too smooth movement)
        movement_variations = []
        for i in range(2, len(self.crosshair_positions)):
            if all(pos for pos in self.crosshair_positions[i-2:i+1]):
                # Calculate acceleration changes
                prev_vel = (self.crosshair_positions[i-1][0] - self.crosshair_positions[i-2][0],
                           self.crosshair_positions[i-1][1] - self.crosshair_positions[i-2][1])
                curr_vel = (self.crosshair_positions[i][0] - self.crosshair_positions[i-1][0],
                           self.crosshair_positions[i][1] - self.crosshair_positions[i-1][1])
                
                accel = (curr_vel[0] - prev_vel[0], curr_vel[1] - prev_vel[1])
                movement_variations.append(math.sqrt(accel[0]**2 + accel[1]**2))
        
        if movement_variations:
            avg_variation = np.mean(movement_variations)
            if avg_variation < 0.5:  # Too consistent = suspicious
                suspicious_score += 0.3
        
        return min(suspicious_score, 1.0)
    
    def analyze_wallhack_patterns(self) -> float:
        """
        Analyze for wallhack by checking pre-aiming at walls
        """
        suspicious_score = 0.0
        
        # This would require game-specific wall detection
        # For demonstration, we'll check for crosshair movements that seem to track through walls
        
        # Look for patterns where crosshair follows enemies before they're visible
        for i, (crosshair_pos, enemies) in enumerate(zip(self.crosshair_positions, self.target_positions)):
            if crosshair_pos and enemies:
                for enemy_pos in enemies:
                    distance = math.sqrt((crosshair_pos[0] - enemy_pos[0])**2 + 
                                       (crosshair_pos[1] - enemy_pos[1])**2)
                    
                    # If crosshair is very close to enemy position consistently
                    if distance < 50:
                        suspicious_score += 0.05
        
        return min(suspicious_score, 1.0)
    
    def analyze_recoil_patterns(self) -> float:
        """
        Analyze recoil compensation patterns
        """
        if len(self.shots_fired) < 5:
            return 0.0
        
        suspicious_score = 0.0
        
        # Check for perfect recoil compensation (too consistent)
        recoil_compensations = []
        
        for shot_frame in self.shots_fired:
            if shot_frame < len(self.crosshair_positions) - 5:
                # Check crosshair movement after shot
                pre_shot = self.crosshair_positions[shot_frame]
                post_shot = self.crosshair_positions[shot_frame + 3] if shot_frame + 3 < len(self.crosshair_positions) else None
                
                if pre_shot and post_shot:
                    recoil_comp = (post_shot[1] - pre_shot[1])  # Vertical compensation
                    recoil_compensations.append(recoil_comp)
        
        if recoil_compensations:
            # Check if recoil compensation is too consistent
            std_dev = np.std(recoil_compensations)
            if std_dev < 2.0:  # Too consistent
                suspicious_score += 0.4
        
        return min(suspicious_score, 1.0)
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int, prev_frame: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze a single frame for cheat indicators
        """
        self.frame_count += 1
        
        # Detect crosshair position
        crosshair_pos = self.detect_crosshair(frame)
        self.crosshair_positions.append(crosshair_pos)
        
        # Detect enemies
        enemies = self.detect_enemies(frame)
        self.target_positions.append(enemies)
        
        # Detect shots fired
        if prev_frame is not None:
            shot_detected = self.detect_muzzle_flash(frame, prev_frame)
            if shot_detected:
                self.shots_fired.append(frame_number)
        
        return {
            'frame': frame_number,
            'crosshair_position': crosshair_pos,
            'enemies_detected': len(enemies),
            'shot_fired': shot_detected if prev_frame is not None else False
        }
    
    def process_video(self, video_path: str) -> Dict:
        """
        Process entire video file and return cheat detection results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        frame_number = 0
        prev_frame = None
        
        print("Processing video for cheat detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            frame_analysis = self.analyze_frame(frame, frame_number, prev_frame)
            
            prev_frame = frame.copy()
            frame_number += 1
            
            # Progress indicator
            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames...")
        
        cap.release()
        
        # Perform final analysis
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """
        Generate final cheat detection report
        """
        # Analyze patterns
        aimbot_score = self.analyze_aimbot_patterns()
        wallhack_score = self.analyze_wallhack_patterns()
        recoil_score = self.analyze_recoil_patterns()
        
        # Set detection thresholds
        aimbot_threshold = 0.7
        wallhack_threshold = 0.6
        recoil_threshold = 0.8
        
        report = {
            'analysis_summary': {
                'total_frames_processed': self.frame_count,
                'shots_detected': len(self.shots_fired),
                'crosshair_tracking_points': len([pos for pos in self.crosshair_positions if pos])
            },
            'cheat_detection': {
                'aimbot': {
                    'suspicious': aimbot_score > aimbot_threshold,
                    'confidence_score': aimbot_score,
                    'threshold': aimbot_threshold,
                    'details': 'Analyzing crosshair snap movements and tracking precision'
                },
                'wallhack': {
                    'suspicious': wallhack_score > wallhack_threshold,
                    'confidence_score': wallhack_score,
                    'threshold': wallhack_threshold,
                    'details': 'Analyzing pre-aiming and wall tracking patterns'
                },
                'recoil_control': {
                    'suspicious': recoil_score > recoil_threshold,
                    'confidence_score': recoil_score,
                    'threshold': recoil_threshold,
                    'details': 'Analyzing recoil compensation consistency'
                }
            },
            'overall_assessment': {
                'highly_suspicious': any([aimbot_score > aimbot_threshold, 
                                        wallhack_score > wallhack_threshold, 
                                        recoil_score > recoil_threshold]),
                'total_suspicion_score': (aimbot_score + wallhack_score + recoil_score) / 3,
                'recommendation': 'Manual review recommended' if max(aimbot_score, wallhack_score, recoil_score) > 0.5 else 'Likely legitimate gameplay'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report

# Usage example
def main():
    # Initialize detector
    detector = FPSCheatDetector()
    
    # Process video file
    video_path = "gameplay_recording.mp4"  # Replace with your video path
    
    try:
        results = detector.process_video(video_path)
        
        # Print results
        print("\n" + "="*50)
        print("CHEAT DETECTION ANALYSIS REPORT")
        print("="*50)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        print(f"Frames Processed: {results['analysis_summary']['total_frames_processed']}")
        print(f"Shots Detected: {results['analysis_summary']['shots_detected']}")
        
        print("\nCHEAT DETECTION RESULTS:")
        print("-" * 30)
        
        for cheat_type, data in results['cheat_detection'].items():
            status = "SUSPICIOUS" if data['suspicious'] else "CLEAN"
            print(f"{cheat_type.upper()}: {status}")
            print(f"  Confidence: {data['confidence_score']:.2f}")
            print(f"  Threshold: {data['threshold']:.2f}")
            print(f"  Details: {data['details']}\n")
        
        print("OVERALL ASSESSMENT:")
        print(f"Highly Suspicious: {results['overall_assessment']['highly_suspicious']}")
        print(f"Total Suspicion Score: {results['overall_assessment']['total_suspicion_score']:.2f}")
        print(f"Recommendation: {results['overall_assessment']['recommendation']}")
        
        # Save detailed report
        with open('cheat_detection_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nDetailed report saved to 'cheat_detection_report.json'")
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()