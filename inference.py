
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import cv2
import collections 
import traceback
import mediapipe as mp
import argparse
from datetime import datetime
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from mediapipe.framework.formats import landmark_pb2
from utils.tools import Logger, read_yaml
from utils.trainer import Trainer

custom_connections = [
        (0, 1), (0, 2),  # Nose to shoulders
        (1, 3), (3, 5),  # Left arm
        (2, 4), (4, 6),  # Right arm
        (1, 7), (2, 8),  # Shoulders to hips
        (7, 9), (9, 11),  # Left leg
        (8, 10), (10, 12)  # Right leg
    ]

class PoseTransformerInference:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.sequence_length = None
        self.n_keypoints = None
        self.n_channels = None
        self.input_features = None
        self.model = None
        self.trainer = None
        self.setup_model()
        
     
        
        self.selected_landmarks = [
                0,  
                11, 
                12,  
                13,  
                14,  # Right Elbow
                15,  # Left Wrist
                16,  # Right Wrist
                23,  # Left Hip
                24,  # Right Hip
                25,  # Left Knee
                26,  # Right Knee
                27,  # Left Ankle
                28   # Right Ankle
            ]

    def setup_model(self):
        """Initialize model with proper transformer configuration"""
        try:
            # Extract configuration details
            dataset_config = self.config[self.config['DATASET']]
            self.sequence_length = dataset_config['FRAMES'] // self.config.get('SUBSAMPLE', 1)
            self.n_keypoints = dataset_config['KEYPOINTS']
            self.n_channels = self.config['CHANNELS']
            self.input_features = self.n_keypoints * self.n_channels
            
            # Initialize Trainer for model structure and load weights
            trainer = Trainer(self.config, self.logger)
            trainer.train_len = 1
            trainer.test_len = 1
            self.trainer = trainer
            self.model = trainer.get_model()  # Create model architecture
            
            # Load pretrained weights
            weights_path = self.config['WEIGHTS_PATH']
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at {weights_path}")
            self.model.load_weights(weights_path)
            print(f"Model loaded with weights from {weights_path}")

        except Exception as e:
            self.logger.save_log(f"Model setup failed: {e}")
            raise
    def preprocess_keypoints(self, keypoints):
        """Prepare keypoints for transformer input"""
        try:
            # Reshape keypoints to the required format
            keypoints = np.array(keypoints).reshape(-1, self.n_keypoints * self.n_channels)  # Flatten (13, 4) -> (52)
            
            # Normalize spatial coordinates
            keypoints[..., :2] = keypoints[..., :2] * 2 - 1  
            
            # Add batch dimension
            return np.expand_dims(keypoints, axis=0)  

        except Exception as e:
            self.logger.save_log(f"Preprocessing failed: {e}")
            return None


    def interpolate_low_visibility_points(self, keypoints, mask):
        """Interpolate low visibility keypoints using temporal information"""
        for i in range(keypoints.shape[1]):  # For each keypoint
            if np.any(mask[:, i]):
                valid_frames = ~mask[:, i]
                if np.any(valid_frames):
                    valid_indices = np.where(valid_frames)[0]
                    invalid_indices = np.where(mask[:, i])[0]
                    for coord in range(3):  # x, y, z coordinates
                        keypoints[invalid_indices, i, coord] = np.interp(
                            invalid_indices,
                            valid_indices,
                            keypoints[valid_frames, i, coord]
                        )

    def create_attention_mask(self, sequence_length):
        """Create attention mask for transformer"""
        return tf.ones((1, sequence_length, sequence_length))

    
    
    def predict_single_sequence(self, keypoints_buffer):
        try:
            # Validate input
            if not isinstance(keypoints_buffer, (list, np.ndarray)):
                raise ValueError("Invalid keypoints buffer type")
                
            # Pre-allocate arrays
            keypoints_array = np.array(keypoints_buffer)
            valid_keypoints = np.zeros((len(keypoints_buffer), self.n_keypoints * self.n_channels))
            
            # Process frames with vectorized operations
            for frame_idx in range(len(keypoints_buffer)):
                frame_data = keypoints_array[frame_idx]
                
                # Vectorized visibility check
                visibilities = frame_data[3::4]
                thresholds = np.full_like(visibilities, 0.2)
                thresholds[:12] = 0.3  # Upper body
                thresholds[12:16] = 0.2  # Mid body
                thresholds[16:] = 0.1  # Lower body
                
                # Create mask for low visibility points
                low_vis_mask = visibilities < thresholds
                
                if frame_idx > 0 and np.any(low_vis_mask):
                    # Use previous frame data for low visibility points
                    valid_keypoints[frame_idx] = valid_keypoints[frame_idx-1]
                    
                    # Update only high visibility points
                    high_vis_mask = ~low_vis_mask
                    valid_indices = np.repeat(high_vis_mask, 4)
                    valid_keypoints[frame_idx, valid_indices] = frame_data[valid_indices]
                else:
                    valid_keypoints[frame_idx] = frame_data

            # Process input with proper shape validation
            try:
                processed_input = valid_keypoints.reshape(1, self.sequence_length, -1)
            except ValueError as e:
                self.logger.save_log(f"Invalid reshape: {e}")
                return None, 0.0
                
            # Dynamic temperature scaling
            prediction = self.model.predict(processed_input, verbose=0)
            temperature = max(0.5, min(1.0, 1.0 - np.std(prediction)))  # Dynamic temperature
            class_probs = tf.nn.softmax(prediction/temperature, axis=-1)[0]
            
            confidence = float(np.max(class_probs))
            predicted_class = np.argmax(class_probs)
            
            # Circular buffer for prediction history
            if not hasattr(self, '_prediction_history'):
                self._prediction_history = collections.deque(maxlen=10)
            
            # Adaptive confidence threshold
            min_confidence = 0.6
            if len(self._prediction_history) > 0:
                # Increase threshold if predictions are unstable
                recent_unique = len(set(self._prediction_history))
                min_confidence += 0.1 * min(recent_unique - 1, 2)
            
            if confidence > min_confidence:
                self._prediction_history.append(predicted_class)
                
                # Weighted voting for stability
                if len(self._prediction_history) >= 3:
                    recent_preds = list(self._prediction_history)[-5:]
                    weights = np.linspace(0.6, 1.0, len(recent_preds))
                    
                    pred_counts = {}
                    for pred, weight in zip(recent_preds, weights):
                        pred_counts[pred] = pred_counts.get(pred, 0) + weight
                    
                    most_common = max(pred_counts.items(), key=lambda x: x[1])
                    if most_common[1] >= 2.0:  # Weighted threshold
                        return most_common[0], confidence
            
            return None, confidence

        except Exception as e:
            self.logger.save_log(f"Prediction failed: {str(e)}")
            self.logger.save_log(traceback.format_exc())
            return None, 0.0
    
        
    

    def process_frame(self, frame, pose_detector):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(frame_rgb)
            if not results.pose_landmarks:
                return None

            frame_height, frame_width = frame.shape[:2]
            keypoints = []
            
           
            if hasattr(self, '_frame_counter'):
                self._frame_counter += 1
                if self._frame_counter > 100:  
                    self._prev_keypoints = None
                    self._frame_counter = 0
            else:
                self._frame_counter = 0
            
           
            prev_keypoints = getattr(self, '_prev_keypoints', None)
            
            
            bad_frames_counter = getattr(self, '_bad_frames', 0)
            
            for idx in self.selected_landmarks:
                landmark = results.pose_landmarks.landmark[idx]
                
                
                alpha = min(0.8, max(0.3, landmark.visibility))  
                
                # Normalize coordinates
                x = (landmark.x - 0.5) * 2
                y = (landmark.y - 0.5) * 2
                z = landmark.z
                visibility = landmark.visibility
                
                if prev_keypoints is not None and len(prev_keypoints) == len(self.selected_landmarks) * 4:
                    prev_idx = self.selected_landmarks.index(idx) * 4
                    
                    # Only interpolate if visibility isn't too low
                    if visibility > 0.1:
                        x = alpha * x + (1 - alpha) * prev_keypoints[prev_idx]
                        y = alpha * y + (1 - alpha) * prev_keypoints[prev_idx + 1]
                        z = alpha * z + (1 - alpha) * prev_keypoints[prev_idx + 2]
                        visibility = max(visibility, 0.2)
                    else:
                        bad_frames_counter += 1
                
                keypoints.extend([x, y, z, visibility])

            # Reset if too many bad frames
            if bad_frames_counter > 10:
                self._prev_keypoints = None
                bad_frames_counter = 0
            
            self._bad_frames = bad_frames_counter
            self._prev_keypoints = keypoints

            return np.array(keypoints)

        except Exception as e:
            self.logger.save_log(f"Frame processing failed: {e}")
            return None
    
    def smooth_predictions(self, predictions, window_size=7):
        if len(predictions) < window_size:
            return predictions[-1]
        
        # Use weighted voting
        recent_predictions = predictions[-window_size:]
        weights = np.linspace(0.5, 1.0, window_size)
        
        pred_counts = {}
        for pred, weight in zip(recent_predictions, weights):
            pred_counts[pred] = pred_counts.get(pred, 0) + weight
        
        return max(pred_counts.items(), key=lambda x: x[1])[0]
    

    def run_inference(self, input_source=0):
        """Run real-time inference on video source and display skeleton overlay."""
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles  # For drawing pose landmarks

        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            self.logger.save_log("Failed to open video source")
            return

        keypoints_buffer = []
        predictions_buffer = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
               

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_detector.process(frame_rgb)

                if results.pose_landmarks:
                    
                    filtered_landmarks = [results.pose_landmarks.landmark[i] for i in self.selected_landmarks]
                    filtered_landmark_list = landmark_pb2.NormalizedLandmarkList(landmark=filtered_landmarks)
                    
            
                    

                    # Draw only selected landmarks and their connections
                    mp_drawing.draw_landmarks(
                        frame,
                        filtered_landmark_list,
                        custom_connections,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )

                # Extract keypoints and prepare for model input
                    keypoints = self.process_frame(frame, pose_detector)
                    if keypoints is None:
                        continue

                    # Maintain sliding window of keypoints
                    keypoints_buffer.append(keypoints)
                    if len(keypoints_buffer) > self.sequence_length:
                        keypoints_buffer.pop(0)

                    # Make prediction when buffer is full
                    if len(keypoints_buffer) == self.sequence_length:
                        pred_class, confidence = self.predict_single_sequence(keypoints_buffer)
                        if pred_class is not None:
                            predictions_buffer.append(pred_class)
                            if len(predictions_buffer) > 5:
                                predictions_buffer.pop(0)
                            smoothed_pred = max(set(predictions_buffer), key=predictions_buffer.count)
                            self.display_results(frame, smoothed_pred, confidence)

              

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            pose_detector.close()

    
    def eval(self):
        self.trainer.get_data()
        self.trainer.evaluate()
    
    def display_results(self, frame, prediction, confidence):
        """Display prediction results on frame."""
        action_labels = self.config[self.config['DATASET']]['LABELS']
        action_name = action_labels[prediction] if prediction < len(action_labels) else "Unknown"
        
        # Draw prediction
        cv2.putText(
            frame,
            f"{action_name} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if confidence > 0.7 else (0, 165, 255),
            2
        )
        cv2.imshow('Pose Recognition', frame)

if __name__ == "__main__":
    # Load configuration and initialize logger
    config = read_yaml('utils/inference_config.yaml')
    log_path = os.path.join(config['LOG_DIR'], 'inference_log.txt')
    logger = Logger(log_path)

    # Run inference
    inference = PoseTransformerInference(config, logger)
    inference.run_inference(0)
    # inference.eval()
