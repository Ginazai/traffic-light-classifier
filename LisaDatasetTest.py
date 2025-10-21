"""
Traffic Light Detection and Classification Pipeline for ESP32-S3
Two-stage approach: Detection (localize traffic lights) → Classification (identify color)
Using LISA Traffic Light Dataset with bounding boxes for detection
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Configuration
CONFIG = {
    'img_size': (96, 96),
    'detection_img_size': (224, 224),  # Larger for detection
    'batch_size': 16,
    'epochs': 80,
    'learning_rate': 0.0003,
    'classes': ['red', 'green'],
    'model_name': 'traffic_light_classifier',
    'detection_model_name': 'traffic_light_detector',
    'data_dir': 'lisa_dataset',
    'output_dir': 'output_models'
}

class LISADatasetProcessor:
    """Process LISA Traffic Light Dataset for both detection and classification"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.annotations = []
        self.images = []
        self.labels = []
        self.bboxes = []  # Store bounding boxes for detection
        
    def load_real_lisa_data(self, lisa_path):
        """Load real LISA dataset"""
        if not os.path.exists(lisa_path):
            print(f"LISA dataset not found at {lisa_path}")
            return False
            
        annotations_base = Path(lisa_path) / "Annotations" / "Annotations"
        images_base = Path(lisa_path)
        
        if not annotations_base.exists():
            print("LISA annotations directory not found")
            return False
            
        print(f"Loading LISA dataset from: {lisa_path}")
        
        sequences = ['daySequence1', 'daySequence2', 'dayTrain', 
                    'nightSequence1', 'nightSequence2', 'nightTrain']
        
        total_processed = 0
        
        for sequence in sequences:
            print(f"Processing {sequence}...")
            
            if sequence in ["dayTrain", "nightTrain"]:
                clip_counter = 1
                while True:
                    clip_name = f'dayClip{clip_counter}' if sequence == "dayTrain" else f'nightClip{clip_counter}'
                    annotation_dir = annotations_base / sequence / clip_name
                    
                    if not annotation_dir.exists():
                        break
                    
                    processed = self._process_sequence_annotations(
                        annotation_dir, images_base, sequence, clip_name
                    )
                    total_processed += processed
                    clip_counter += 1
            else:
                annotation_dir = annotations_base / sequence
                if annotation_dir.exists():
                    processed = self._process_sequence_annotations(
                        annotation_dir, images_base, sequence
                    )
                    total_processed += processed
        
        print(f"Total annotations loaded: {len(self.annotations)}")
        
        if len(self.annotations) > 0:
            self._extract_detection_and_classification_data(images_base)
        
        return len(self.images) > 0
    
    def _process_sequence_annotations(self, annotation_dir, images_base, sequence_name, clip_name=None):
        """Process annotations from a sequence"""
        csv_files = list(annotation_dir.glob("frameAnnotationsBOX.csv"))
        
        if not csv_files:
            print(f"  No CSV files found in {annotation_dir}")
            return 0
        
        processed_count = 0
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, delimiter=';', on_bad_lines='skip')
                print(f"  - CSV loaded: {len(df)} annotations from {csv_file.name}")
                
                for _, row in df.iterrows():
                    annotation_tag = row['Annotation tag'] if 'Annotation tag' in df.columns else row.get('Annotation_tag', '')
                    
                    if annotation_tag in ['stop', 'stopLeft']:
                        label = 'red'
                    elif annotation_tag in ['go', 'goLeft', 'goForward']:
                        label = 'green'
                    else:
                        continue
                    
                    filename = row['Filename'] if 'Filename' in df.columns else row.get('filename', '')
                    if '/' in filename:
                        filename = filename.split('/')[-1]
                    
                    try:
                        x1 = float(row['Upper left corner X'] if 'Upper left corner X' in df.columns else row.get('Upper_left_X', 0))
                        y1 = float(row['Upper left corner Y'] if 'Upper left corner Y' in df.columns else row.get('Upper_left_Y', 0))
                        x2 = float(row['Lower right corner X'] if 'Lower right corner X' in df.columns else row.get('Lower_right_X', 0))
                        y2 = float(row['Lower right corner Y'] if 'Lower right corner Y' in df.columns else row.get('Lower_right_Y', 0))
                        
                        if x1 >= x2 or y1 >= y2:
                            continue
                            
                    except (ValueError, TypeError, KeyError):
                        continue
                    
                    annotation_data = {
                        'filename': filename,
                        'label': label,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'sequence': sequence_name,
                        'annotation_tag': annotation_tag
                    }
                    
                    if clip_name:
                        annotation_data['clip'] = clip_name
                    
                    self.annotations.append(annotation_data)
                    processed_count += 1
                    
            except Exception as e:
                print(f"  Error reading {csv_file}: {e}")
                continue
        
        print(f"  - Processed: {processed_count} annotations")
        return processed_count

    def _extract_detection_and_classification_data(self, images_base):
        """
        Extract data for YOLO-style detection (detects AND classifies in one shot)
        YOLO simultaneously predicts bounding boxes and class labels
        """
        print("Extracting data for YOLO detection pipeline...")
        print(f"Original annotations: {len(self.annotations)}")

        # Count class distribution
        class_counts = {'red': 0, 'green': 0}
        for annotation in self.annotations:
            label = annotation['label']
            if label in class_counts:
                class_counts[label] += 1

        print("Original class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count:,}")

        # Target samples per class
        target_per_class = {'red': 6000, 'green': 6000}

        # Group by image to avoid duplicates
        image_to_annotations = {}
        for annotation in self.annotations:
            sequence = annotation.get('sequence', 'dayTrain')
            clip_name = annotation.get('clip', '')
            filename = annotation['filename']
            image_key = f"{sequence}_{clip_name}_{filename}"
        
            if image_key not in image_to_annotations:
                image_to_annotations[image_key] = []
            image_to_annotations[image_key].append(annotation)

        print(f"Unique images available: {len(image_to_annotations)}")

        # Calculate sampling intervals
        image_class_counts = {'red': 0, 'green': 0}
        for annotations_list in image_to_annotations.values():
            class_counter = {'red': 0, 'green': 0}
            for ann in annotations_list:
                if ann['label'] in class_counter:
                    class_counter[ann['label']] += 1
        
            if sum(class_counter.values()) > 0:
                dominant_class = max(class_counter, key=class_counter.get)
                image_class_counts[dominant_class] += 1

        sampling_intervals = {}
        for cls in ['red', 'green']:
            if image_class_counts[cls] > 0:
                if image_class_counts[cls] <= target_per_class[cls]:
                    sampling_intervals[cls] = 1
                else:
                    sampling_intervals[cls] = max(1, image_class_counts[cls] // target_per_class[cls])
            else:
                sampling_intervals[cls] = 1

        print("\nSampling strategy:")
        for cls, interval in sampling_intervals.items():
            expected = image_class_counts[cls] // interval if image_class_counts[cls] > 0 else 0
            print(f"  {cls}: every {interval}th image → ~{expected} samples")

        # Sample images
        sampled_images = []
        class_sampled = {'red': 0, 'green': 0}
        class_counters = {'red': 0, 'green': 0}
    
        for image_key, annotations_list in image_to_annotations.items():
            class_counter = {'red': 0, 'green': 0}
            for ann in annotations_list:
                if ann['label'] in class_counter:
                    class_counter[ann['label']] += 1
    
            if sum(class_counter.values()) == 0:
                continue
    
            dominant_class = max(class_counter, key=class_counter.get)
            class_counters[dominant_class] += 1

            if (class_counters[dominant_class] % sampling_intervals[dominant_class] == 0 and 
                class_sampled[dominant_class] < target_per_class[dominant_class]):
                
                best_annotation = max(annotations_list, 
                                    key=lambda ann: (ann.get('x2', 0) - ann.get('x1', 0)) * 
                                                  (ann.get('y2', 0) - ann.get('y1', 0)))
            
                sampled_images.append(best_annotation)
                class_sampled[dominant_class] += 1

        print(f"\nSampled {len(sampled_images)} images total")

        # Process images - YOLO needs full images with bounding boxes
        counters = {
            'successful_detections': 0,
            'successful_classifications': 0,
            'successful_by_class': {'red': 0, 'green': 0}
        }

        for i, annotation in enumerate(sampled_images):
            if i % 100 == 0 and i > 0:
                print(f"Progress ({i}/{len(sampled_images)}): {i / len(sampled_images)*100:.1f}%")
    
            sequence = annotation.get('sequence', 'dayTrain')
            clip_name = annotation.get('clip', None)
    
            possible_paths = []
            if clip_name:
                possible_paths.extend([
                    images_base / sequence / sequence / clip_name / "frames" / annotation['filename'],
                    images_base / sequence / clip_name / "frames" / annotation['filename']
                ])
            else:
                possible_paths.extend([
                    images_base / sequence / sequence / "frames" / annotation['filename'],
                    images_base / sequence / "frames" / annotation['filename']
                ])
    
            for image_path in possible_paths:
                if image_path.exists():
                    try:
                        img = cv2.imread(str(image_path))
                        if img is None:
                            continue
                
                        height, width = img.shape[:2]
                
                        # YOLO APPROACH: Store full image with bounding box
                        img_resized = cv2.resize(img, CONFIG['detection_img_size'])
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                        
                        # Scale bounding box to resized image dimensions
                        scale_x = CONFIG['detection_img_size'][0] / width
                        scale_y = CONFIG['detection_img_size'][1] / height
                        
                        scaled_x1 = annotation['x1'] * scale_x
                        scaled_y1 = annotation['y1'] * scale_y
                        scaled_x2 = annotation['x2'] * scale_x
                        scaled_y2 = annotation['y2'] * scale_y
                        
                        bbox_scaled = np.array([scaled_x1, scaled_y1, scaled_x2, scaled_y2])
                        
                        # Store full image, bbox, and label for YOLO training
                        self.images.append(img_rgb)
                        self.bboxes.append(bbox_scaled)
                        self.labels.append(annotation['label'])
                        
                        counters['successful_detections'] += 1
                        counters['successful_by_class'][annotation['label']] += 1
                
                        break
                
                    except Exception as e:
                        continue

        print(f"\nExtraction complete:")
        print(f"  Total YOLO samples: {counters['successful_detections']}")
        for cls, count in counters['successful_by_class'].items():
            print(f"    {cls}: {count}")

        self.annotations.clear()
    
    def load_images_and_labels(self):
        """Load processed images and labels"""
        if len(self.images) > 0 and len(self.labels) > 0:
            print("Using processed LISA dataset")
            return np.array(self.images), np.array(self.labels)
        
        return np.array([]), np.array([])

    def load_detection_data(self):
        """Return detection-specific data: images with bounding boxes and labels"""
        if len(self.images) > 0 and len(self.bboxes) > 0:
            # Convert string labels to integers for YOLO
            label_map = {'red': 0, 'green': 1}
            labels_encoded = np.array([label_map[label] for label in self.labels])
            return np.array(self.images), np.array(self.bboxes), labels_encoded
        return np.array([]), np.array([]), np.array([])


class TrafficLightDetectionModel:
    """YOLOv5-inspired lightweight detection model for traffic lights"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.grid_size = 7  # 7x7 grid cells
        self.num_boxes = 2  # 2 anchor boxes per cell
        self.num_classes = 2  # red, green (detection + classification in one)
        
    def create_yolo_detection_model(self, input_shape):
        """
        Create YOLO-style detection model with IMPROVED MobileNetV2 backbone
        Output: (grid_size, grid_size, num_boxes * (5 + num_classes))
        Each box: [x, y, w, h, confidence, class_probs...]
        """
        
        inputs = keras.Input(shape=input_shape + (3,))
        
        # Use pretrained MobileNetV2 as backbone (MUCH better features)
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape + (3,),
            include_top=False,
            weights='imagenet',  # Pretrained on ImageNet
            alpha=0.75  # Width multiplier (0.75 = good balance for ESP32)
        )
        
        # Fine-tune the last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:  # Freeze early layers
            layer.trainable = False
        
        # Extract features from backbone
        x = base_model(inputs, training=False)
        
        # Detection head with residual connections
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Final prediction layer
        outputs = layers.Conv2D(
            self.num_boxes * (5 + self.num_classes), 
            (1, 1), 
            padding='same'
        )(x)
        
        # Reshape to (grid_size, grid_size, num_boxes, 5 + num_classes)
        outputs = layers.Reshape((self.grid_size, self.grid_size, 
                                 self.num_boxes, 5 + self.num_classes))(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def yolo_loss(self, y_true, y_pred):
        """
        SIMPLIFIED YOLO loss function that actually works
        Focus on the essentials: bbox localization + confidence + classification
        """
        
        # Split predictions (apply activations)
        pred_xy = tf.sigmoid(y_pred[..., 0:2])  # x, y offsets [0,1]
        pred_wh = tf.sigmoid(y_pred[..., 2:4])  # w, h [0,1] - CHANGED to sigmoid
        pred_conf = tf.sigmoid(y_pred[..., 4:5])  # objectness confidence
        pred_class = y_pred[..., 5:]  # class logits (we'll use softmax in loss)
        
        # Split ground truth
        true_xy = y_true[..., 0:2]
        true_wh = y_true[..., 2:4]
        true_conf = y_true[..., 4:5]
        true_class = y_true[..., 5:]
        
        # Object mask (which cells have objects)
        obj_mask = true_conf  # Already 0 or 1
        noobj_mask = 1.0 - obj_mask
        
        # 1. Localization loss (MSE for coordinates)
        xy_loss = obj_mask * tf.reduce_sum(
            tf.square(true_xy - pred_xy), 
            axis=-1, keepdims=True
        )
        
        wh_loss = obj_mask * tf.reduce_sum(
            tf.square(true_wh - pred_wh),  # Direct MSE, no sqrt
            axis=-1, keepdims=True
        )
        
        # 2. Confidence loss (binary cross-entropy style)
        conf_loss_obj = obj_mask * tf.square(1.0 - pred_conf)  # Want confidence = 1
        conf_loss_noobj = noobj_mask * tf.square(0.0 - pred_conf)  # Want confidence = 0
        
        # 3. Classification loss (categorical cross-entropy)
        class_loss = obj_mask * tf.keras.losses.categorical_crossentropy(
            true_class, 
            pred_class,
            from_logits=True  # pred_class are raw logits
        )[..., tf.newaxis]
        
        # Weighted sum of losses
        total_loss = (
            10.0 * tf.reduce_mean(xy_loss) +      # High weight for location
            10.0 * tf.reduce_mean(wh_loss) +      # High weight for size
            5.0 * tf.reduce_mean(conf_loss_obj) + # Penalize missed objects
            0.5 * tf.reduce_mean(conf_loss_noobj) + # Low penalty for background
            2.0 * tf.reduce_mean(class_loss)      # Classification matters
        )
        
        return total_loss
    
    def encode_yolo_target(self, bboxes, labels, img_size):
        """
        Convert bounding boxes to YOLO target format
        bboxes: [x1, y1, x2, y2] in pixel coordinates
        labels: class labels (0=red, 1=green)
        Returns: (grid_size, grid_size, num_boxes, 5 + num_classes)
        """
        
        batch_size = len(bboxes)
        targets = np.zeros((batch_size, self.grid_size, self.grid_size, 
                          self.num_boxes, 5 + self.num_classes))
        
        for b in range(batch_size):
            bbox = bboxes[b]
            label = labels[b]
            
            # Convert to YOLO format (center x, center y, width, height)
            x_center = (bbox[0] + bbox[2]) / 2.0 / img_size[0]
            y_center = (bbox[1] + bbox[3]) / 2.0 / img_size[1]
            width = (bbox[2] - bbox[0]) / img_size[0]
            height = (bbox[3] - bbox[1]) / img_size[1]
            
            # Clamp to valid range
            x_center = np.clip(x_center, 0, 0.999)
            y_center = np.clip(y_center, 0, 0.999)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            # Find which grid cell this object belongs to
            grid_x = int(x_center * self.grid_size)
            grid_y = int(y_center * self.grid_size)
            
            # Offset within the cell
            x_offset = x_center * self.grid_size - grid_x
            y_offset = y_center * self.grid_size - grid_y
            
            # Assign to first anchor box (simplified - proper YOLO uses IoU matching)
            targets[b, grid_y, grid_x, 0, 0] = x_offset
            targets[b, grid_y, grid_x, 0, 1] = y_offset
            targets[b, grid_y, grid_x, 0, 2] = width
            targets[b, grid_y, grid_x, 0, 3] = height
            targets[b, grid_y, grid_x, 0, 4] = 1.0  # Confidence
            
            # One-hot encode class
            targets[b, grid_y, grid_x, 0, 5 + label] = 1.0
        
        return targets
    
    def train_model(self, X_train, bboxes_train, labels_train, X_val, bboxes_val, labels_val):
        """Train YOLO-style detection model"""
        
        self.model = self.create_yolo_detection_model(CONFIG['detection_img_size'])
        
        # Encode targets to YOLO format
        print("Encoding YOLO targets...")
        y_train = self.encode_yolo_target(bboxes_train, labels_train, CONFIG['detection_img_size'])
        y_val = self.encode_yolo_target(bboxes_val, labels_val, CONFIG['detection_img_size'])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(
            optimizer=optimizer,
            loss=self.yolo_loss,
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                monitor='val_loss'
            ),
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.95 ** epoch)
            )
        ]
        
        print("Training YOLO detection model...")
        print(f"Model output shape: {self.model.output_shape}")
        self.model.summary()
        
        # Normalize images
        X_train_norm = X_train.astype(np.float32) / 255.0
        X_val_norm = X_val.astype(np.float32) / 255.0
        
        self.history = self.model.fit(
            X_train_norm, y_train,
            batch_size=CONFIG['batch_size'],
            epochs=CONFIG['epochs'],
            validation_data=(X_val_norm, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def decode_predictions(self, predictions, confidence_threshold=0.5):
        """
        Decode YOLO predictions to bounding boxes
        Returns: list of (bbox, class_id, confidence) tuples
        """
        
        detections = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for b in range(self.num_boxes):
                    # Extract prediction
                    pred = predictions[i, j, b]
                    
                    x_offset = tf.sigmoid(pred[0]).numpy()
                    y_offset = tf.sigmoid(pred[1]).numpy()
                    width = pred[2]
                    height = pred[3]
                    confidence = tf.sigmoid(pred[4]).numpy()
                    class_probs = tf.nn.softmax(pred[5:]).numpy()
                    
                    if confidence < confidence_threshold:
                        continue
                    
                    # Convert to absolute coordinates
                    x_center = (j + x_offset) / self.grid_size
                    y_center = (i + y_offset) / self.grid_size
                    
                    # Convert to bbox format [x1, y1, x2, y2]
                    x1 = (x_center - width / 2) * CONFIG['detection_img_size'][0]
                    y1 = (y_center - height / 2) * CONFIG['detection_img_size'][1]
                    x2 = (x_center + width / 2) * CONFIG['detection_img_size'][0]
                    y2 = (y_center + height / 2) * CONFIG['detection_img_size'][1]
                    
                    bbox = [
                        np.clip(x1, 0, CONFIG['detection_img_size'][0]),
                        np.clip(y1, 0, CONFIG['detection_img_size'][1]),
                        np.clip(x2, 0, CONFIG['detection_img_size'][0]),
                        np.clip(y2, 0, CONFIG['detection_img_size'][1])
                    ]
                    
                    class_id = np.argmax(class_probs)
                    class_confidence = class_probs[class_id]
                    final_confidence = confidence * class_confidence
                    
                    detections.append((bbox, class_id, final_confidence))
        
        # Non-maximum suppression (simple version)
        detections = self.nms(detections, iou_threshold=0.5)
        
        return detections
    
    def nms(self, detections, iou_threshold=0.5):
        """Non-maximum suppression to remove duplicate detections"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        
        keep = []
        while len(detections) > 0:
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            # Remove overlapping boxes
            filtered = []
            for det in detections:
                if self.compute_iou(best[0], det[0]) < iou_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    def compute_iou(self, box1, box2):
        """Compute Intersection over Union of two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def predict_bbox(self, image):
        """Predict bounding box for an image"""
        if self.model is None:
            return None
        
        img_resized = cv2.resize(image, CONFIG['detection_img_size'])
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        predictions = self.model.predict(np.expand_dims(img_normalized, axis=0), verbose=0)
        detections = self.decode_predictions(predictions[0])
        
        return detections
    
    def convert_to_tflite(self, output_path):
        """Convert detection model to TFLite"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimize for size (important for ESP32)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Detection TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model


class TrafficLightClassificationModel:
    """Classification model for traffic light colors"""
    
    def __init__(self):
        self.model = None
        self.history = None
        
    def create_simple_cnn(self, input_shape, num_classes):
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape + (3,)),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.2),  # Added dropout
        
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.3),  # Added dropout
        
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.4),  # Added dropout
        
            layers.Flatten(),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train classification model"""
        
        # Manual class mapping
        desired_classes = ['red', 'green']
        class_to_idx = {cls: idx for idx, cls in enumerate(desired_classes)}
        idx_to_class = {idx: cls for idx, cls in enumerate(desired_classes)}

        def manual_encode(labels):
            return np.array([class_to_idx[label] for label in labels])

        # Encode labels
        y_train_encoded = manual_encode(y_train)
        y_val_encoded = manual_encode(y_val)

        num_classes = len(desired_classes)
        y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val_encoded, num_classes)

        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight

        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_encoded),
            y=y_train_encoded
        )

        class_weights = dict(enumerate(class_weights_array))
        class_weights[0] *= 2.5  # red
        class_weights[1] *= 2.25  # green

        self.model = self.create_simple_cnn(CONFIG['img_size'], num_classes)
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        optimizer = keras.optimizers.Adam(learning_rate=0.0005)
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=8,
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.3,
                patience=4,
                min_lr=1e-7,
                monitor='val_accuracy'
            )
        ]

        train_datagen = ImageDataGenerator(
            rotation_range=3,
            brightness_range=[0.6, 1.4],
            channel_shift_range=10,
            horizontal_flip=False,
            vertical_flip=False
        )

        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )

        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow(
            X_val, y_val_cat,
            batch_size=CONFIG['batch_size'],
            shuffle=False
        )

        print("Training classification model...")
        steps_per_epoch = len(X_train) // CONFIG['batch_size']
        validation_steps = len(X_val) // CONFIG['batch_size']

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=CONFIG['epochs'],
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return self.history
    
    def convert_to_tflite(self, output_path):
        """Convert classification model to TFLite"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Classification TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model


def main():
    """Main training pipeline for YOLO-style detection (detection + classification in one)"""
    
    print("=== Traffic Light YOLO Detection Pipeline ===")
    print("Note: YOLO simultaneously detects AND classifies traffic lights\n")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    processor = LISADatasetProcessor(CONFIG['data_dir'])
    
    lisa_path = input("Enter path to LISA dataset: ").strip()
    if not lisa_path or not processor.load_real_lisa_data(lisa_path):
        print("No LISA dataset found")
        return
    
    # Load YOLO detection data (includes bboxes and class labels)
    images, bboxes, labels = processor.load_detection_data()
    
    if len(images) == 0:
        print("No images found!")
        return
    
    print(f"Loaded {len(images)} images for YOLO training")
    print(f"Classes: {np.unique(labels)} (0=red, 1=green)")
    
    # Split data for training
    X_train, X_temp, bboxes_train, bboxes_temp, labels_train, labels_temp = train_test_split(
        images, bboxes, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, bboxes_val, bboxes_test, labels_val, labels_test = train_test_split(
        X_temp, bboxes_temp, labels_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Train YOLO detection model
    print("\n" + "="*50)
    print("TRAINING YOLO DETECTION MODEL")
    print("="*50)
    
    yolo_model = TrafficLightDetectionModel()
    yolo_model.train_model(X_train, bboxes_train, labels_train, 
                          X_val, bboxes_val, labels_val)
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATING YOLO MODEL ON TEST SET")
    print("="*50)
    
    correct_detections = 0
    total_detections = 0
    detection_failures = 0
    
    for i in range(min(100, len(X_test))):  # Test on 100 samples
        detections = yolo_model.predict_bbox(X_test[i])
        
        if len(detections) > 0:
            # Get best detection
            best_det = max(detections, key=lambda x: x[2])
            pred_class = best_det[1]
            true_class = labels_test[i]
            
            total_detections += 1
            if pred_class == true_class:
                correct_detections += 1
        else:
            detection_failures += 1
    
    accuracy = correct_detections / total_detections if total_detections > 0 else 0
    detection_rate = total_detections / (total_detections + detection_failures)
    
    print(f"Detection rate: {detection_rate:.2%} ({total_detections}/{total_detections + detection_failures})")
    print(f"Classification accuracy (when detected): {accuracy:.2%}")
    print(f"End-to-end accuracy: {(correct_detections / (total_detections + detection_failures)):.2%}")
    
    # Convert to TFLite
    yolo_tflite_path = f"{CONFIG['output_dir']}/{CONFIG['detection_model_name']}_yolo.tflite"
    yolo_model.convert_to_tflite(yolo_tflite_path)
    
    # Save pipeline info
    pipeline_info = {
        'pipeline': 'YOLO single-stage detection + classification',
        'model': CONFIG['detection_model_name'] + '_yolo.tflite',
        'input_size': CONFIG['detection_img_size'],
        'grid_size': yolo_model.grid_size,
        'num_boxes_per_cell': yolo_model.num_boxes,
        'classes': ['red', 'green'],
        'class_mapping': {'0': 'red', '1': 'green'},
        'confidence_threshold': 0.5,
        'workflow': [
            '1. Input full scene image (224x224)',
            '2. Run YOLO model → get bounding boxes + class predictions',
            '3. Apply NMS (non-maximum suppression)',
            '4. Return detected traffic light with class and confidence',
            '5. Audio feedback: "Red light ahead" or "Green light ahead"'
        ],
        'advantages': [
            'Single model (simpler deployment)',
            'Faster inference (one pass)',
            'Joint optimization (detection + classification trained together)',
            'Better real-time performance'
        ],
        'esp32_deployment': {
            'model_file': CONFIG['detection_model_name'] + '_yolo.tflite',
            'input_shape': [1] + list(CONFIG['detection_img_size']) + [3],
            'output_shape': [1, yolo_model.grid_size, yolo_model.grid_size, 
                           yolo_model.num_boxes, 5 + yolo_model.num_classes],
            'preprocessing': 'Normalize to [0, 1], resize to 224x224',
            'postprocessing': 'Decode YOLO output, apply NMS, threshold confidence'
        },
        'test_performance': {
            'detection_rate': float(detection_rate),
            'classification_accuracy_when_detected': float(accuracy),
            'end_to_end_accuracy': float(correct_detections / (total_detections + detection_failures))
        }
    }
    
    with open(f"{CONFIG['output_dir']}/yolo_pipeline_info.json", 'w') as f:
        json.dump(pipeline_info, f, indent=2)
    
    print(f"\n=== YOLO Pipeline Training Complete ===")
    print(f"Model: {yolo_tflite_path}")
    print(f"Pipeline info: {CONFIG['output_dir']}/yolo_pipeline_info.json")
    print(f"\nThis single model both DETECTS and CLASSIFIES traffic lights!")
    print(f"Much simpler than two-stage approach for embedded deployment.")
    
    # Visualize some predictions
    visualize_yolo_predictions(yolo_model, X_test[:5], labels_test[:5])


def visualize_yolo_predictions(model, images, true_labels):
    """Visualize YOLO predictions on sample images"""
    
    class_names = ['red', 'green']
    
    fig, axes = plt.subplots(1, min(5, len(images)), figsize=(15, 3))
    if len(images) == 1:
        axes = [axes]
    
    for i, (img, true_label) in enumerate(zip(images[:5], true_labels[:5])):
        detections = model.predict_bbox(img)
        
        # Display image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if len(detections) > 0:
            # Draw best detection
            best_det = max(detections, key=lambda x: x[2])
            bbox, pred_class, confidence = best_det
            
            # Draw bounding box
            rect = plt.Rectangle(
                (bbox[0], bbox[1]), 
                bbox[2] - bbox[0], 
                bbox[3] - bbox[1],
                fill=False, 
                color='lime' if pred_class == true_label else 'red',
                linewidth=2
            )
            axes[i].add_patch(rect)
            
            # Add label
            label_text = f"{class_names[pred_class]} {confidence:.2f}"
            axes[i].text(bbox[0], bbox[1] - 5, label_text, 
                        color='white', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='green' if pred_class == true_label else 'red', alpha=0.7))
            
            title = f"True: {class_names[true_label]}\nPred: {class_names[pred_class]}"
        else:
            title = f"True: {class_names[true_label]}\nNo detection"
        
        axes[i].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/yolo_predictions.png", dpi=150, bbox_inches='tight')
    print(f"\nSample predictions saved to: {CONFIG['output_dir']}/yolo_predictions.png")
    plt.show()


if __name__ == "__main__":
    main()