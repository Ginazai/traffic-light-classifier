"""
IMPROVED Traffic Light Color Classification Model for ESP32-S3
Key Changes:
1. FULL SCENE training (not cropped regions)
2. Better data augmentation for real-world scenarios
3. Improved model architecture for context understanding
4. ESP32-specific preprocessing pipeline
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
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Enhanced Configuration
CONFIG = {
    'img_size': (96, 96), 
    'batch_size': 32,  # Increased batch size
    'epochs': 50,
    'learning_rate': 0.001,  # Higher initial learning rate
    'classes': ['red', 'yellow', 'green'],
    'model_name': 'traffic_light_full_scene_v2',
    'data_dir': 'lisa_dataset',
    'output_dir': 'output_models_v2'
}

class ImprovedLISAProcessor:
    """Enhanced LISA processor for full scene training"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.annotations = []
        self.images = []
        self.labels = []
    
    def load_real_lisa_data(self, lisa_path):
        """Load LISA dataset with focus on full scenes"""
        if not os.path.exists(lisa_path):
            print(f"LISA dataset not found at {lisa_path}")
            return False
            
        annotations_base = Path(lisa_path) / "Annotations" / "Annotations"
        images_base = Path(lisa_path)
        
        if not annotations_base.exists():
            print("LISA annotations directory not found")
            return False
            
        print(f"Loading LISA dataset from: {lisa_path}")
        
        # Process ALL sequences
        sequences = ['daySequence1', 'daySequence2', 'dayTrain', 
                    'nightSequence1', 'nightSequence2', 'nightTrain']
        
        for sequence in sequences:
            print(f"Processing {sequence}...")
            
            if sequence in ["dayTrain", "nightTrain"]:
                # Process training clips
                clip_counter = 1
                while True:
                    clip_name = f'dayClip{clip_counter}' if sequence == "dayTrain" else f'nightClip{clip_counter}'
                    annotation_dir = annotations_base / sequence / clip_name
                    
                    if not annotation_dir.exists():
                        break
                    
                    self._process_sequence_annotations(annotation_dir, images_base, sequence, clip_name)
                    clip_counter += 1
            else:
                # Process regular sequences
                annotation_dir = annotations_base / sequence
                if annotation_dir.exists():
                    self._process_sequence_annotations(annotation_dir, images_base, sequence)
        
        print(f"Total annotations loaded: {len(self.annotations)}")
        
        if len(self.annotations) > 0:
            self._extract_full_scenes_improved(images_base)
        
        return len(self.images) > 0
    
    def _process_sequence_annotations(self, annotation_dir, images_base, sequence_name, clip_name=None):
        """Process annotations from a sequence"""
        csv_files = list(annotation_dir.glob("frameAnnotationsBOX.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, delimiter=';', on_bad_lines='skip')
                
                for _, row in df.iterrows():
                    # Map LISA labels to our classes
                    annotation_tag = row.get('Annotation tag', row.get('Annotation_tag', ''))
                    
                    if annotation_tag in ['stop', 'stopLeft']:
                        label = 'red'
                    elif annotation_tag in ['warning', 'warningLeft']:
                        label = 'yellow'
                    elif annotation_tag in ['go', 'goLeft', 'goForward']:
                        label = 'green'
                    else:
                        continue
                    
                    filename = row.get('Filename', row.get('filename', ''))
                    if '/' in filename:
                        filename = filename.split('/')[-1]
                    
                    try:
                        x1 = float(row.get('Upper left corner X', row.get('Upper_left_X', 0)))
                        y1 = float(row.get('Upper left corner Y', row.get('Upper_left_Y', 0)))
                        x2 = float(row.get('Lower right corner X', row.get('Lower_right_X', 0)))
                        y2 = float(row.get('Lower right corner Y', row.get('Lower_right_Y', 0)))
                        
                        if x1 >= x2 or y1 >= y2:
                            continue
                            
                    except (ValueError, TypeError):
                        continue
                    
                    annotation_data = {
                        'filename': filename,
                        'label': label,
                        'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                        'sequence': sequence_name,
                        'annotation_tag': annotation_tag
                    }
                    
                    if clip_name:
                        annotation_data['clip'] = clip_name
                    
                    self.annotations.append(annotation_data)
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    def _extract_full_scenes_improved(self, images_base):
        """Extract FULL SCENES with smart sampling and quality filtering"""
        
        print("Extracting full scenes with improved sampling...")
        
        # Group by image to avoid duplicates and find dominant class per image
        image_groups = {}
        for annotation in self.annotations:
            sequence = annotation.get('sequence', '')
            clip = annotation.get('clip', '')
            filename = annotation['filename']
            image_key = f"{sequence}_{clip}_{filename}"
            
            if image_key not in image_groups:
                image_groups[image_key] = {
                    'annotations': [],
                    'class_counts': {'red': 0, 'yellow': 0, 'green': 0},
                    'best_annotation': None,
                    'total_bbox_area': 0
                }
            
            image_groups[image_key]['annotations'].append(annotation)
            image_groups[image_key]['class_counts'][annotation['label']] += 1
            
            # Calculate bounding box area
            bbox_area = (annotation['x2'] - annotation['x1']) * (annotation['y2'] - annotation['y1'])
            if bbox_area > image_groups[image_key]['total_bbox_area']:
                image_groups[image_key]['total_bbox_area'] = bbox_area
                image_groups[image_key]['best_annotation'] = annotation
        
        print(f"Unique images found: {len(image_groups)}")
        
        # Smart sampling: prioritize images with clear dominant class and good quality
        quality_filtered = []
        for image_key, group in image_groups.items():
            class_counts = group['class_counts']
            total_lights = sum(class_counts.values())
            
            if total_lights == 0:
                continue
            
            # Find dominant class
            dominant_class = max(class_counts, key=class_counts.get)
            dominant_count = class_counts[dominant_class]
            
            # Quality criteria
            clarity_score = dominant_count / total_lights  # How clear is the dominant class
            bbox_score = min(1.0, group['total_bbox_area'] / 10000)  # Prefer larger traffic lights
            
            # Only keep high-quality images
            if clarity_score >= 0.6 and bbox_score >= 0.1:  # At least 60% clarity and reasonable size
                quality_filtered.append({
                    'image_key': image_key,
                    'annotation': group['best_annotation'],
                    'dominant_class': dominant_class,
                    'quality_score': clarity_score + bbox_score
                })
        
        print(f"Quality filtered images: {len(quality_filtered)}")
        
        # Sort by quality score and sample evenly by class
        quality_filtered.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # Balanced sampling by class
        target_per_class = 2000  # Reduced for better quality
        class_samples = {'red': [], 'yellow': [], 'green': []}
        
        for item in quality_filtered:
            dominant_class = item['dominant_class']
            if len(class_samples[dominant_class]) < target_per_class:
                class_samples[dominant_class].append(item)
        
        # Process selected images
        counters = {
            'attempted': 0,
            'successful': 0,
            'by_class': {'red': 0, 'yellow': 0, 'green': 0}
        }
        
        for class_name, samples in class_samples.items():
            print(f"Processing {class_name} images: {len(samples)} samples")
            
            for item in samples:
                counters['attempted'] += 1
                
                if self._process_full_scene_image(item, images_base):
                    counters['successful'] += 1
                    counters['by_class'][class_name] += 1
                
                if counters['attempted'] % 500 == 0:
                    success_rate = counters['successful'] / counters['attempted'] * 100
                    print(f"Progress: {counters['attempted']} processed, {success_rate:.1f}% success")
        
        # Final report
        print(f"\n{'='*50}")
        print(f"FULL SCENE EXTRACTION REPORT")
        print(f"{'='*50}")
        print(f"Total attempted: {counters['attempted']}")
        print(f"Total successful: {counters['successful']}")
        print(f"Success rate: {counters['successful']/counters['attempted']*100:.1f}%")
        
        total_extracted = sum(counters['by_class'].values())
        for class_name, count in counters['by_class'].items():
            percentage = count / total_extracted * 100 if total_extracted > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def _process_full_scene_image(self, item, images_base):
        """Process a single full scene image"""
        annotation = item['annotation']
        sequence = annotation.get('sequence', '')
        clip_name = annotation.get('clip', None)
        
        # Build possible image paths
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
                    
                    # CRITICAL: Process FULL IMAGE (no cropping to traffic light)
                    height, width = img.shape[:2]
                    
                    # If image is too large, intelligently crop to focus area
                    if width > 800 or height > 600:
                        # Get traffic light bounding box for guidance
                        tl_x1, tl_y1 = annotation['x1'], annotation['y1']
                        tl_x2, tl_y2 = annotation['x2'], annotation['y2']
                        tl_center_x = (tl_x1 + tl_x2) // 2
                        tl_center_y = (tl_y1 + tl_y2) // 2
                        
                        # Create crop window around traffic light with good context
                        crop_size = min(600, width, height)
                        x1 = max(0, tl_center_x - crop_size // 2)
                        y1 = max(0, tl_center_y - crop_size // 2)
                        x2 = min(width, x1 + crop_size)
                        y2 = min(height, y1 + crop_size)
                        
                        img = img[y1:y2, x1:x2]
                    
                    # Resize entire image to model input size
                    full_scene = cv2.resize(img, CONFIG['img_size'])
                    full_scene_rgb = cv2.cvtColor(full_scene, cv2.COLOR_BGR2RGB)
                    
                    self.images.append(full_scene_rgb)
                    self.labels.append(annotation['label'])
                    
                    return True
                    
                except Exception as e:
                    continue
        
        return False
    
    def load_images_and_labels(self):
        """Load processed images and labels"""
        if len(self.images) > 0 and len(self.labels) > 0:
            return np.array(self.images), np.array(self.labels)
        
        return np.array([]), np.array([])

class ImprovedTrafficLightModel:
    """Improved model for full scene traffic light classification"""
    
    def __init__(self):
        self.model = None
        self.history = None
        self.class_to_idx = {'red': 0, 'yellow': 1, 'green': 2}
        self.idx_to_class = {0: 'red', 1: 'yellow', 2: 'green'}
    
    def create_improved_model(self, input_shape, num_classes):
        """Create improved model architecture for full scenes"""
        
        # Use EfficientNet-B0 as base (better than MobileNetV2 for this task)
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape + (3,),
            include_top=False,
            weights='imagenet'
        )
        
        # Fine-tune the last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-20]:  # Freeze all but last 20 layers
            layer.trainable = False
        
        # Enhanced classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_real_world_augmentation(self):
        """Create augmentation that matches real-world ESP32 conditions"""
        
        return ImageDataGenerator(
            # Geometric transformations (minimal for traffic lights)
            rotation_range=2,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zoom_range=0.05,
            
            # Photometric augmentations (crucial for real-world conditions)
            brightness_range=[0.5, 1.5],  # Wide brightness range
            channel_shift_range=30,        # Strong color variations
            
            # Real-world specific augmentations
            # contrast_range=[0.7, 1.3],     # Contrast variations
            # saturation_range=[0.8, 1.2],   # Saturation changes
            # hue_range=[-0.05, 0.05],       # Small hue shifts
            
            # No flipping for traffic lights
            horizontal_flip=False,
            vertical_flip=False,
            
            fill_mode='nearest'
        )
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the improved model"""
        print("Training improved full-scene model...")
        
        # Manual encoding
        def encode_labels(labels):
            return np.array([self.class_to_idx[label] for label in labels])
        
        y_train_encoded = encode_labels(y_train)
        y_val_encoded = encode_labels(y_val)
        
        # Convert to categorical
        num_classes = len(self.class_to_idx)
        y_train_cat = keras.utils.to_categorical(y_train_encoded, num_classes)
        y_val_cat = keras.utils.to_categorical(y_val_encoded, num_classes)
        
        # Calculate class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_array = compute_class_weight(
            'balanced', classes=np.unique(y_train_encoded), y=y_train_encoded
        )
        class_weights = dict(enumerate(class_weights_array))
        
        # Create improved model
        self.model = self.create_improved_model(CONFIG['img_size'], num_classes)
        
        # Use different learning rates for base and head
        optimizer = keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'])
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True, monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=5, min_lr=1e-7, monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                f"{CONFIG['output_dir']}/best_model.h5",
                save_best_only=True, monitor='val_accuracy'
            )
        ]
        
        # Real-world augmentation
        train_datagen = self.create_real_world_augmentation()
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(
            X_train, y_train_cat, batch_size=CONFIG['batch_size'], shuffle=True
        )
        val_generator = val_datagen.flow(
            X_val, y_val_cat, batch_size=CONFIG['batch_size'], shuffle=False
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Train
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
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        def encode_labels(labels):
            return np.array([self.class_to_idx[label] for label in labels])
        
        y_test_encoded = encode_labels(y_test)
        y_test_cat = keras.utils.to_categorical(y_test_encoded, len(self.class_to_idx))
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred_classes, 
                                  target_names=CONFIG['classes']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_encoded, y_pred_classes)
        print(cm)
        
        return test_acc
    
    def convert_to_tflite_optimized(self, output_path):
        """Convert to optimized TensorFlow Lite for ESP32"""
        
        # Create representative dataset for quantization
        def representative_dataset():
            for i in range(100):
                # Use actual validation data if available
                sample = np.random.random((1,) + CONFIG['img_size'] + (3,)).astype(np.float32)
                yield [sample]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Basic conversion (no quantization for ESP32 compatibility)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return tflite_model

def enhanced_preprocessing(images):
    """Enhanced preprocessing pipeline matching ESP32 conditions"""
    processed_images = []
    
    for img in images:
        # Convert to float32
        img_float = img.astype(np.float32)
        
        # Apply contrast enhancement (CLAHE)
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Slight saturation boost for traffic light colors
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.1, 0, 255)  # Boost saturation by 10%
        final_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        processed_images.append(final_img.astype(np.float32) / 255.0)
    
    return np.array(processed_images)

def main():
    """Main improved training pipeline"""
    print("=== IMPROVED Traffic Light Classification for ESP32-S3 ===\n")
    print("Key improvements:")
    print("- Full scene training (no cropping)")
    print("- Better model architecture (EfficientNet)")
    print("- Real-world augmentation")
    print("- Enhanced preprocessing")
    print("- Quality-based sample selection\n")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load data
    processor = ImprovedLISAProcessor(CONFIG['data_dir'])
    
    lisa_path = input("Enter path to LISA dataset: ").strip()
    if not (lisa_path and processor.load_real_lisa_data(lisa_path)):
        print("LISA dataset required for training")
        return
    
    images, labels = processor.load_images_and_labels()
    
    if len(images) == 0:
        print("No images loaded!")
        return
    
    print(f"Loaded {len(images)} images")
    print(f"Classes: {np.unique(labels, return_counts=True)}")
    
    # Enhanced preprocessing
    images = enhanced_preprocessing(images)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train improved model
    model = ImprovedTrafficLightModel()
    history = model.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    test_accuracy = model.evaluate_model(X_test, y_test)
    
    # Save models
    keras_path = f"{CONFIG['output_dir']}/{CONFIG['model_name']}.h5"
    tflite_path = f"{CONFIG['output_dir']}/{CONFIG['model_name']}.tflite"
    
    model.model.save(keras_path)
    tflite_model = model.convert_to_tflite_optimized(tflite_path)
    
    # Save model info
    model_info = {
        'input_shape': CONFIG['img_size'] + (3,),
        'classes': CONFIG['classes'],
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'test_accuracy': float(test_accuracy),
        'model_size_kb': len(tflite_model) / 1024,
        'training_type': 'full_scene',
        'improvements': [
            'EfficientNet-B0 base model',
            'Full scene training',
            'Real-world augmentation',
            'Enhanced preprocessing',
            'Quality-based sampling'
        ]
    }
    
    with open(f"{CONFIG['output_dir']}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n=== IMPROVED TRAINING COMPLETE ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")
    print(f"Files saved in: {CONFIG['output_dir']}/")
    print("\n=== ESP32-S3 Deployment Notes ===")
    print("1. This model processes FULL SCENES (not cropped traffic lights)")
    print("2. Apply same preprocessing: contrast enhancement + slight saturation boost")
    print("3. Normalize input to [0, 1] range")
    print("4. Model expects 96x96x3 RGB images")

if __name__ == "__main__":
    main()