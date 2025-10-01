"""
Traffic Light Color Classification Model for ESP32-S3
Using LISA Traffic Light Dataset with TensorFlow Lite

This script creates a lightweight model to classify traffic light colors:
- Red, Yellow, Green
- Optimized for ESP32-S3 deployment
- Uses MobileNetV2 as base for efficiency
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
import zipfile
import requests
from PIL import Image
import json

# Configuration
CONFIG = {
    'img_size': (96, 96), 
    'batch_size': 16,
    'epochs': 80,
    'learning_rate': 0.0003,
    'classes': ['red', 'yellow', 'green'],
    'model_name': 'traffic_light_classifier',
    'data_dir': 'lisa_dataset',
    'output_dir': 'output_models'
}

class LISADatasetProcessor:
    """Process LISA Traffic Light Dataset"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.annotations = []
        self.images = []
        self.labels = []
    
    def load_real_lisa_data(self, lisa_path):
        """
        Load real LISA dataset - IMPROVED VERSION that processes ALL sequences
        """
        if not os.path.exists(lisa_path):
            print(f"LISA dataset not found at {lisa_path}")
            return False
            
        annotations_base = Path(lisa_path) / "Annotations" / "Annotations"
        images_base = Path(lisa_path)
        
        if not annotations_base.exists():
            print("LISA annotations directory not found")
            return False
            
        print(f"Loading LISA dataset from: {lisa_path}")
        
        # Process ALL sequences (day and night)
        sequences = ['daySequence1', 'daySequence2', 'dayTrain', 
                    'nightSequence1', 'nightSequence2', 'nightTrain']
        
        total_processed = 0
        
        for sequence in sequences:
            print(f"Processing {sequence}...")
            
            if sequence in ["dayTrain", "nightTrain"]:
                # For training sequences, process multiple clips
                clip_counter = 1
                while True:
                    if sequence == "dayTrain":
                        clip_name = f'dayClip{clip_counter}'
                    else:  # nightTrain
                        clip_name = f'nightClip{clip_counter}'
                    
                    annotation_dir = annotations_base / sequence / clip_name
                    
                    if not annotation_dir.exists():
                        break  # No more clips
                    
                    processed = self._process_sequence_annotations(
                        annotation_dir, images_base, sequence, clip_name
                    )
                    total_processed += processed
                    clip_counter += 1
                    
            else:
                # For regular sequences (daySequence1, daySequence2, etc.)
                annotation_dir = annotations_base / sequence
                
                if annotation_dir.exists():
                    processed = self._process_sequence_annotations(
                        annotation_dir, images_base, sequence
                    )
                    total_processed += processed
                else:
                    print(f"  Warning: {sequence} not found")
        
        print(f"Total annotations loaded: {len(self.annotations)}")
        print(f"Total images processed: {total_processed}")
        
        # Process images and extract traffic light regions
        if len(self.annotations) > 0:
            self._extract_center_crop_regions(images_base)
        
        return len(self.images) > 0
    
    def _process_sequence_annotations(self, annotation_dir, images_base, sequence_name, clip_name=None):
        """Process annotations from a single sequence or clip"""
        
        # Look for CSV annotation file
        csv_files = list(annotation_dir.glob("frameAnnotationsBOX.csv"))
        
        if not csv_files:
            print(f"  No CSV files found in {annotation_dir}")
            return 0
        
        processed_count = 0
        
        for csv_file in csv_files:
            try:
                # Read CSV with error handling
                df = pd.read_csv(csv_file, delimiter=';', on_bad_lines='skip')
                
                print(f"  - CSV loaded: {len(df)} annotations from {csv_file.name}")
                
                for _, row in df.iterrows():
                    # Map LISA labels to our classes
                    annotation_tag = row['Annotation tag'] if 'Annotation tag' in df.columns else row.get('Annotation_tag', '')
                    
                    if annotation_tag in ['stop', 'stopLeft']:
                        label = 'red'
                    elif annotation_tag in ['warning', 'warningLeft']:
                        label = 'yellow'
                    elif annotation_tag in ['go', 'goLeft', 'goForward']:
                        label = 'green'
                    else:
                        continue
                    
                    # Extract filename (handle different path formats)
                    filename = row['Filename'] if 'Filename' in df.columns else row.get('filename', '')
                    if '/' in filename:
                        filename = filename.split('/')[-1]
                    
                    # Validate bounding box coordinates
                    try:
                        x1 = float(row['Upper left corner X']) if 'Upper left corner X' in df.columns else float(row.get('Upper_left_X', 0))
                        y1 = float(row['Upper left corner Y']) if 'Upper left corner Y' in df.columns else float(row.get('Upper_left_Y', 0))
                        x2 = float(row['Lower right corner X']) if 'Lower right corner X' in df.columns else float(row.get('Lower_right_X', 0))
                        y2 = float(row['Lower right corner Y']) if 'Lower right corner Y' in df.columns else float(row.get('Lower_right_Y', 0))
                        
                        # Validate bbox
                        if x1 >= x2 or y1 >= y2:
                            continue
                            
                    except (ValueError, TypeError, KeyError):
                        continue
                    
                    # Store annotation with sequence and clip info
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

    def _extract_center_crop_regions(self, images_base):
        """Extract center-cropped regions - matching ESP32 deployment approach"""

        print("Extracting center-cropped regions (ESP32-style)...")
        print(f"Original annotations: {len(self.annotations)}")

        # Count class distribution
        class_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for annotation in self.annotations:
            label = annotation['label']
            if label in class_counts:
                class_counts[label] += 1

        print("Original class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count:,}")

        # Target samples per class
        target_per_class = {
            'red': 4000,
            'yellow': 4000,
            'green': 4000
        }

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

        # Sample images by dominant class
        sampled_images = []
        class_sampled = {'red': 0, 'yellow': 0, 'green': 0}

        # Calculate how many unique images per class exist
        image_class_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for annotations_list in image_to_annotations.values():
            class_counter = {'red': 0, 'yellow': 0, 'green': 0}
            if class_sampled['yellow'] < target_per_class['yellow']:
                yellow_shortfall = target_per_class['yellow'] - class_sampled['yellow']
                print(f"Yellow shortfall: {yellow_shortfall}, adding more yellow samples...")
    
                # Go through ALL yellow images again
                for image_key, annotations_list in image_to_annotations.items():
                    class_counter = {'red': 0, 'yellow': 0, 'green': 0}
                    for ann in annotations_list:
                        if ann['label'] in class_counter:
                            class_counter[ann['label']] += 1
        
                    dominant_class = max(class_counter, key=class_counter.get)
        
                    if dominant_class == 'yellow' and class_sampled['yellow'] < target_per_class['yellow']:
                        best_annotation = max(annotations_list, 
                                            key=lambda ann: (ann.get('x2', 0) - ann.get('x1', 0)) * 
                                                          (ann.get('y2', 0) - ann.get('y1', 0)))
                        sampled_images.append(best_annotation)
                        class_sampled['yellow'] += 1

            for ann in annotations_list:
                if ann['label'] in class_counter:
                    class_counter[ann['label']] += 1
        
            if sum(class_counter.values()) > 0:
                dominant_class = max(class_counter, key=class_counter.get)
                image_class_counts[dominant_class] += 1

        print("Unique images per class:")
        for cls, count in image_class_counts.items():
            print(f"  {cls}: {count}")

        # Calculate sampling intervals
        sampling_intervals = {}
        for cls in ['red', 'yellow', 'green']:
            if image_class_counts[cls] > 0:
                # If we have fewer images than target, take all (interval=1)
                # Otherwise calculate interval to reach target
                if image_class_counts[cls] <= target_per_class[cls]:
                    sampling_intervals[cls] = 1  # Take all
                else:
                    sampling_intervals[cls] = max(1, image_class_counts[cls] // target_per_class[cls])
            else:
                sampling_intervals[cls] = 1

        print("\nSampling strategy:")
        for cls, interval in sampling_intervals.items():
            expected = image_class_counts[cls] // interval if image_class_counts[cls] > 0 else 0
            print(f"  {cls}: every {interval}th image → ~{expected} samples")

        # Sample images
        class_counters = {'red': 0, 'yellow': 0, 'green': 0}

        for image_key, annotations_list in image_to_annotations.items():
            class_counter = {'red': 0, 'yellow': 0, 'green': 0}
            for ann in annotations_list:
                if ann['label'] in class_counter:
                    class_counter[ann['label']] += 1
    
            if sum(class_counter.values()) == 0:
                continue
    
            dominant_class = max(class_counter, key=class_counter.get)
            class_counters[dominant_class] += 1

            # Sample based on interval
            should_sample = (class_counters[dominant_class] % sampling_intervals[dominant_class] == 0)
        
            # For yellow: if we have room, be more aggressive
            if dominant_class == 'yellow' and class_sampled['yellow'] < target_per_class['yellow']:
                # Take every yellow until we hit target
                should_sample = True

            if should_sample and class_sampled[dominant_class] < target_per_class[dominant_class]:
                # Use the annotation with the largest bounding box
                best_annotation = max(annotations_list, 
                                    key=lambda ann: (ann.get('x2', 0) - ann.get('x1', 0)) * 
                                                  (ann.get('y2', 0) - ann.get('y1', 0)))
            
                sampled_images.append(best_annotation)
                class_sampled[dominant_class] += 1
            
                # ONLY duplicate yellow if we're still far from target
                if dominant_class == 'yellow' and class_sampled['yellow'] < target_per_class['yellow']:
                    sampled_images.append(best_annotation)
                    class_sampled['yellow'] += 1

        print(f"\nSampled {len(sampled_images)} images total")
        print("Samples per class:")
        for cls, count in class_sampled.items():
            print(f"  {cls}: {count}")

        # Extract center crops
        counters = {
            'successful_extractions': 0,
            'successful_by_class': {'red': 0, 'yellow': 0, 'green': 0}
        }

        for i, annotation in enumerate(sampled_images):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{len(sampled_images)}")
    
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
                
                        # Extract center crop based on traffic light location
                        tl_center_x = (annotation['x1'] + annotation['x2']) // 2
                        tl_center_y = (annotation['y1'] + annotation['y2']) // 2
                
                        # Calculate tight crop based on bbox size
                        bbox_width = annotation['x2'] - annotation['x1']
                        bbox_height = annotation['y2'] - annotation['y1']

                        # Crop should be 2-3x the traffic light size, not half the image
                        crop_size = max(bbox_width, bbox_height) * 4
                        crop_size = min(crop_size, 300)  # Cap at 300px
                        crop_size = max(crop_size, 100)  # Minimum 100px
                
                        x1 = max(0, tl_center_x - crop_size // 2)
                        y1 = max(0, tl_center_y - crop_size // 2)
                        x2 = min(width, tl_center_x + crop_size // 2)
                        y2 = min(height, tl_center_y + crop_size // 2)
                
                        center_crop = img[y1:y2, x1:x2]
                
                        if center_crop.size == 0:
                            continue
                
                        crop_resized = cv2.resize(center_crop, CONFIG['img_size'])
                        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

                        # In _extract_center_crop_regions, after creating crop_rgb:
                        if counters['successful_by_class'][annotation['label']] < 10:
                            # Save first 10 of each class for inspection
                            debug_dir = Path('debug_crops') / annotation['label']
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            debug_path = debug_dir / f"{counters['successful_by_class'][annotation['label']]:03d}.png"
                            cv2.imwrite(str(debug_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                
                        self.images.append(crop_rgb)
                        self.labels.append(annotation['label'])
                        counters['successful_extractions'] += 1
                        counters['successful_by_class'][annotation['label']] += 1
                
                        break
                
                    except Exception as e:
                        continue

        print(f"\nExtracted {counters['successful_extractions']} center crops")
        for cls, count in counters['successful_by_class'].items():
            print(f"  {cls}: {count}")

        self.annotations.clear()
    
    def _extract_traffic_light_regions_improved(self, images_base):
        """Extract traffic light regions with BALANCED SAMPLING by class"""
    
        print("Extracting traffic light regions with balanced sampling...")
        print(f"Original annotations: {len(self.annotations)}")
    
        # PASO 1: Analizar distribución de clases ANTES del sampling
        class_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for annotation in self.annotations:
            label = annotation['label']
            if label in class_counts:
                class_counts[label] += 1
    
        print("Original class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count:,}")
    
        # PASO 2: Sampling balanceado por clase
        target_samples_per_class = 3000  # Objetivo: 3000 por clase
    
        sampled_annotations = []
        class_sampled = {'red': 0, 'yellow': 0, 'green': 0}
    
        # Calcular intervalos de sampling por clase
        sampling_intervals = {}
        for cls in class_counts:
            if class_counts[cls] > 0:
                # Intervalo para conseguir target_samples_per_class de cada clase
                sampling_intervals[cls] = max(1, class_counts[cls] // target_samples_per_class)
            else:
                sampling_intervals[cls] = 1
    
        print(f"\nSampling intervals:")
        for cls, interval in sampling_intervals.items():
            expected = class_counts[cls] // interval if class_counts[cls] > 0 else 0
            print(f"  {cls}: every {interval}th sample → ~{expected} samples")
    
        # Aplicar sampling balanceado
        class_counters = {'red': 0, 'yellow': 0, 'green': 0}
    
        for annotation in self.annotations:
            label = annotation['label']
            if label not in class_counters:
                continue
            
            class_counters[label] += 1
        
            # Tomar muestra si cumple el intervalo Y no hemos alcanzado el máximo
            if (class_counters[label] % sampling_intervals[label] == 0 and 
                class_sampled[label] < target_samples_per_class):
            
                sampled_annotations.append(annotation)
                class_sampled[label] += 1
    
        print(f"\nAfter balanced sampling:")
        total_sampled = sum(class_sampled.values())
        for cls, count in class_sampled.items():
            percentage = (count / total_sampled * 100) if total_sampled > 0 else 0
            print(f"  {cls}: {count} ({percentage:.1f}%)")
    
        print(f"Total sampled: {total_sampled} (vs original {len(self.annotations)})")
    
        # PASO 3: Procesar imágenes con tracking detallado
        counters = {
            'sampled_annotations': len(sampled_annotations),
            'image_not_found': 0,
            'invalid_bbox_dimensions': 0,
            'invalid_bbox_bounds': 0,
            'roi_too_small': 0,
            'processing_errors': 0,
            'successful_extractions': 0,
            'successful_by_class': {'red': 0, 'yellow': 0, 'green': 0}
        }
    
        for i, annotation in enumerate(sampled_annotations):
            # Progress tracking
            if i % 100 == 0 and i > 0:
                success_rate = counters['successful_extractions'] / i * 100
                print(f"Progress: {i}/{len(sampled_annotations)} ({i/len(sampled_annotations)*100:.1f}%) - Success: {success_rate:.1f}%")
            
                # Class-wise progress
                print("  Class progress:", end=" ")
                for cls in ['red', 'yellow', 'green']:
                    print(f"{cls}:{counters['successful_by_class'][cls]}", end=" ")
                print()
        
            image_found = False
        
            # Build image paths
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
                    images_base / sequence / "frames" / annotation['filename'],
                    images_base / "frames" / annotation['filename']
                ])
        
            # Try to find and process image
            for image_path in possible_paths:
                if image_path.exists():
                    try:
                        img = cv2.imread(str(image_path))
                        if img is None:
                            continue
                    
                        height, width = img.shape[:2]
                        x1, y1, x2, y2 = annotation['x1'], annotation['y1'], annotation['x2'], annotation['y2']
                    
                        # Validate bbox dimensions
                        if x1 >= x2 or y1 >= y2:
                            counters['invalid_bbox_dimensions'] += 1
                            break
                    
                        # Validate bbox bounds
                        if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
                            counters['invalid_bbox_bounds'] += 1
                            break
                    
                        # Clamp and pad bbox
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(x1 + 1, min(x2, width))
                        y2 = max(y1 + 1, min(y2, height))
                    
                        # Smart padding based on class (green lights are often smaller)
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                    
                        if annotation['label'] == 'green' or bbox_width < 15 or bbox_height < 15:
                            # More aggressive padding for small/green lights
                            padding_x = max(20, int(bbox_width * 0.8))
                            padding_y = max(20, int(bbox_height * 0.8))
                        else:
                            # Normal padding for red/yellow lights
                            padding_x = max(5, int(bbox_width * 0.2))
                            padding_y = max(5, int(bbox_height * 0.2))
                    
                        x1 = max(0, x1 - padding_x)
                        y1 = max(0, y1 - padding_y)
                        x2 = min(width, x2 + padding_x)
                        y2 = min(height, y2 + padding_y)
                    
                        # Extract ROI
                        roi = img[y1:y2, x1:x2]
                    
                        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                            counters['roi_too_small'] += 1
                            break
                    
                        # Process ROI
                        roi_resized = cv2.resize(roi, CONFIG['img_size'])
                        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    
                        self.images.append(roi_rgb)
                        self.labels.append(annotation['label'])
                        counters['successful_extractions'] += 1
                        counters['successful_by_class'][annotation['label']] += 1
                    
                        image_found = True
                        break
                    
                    except Exception as e:
                        counters['processing_errors'] += 1
                        if counters['processing_errors'] <= 3:
                            print(f"    Processing error: {e}")
                        break
        
            if not image_found:
                counters['image_not_found'] += 1
    
        # PASO 4: Reporte final detallado
        print("\n" + "="*50)
        print("FINAL EXTRACTION REPORT")
        print("="*50)
    
        print(f"Sampled annotations:        {counters['sampled_annotations']:,}")
        print(f"Successful extractions:     {counters['successful_extractions']:,}")
        print(f"Success rate:               {counters['successful_extractions']/counters['sampled_annotations']*100:.1f}%")
    
        print(f"\nLoss breakdown:")
        print(f"  Images not found:         {counters['image_not_found']:,}")
        print(f"  Invalid bbox dimensions:  {counters['invalid_bbox_dimensions']:,}")
        print(f"  Invalid bbox bounds:      {counters['invalid_bbox_bounds']:,}")
        print(f"  ROI too small:            {counters['roi_too_small']:,}")
        print(f"  Processing errors:        {counters['processing_errors']:,}")
    
        print(f"\nFinal class distribution:")
        total_final = sum(counters['successful_by_class'].values())
        for cls in ['red', 'yellow', 'green']:
            count = counters['successful_by_class'][cls]
            percentage = (count / total_final * 100) if total_final > 0 else 0
            print(f"  {cls}: {count:,} ({percentage:.1f}%)")
    
        # Clear memory
        self.annotations.clear()

    def _extract_full_scenes(self, images_base):
        """Extract FULL SCENES instead of cropped traffic light regions - BALANCED SAMPLING by class"""

        print("Extracting FULL SCENES with balanced sampling...")
        print(f"Original annotations: {len(self.annotations)}")

        # PASO 1: Analizar distribución de clases ANTES del sampling
        class_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for annotation in self.annotations:
            label = annotation['label']
            if label in class_counts:
                class_counts[label] += 1

        print("Original class distribution:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count:,}")

        # PASO 2: Sampling balanceado por clase - reducido para full scenes
        target_samples_per_class = 2000  # Reducido porque full scenes son más informativos

        # Group annotations by image to avoid duplicates per image
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

        # Sample images (not individual annotations) by class
        sampled_images = []
        class_sampled = {'red': 0, 'yellow': 0, 'green': 0}

        # Calculate sampling intervals per class based on unique images
        image_class_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for annotations_list in image_to_annotations.values():
            # Count each image by its dominant class
            class_counter = {'red': 0, 'yellow': 0, 'green': 0}
            for ann in annotations_list:
                if ann['label'] in class_counter:
                    class_counter[ann['label']] += 1
        
            # Assign image to class with most annotations
            dominant_class = max(class_counter, key=class_counter.get)
            if class_counter[dominant_class] > 0:
                image_class_counts[dominant_class] += 1

        sampling_intervals = {}
        for cls in class_counts:
            if image_class_counts[cls] > 0:
                sampling_intervals[cls] = max(1, image_class_counts[cls] // target_samples_per_class)
            else:
                sampling_intervals[cls] = 1

        print(f"\nImage-level sampling intervals:")
        for cls, interval in sampling_intervals.items():
            expected = image_class_counts[cls] // interval if image_class_counts[cls] > 0 else 0
            print(f"  {cls}: every {interval}th image → ~{expected} images")

        # Sample images by class
        class_counters = {'red': 0, 'yellow': 0, 'green': 0}
    
        for image_key, annotations_list in image_to_annotations.items():
            # Determine dominant class for this image
            class_counter = {'red': 0, 'yellow': 0, 'green': 0}
            for ann in annotations_list:
                if ann['label'] in class_counter:
                    class_counter[ann['label']] += 1
        
            if sum(class_counter.values()) == 0:
                continue
            
            dominant_class = max(class_counter, key=class_counter.get)
            class_counters[dominant_class] += 1
        
            # Take sample if it meets interval AND we haven't reached maximum
            if (class_counters[dominant_class] % sampling_intervals[dominant_class] == 0 and 
                class_sampled[dominant_class] < target_samples_per_class):
            
                # Use the annotation with the largest bounding box (most prominent traffic light)
                best_annotation = max(annotations_list, 
                                    key=lambda ann: (ann.get('x2', 0) - ann.get('x1', 0)) * 
                                                  (ann.get('y2', 0) - ann.get('y1', 0)))
            
                sampled_images.append(best_annotation)
                class_sampled[dominant_class] += 1

        print(f"\nAfter balanced image sampling:")
        total_sampled = sum(class_sampled.values())
        for cls, count in class_sampled.items():
            percentage = (count / total_sampled * 100) if total_sampled > 0 else 0
            print(f"  {cls}: {count} ({percentage:.1f}%)")

        print(f"Total sampled images: {total_sampled} (vs original {len(image_to_annotations)} unique images)")

        # PASO 3: Procesar imágenes COMPLETAS con tracking detallado
        counters = {
            'sampled_images': len(sampled_images),
            'image_not_found': 0,
            'processing_errors': 0,
            'successful_extractions': 0,
            'successful_by_class': {'red': 0, 'yellow': 0, 'green': 0}
        }

        for i, annotation in enumerate(sampled_images):
            # Progress tracking
            if i % 100 == 0 and i > 0:
                success_rate = counters['successful_extractions'] / i * 100
                print(f"Progress: {i}/{len(sampled_images)} ({i/len(sampled_images)*100:.1f}%) - Success: {success_rate:.1f}%")
        
                # Class-wise progress
                print("  Class progress:", end=" ")
                for cls in ['red', 'yellow', 'green']:
                    print(f"{cls}:{counters['successful_by_class'][cls]}", end=" ")
                print()
    
            image_found = False
    
            # Build image paths
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
                    images_base / sequence / "frames" / annotation['filename'],
                    images_base / "frames" / annotation['filename']
                ])
    
            # Try to find and process image
            for image_path in possible_paths:
                if image_path.exists():
                    try:
                        img = cv2.imread(str(image_path))
                        if img is None:
                            continue
                
                        height, width = img.shape[:2]
                    
                        # CRITICAL CHANGE: Process FULL IMAGE instead of cropped region
                    
                        # Optional: Focus on relevant part of image if it's very large
                        if width > 640 or height > 480:
                            # If image is very large, crop to center region to focus on traffic lights
                            center_x, center_y = width // 2, height // 2
                            crop_width = min(640, width)
                            crop_height = min(480, height)
                        
                            x1 = max(0, center_x - crop_width // 2)
                            y1 = max(0, center_y - crop_height // 2)
                            x2 = min(width, x1 + crop_width)
                            y2 = min(height, y1 + crop_height)
                        
                            img = img[y1:y2, x1:x2]
                    
                        # Resize ENTIRE image to model input size
                        # This preserves the full context and scene information
                        full_scene_resized = cv2.resize(img, CONFIG['img_size'])
                        full_scene_rgb = cv2.cvtColor(full_scene_resized, cv2.COLOR_BGR2RGB)
                
                        self.images.append(full_scene_rgb)
                        self.labels.append(annotation['label'])
                        counters['successful_extractions'] += 1
                        counters['successful_by_class'][annotation['label']] += 1
                
                        image_found = True
                        break
                
                    except Exception as e:
                        counters['processing_errors'] += 1
                        if counters['processing_errors'] <= 3:
                            print(f"    Processing error: {e}")
                        break
    
            if not image_found:
                counters['image_not_found'] += 1

        # PASO 4: Reporte final detallado
        print("\n" + "="*50)
        print("FINAL FULL SCENE EXTRACTION REPORT")
        print("="*50)

        print(f"Sampled images:             {counters['sampled_images']:,}")
        print(f"Successful extractions:     {counters['successful_extractions']:,}")
        print(f"Success rate:               {counters['successful_extractions']/counters['sampled_images']*100:.1f}%")

        print(f"\nLoss breakdown:")
        print(f"  Images not found:         {counters['image_not_found']:,}")
        print(f"  Processing errors:        {counters['processing_errors']:,}")

        print(f"\nFinal class distribution:")
        total_final = sum(counters['successful_by_class'].values())
        for cls in ['red', 'yellow', 'green']:
            count = counters['successful_by_class'][cls]
            percentage = (count / total_final * 100) if total_final > 0 else 0
            print(f"  {cls}: {count:,} ({percentage:.1f}%)")

        print("\nKEY DIFFERENCE: Training on full scenes instead of cropped traffic light regions")
        print("This should better match your ESP32 deployment scenario!")

        # Clear memory
        self.annotations.clear()
    
    def _print_dataset_structure_info(self, images_base):
        """Print information about the dataset structure for debugging"""
        print("\nDataset structure analysis:")
        
        # Check what directories exist
        for item in images_base.iterdir():
            if item.is_dir():
                print(f"Directory found: {item.name}")
                
                # Check subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"  Subdirectory: {item.name}/{subitem.name}")
                        
                        # Check for frames directory
                        frames_dir = subitem / "frames"
                        if frames_dir.exists():
                            frame_count = len(list(frames_dir.glob("*.png"))) + len(list(frames_dir.glob("*.jpg")))
                            print(f"    Frames directory found with {frame_count} images")
        
        # Print sample annotations for debugging
        if len(self.annotations) > 0:
            print(f"\nSample annotations:")
            for i, ann in enumerate(self.annotations[:3]):
                print(f"  {i+1}: {ann}")
                if i >= 2:
                    break
    
    def load_images_and_labels(self):
        """Load images and labels from directory structure or LISA data"""
        
        # If LISA data was processed, use it
        if len(self.images) > 0 and len(self.labels) > 0:
            print("Using processed LISA dataset")
            return np.array(self.images), np.array(self.labels)
        
        # Otherwise, load from directory structure (synthetic data)
        print("Loading from directory structure...")
        images = []
        labels = []
        
        for class_name in CONFIG['classes']:
            class_dir = self.data_dir / class_name
            
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, CONFIG['img_size'])
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(class_name)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)

class TrafficLightModel:
    """Traffic Light Classification Model"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
    def create_model(self, input_shape, num_classes):
        """Create lightweight MobileNetV2-based model for ESP32-S3"""
        
        # Use MobileNetV2 as base (efficient for mobile devices)
        base_model = keras.applications.MobileNetV2(
            input_shape=input_shape + (3,),
            include_top=False,
            weights='imagenet',
            alpha=0.5  # Reduced width multiplier for lighter model
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_simple_cnn(self, input_shape, num_classes):
        """Create simple CNN model"""
        
        model = keras.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape + (3,)),
            layers.MaxPooling2D(2, 2),
        
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
        
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
        
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_class_balanced_generator(self, X_train, y_train_cat, y_train_labels):
        # Split by class
        red_mask = y_train_labels == 'red'
        yellow_mask = y_train_labels == 'yellow'
        green_mask = y_train_labels == 'green'
    
        # RED gets extreme augmentation
        red_datagen = ImageDataGenerator(
            rotation_range=5,
            brightness_range=[0.4, 1.8],  # Very wide
            channel_shift_range=50,       # Extreme
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False
        )
    
        # Standard for yellow/green
        standard_datagen = self.create_traffic_light_augmentation()
    
        # Create generators with balanced batch sizes
        red_gen = red_datagen.flow(X_train[red_mask], y_train_cat[red_mask], 
                                    batch_size=8, shuffle=True)
        yellow_gen = standard_datagen.flow(X_train[yellow_mask], y_train_cat[yellow_mask], 
                                           batch_size=4, shuffle=True)
        green_gen = standard_datagen.flow(X_train[green_mask], y_train_cat[green_mask], 
                                          batch_size=4, shuffle=True)
    
        # Combined generator
        while True:
            r_batch = next(red_gen)
            y_batch = next(yellow_gen)
            g_batch = next(green_gen)
        
            X_combined = np.concatenate([r_batch[0], y_batch[0], g_batch[0]])
            y_combined = np.concatenate([r_batch[1], y_batch[1], g_batch[1]])
        
            indices = np.random.permutation(len(X_combined))
            yield X_combined[indices], y_combined[indices]

    def create_traffic_light_augmentation(self):
        """Specialized augmentation for traffic lights"""
    
        train_datagen = ImageDataGenerator(
            # Geometric augmentations (conservative for traffic lights)
            rotation_range=3,            # Very small rotation only
            width_shift_range=0.03,      # Minimal shifts
            height_shift_range=0.03,
            zoom_range=0.03,             # Minimal zoom
            # shear_range=0.02,            # Very small shear
        
            # Photometric augmentations (more important for traffic lights)
            brightness_range=[0.8, 1.2], # More aggressive brightness variation
            channel_shift_range=10,       # Color channel variations
        
            # No flipping for traffic lights
            horizontal_flip=False,
            vertical_flip=False,
        
            # Fill mode
            fill_mode='nearest'
        )
    
        return train_datagen

    def create_class_specific_augmentation(self, class_name):
        """Different augmentation for each class"""
        if class_name == 'green':
            # More aggressive augmentation for green to increase variety
            return ImageDataGenerator(
                rotation_range=5,
                brightness_range=[0.6, 1.4],  # Wider brightness range
                channel_shift_range=20,        # More color variation
                zoom_range=0.1,
                width_shift_range=0.05,
                height_shift_range=0.05,
                horizontal_flip=False,
                vertical_flip=False
            )
        else:
            # Standard augmentation for red/yellow
            return self.create_traffic_light_augmentation()

    def debug_class_mapping(self, label_encoder, classes_config):
        """Debug and verify class mapping"""
        print("\n" + "="*50)
        print("CLASS MAPPING VERIFICATION")
        print("="*50)
    
        print("Configured classes:", classes_config)
        print("LabelEncoder classes:", label_encoder.classes_)
        print("Class to index mapping:")
    
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  {class_name} -> {i}")
    
        # Test encoding/decoding
        print("\nTesting encoding:")
        for class_name in ['red', 'yellow', 'green']:
            if class_name in label_encoder.classes_:
                encoded = label_encoder.transform([class_name])[0]
                decoded = label_encoder.inverse_transform([encoded])[0]
                print(f"  {class_name} -> {encoded} -> {decoded}")
            else:
                print(f"  {class_name} -> NOT FOUND IN ENCODER!")
    
        # Verify consistency
        is_consistent = (list(label_encoder.classes_) == classes_config)
        print(f"\nClass order consistent with config: {is_consistent}")
    
        if not is_consistent:
            print("WARNING: Class order mismatch detected!")
            print("This will cause prediction errors!")
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the model"""
        print("Setting up manual class mapping (no LabelEncoder)...")

        # Manual class mapping with YOUR desired order
        desired_classes = ['red', 'yellow', 'green']  # Your desired order
        class_to_idx = {cls: idx for idx, cls in enumerate(desired_classes)}
        idx_to_class = {idx: cls for idx, cls in enumerate(desired_classes)}

        print(f"Manual class mapping: {class_to_idx}")

        # Manual encoding function
        def manual_encode(labels):
            return np.array([class_to_idx[label] for label in labels])

        def manual_decode(indices):
            return np.array([idx_to_class[idx] for idx in indices])

        # Store these for later use in evaluation
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.encode_labels = manual_encode
        self.decode_labels = manual_decode

        # Encode labels manually
        y_train_encoded = manual_encode(y_train)
        y_val_encoded = manual_encode(y_val)

        # Verify encoding
        print(f"\nLabel distribution after manual encoding:")
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        for idx, count in zip(unique, counts):
            class_name = idx_to_class[idx]
            print(f"  Class {idx} ({class_name}): {count} samples")

        # Convert to categorical
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

        class_weights[0] *= 3.0   # red (HEAVY - it's failing badly)
        class_weights[1] *= 2.0   # yellow (moderate boost)
        class_weights[2] *= 0.7   # green (reduce - it's overconfident)
    
        print(f"Moderate class weights: {class_weights}")
    
        # Create improved model
        # self.model = self.create_model(CONFIG['img_size'], len(CONFIG['classes']))
        self.model = self.create_simple_cnn(CONFIG['img_size'], len(CONFIG['classes']))
    
        # Better optimizer with lower learning rate
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0005,  # Lower learning rate
            beta_1=0.9,
            beta_2=0.999
        )
    
        # Compile with label smoothing to prevent overconfidence
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Label smoothing
            metrics=['accuracy']
        )
    
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=8, 
                restore_best_weights=True,
                monitor='val_accuracy'  # Monitor validation accuracy
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.3, 
                patience=4,
                min_lr=1e-7,
                monitor='val_accuracy'
            ),
            # Add learning rate scheduling
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.0005 * (0.95 ** epoch)  # Exponential decay
            )
        ]
    
        # Enhanced data augmentation
        train_datagen = self.create_traffic_light_augmentation()
        val_datagen = ImageDataGenerator()  # No augmentation for validation
    
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat,
            batch_size=CONFIG['batch_size'],
            shuffle=True
        )
        # Replace the standard train_generator with:
        #train_generator = self.create_class_balanced_generator(X_train, y_train_cat, y_train)
    
        val_generator = val_datagen.flow(
            X_val, y_val_cat,
            batch_size=CONFIG['batch_size'],
            shuffle=False
        )
    
        print("Improved model architecture:")
        self.model.summary()
    
        # Calculate steps
        steps_per_epoch = len(X_train) // CONFIG['batch_size']
        validation_steps = len(X_val) // CONFIG['batch_size']
    
        # Train with more epochs and better monitoring
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=CONFIG['epochs'],  # More epochs with early stopping
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
    
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
         # Use manual encoding instead of label_encoder
        y_test_encoded = self.encode_labels(y_test)
        y_test_cat = keras.utils.to_categorical(y_test_encoded, len(self.class_to_idx))
    
        test_loss, test_acc = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
    
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
    
        # Classification report
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred_classes, 
                                  target_names=CONFIG['classes']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_encoded, y_pred_classes)
        print(cm)
        
        return test_acc
    
    def convert_to_tflite(self, output_path):
        """Convert model to TensorFlow Lite for ESP32-S3"""
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # For ESP32-S3 compatibility - avoid optimizations that cause issues
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Commented out
        
        # Keep as float32 for better ESP32-S3 compatibility
        #converter.target_spec.supported_types = [tf.float16]  # Commented out
        
        # Use only basic TFLITE operations for microcontroller compatibility
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        
        # Disable experimental features that might cause issues
        #converter.experimental_new_converter = False
        
        tflite_model = converter.convert()
        
        # Save the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Float32 TFLite model saved to: {output_path}")
        print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        # Create a quantized version as well for comparison
        self._create_quantized_version(output_path)
        
        return tflite_model
    
    def _create_quantized_version(self, base_output_path):
        """Create a separate quantized version for testing"""
        quantized_path = base_output_path.replace('.tflite', '_quantized.tflite')
        
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
            quantized_model = converter.convert()
            
            with open(quantized_path, 'wb') as f:
                f.write(quantized_model)
            
            print(f"Quantized model saved to: {quantized_path}")
            print(f"Quantized model size: {len(quantized_model) / 1024:.2f} KB")
            
        except Exception as e:
            print(f"Quantized conversion failed: {e}")
            print("Using float32 version only")
    
    def _representative_dataset(self):
        """Representative dataset for quantization"""
        # Use a subset of training data for calibration
        for i in range(10):
            yield [np.random.random((1,) + CONFIG['img_size'] + (3,)).astype(np.float32)]
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/training_history.png", dpi=150, bbox_inches='tight')
        plt.show()

def preprocess_images(images):
    processed_images = []
    
    for img in images:
        # Convert to HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # AGGRESSIVE saturation boost
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 2.0, 0, 255)  # 2.0x not 1.8x
        
        # Value enhancement
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.2, 0, 255)
        
        img_enhanced = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # RED-SPECIFIC boost
        # Find red-ish pixels and enhance them
        red_mask = (img_enhanced[:,:,0] > 100) & (img_enhanced[:,:,0] > img_enhanced[:,:,1] * 1.2)
        img_enhanced[:,:,0][red_mask] = np.clip(img_enhanced[:,:,0][red_mask] * 1.3, 0, 255)
        
        # Contrast
        img_enhanced = np.clip(img_enhanced * 1.15, 0, 255).astype(np.uint8)
        
        processed_images.append(img_enhanced)
    
    return np.array(processed_images, dtype=np.float32) / 255.0
    return np.array(processed_images, dtype=np.float32) / 255.0

def main():
    """Main training pipeline"""
    
    print("=== Traffic Light Classification for ESP32-S3 ===\n")
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Initialize dataset processor
    processor = LISADatasetProcessor(CONFIG['data_dir'])
    
    # Try to load real LISA data, otherwise use sample data
    lisa_path = input("Enter path to LISA dataset (or press Enter to use sample data): ").strip()
    if lisa_path and processor.load_real_lisa_data(lisa_path):
        print("Using real LISA dataset")
    else:
        print("No LISA path provided")
        return  # Salir si no hay datos LISA
    
    # Load images and labels
    print("Loading images and labels...")
    images, labels = processor.load_images_and_labels()
    
    if len(images) == 0:
        print("No images found! Please check the dataset.")
        return
    
    print(f"Loaded {len(images)} images")
    print(f"Classes: {np.unique(labels)}")
    
    # Preprocess images
    images = preprocess_images(images)
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    print(f"Test set: {len(X_test)} images")
    
    # Initialize and train model
    model = TrafficLightModel()
    
    print("\nTraining model...")
    history = model.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_accuracy = model.evaluate_model(X_test, y_test)
    
    # Plot training history
    model.plot_training_history()
    
    # Save Keras model
    keras_model_path = f"{CONFIG['output_dir']}/{CONFIG['model_name']}.h5"
    model.model.save(keras_model_path)
    print(f"Keras model saved to: {keras_model_path}")
    
    # Convert to TensorFlow Lite
    tflite_model_path = f"{CONFIG['output_dir']}/{CONFIG['model_name']}.tflite"
    tflite_model = model.convert_to_tflite(tflite_model_path)
    
    # Save label encoder
    import pickle
    label_encoder_path = f"{CONFIG['output_dir']}/label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(model.label_encoder, f)
    # print(f"Label encoder classes: {model.label_encoder.classes_}")
    # Save model info
    model_info = {
        'input_shape': CONFIG['img_size'] + (3,),
        'classes': ['red', 'yellow', 'green'],  # Your desired order
        'class_to_idx': {'red': 0, 'yellow': 1, 'green': 2},
        'idx_to_class': {0: 'red', 1: 'yellow', 2: 'green'},
        'test_accuracy': float(test_accuracy),
        'model_size_kb': len(tflite_model) / 1024
    }
    
    with open(f"{CONFIG['output_dir']}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n=== Training Complete ===")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Model Size: {len(tflite_model) / 1024:.2f} KB")
    print(f"Files saved in: {CONFIG['output_dir']}/")
    
    # ESP32-S3 deployment instructions
    print("\n=== ESP32-S3 Deployment Instructions ===")
    print("1. Copy the .tflite file to your ESP32-S3 project")
    print("2. Use TensorFlow Lite for Microcontrollers library")
    print("3. Input image size: 96x96x3")
    print("4. Classes: red=0, yellow=1, green=2")
    print("5. Normalize input images to [0, 1] range")

def test_tflite_model():
    """Test the converted TensorFlow Lite model"""
    
    tflite_model_path = f"{CONFIG['output_dir']}/{CONFIG['model_name']}.tflite"
    
    if not os.path.exists(tflite_model_path):
        print("TensorFlow Lite model not found!")
        return
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("TensorFlow Lite Model Info:")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test with random input
    test_input = np.random.random(input_details[0]['shape']).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output[0])
    confidence = output[0][predicted_class]
    
    print(f"Test prediction: Class {predicted_class} ({CONFIG['classes'][predicted_class]}) with confidence {confidence:.4f}")

if __name__ == "__main__":
    main()
    
    # Test the TFLite model
    print("\nTesting TensorFlow Lite model...")
    test_tflite_model()