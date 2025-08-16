from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from yolo.object_detection import YoloDetector
from model_monitor import start_model_monitoring
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import random
import json
import shutil
from pathlib import Path
import yaml
from datetime import datetime
from fastapi.responses import Response
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

yolo = YoloDetector()

# Training data directory
TRAINING_DATA_DIR = Path("training_data")
TRAINING_DATA_DIR.mkdir(exist_ok=True)

# Start model monitoring
try:
    model_monitor = start_model_monitoring(yolo, ["./yolov8s-world.pt"])
    print("=" * 60)
    print("🚀 YOLO-World Object Detection API Starting...")
    print("=" * 60)
    print("✅ Model monitoring started successfully")
    print("📁 Monitoring model files for changes...")
    print("🔄 Automatic model reload enabled")
    print("=" * 60)
except Exception as e:
    print(f"⚠️  Warning: Could not start model monitoring: {e}")
    print("📝 Manual model reload will be required")
    model_monitor = None

# Enhanced FastAPI app with better OpenAPI documentation
app = FastAPI(
    title="YOLO-World Object Detection API",
    description="""
    🔍 **YOLO-World Object Detection API**

    This API provides real-time object detection capabilities using YOLO-World model.
    You can:

    * 🎯 **Configure detection classes** - Add custom object classes for detection
    * 📷 **Detect objects in images** - Upload images and get object detection results
    * ⚙️ **Manage model settings** - View and modify model configuration
    * 🎚️ **Adjust confidence** - Control detection sensitivity

    ## Usage Flow
    1. **Add detection classes** using `POST /model/classes`
    2. **Upload an image** using `POST /detect` or `POST /detect/with-confidence`
    3. **Get detection results** with bounding boxes, confidence scores, and class labels

    ## Supported Image Formats
    - JPEG (.jpg, .jpeg)
    - PNG (.png)
    - Other common image formats supported by PIL
    """,
    version="1.0.0",
    contact={
        "name": "YOLO-World API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "info",
            "description": "API information and health checks"
        },
        {
            "name": "model",
            "description": "Model configuration and class management"
        },
        {
            "name": "detection",
            "description": "Object detection operations"
        },
        {
            "name": "labeling",
            "description": "Labeling operations"
        },
        {
            "name": "training",
            "description": "Training operations"
        }
    ]
)


# Enhanced Pydantic models with better documentation
class ClassesRequest(BaseModel):
    """Request model for adding detection classes"""
    classes: List[str] = Field(
        ...,
        description="List of object class names to add for detection",
    )

class ClassesResponse(BaseModel):
    """Response model for class-related operations"""
    classes: List[str] = Field(
        ...,
        description="List of currently configured detection classes"
    )
    message: str = Field(
        ...,
        description="Status message about the operation"
    )

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_path: str = Field(
        ...,
        description="Path to the YOLO model file"
    )
    vocab_file: str = Field(
        ...,
        description="Path to the vocabulary file"
    )
    current_classes: List[str] = Field(
        ...,
        description="Currently configured detection classes"
    )
    total_classes: int = Field(
        ...,
        description="Total number of configured classes"
    )

class Detection(BaseModel):
    """Single object detection result"""
    class_name: str = Field(
        ...,
        alias="class",
        description="Detected object class name"
    )
    confidence: float = Field(
        ...,
        description="Detection confidence score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    bbox: List[float] = Field(
        ...,
        description="Bounding box coordinates [x1, y1, x2, y2] in pixels",
    )

class DetectionResponse(BaseModel):
    """Response model for object detection"""
    detections: List[Detection] = Field(
        ...,
        description="List of detected objects"
    )
    message: str = Field(
        ...,
        description="Status message about the detection operation"
    )
    processed_image: str = Field(
        ...,
        description="Base64 encoded image with bounding boxes drawn"
    )

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str = Field(
        ...,
        description="Status or informational message"
    )

class LabelingData(BaseModel):
    """Labeling data for training"""
    boxes: List[Dict] = Field(
        ...,
        description="List of bounding boxes with labels"
    )
    image_width: int = Field(
        ...,
        description="Original image width"
    )
    image_height: int = Field(
        ...,
        description="Original image height"
    )

class LabelingResponse(BaseModel):
    """Response for labeling submission"""
    message: str = Field(
        ...,
        description="Status message"
    )
    saved_path: str = Field(
        ...,
        description="Path where the labeling data was saved"
    )
    total_labels: int = Field(
        ...,
        description="Total number of labels in the dataset"
    )


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/",
    tags=["info"],
    summary="API Information",
    description="Get basic information about the API and available endpoints",
    response_model=Dict[str, Any]
)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "YOLO-World Object Detection API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "GET /": "API information",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)",
            "GET /openapi.json": "OpenAPI specification",
            "GET /model/info": "Get model information",
            "GET /model/classes": "Get current detection classes",
            "POST /model/classes": "Add new detection classes",
            "DELETE /model/classes": "Clear all detection classes",
            "POST /detect": "Detect objects in uploaded image",
            "POST /detect/with-confidence": "Detect objects with custom confidence",
            "POST /labeling/submit": "Submit labeling data",
            "POST /training/start": "Start model fine-tuning",
            "GET /training/data/stats": "Get training data statistics"
        }
    }

@app.get(
    "/model/info",
    tags=["model"],
    summary="Get Model Information",
    description="Get current model information and statistics",
    response_model=Dict[str, Any]
)
async def get_model_info():
    """Get current model information"""
    try:
        # Get model classes with proper error handling
        classes = []
        try:
            if hasattr(yolo, 'get_classes'):
                classes = yolo.get_classes()
            elif hasattr(yolo, 'get_current_classes'):
                classes = yolo.get_current_classes()
            else:
                # Fallback: try to get classes from vocab file
                vocab_file = Path("custom_vocab.json")
                if vocab_file.exists():
                    with open(vocab_file, 'r') as f:
                        vocab_data = json.load(f)
                        classes = list(vocab_data.keys()) if isinstance(vocab_data, dict) else []
        except Exception as e:
            print(f"Warning: Could not get classes: {e}")
            classes = []

        # Ensure classes is always a list
        if not isinstance(classes, list):
            classes = []

        # Get model file info
        model_path = Path("yolov8s-world.pt")
        model_size = model_path.stat().st_size / (1024*1024) if model_path.exists() else 0

        # Get model status
        model_status = "Loaded" if hasattr(yolo, 'model') and yolo.model else "Not loaded"

        # Get detection statistics (placeholder for now)
        total_detections = 0  # TODO: Implement actual detection counting
        accuracy = "95%"  # TODO: Implement actual accuracy calculation

        return {
            "model_name": "YOLOv8s-World",
            "model_path": str(model_path.absolute()) if model_path.exists() else "Not found",
            "model_size_mb": round(model_size, 2),
            "status": model_status,
            "classes": classes,
            "total_detections": total_detections,
            "accuracy": accuracy,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        print(f"Error getting model info: {e}")
        import traceback
        traceback.print_exc()
        # Return safe default values on error
        return {
            "model_name": "Unknown",
            "model_path": "Not found",
            "model_size_mb": 0,
            "status": "Error",
            "classes": [],
            "total_detections": 0,
            "accuracy": "0%",
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "error": str(e)
        }

@app.get(
    "/model/classes",
    tags=["model"],
    summary="Get Detection Classes",
    description="Retrieve all currently configured object detection classes",
    response_model=ClassesResponse
)
async def get_current_classes():
    """Get currently configured detection classes"""
    try:
        current_classes = yolo.get_current_classes()
        return ClassesResponse(
            classes=current_classes,
            message=f"Retrieved {len(current_classes)} detection classes"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting classes: {str(e)}")

@app.post(
    "/model/classes",
    tags=["model"],
    summary="Add Detection Classes",
    description="""
    Add new object detection classes to the model.

    **Important Notes:**
    - Classes are case-sensitive
    - Duplicate classes are automatically filtered out
    - Empty strings and whitespace-only strings are ignored
    - Classes are automatically saved to the vocabulary file

    **Example classes:** person, car, bicycle, dog, cat, chair, table, etc.
    """,
    response_model=ClassesResponse,
    responses={
        200: {
            "description": "Classes added successfully",
            "content": {
                "application/json": {
                    "example": {
                        "classes": ["person", "car", "bicycle"],
                        "message": "Successfully added classes. Total classes: 3"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input - empty classes list or no valid classes",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Classes list cannot be empty"
                    }
                }
            }
        }
    }
)
async def add_detection_classes(request: ClassesRequest):
    """Add new detection classes to the model"""
    try:
        if not request.classes:
            raise HTTPException(status_code=400, detail="Classes list cannot be empty")

        # Filter out empty strings
        valid_classes = [cls.strip() for cls in request.classes if cls.strip()]
        if not valid_classes:
            raise HTTPException(status_code=400, detail="No valid classes provided")

        yolo.add_classes(valid_classes)
        updated_classes = yolo.get_current_classes()

        return ClassesResponse(
            classes=updated_classes,
            message=f"Successfully added classes. Total classes: {len(updated_classes)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding classes: {str(e)}")

@app.delete(
    "/model/classes",
    tags=["model"],
    summary="Clear All Detection Classes",
    description="""
    Remove all currently configured detection classes.

    **Warning:** This action will:
    - Clear all detection classes from memory
    - Update the vocabulary file
    - Make object detection unavailable until new classes are added
    """,
    response_model=MessageResponse
)
async def clear_detection_classes():
    """Clear all detection classes"""
    try:
        yolo.current_classes.clear()
        yolo._update_model_classes()
        yolo._save_custom_vocab()

        return MessageResponse(message="All detection classes cleared successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing classes: {str(e)}")

def draw_bounding_boxes(image_path: str, detections_data: List[Dict]) -> str:
    """
    画像にバウンディングボックスを描画し、Base64エンコードした文字列を返す
    """
    try:
        # 画像を開く
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # カラーパレット
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
            '#FECA57', '#FF9FF3', '#A8E6CF', '#FFD93D'
        ]

        # フォントを設定（デフォルトフォントを使用）
        try:
            # より大きなフォントを試す
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

        for i, detection in enumerate(detections_data):
            # バウンディングボックスの座標
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # 色を選択
            color = colors[i % len(colors)]

            # バウンディングボックスを描画（太い線）
            line_width = 4
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # ラベルテキストの準備
            label = f"{detection['class']} {detection['confidence']*100:.0f}%"

            # テキストサイズを計算
            try:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                # fallback for older PIL versions
                try:
                    # Try to get text size using textbbox with different method
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                except:
                    # Final fallback - estimate text size
                    text_width = len(label) * 10  # Rough estimate
                    text_height = 20  # Rough estimate

            # ラベル背景を描画
            label_y = max(0, y1 - text_height - 10)
            draw.rectangle(
                [x1, label_y, x1 + text_width + 10, y1],
                fill=color
            )

            # ラベルテキストを描画
            draw.text(
                (x1 + 5, label_y + 2),
                label,
                fill='white',
                font=font
            )

        # 画像をBase64にエンコード
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        # エラーの場合は元の画像をそのまま返す
        with open(image_path, "rb") as img_file:
            img_str = base64.b64encode(img_file.read()).decode()
        return img_str

@app.post(
    "/detect",
    tags=["detection"],
    summary="Detect Objects in Image",
    description="""
    Upload an image and detect objects using the configured detection classes.
    Returns the detection results along with the processed image containing bounding boxes.

    **Requirements:**
    - At least one detection class must be configured (use POST /model/classes)
    - Image file must be a valid image format (JPEG, PNG, etc.)
    - File size should be reasonable for processing

    **Returns:**
    - List of detected objects with bounding boxes
    - Confidence scores for each detection
    - Class labels for identified objects
    - Processed image with bounding boxes drawn (Base64 encoded)

    **Bounding Box Format:**
    - [x1, y1, x2, y2] where (x1,y1) is top-left corner and (x2,y2) is bottom-right corner
    - Coordinates are in pixels relative to the original image dimensions
    """,
    responses={
        200: {
            "description": "Detection completed successfully with processed image",
            "content": {
                "application/json": {
                    "example": {
                        "detections": [
                            {
                                "class": "person",
                                "confidence": 0.85,
                                "bbox": [100, 50, 200, 300]
                            }
                        ],
                        "message": "Object detection completed. Found 1 objects.",
                        "processed_image": "base64_encoded_image_string"
                    }
                }
            }
        },
        400: {
            "description": "Invalid file format or empty file",
        },
        500: {
            "description": "Processing error"
        }
    }
)
async def detect_object(
    image: UploadFile
):
    """Detect objects in uploaded image and return processed image with bounding boxes"""
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read the uploaded file content
        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        try:
            # Perform object detection
            result = yolo.predict_image(temp_file_path)

            if result is None:
                return {
                    "detections": [],
                    "message": "No detection classes set. Please configure the model first using POST /model/classes",
                    "processed_image": ""
                }

            # Extract detection information
            detections = []
            detections_data = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    detection = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": [float(coord) for coord in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)
                    detections_data.append(detection)

            # 画像にバウンディングボックスを描画
            processed_image_b64 = draw_bounding_boxes(temp_file_path, detections_data)

            return {
                "detections": detections,
                "message": f"Object detection completed. Found {len(detections)} objects.",
                "processed_image": processed_image_b64
            }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post(
    "/detect/with-confidence",
    tags=["detection"],
    summary="Detect Objects with Custom Confidence",
    description="""
    Upload an image and detect objects with a custom confidence threshold.

    **Confidence Threshold:**
    - Range: 0.0 to 1.0
    - Higher values = fewer, more confident detections
    - Lower values = more detections, potentially less accurate
    - Default: 0.25

    **Recommended Values:**
    - 0.1-0.3: Maximum detections (may include false positives)
    - 0.3-0.5: Balanced detection (default range)
    - 0.5-0.8: High confidence detections only
    - 0.8-1.0: Very conservative detection
    """,
    responses={
        200: {
            "description": "Detection completed successfully with custom confidence",
        },
        400: {
            "description": "Invalid confidence value or file format",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Confidence must be between 0.0 and 1.0"
                    }
                }
            }
        }
    }
)
async def detect_object_with_confidence(
    image: UploadFile,
    confidence: float = Form(0.25)
):
    """Detect objects in uploaded image with custom confidence threshold"""
    try:
        # Validate confidence threshold
        if not 0.0 <= confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")

        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read the uploaded file content
        image_bytes = await image.read()

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        try:
            # Perform object detection with custom confidence
            result = yolo.predict_image(temp_file_path, conf_threshold=confidence)

            if result is None:
                return {
                    "detections": [],
                    "message": "No detection classes set. Please configure the model first using POST /model/classes"
                }

            # Extract detection information
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    detection = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)

            return {
                "detections": detections,
                "message": f"Object detection completed with confidence {confidence}. Found {len(detections)} objects."
            }

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def save_labeling_data(image_path: str, labeling_data: LabelingData, image_filename: str):
    """
    ラベリングデータをYOLO形式で保存
    """
    # Create subdirectories
    images_dir = TRAINING_DATA_DIR / "images"
    labels_dir = TRAINING_DATA_DIR / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{os.path.splitext(image_filename)[0]}"

    # Save image
    image_save_path = images_dir / f"{base_name}.jpg"
    shutil.copy2(image_path, image_save_path)

    # Convert bounding boxes to YOLO format and save annotation
    label_save_path = labels_dir / f"{base_name}.txt"

    with open(label_save_path, 'w') as f:
        for box in labeling_data.boxes:
            # Convert absolute coordinates to YOLO format (normalized)
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

            # Calculate center and dimensions (normalized)
            center_x = (x1 + x2) / 2 / labeling_data.image_width
            center_y = (y1 + y2) / 2 / labeling_data.image_height
            width = abs(x2 - x1) / labeling_data.image_width
            height = abs(y2 - y1) / labeling_data.image_height

            # Get class ID (for now, use simple mapping)
            class_name = box['label']
            class_id = get_or_create_class_id(class_name)

            # Write YOLO format: class_id center_x center_y width height
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    return str(image_save_path), str(label_save_path)

def get_or_create_class_id(class_name: str) -> int:
    """
    クラス名からクラスIDを取得または作成
    """
    classes_file = TRAINING_DATA_DIR / "classes.txt"

    # Load existing classes
    classes = []
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]

    # Find class ID
    if class_name in classes:
        return classes.index(class_name)
    else:
        # Add new class
        classes.append(class_name)
        with open(classes_file, 'w', encoding='utf-8') as f:
            for cls in classes:
                f.write(f"{cls}\n")
        return len(classes) - 1

def count_total_labels():
    """
    保存されているラベルの総数をカウント
    """
    labels_dir = TRAINING_DATA_DIR / "labels"
    if not labels_dir.exists():
        return 0

    total = 0
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            total += len([line for line in f if line.strip()])

    return total

def create_training_config():
    """
    YOLOv8用のトレーニング設定ファイルを作成
    """
    classes_file = TRAINING_DATA_DIR / "classes.txt"
    if not classes_file.exists():
        return None

    # Load classes
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    if not classes:
        return None

    # Create data.yaml
    config = {
        'path': str(TRAINING_DATA_DIR.absolute()),
        'train': 'images',
        'val': 'images',  # For simplicity, using same for validation
        'nc': len(classes),
        'names': classes
    }

    config_path = TRAINING_DATA_DIR / "data.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    return str(config_path)

# Global training status
training_status = {
    "is_training": False,
    "current_run": None,
    "progress": "Not started"
}

# Thread pool for running training
training_executor = ThreadPoolExecutor(max_workers=1)

def run_training_in_background(config_path: str, epochs: int):
    """Run training in a background thread"""
    global training_status

    try:
        training_status["is_training"] = True
        training_status["progress"] = "Starting training..."

        print(f"🚀 Starting background training with {epochs} epochs...")

        # Run training in thread
        results = yolo.fine_tune_model(config_path, epochs=epochs)

        # Get the path to the best trained model
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            # Find the most recent training run
            train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
            if train_dirs:
                latest_run = max(train_dirs, key=lambda x: x.stat().st_mtime)
                best_model_path = latest_run / "weights" / "best.pt"

                if best_model_path.exists():
                    # Load the fine-tuned model with automatic reload
                    print("🔄 Auto-reloading fine-tuned model...")
                    print(f"📁 Model path: {best_model_path}")
                    yolo.load_trained_model(str(best_model_path))
                    print("✅ Fine-tuned model auto-reloaded successfully!")
                    print(f"⏰ Model reload completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"📊 Model file size: {best_model_path.stat().st_size / (1024*1024):.2f} MB")

        training_status["progress"] = "Training completed successfully!"
        print("✅ Background training completed!")

    except Exception as e:
        training_status["progress"] = f"Training failed: {str(e)}"
        print(f"❌ Background training failed: {e}")
    finally:
        training_status["is_training"] = False

@app.post(
    "/training/start",
    tags=["training"],
    summary="Start Model Fine-Tuning",
    description="""
    Start model fine-tuning with collected labeling data.

    **Requirements:**
    - At least some labeled training data must be available
    - Training data should be in proper YOLO format

    **Process:**
    1. Validates training data availability
    2. Creates training configuration
    3. Starts training in background
    4. Returns training status immediately
    """,
    response_model=MessageResponse
)
async def start_model_training(epochs: int = 50):
    """Start model fine-tuning with collected labeling data"""
    try:
        # Check if already training
        if training_status["is_training"]:
            raise HTTPException(
                status_code=400,
                detail="Training is already in progress. Please wait for it to complete."
            )

        # Validate epochs parameter
        if epochs <= 0 or epochs > 500:
            raise HTTPException(
                status_code=400,
                detail="Epochs must be between 1 and 500"
            )

        # Check if training data exists
        images_dir = TRAINING_DATA_DIR / "images"
        labels_dir = TRAINING_DATA_DIR / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="No training data available. Please submit some labeled data first."
            )

        # Count training samples
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))

        if len(image_files) == 0 or len(label_files) == 0:
            raise HTTPException(
                status_code=400,
                detail="Insufficient training data. Please add more labeled images."
            )

        # Create training config
        config_path = create_training_config()
        if not config_path:
            raise HTTPException(
                status_code=400,
                detail="Could not create training configuration. Please check your labeling data."
            )

        # Start training in background thread
        training_status["current_run"] = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Submit training to thread pool
        training_executor.submit(run_training_in_background, config_path, epochs)

        total_labels = count_total_labels()

        return MessageResponse(
            message=f"Training started successfully! Training on {len(image_files)} images with {total_labels} labels for {epochs} epochs. Training is running in the background."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

@app.get(
    "/training/status",
    tags=["training"],
    summary="Get Training Status",
    description="Get current training status and progress",
    response_model=Dict[str, Any]
)
async def get_training_status():
    """Get current training status"""
    return {
        "is_training": training_status["is_training"],
        "current_run": training_status["current_run"],
        "progress": training_status["progress"]
    }

@app.get(
    "/training/data/stats",
    tags=["training"],
    summary="Get Training Data Statistics",
    description="Get statistics about the collected training data",
    response_model=Dict[str, Any]
)
async def get_training_data_stats():
    """Get statistics about training data"""
    try:
        images_dir = TRAINING_DATA_DIR / "images"
        labels_dir = TRAINING_DATA_DIR / "labels"
        classes_file = TRAINING_DATA_DIR / "classes.txt"

        stats = {
            "total_images": 0,
            "total_labels": 0,
            "classes": [],
            "class_counts": {},
            "data_directory": str(TRAINING_DATA_DIR.absolute())
        }

        # Count images
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            stats["total_images"] = len(image_files)

        # Count labels and class distribution
        if labels_dir.exists():
            total_labels = 0
            class_counts = {}

            # Load class names
            classes = []
            if classes_file.exists():
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes = [line.strip() for line in f if line.strip()]

            stats["classes"] = classes

            # Count labels per class
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_labels += 1
                            try:
                                class_id = int(line.split()[0])
                                if 0 <= class_id < len(classes):
                                    class_name = classes[class_id]
                                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            except (ValueError, IndexError):
                                continue

            stats["total_labels"] = total_labels
            stats["class_counts"] = class_counts

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training stats: {str(e)}")

@app.get(
    "/training/results",
    tags=["training"],
    summary="Get Training Results",
    description="Get results from the most recent training run",
    response_model=Dict[str, Any]
)
async def get_training_results():
    """Get results from the most recent training run"""
    try:
        # Try multiple possible paths for runs directory
        possible_paths = [
            Path("runs/detect"),  # From backend directory
            Path("../runs/detect"),  # From backend/src directory
            Path("../../runs/detect"),  # From backend/src directory (alternative)
        ]

        runs_dir = None
        for path in possible_paths:
            print(f"🔍 Trying path: {path}")
            if path.exists():
                runs_dir = path
                print(f"✅ Found runs directory at: {path}")
                break

        if runs_dir is None:
            print(f"❌ No runs directory found in any of the possible paths")
            return {
                "message": "No training runs found",
                "training_runs": []
            }

        print(f"🔍 Current working directory: {Path.cwd()}")
        print(f"🔍 Absolute path to runs: {runs_dir.absolute()}")

        # Find all training runs
        train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
        print(f"🔍 Found training directories: {[d.name for d in train_dirs]}")

        if not train_dirs:
            print("❌ No training directories found")
            return {
                "message": "No training runs found",
                "training_runs": []
            }

        # Sort by modification time (most recent first)
        train_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        training_runs = []

        for train_dir in train_dirs:
            print(f"🔍 Processing training directory: {train_dir.name}")
            run_info = {
                "name": train_dir.name,
                "path": str(train_dir),
                "modified": train_dir.stat().st_mtime,
                "modified_date": datetime.fromtimestamp(train_dir.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                "has_weights": False,
                "has_results": False,
                "has_plots": False,
                "metrics": None,
                "best_model_size": 0,
                "last_model_size": 0,
                "plot_files": [],
                "training_args": None
            }

            # Check for weights
            weights_dir = train_dir / "weights"
            if weights_dir.exists():
                best_model = weights_dir / "best.pt"
                last_model = weights_dir / "last.pt"
                run_info["has_weights"] = True
                run_info["best_model_size"] = best_model.stat().st_size / (1024*1024) if best_model.exists() else 0
                run_info["last_model_size"] = last_model.stat().st_size / (1024*1024) if last_model.exists() else 0
                print(f"  ✅ Weights found: best={run_info['best_model_size']:.2f}MB, last={run_info['last_model_size']:.2f}MB")

            # Check for results CSV
            results_csv = train_dir / "results.csv"
            if results_csv.exists():
                run_info["has_results"] = True
                print(f"  ✅ Results CSV found: {results_csv}")
                # Read and parse results CSV
                try:
                    import pandas as pd
                    df = pd.read_csv(results_csv)
                    if not df.empty:
                        # Get final metrics
                        final_row = df.iloc[-1]
                        run_info["metrics"] = {
                            "final_epoch": int(final_row['epoch']),
                            "final_precision": float(final_row['metrics/precision(B)']),
                            "final_recall": float(final_row['metrics/recall(B)']),
                            "final_map50": float(final_row['metrics/mAP50(B)']),
                            "final_map50_95": float(final_row['metrics/mAP50-95(B)']),
                            "total_epochs": len(df),
                            "training_time": float(final_row['time']) if 'time' in final_row else 0
                        }
                        print(f"  ✅ Metrics parsed: {run_info['metrics']}")
                except Exception as e:
                    print(f"❌ Error parsing results CSV: {e}")
                    run_info["metrics"] = None

            # Check for plot images
            plot_files = list(train_dir.glob("*.png"))
            if plot_files:
                run_info["has_plots"] = True
                run_info["plot_files"] = [f.name for f in plot_files]
                print(f"  ✅ Plot files found: {run_info['plot_files']}")

            # Check for args.yaml
            args_file = train_dir / "args.yaml"
            if args_file.exists():
                try:
                    with open(args_file, 'r') as f:
                        import yaml
                        args = yaml.safe_load(f)
                        run_info["training_args"] = {
                            "epochs": args.get('epochs', 'Unknown'),
                            "batch_size": args.get('batch', 'Unknown'),
                            "image_size": args.get('imgsz', 'Unknown'),
                            "model": args.get('model', 'Unknown')
                        }
                        print(f"  ✅ Training args parsed: {run_info['training_args']}")
                except Exception as e:
                    print(f"❌ Error parsing args.yaml: {e}")
                    run_info["training_args"] = None

            training_runs.append(run_info)
            print(f"  ✅ Added run info for {train_dir.name}")

        print(f"🎯 Total training runs processed: {len(training_runs)}")
        return {
            "message": f"Found {len(training_runs)} training runs",
            "training_runs": training_runs,
            "latest_run": training_runs[0] if training_runs else None
        }

    except Exception as e:
        print(f"❌ Error in get_training_results: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting training results: {str(e)}")

@app.get(
    "/training/results/{run_name}/plots/{plot_name}",
    tags=["training"],
    summary="Get Training Plot Image",
    description="Get a specific plot image from a training run",
    response_class=Response
)
async def get_training_plot(run_name: str, plot_name: str):
    """Get a specific plot image from a training run"""
    try:
        # Validate run name for security
        if not run_name.startswith('train'):
            raise HTTPException(status_code=400, detail="Invalid run name")

        # Validate plot name for security
        allowed_plots = ['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png',
                        'R_curve.png', 'P_curve.png', 'F1_curve.png', 'PR_curve.png']
        if plot_name not in allowed_plots:
            raise HTTPException(status_code=400, detail="Invalid plot name")

        # Try multiple possible paths for the plot
        possible_paths = [
            Path(f"runs/detect/{run_name}/{plot_name}"),  # From backend directory
            Path(f"../runs/detect/{run_name}/{plot_name}"),  # From backend/src directory
            Path(f"../../runs/detect/{run_name}/{plot_name}"),  # From backend/src directory (alternative)
        ]

        plot_path = None
        for path in possible_paths:
            print(f"🔍 Trying plot path: {path}")
            if path.exists():
                plot_path = path
                print(f"✅ Found plot at: {path}")
                break

        if plot_path is None:
            print(f"❌ Plot not found in any of the possible paths")
            raise HTTPException(status_code=404, detail="Plot not found")

        print(f"🔍 Absolute path to plot: {plot_path.absolute()}")

        # Read and return the image
        with open(plot_path, "rb") as f:
            image_data = f.read()

        # Determine content type based on file extension
        content_type = "image/png" if plot_name.endswith('.png') else "image/jpeg"

        return Response(content=image_data, media_type=content_type)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting plot: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting plot: {str(e)}")


@app.post(
    "/model/reload",
    tags=["model"],
    summary="Reload Model",
    description="Manually reload the current model or load a new model from a specified path",
    response_model=MessageResponse
)
async def reload_model(model_path: str = None):
    """Reload the model manually"""
    try:
        if model_path:
            # 新しいモデルパスが指定された場合
            print(f"Manual model reload requested for new model: {model_path}")
            yolo.reload_model(model_path)
            message = f"Model reloaded successfully from new path: {model_path}"
        else:
            # 現在のモデルを再読み込み
            print("Manual model reload requested for current model")
            yolo.reload_model()
            message = "Current model reloaded successfully"

        print(f"Model reload completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return MessageResponse(message=message)

    except Exception as e:
        print(f"Error during model reload: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


@app.get(
    "/model/status",
    tags=["model"],
    summary="Get Model Status",
    description="Get current model information and status",
    response_model=Dict[str, Any]
)
async def get_model_status():
    """Get current model status and information"""
    try:
        model_info = yolo.get_model_info()
        return {
            "status": "active",
            "model_info": model_info,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model status: {str(e)}")


@app.get(
    "/model/monitor/status",
    tags=["model"],
    summary="Get Model Monitor Status",
    description="Get the status of the automatic model monitoring system",
    response_model=Dict[str, Any]
)
async def get_model_monitor_status():
    """Get model monitoring system status"""
    try:
        if model_monitor is None:
            return {
                "status": "inactive",
                "message": "Model monitoring is not available",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        monitor_status = model_monitor.get_status()
        return {
            "status": "active" if monitor_status["is_monitoring"] else "inactive",
            "monitor_status": monitor_status,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting monitor status: {str(e)}")


@app.post(
    "/model/monitor/start",
    tags=["model"],
    summary="Start Model Monitoring",
    description="Start the automatic model monitoring system",
    response_model=MessageResponse
)
async def start_model_monitor():
    """Start model monitoring"""
    try:
        global model_monitor

        if model_monitor is None:
            model_monitor = start_model_monitoring(yolo)
            message = "Model monitoring started successfully"
        elif not model_monitor.get_status()["is_monitoring"]:
            model_monitor.start_monitoring()
            message = "Model monitoring restarted successfully"
        else:
            message = "Model monitoring is already active"

        print(f"Model monitoring status: {message}")
        return MessageResponse(message=message)

    except Exception as e:
        print(f"Error starting model monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model monitoring: {str(e)}")


@app.post(
    "/model/monitor/stop",
    tags=["model"],
    summary="Stop Model Monitoring",
    description="Stop the automatic model monitoring system",
    response_model=MessageResponse
)
async def stop_model_monitor():
    """Stop model monitoring"""
    try:
        global model_monitor

        if model_monitor is None:
            message = "Model monitoring is not available"
        elif model_monitor.get_status()["is_monitoring"]:
            model_monitor.stop_monitoring()
            message = "Model monitoring stopped successfully"
        else:
            message = "Model monitoring is already stopped"

        print(f"Model monitoring status: {message}")
        return MessageResponse(message=message)

    except Exception as e:
        print(f"Error stopping model monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop model monitoring: {str(e)}")


@app.post(
    "/labeling/submit",
    tags=["labeling"],
    summary="Submit Labeling Data",
    description="""
    Submit manually labeled image data for model training.

    This endpoint accepts an image and its corresponding labeling data (bounding boxes with labels)
    and saves them in YOLO training format for future model fine-tuning.

    **Data Format:**
    - Image: Any standard image format (JPEG, PNG, etc.)
    - Labeling data: JSON with bounding boxes and labels
    - Coordinates should be in absolute pixel values

    **Training Data Storage:**
    - Images are saved in `training_data/images/`
    - Labels are saved in `training_data/labels/` in YOLO format
    - Class mappings are maintained in `training_data/classes.txt`
    """,
    response_model=LabelingResponse
)
async def submit_labeling_data(
    image: UploadFile,
    labeling_data: str = Form(...)
):
    """Submit manually labeled data for training"""
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Parse labeling data
        try:
            labeling_json = json.loads(labeling_data)
            labeling_obj = LabelingData(**labeling_json)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid labeling data format: {str(e)}")

        # Read image
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file_path = temp_file.name

        try:
            # Save labeling data
            image_path, label_path = save_labeling_data(
                temp_file_path,
                labeling_obj,
                image.filename or "labeled_image.jpg"
            )

            # Create training configuration
            config_path = create_training_config()

            # Count total labels
            total_labels = count_total_labels()

            # Add new classes to the model if they don't exist
            new_classes = [box['label'] for box in labeling_obj.boxes]
            unique_classes = list(set(new_classes))

            try:
                yolo.add_classes(unique_classes)
            except Exception as e:
                print(f"Warning: Could not add classes to model: {e}")

            return LabelingResponse(
                message=f"Labeling data saved successfully. Added {len(labeling_obj.boxes)} labels.",
                saved_path=image_path,
                total_labels=total_labels
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing labeling data: {str(e)}")

