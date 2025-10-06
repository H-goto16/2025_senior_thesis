from fastapi import FastAPI, UploadFile, HTTPException, Form, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from yolo.object_detection import YoloDetector
from training.training_manager import TrainingManager
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
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

yolo = YoloDetector()
training_manager = TrainingManager()

# Training data directory
TRAINING_DATA_DIR = Path("training_data")
TRAINING_DATA_DIR.mkdir(exist_ok=True)

# Global training status
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "status_message": "Ready",
    "progress": 0.0
}

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

class TrainingStatusResponse(BaseModel):
    """Training status response"""
    is_training: bool = Field(
        ...,
        description="Whether training is currently in progress"
    )
    current_epoch: int = Field(
        ...,
        description="Current training epoch"
    )
    total_epochs: int = Field(
        ...,
        description="Total number of epochs"
    )
    status_message: str = Field(
        ...,
        description="Current status message"
    )
    progress: float = Field(
        ...,
        description="Training progress (0.0 to 1.0)"
    )

class TrainingHistoryResponse(BaseModel):
    """Training history response"""
    history: List[Dict[str, Any]] = Field(
        ...,
        description="List of training sessions"
    )

class TrainingMetricsResponse(BaseModel):
    """Training metrics response"""
    metrics: Optional[Dict[str, Any]] = Field(
        ...,
        description="Training metrics data"
    )
    plots: Dict[str, str] = Field(
        ...,
        description="Plotly JSON plots"
    )

class ModelListResponse(BaseModel):
    """Available models response"""
    models: List[Dict[str, Any]] = Field(
        ...,
        description="List of available models"
    )

class DatasetAnalysisResponse(BaseModel):
    """Dataset analysis response"""
    analysis: Dict[str, Any] = Field(
        ...,
        description="Dataset analysis results"
    )


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    description="Retrieve current model configuration, loaded classes, and status",
    response_model=ModelInfoResponse
)
async def get_model_info():
    """Get current model configuration and status"""
    try:
        current_classes = yolo.get_current_classes()
        return ModelInfoResponse(
            model_path=yolo.model_path,
            vocab_file=str(yolo.vocab_file),
            current_classes=current_classes,
            total_classes=len(current_classes)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

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
    confidence: float = Query(0.25)
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

@app.post(
    "/training/start",
    tags=["training"],
    summary="Start Model Fine-tuning",
    description="""
    Start fine-tuning the YOLO model with the collected labeling data.

    **Requirements:**
    - At least some labeled training data must be available
    - Training data should be in proper YOLO format

    **Process:**
    1. Validates training data availability
    2. Creates training configuration
    3. Starts model fine-tuning process
    4. Returns training status

    **Note:** This is a long-running process. In production, this should be implemented as an async task.
    """,
    response_model=MessageResponse
)
async def start_model_training(epochs: int = 50):
    """Start model fine-tuning with collected labeling data"""
    try:
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

        # Start actual training
        try:
            print(f"Starting model fine-tuning with {len(image_files)} images and {epochs} epochs...")
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
                        # Load the fine-tuned model
                        yolo.load_trained_model(str(best_model_path))
                        print(f"Fine-tuned model loaded from: {best_model_path}")

            total_labels = count_total_labels()

            return MessageResponse(
                message=f"Model fine-tuning completed successfully! Trained on {len(image_files)} images with {total_labels} labels for {epochs} epochs."
            )

        except Exception as e:
            print(f"Error during training: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

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

# New advanced training and visualization endpoints

@app.get(
    "/training/status",
    tags=["training"],
    summary="Get Training Status",
    description="Get current training status and progress",
    response_model=TrainingStatusResponse
)
async def get_training_status():
    """Get current training status"""
    return TrainingStatusResponse(**training_status)

@app.get(
    "/training/history",
    tags=["visualization"],
    summary="Get Training History",
    description="Get history of all training sessions with metrics",
    response_model=TrainingHistoryResponse
)
async def get_training_history():
    """Get training history"""
    try:
        history = training_manager.get_training_history()
        return TrainingHistoryResponse(history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training history: {str(e)}")

@app.get(
    "/training/metrics/{run_name}",
    tags=["visualization"],
    summary="Get Training Metrics",
    description="Get detailed training metrics and plots for a specific run",
    response_model=TrainingMetricsResponse
)
async def get_training_metrics(run_name: str):
    """Get detailed training metrics for a specific run"""
    try:
        metrics = training_manager.get_training_metrics(run_name)
        plots = training_manager.generate_training_plots(run_name)

        return TrainingMetricsResponse(
            metrics=metrics,
            plots=plots
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training metrics: {str(e)}")

@app.get(
    "/models/available",
    tags=["models"],
    summary="Get Available Models",
    description="Get list of all available trained models",
    response_model=ModelListResponse
)
async def get_available_models():
    """Get list of available models"""
    try:
        models = yolo.get_available_models()
        return ModelListResponse(models=models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting available models: {str(e)}")

@app.post(
    "/models/load/{model_path:path}",
    tags=["models"],
    summary="Load Model",
    description="Load a specific trained model",
    response_model=MessageResponse
)
async def load_model(model_path: str):
    """Load a specific model"""
    try:
        # Validate model path exists
        if not Path(model_path).exists():
            raise HTTPException(status_code=404, detail="Model file not found")

        yolo.load_trained_model(model_path)

        return MessageResponse(
            message=f"Model loaded successfully from: {model_path}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.post(
    "/models/backup",
    tags=["models"],
    summary="Backup Current Model",
    description="Create a backup of the currently loaded model",
    response_model=MessageResponse
)
async def backup_current_model():
    """Backup the current model"""
    try:
        backup_path = yolo.backup_current_model()
        return MessageResponse(
            message=f"Model backed up successfully to: {backup_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error backing up model: {str(e)}")

@app.get(
    "/models/validate",
    tags=["models"],
    summary="Validate Model Performance",
    description="Validate current model performance",
    response_model=Dict[str, Any]
)
async def validate_model_performance(test_data_path: Optional[str] = None):
    """Validate model performance"""
    try:
        results = yolo.validate_model_performance(test_data_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating model: {str(e)}")

@app.get(
    "/models/comparison",
    tags=["visualization"],
    summary="Compare Models",
    description="Get comparison of all trained models",
    response_model=Dict[str, Any]
)
async def get_model_comparison():
    """Get model comparison data"""
    try:
        comparison = training_manager.get_model_comparison()
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model comparison: {str(e)}")

@app.get(
    "/data/analysis",
    tags=["data"],
    summary="Analyze Dataset",
    description="Get comprehensive analysis of the training dataset",
    response_model=DatasetAnalysisResponse
)
async def get_dataset_analysis():
    """Get dataset analysis"""
    try:
        analysis = training_manager.get_dataset_analysis()
        return DatasetAnalysisResponse(analysis=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")

@app.get(
    "/data/export",
    tags=["data"],
    summary="Export Training Data",
    description="Export all training data as a downloadable archive",
    response_class=FileResponse
)
async def export_training_data(format: str = "zip"):
    """Export training data"""
    try:
        if format not in ["zip"]:
            raise HTTPException(status_code=400, detail="Unsupported export format")

        export_path = training_manager.export_training_data(format)

        if not Path(export_path).exists():
            raise HTTPException(status_code=500, detail="Failed to create export file")

        return FileResponse(
            path=export_path,
            filename=Path(export_path).name,
            media_type='application/zip'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

@app.delete(
    "/training/cleanup",
    tags=["training"],
    summary="Cleanup Old Training Runs",
    description="Clean up old training runs, keeping only the latest N",
    response_model=MessageResponse
)
async def cleanup_training_runs(keep_latest: int = 5):
    """Clean up old training runs"""
    try:
        if keep_latest < 1:
            raise HTTPException(status_code=400, detail="keep_latest must be at least 1")

        training_manager.cleanup_old_runs(keep_latest)

        return MessageResponse(
            message=f"Cleanup completed. Kept latest {keep_latest} training runs."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")

def run_training_background(data_config_path: str, epochs: int):
    """Background training function"""
    global training_status

    try:
        training_status.update({
            "is_training": True,
            "current_epoch": 0,
            "total_epochs": epochs,
            "status_message": "Starting training...",
            "progress": 0.0
        })

        # Start training
        results = yolo.fine_tune_model(data_config_path, epochs=epochs)

        # Load the best model after training
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('train')]
            if train_dirs:
                latest_run = max(train_dirs, key=lambda x: x.stat().st_mtime)
                best_model_path = latest_run / "weights" / "best.pt"

                if best_model_path.exists():
                    yolo.load_trained_model(str(best_model_path))

        training_status.update({
            "is_training": False,
            "current_epoch": epochs,
            "total_epochs": epochs,
            "status_message": "Training completed successfully!",
            "progress": 1.0
        })

    except Exception as e:
        training_status.update({
            "is_training": False,
            "status_message": f"Training failed: {str(e)}",
            "progress": 0.0
        })

@app.post(
    "/training/start-async",
    tags=["training"],
    summary="Start Asynchronous Training",
    description="Start model training in the background",
    response_model=MessageResponse
)
async def start_async_training(background_tasks: BackgroundTasks, epochs: int = 50):
    """Start training in background"""
    try:
        if training_status["is_training"]:
            raise HTTPException(status_code=400, detail="Training is already in progress")

        # Validate training data
        images_dir = TRAINING_DATA_DIR / "images"
        labels_dir = TRAINING_DATA_DIR / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            raise HTTPException(
                status_code=400,
                detail="No training data available. Please submit some labeled data first."
            )

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
                detail="Could not create training configuration."
            )

        # Start training in background
        background_tasks.add_task(run_training_background, config_path, epochs)

        return MessageResponse(
            message=f"Training started in background with {epochs} epochs. Use /training/status to monitor progress."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting training: {str(e)}")

