# YOLO-World Object Detection App

This is a full-stack application for real-time object detection using YOLO-World model with React Native frontend and FastAPI backend.

## Features

### 🎯 Object Detection
- Real-time object detection using YOLO-World model
- Camera integration for live photo capture
- Gallery image selection support
- Configurable detection classes
- Confidence threshold adjustment

### 🏷️ Manual Labeling System
- Interactive image labeling interface
- Drag-and-drop bounding box creation
- Support for existing and new class labels
- YOLO format data storage for training

### 🤖 Model Fine-tuning
- Automatic model improvement with labeled data
- Training data management and statistics
- Custom class integration
- Model retraining capabilities

### 🔄 Automatic Model Management
- **Auto-reload after fine-tuning**: Models are automatically reloaded after training completion
- **File change monitoring**: Automatic detection of model file changes
- **Real-time model updates**: Seamless model switching without restart
- **Comprehensive logging**: Detailed logs for all model operations

## Architecture

- **Frontend**: React Native with Expo
- **Backend**: FastAPI with YOLO-World
- **Training**: YOLOv8 fine-tuning pipeline
- **Monitoring**: Real-time model file change detection

## Getting Started

### Prerequisites
- Node.js 18+
- Python 3.8+
- pnpm (recommended) or npm

### Quick Setup
```bash
make setup
```

### Development
```bash
# Start both frontend and backend
make dev

# Or start individually
make server    # Backend only
make frontend  # Frontend only
```

## New Features Usage

### Manual Labeling
1. Capture or select an image
2. Click "Manual Labeling" button
3. Draw bounding boxes by dragging on the image
4. Assign labels to each bounding box
5. Submit the labeled data

### Model Training
```bash
# Start model fine-tuning
make train-model

# View training statistics
make view-training-stats

# Clean training data
make clean-training-data
```

### Model Management
```bash
# Check model status
curl http://localhost:8000/model/status

# Check monitoring status
curl http://localhost:8000/model/monitor/status

# Manually reload model
curl -X POST http://localhost:8000/model/reload

# Start/stop monitoring
curl -X POST http://localhost:8000/model/monitor/start
curl -X POST http://localhost:8000/model/monitor/stop
```

### API Endpoints

#### Labeling
- `POST /labeling/submit` - Submit labeled training data
- `GET /training/data/stats` - Get training dataset statistics
- `POST /training/start` - Start model fine-tuning

#### Model Management (New)
- `GET /model/status` - Get current model information
- `POST /model/reload` - Manually reload the model
- `GET /model/monitor/status` - Get monitoring system status
- `POST /model/monitor/start` - Start automatic monitoring
- `POST /model/monitor/stop` - Stop automatic monitoring

#### Detection (Existing)
- `GET /model/classes` - Get current detection classes
- `POST /model/classes` - Add new detection classes
- `POST /detect` - Detect objects in image
- `POST /detect/with-confidence` - Detect with custom confidence

## Training Data Format

The application stores training data in YOLO format:
- `training_data/images/` - Training images
- `training_data/labels/` - YOLO format annotations
- `training_data/classes.txt` - Class name mappings
- `training_data/data.yaml` - Training configuration

## Model Monitoring

The application automatically monitors model files for changes:
- **Automatic detection**: Detects when model files are modified
- **Instant reload**: Automatically reloads models when changes are detected
- **Comprehensive logging**: All operations are logged to `yolo_detector.log` and `model_monitor.log`
- **Manual control**: Start/stop monitoring via API endpoints

## Workflow

1. **Initial Detection**: Try automatic detection with existing classes
2. **Add Classes**: If objects aren't detected, add new classes
3. **Manual Labeling**: For persistent detection issues, create manual labels
4. **Model Training**: Fine-tune the model with labeled data
5. **Automatic Updates**: Models are automatically reloaded after training
6. **Continuous Monitoring**: File changes trigger automatic model updates

## Testing
```bash
make test
```

## API Documentation
Access interactive API docs at: http://localhost:8000/docs
