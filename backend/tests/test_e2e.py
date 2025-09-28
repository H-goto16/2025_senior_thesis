import pytest
import json
import tempfile
import os
import time
import requests
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
from io import BytesIO
from PIL import Image

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import app


class TestE2EWorkflow:
    """End-to-End test class for complete workflow testing"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing"""
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return ("test.jpg", img_bytes, "image/jpeg")

    @pytest.fixture
    def mock_yolo(self):
        """Mock the global yolo instance"""
        with patch('main.yolo') as mock:
            # Mock successful training
            mock.fine_tune_model.return_value = {"success": True}
            mock.load_trained_model.return_value = True
            mock.get_current_classes.return_value = ["person", "car", "bicycle"]
            # Create a mock box object
            mock_box = Mock()
            mock_box.cls = [0]
            mock_box.conf = [0.85]
            mock_box.xyxy = [[100, 50, 200, 300]]

            mock_result = Mock()
            mock_result.boxes = [mock_box]
            mock_result.names = {0: "person"}

            mock.predict_image.return_value = mock_result
            mock.get_available_models.return_value = [
                {"path": "model1.pt", "created": "2024-01-01", "size": "100MB"},
                {"path": "model2.pt", "created": "2024-01-02", "size": "120MB"}
            ]
            mock.backup_current_model.return_value = "/path/to/backup.pt"
            mock.validate_model_performance.return_value = {
                "map50": 0.85,
                "map50_95": 0.65,
                "precision": 0.90,
                "recall": 0.80
            }
            yield mock

    @pytest.fixture
    def mock_training_manager(self):
        """Mock the global training_manager instance"""
        with patch('main.training_manager') as mock:
            mock.get_training_history.return_value = [
                {
                    "run_name": "train1",
                    "timestamp": "2024-01-01T10:00:00",
                    "epochs": 50,
                    "final_metrics": {"map50": 0.85, "map50_95": 0.65}
                }
            ]
            mock.get_training_metrics.return_value = {
                "epochs": [1, 2, 3],
                "train_loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4]
            }
            mock.generate_training_plots.return_value = {
                "loss_plot": {"data": [], "layout": {}},
                "metrics_plot": {"data": [], "layout": {}}
            }
            mock.get_dataset_analysis.return_value = {
                "total_images": 100,
                "total_annotations": 250,
                "class_distribution": {"person": 100, "car": 80, "bicycle": 70},
                "image_sizes": [(640, 480), (800, 600)],
                "annotation_stats": {"avg_per_image": 2.5},
                "class_distribution_plot": None
            }
            mock.get_model_comparison.return_value = {
                "models": [
                    {"name": "model1", "map50": 0.85},
                    {"name": "model2", "map50": 0.82}
                ]
            }
            mock.export_training_data.return_value = "/tmp/training_data.zip"
            mock.cleanup_old_runs.return_value = None
            yield mock

    def test_complete_training_workflow(self, client, mock_yolo, mock_training_manager, sample_image_file):
        """Test complete training workflow from setup to validation"""

        # Step 1: Check API status
        response = client.get("/")
        assert response.status_code == 200
        assert "YOLO-World Object Detection API" in response.json()["message"]

        # Step 2: Add detection classes
        classes_data = {"classes": ["person", "car", "bicycle"]}
        response = client.post("/model/classes", json=classes_data)
        assert response.status_code == 200
        assert "Successfully added classes" in response.json()["message"]

        # Step 3: Get model info
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "current_classes" in data
        assert len(data["current_classes"]) > 0

        # Step 4: Submit labeling data
        labeling_data = {
            "boxes": [
                {"x1": 100, "y1": 50, "x2": 200, "y2": 300, "label": "person"}
            ],
            "image_width": 640,
            "image_height": 480
        }

        filename, file_content, content_type = sample_image_file
        files = {"image": (filename, file_content, content_type)}
        data = {"labeling_data": json.dumps(labeling_data)}

        response = client.post("/labeling/submit", files=files, data=data)
        assert response.status_code == 200
        assert "Labeling data saved successfully" in response.json()["message"]

        # Step 5: Check training data stats
        response = client.get("/training/data/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "total_images" in stats
        assert "total_labels" in stats

        # Step 6: Start async training
        response = client.post("/training/start-async?epochs=10")
        assert response.status_code == 200
        assert "Training started in background" in response.json()["message"]

        # Step 7: Check training status
        response = client.get("/training/status")
        assert response.status_code == 200
        status = response.json()
        assert "is_training" in status
        assert "status_message" in status

        # Step 8: Get training history
        response = client.get("/training/history")
        assert response.status_code == 200
        history = response.json()
        assert "history" in history

        # Step 9: Get dataset analysis
        response = client.get("/data/analysis")
        assert response.status_code == 200
        analysis = response.json()
        assert "analysis" in analysis
        assert "total_images" in analysis["analysis"]

        # Step 10: Get available models
        response = client.get("/models/available")
        assert response.status_code == 200
        models = response.json()
        assert "models" in models

    def test_object_detection_workflow(self, client, mock_yolo, sample_image_file):
        """Test object detection workflow"""

        # Step 1: Add classes
        classes_data = {"classes": ["person", "car"]}
        response = client.post("/model/classes", json=classes_data)
        assert response.status_code == 200

        # Step 2: Perform detection
        filename, file_content, content_type = sample_image_file
        files = {"image": (filename, file_content, content_type)}

        response = client.post("/detect", files=files)
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "message" in data

        # Step 3: Perform detection with custom confidence
        file_content.seek(0)  # Reset file pointer
        files = {"image": (filename, file_content, content_type)}
        data = {"confidence": 0.5}

        response = client.post("/detect/with-confidence", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        assert "detections" in result

    def test_model_management_workflow(self, client, mock_yolo):
        """Test model management workflow"""

        # Step 1: Get available models
        response = client.get("/models/available")
        assert response.status_code == 200
        models = response.json()
        assert "models" in models

        # Step 2: Backup current model
        response = client.post("/models/backup")
        assert response.status_code == 200
        assert "backed up successfully" in response.json()["message"]

        # Step 3: Validate model performance
        response = client.get("/models/validate")
        assert response.status_code == 200
        validation = response.json()
        assert "map50" in validation

        # Step 4: Get model comparison
        response = client.get("/models/comparison")
        assert response.status_code == 200
        comparison = response.json()
        assert "models" in comparison

    def test_data_management_workflow(self, client, mock_training_manager):
        """Test data management workflow"""

        # Step 1: Get dataset analysis
        response = client.get("/data/analysis")
        assert response.status_code == 200
        analysis = response.json()
        assert "analysis" in analysis

        # Step 2: Export training data
        response = client.get("/data/export?format=zip")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"

        # Step 3: Cleanup old training runs
        response = client.delete("/training/cleanup?keep_latest=3")
        assert response.status_code == 200
        assert "Cleanup completed" in response.json()["message"]

    def test_error_handling(self, client):
        """Test error handling scenarios"""

        # Test invalid file upload
        files = {"image": ("test.txt", BytesIO(b"not an image"), "text/plain")}
        response = client.post("/detect", files=files)
        assert response.status_code == 400

        # Test invalid confidence value
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
        data = {"confidence": 1.5}  # Invalid confidence > 1.0

        response = client.post("/detect/with-confidence", files=files, data=data)
        assert response.status_code == 400

        # Test empty classes list
        response = client.post("/model/classes", json={"classes": []})
        assert response.status_code == 400

    def test_api_documentation(self, client):
        """Test API documentation endpoints"""

        # Test OpenAPI JSON
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        assert "openapi" in openapi_spec
        assert "paths" in openapi_spec

        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger-ui" in response.text.lower()

    def test_concurrent_requests(self, client, mock_yolo):
        """Test handling of concurrent requests"""
        import threading
        import time

        # Prepare test data
        classes_data = {"classes": ["person", "car"]}
        results = []

        def make_request():
            response = client.post("/model/classes", json=classes_data)
            results.append(response)

        # Execute multiple concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all requests were handled
        assert len(results) == 5
        for response in results:
            assert response.status_code in [200, 400]  # Either success or validation error

    def test_training_status_updates(self, client, mock_training_manager):
        """Test training status updates"""

        # Initial status should be ready
        response = client.get("/training/status")
        assert response.status_code == 200
        status = response.json()
        assert status["is_training"] == False
        assert status["status_message"] == "Ready"

        # Test training metrics endpoint
        response = client.get("/training/metrics/train1")
        assert response.status_code == 200
        metrics = response.json()
        assert "metrics" in metrics
        assert "plots" in metrics

    def test_comprehensive_api_coverage(self, client, mock_yolo, mock_training_manager):
        """Test comprehensive API endpoint coverage"""

        endpoints_to_test = [
            ("GET", "/"),
            ("GET", "/model/info"),
            ("GET", "/model/classes"),
            ("GET", "/training/status"),
            ("GET", "/training/history"),
            ("GET", "/training/data/stats"),
            ("GET", "/models/available"),
            ("GET", "/data/analysis"),
            ("POST", "/models/backup"),
            ("GET", "/models/validate"),
            ("GET", "/models/comparison"),
            ("DELETE", "/training/cleanup"),
        ]

        for method, endpoint in endpoints_to_test:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint)
            elif method == "DELETE":
                response = client.delete(endpoint)

            # All endpoints should return valid responses (not 404)
            assert response.status_code != 404, f"Endpoint {method} {endpoint} returned 404"
            assert response.status_code < 500, f"Endpoint {method} {endpoint} returned server error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
