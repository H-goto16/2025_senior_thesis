import Constants from 'expo-constants';

const API_BASE_URL = Constants.expoConfig?.extra?.apiUrl || 'http://localhost:8000';

export interface DetectionResult {
  class: string;
  confidence: number;
  bbox: number[];
}

export interface TrainingStatus {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  status_message: string;
  progress: number;
}

export interface TrainingHistory {
  run_name: string;
  timestamp: string;
  epochs: number;
  best_model_path?: string;
  last_model_path?: string;
  final_metrics: {
    train_loss: number;
    val_loss: number;
    map50: number;
    map50_95: number;
  };
  args: any;
  results_path: string;
}

export interface TrainingMetrics {
  epochs: number[];
  train_box_loss: number[];
  train_cls_loss: number[];
  val_box_loss: number[];
  val_cls_loss: number[];
  map50: number[];
  map50_95: number[];
  precision: number[];
  recall: number[];
}

export interface ModelInfo {
  run_name: string;
  timestamp: string;
  best_model?: string;
  last_model?: string;
  args: any;
}

export interface DatasetAnalysis {
  total_images: number;
  total_annotations: number;
  class_distribution: Record<string, number>;
  image_sizes: Array<{ width: number; height: number }>;
  annotation_stats: {
    avg_bbox_area: number;
    min_bbox_area: number;
    max_bbox_area: number;
  };
  class_distribution_plot?: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  private async uploadFile<T>(
    endpoint: string,
    file: File | Blob,
    additionalData?: Record<string, string>
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const formData = new FormData();
    formData.append('image', file);

    if (additionalData) {
      Object.entries(additionalData).forEach(([key, value]) => {
        formData.append(key, value);
      });
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`File upload failed: ${url}`, error);
      throw error;
    }
  }

  // Model management
  async getModelInfo() {
    return this.request('/model/info');
  }

  async getClasses() {
    return this.request('/model/classes');
  }

  async addClasses(classes: string[]) {
    return this.request('/model/classes', {
      method: 'POST',
      body: JSON.stringify({ classes }),
    });
  }

  async clearClasses() {
    return this.request('/model/classes', {
      method: 'DELETE',
    });
  }

  // Detection
  async detectObjects(file: File | Blob) {
    return this.uploadFile<{
      detections: DetectionResult[];
      message: string;
      processed_image: string;
    }>('/detect', file);
  }

  async detectObjectsWithConfidence(file: File | Blob, confidence: number) {
    return this.uploadFile<{
      detections: DetectionResult[];
      message: string;
    }>('/detect/with-confidence', file, { confidence: confidence.toString() });
  }

  // Labeling
  async submitLabelingData(file: File | Blob, labelingData: any) {
    return this.uploadFile<{
      message: string;
      saved_path: string;
      total_labels: number;
    }>('/labeling/submit', file, { labeling_data: JSON.stringify(labelingData) });
  }

  // Training
  async startTraining(epochs: number = 50) {
    return this.request<{ message: string }>('/training/start', {
      method: 'POST',
      body: JSON.stringify({ epochs }),
    });
  }

  async startAsyncTraining(epochs: number = 50) {
    return this.request<{ message: string }>('/training/start-async', {
      method: 'POST',
      body: JSON.stringify({ epochs }),
    });
  }

  async getTrainingStatus(): Promise<TrainingStatus> {
    return this.request('/training/status');
  }

  async getTrainingHistory(): Promise<{ history: TrainingHistory[] }> {
    return this.request('/training/history');
  }

  async getTrainingMetrics(runName: string): Promise<{
    metrics: TrainingMetrics | null;
    plots: Record<string, string>;
  }> {
    return this.request(`/training/metrics/${encodeURIComponent(runName)}`);
  }

  async getTrainingDataStats() {
    return this.request('/training/data/stats');
  }

  async cleanupTrainingRuns(keepLatest: number = 5) {
    return this.request(`/training/cleanup?keep_latest=${keepLatest}`, {
      method: 'DELETE',
    });
  }

  // Models
  async getAvailableModels(): Promise<{ models: ModelInfo[] }> {
    return this.request('/models/available');
  }

  async loadModel(modelPath: string) {
    return this.request(`/models/load/${encodeURIComponent(modelPath)}`, {
      method: 'POST',
    });
  }

  async backupCurrentModel() {
    return this.request('/models/backup', {
      method: 'POST',
    });
  }

  async validateModel(testDataPath?: string) {
    const params = testDataPath ? `?test_data_path=${encodeURIComponent(testDataPath)}` : '';
    return this.request(`/models/validate${params}`);
  }

  async getModelComparison() {
    return this.request('/models/comparison');
  }

  // Data management
  async getDatasetAnalysis(): Promise<{ analysis: DatasetAnalysis }> {
    return this.request('/data/analysis');
  }

  async exportTrainingData(format: string = 'zip') {
    const response = await fetch(`${this.baseUrl}/data/export?format=${format}`);

    if (!response.ok) {
      throw new Error(`Export failed: ${response.statusText}`);
    }

    return response.blob();
  }

  // Model management
  async getAvailableModels() {
    return this.request('/models/available');
  }

  async loadModel(modelPath: string) {
    return this.request(`/models/load/${encodeURIComponent(modelPath)}`, {
      method: 'POST',
    });
  }

  async backupModel() {
    return this.request('/models/backup', {
      method: 'POST',
    });
  }

  async validateModel(testDataPath?: string) {
    const params = testDataPath ? `?test_data_path=${encodeURIComponent(testDataPath)}` : '';
    return this.request(`/models/validate${params}`);
  }

  async getModelComparison() {
    return this.request('/models/comparison');
  }

  // Advanced training
  async startAsyncTraining(epochs: number = 50) {
    return this.request(`/training/start-async?epochs=${epochs}`, {
      method: 'POST',
    });
  }

  async cleanupTrainingRuns(keepLatest: number = 5) {
    return this.request(`/training/cleanup?keep_latest=${keepLatest}`, {
      method: 'DELETE',
    });
  }

  // Health check
  async healthCheck() {
    return this.request('/');
  }
}

export const apiClient = new ApiClient();
export default apiClient;

