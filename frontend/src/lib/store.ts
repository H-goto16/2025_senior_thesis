import { create } from 'zustand';
import { DatasetAnalysis, ModelInfo, TrainingHistory, TrainingStatus } from './api';

interface AppState {
  // Theme
  isDarkMode: boolean;
  toggleDarkMode: () => void;

  // Training
  trainingStatus: TrainingStatus | null;
  trainingHistory: TrainingHistory[];
  setTrainingStatus: (status: TrainingStatus) => void;
  setTrainingHistory: (history: TrainingHistory[]) => void;

  // Models
  availableModels: ModelInfo[];
  currentModel: string | null;
  setAvailableModels: (models: ModelInfo[]) => void;
  setCurrentModel: (model: string | null) => void;

  // Dataset
  datasetAnalysis: DatasetAnalysis | null;
  setDatasetAnalysis: (analysis: DatasetAnalysis) => void;

  // UI State
  selectedTab: string;
  setSelectedTab: (tab: string) => void;

  // Loading states
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // Error handling
  error: string | null;
  setError: (error: string | null) => void;
  clearError: () => void;

  // Detection results
  lastDetectionResults: any[];
  setLastDetectionResults: (results: any[]) => void;

  // Labeling state
  labelingMode: boolean;
  setLabelingMode: (mode: boolean) => void;
  currentImage: string | null;
  setCurrentImage: (image: string | null) => void;
  labels: any[];
  setLabels: (labels: any[]) => void;
  addLabel: (label: any) => void;
  removeLabel: (index: number) => void;
}

export const useAppStore = create<AppState>((set, get) => ({
  // Theme
  isDarkMode: true, // Default to dark mode
  toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),

  // Training
  trainingStatus: null,
  trainingHistory: [],
  setTrainingStatus: (status) => set({ trainingStatus: status }),
  setTrainingHistory: (history) => set({ trainingHistory: history }),

  // Models
  availableModels: [],
  currentModel: null,
  setAvailableModels: (models) => set({ availableModels: models }),
  setCurrentModel: (model) => set({ currentModel: model }),

  // Dataset
  datasetAnalysis: null,
  setDatasetAnalysis: (analysis) => set({ datasetAnalysis: analysis }),

  // UI State
  selectedTab: 'detection',
  setSelectedTab: (tab) => set({ selectedTab: tab }),

  // Loading states
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  // Error handling
  error: null,
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),

  // Detection results
  lastDetectionResults: [],
  setLastDetectionResults: (results) => set({ lastDetectionResults: results }),

  // Labeling state
  labelingMode: false,
  setLabelingMode: (mode) => set({ labelingMode: mode }),
  currentImage: null,
  setCurrentImage: (image) => set({ currentImage: image }),
  labels: [],
  setLabels: (labels) => set({ labels }),
  addLabel: (label) => set((state) => ({ labels: [...state.labels, label] })),
  removeLabel: (index) => set((state) => ({
    labels: state.labels.filter((_, i) => i !== index)
  })),
}));

// Selectors for commonly used state combinations
export const useTheme = () => {
  const isDarkMode = useAppStore((state) => state.isDarkMode);
  const toggleDarkMode = useAppStore((state) => state.toggleDarkMode);
  return { isDarkMode, toggleDarkMode };
};

export const useTraining = () => {
  const trainingStatus = useAppStore((state) => state.trainingStatus);
  const trainingHistory = useAppStore((state) => state.trainingHistory);
  const setTrainingStatus = useAppStore((state) => state.setTrainingStatus);
  const setTrainingHistory = useAppStore((state) => state.setTrainingHistory);

  return {
    trainingStatus,
    trainingHistory,
    setTrainingStatus,
    setTrainingHistory,
  };
};

export const useModels = () => {
  const availableModels = useAppStore((state) => state.availableModels);
  const currentModel = useAppStore((state) => state.currentModel);
  const setAvailableModels = useAppStore((state) => state.setAvailableModels);
  const setCurrentModel = useAppStore((state) => state.setCurrentModel);

  return {
    availableModels,
    currentModel,
    setAvailableModels,
    setCurrentModel,
  };
};

export const useLabeling = () => {
  const labelingMode = useAppStore((state) => state.labelingMode);
  const currentImage = useAppStore((state) => state.currentImage);
  const labels = useAppStore((state) => state.labels);
  const setLabelingMode = useAppStore((state) => state.setLabelingMode);
  const setCurrentImage = useAppStore((state) => state.setCurrentImage);
  const setLabels = useAppStore((state) => state.setLabels);
  const addLabel = useAppStore((state) => state.addLabel);
  const removeLabel = useAppStore((state) => state.removeLabel);

  return {
    labelingMode,
    currentImage,
    labels,
    setLabelingMode,
    setCurrentImage,
    setLabels,
    addLabel,
    removeLabel,
  };
};

export const useUI = () => {
  const selectedTab = useAppStore((state) => state.selectedTab);
  const isLoading = useAppStore((state) => state.isLoading);
  const error = useAppStore((state) => state.error);
  const setSelectedTab = useAppStore((state) => state.setSelectedTab);
  const setIsLoading = useAppStore((state) => state.setIsLoading);
  const setError = useAppStore((state) => state.setError);
  const clearError = useAppStore((state) => state.clearError);

  return {
    selectedTab,
    isLoading,
    error,
    setSelectedTab,
    setIsLoading,
    setError,
    clearError,
  };
};





