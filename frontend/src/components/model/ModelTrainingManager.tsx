import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { CameraButton } from '@/components/ui/CameraButton';
import { PlatformAlert } from '@/components/ui/PlatformAlert';
import { StyledTextInput } from '@/components/ui/StyledTextInput';
import env from '@/env';
import React, { useEffect, useState } from 'react';
import {
  ActivityIndicator,
  Dimensions,
  Image,
  Modal,
  ScrollView,
  TouchableOpacity,
  useColorScheme,
  View
} from 'react-native';

interface TrainingStats {
  total_images: number;
  total_labels: number;
  classes: string[];
  class_counts: Record<string, number>;
  data_directory: string;
}

interface TrainingProgress {
  isTraining: boolean;
  currentEpoch?: number;
  totalEpochs?: number;
  message?: string;
}

interface TrainingMetrics {
  final_epoch: number;
  final_precision: number;
  final_recall: number;
  final_map50: number;
  final_map50_95: number;
  total_epochs: number;
  training_time: number;
}

interface TrainingRun {
  name: string;
  path: string;
  modified: number;
  modified_date: string;
  has_weights: boolean;
  has_results: boolean;
  has_plots: boolean;
  metrics: TrainingMetrics | null;
  best_model_size: number;
  last_model_size: number;
  plot_files: string[];
  training_args: {
    epochs: number | string;
    batch_size: number | string;
    image_size: number | string;
    model: string;
  } | null;
}

interface TrainingResults {
  message: string;
  training_runs: TrainingRun[];
  latest_run: TrainingRun | null;
}

type TabType = 'training' | 'status' | 'results';

const ModelTrainingManager = () => {
  const [activeTab, setActiveTab] = useState<TabType>('training');
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>({ isTraining: false });
  const [trainingResults, setTrainingResults] = useState<TrainingResults | null>(null);
  const [epochs, setEpochs] = useState('50');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageModalVisible, setImageModalVisible] = useState(false);
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  // Fetch training data statistics
  const fetchTrainingStats = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(env?.API_ENDPOINT + "/training/data/stats");
      if (response.ok) {
        const stats = await response.json();
        setTrainingStats(stats);
      } else {
        console.error('Failed to fetch training stats');
      }
    } catch (error) {
      console.error('Error fetching training stats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch training results
  const fetchTrainingResults = async () => {
    try {
      setIsLoadingResults(true);
      const response = await fetch(env?.API_ENDPOINT + "/training/results");
      if (response.ok) {
        const results = await response.json();
        setTrainingResults(results);
      } else {
        console.error('Failed to fetch training results');
      }
    } catch (error) {
      console.error('Error fetching training results:', error);
    } finally {
      setIsLoadingResults(false);
    }
  };

  // Fetch training status
  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch(env?.API_ENDPOINT + "/training/status");
      if (response.ok) {
        const status = await response.json();
        if (status.is_training) {
          setTrainingProgress({
            isTraining: true,
            message: status.progress || 'Training in progress...'
          });
        } else {
          setTrainingProgress({
            isTraining: false,
            message: status.progress || 'Training not started'
          });
        }
      } else {
        console.error('Failed to fetch training status');
      }
    } catch (error) {
      console.error('Error fetching training status:', error);
    }
  };

  // Open image modal
  const openImageModal = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    setImageModalVisible(true);
  };

  // Close image modal
  const closeImageModal = () => {
    setImageModalVisible(false);
    setSelectedImage(null);
  };

  // Start fine-tuning
  const startTraining = async () => {
    console.log('startTraining called with epochs:', epochs);
    const epochCount = parseInt(epochs);
    if (isNaN(epochCount) || epochCount <= 0) {
      console.log('Invalid epoch count:', epochCount);
      PlatformAlert.error('Error', 'Please enter a valid number of epochs');
      return;
    }

    if (epochCount > 500) {
      console.log('Epoch count too high:', epochCount);
      PlatformAlert.error('Error', 'Maximum epochs allowed is 500');
      return;
    }

    if (!trainingStats || trainingStats.total_images === 0) {
      console.log('No training data available:', trainingStats);
      PlatformAlert.error('Error', 'No training data available. Please add some labeled images first.');
      return;
    }

    console.log('All validation passed, showing confirmation dialog');

    PlatformAlert.confirm(
      'Start Training',
      `Are you sure you want to start fine-tuning with ${epochCount} epochs?\n\nThis will train on:\n• ${trainingStats.total_images} images\n• ${trainingStats.total_labels} labels\n• ${trainingStats.classes.length} classes\n\nThis process may take several minutes.`,
      executeTraining
    );
  };

  const executeTraining = async () => {
    try {
      console.log('executeTraining started');
      setTrainingProgress({ isTraining: true, message: 'Starting training...' });

      const url = `${env?.API_ENDPOINT}/training/start?epochs=${epochs}`;
      console.log('Making API request to:', url);

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Training started successfully:', result);
        setTrainingProgress({ isTraining: true, message: result.message });

        // Start polling for training status (less frequent during training)
        const statusInterval = setInterval(async () => {
          await fetchTrainingStatus();
          await fetchTrainingResults();
        }, 10000); // 10 seconds instead of 5

        // Clear interval after 15 minutes (training should be done by then)
        setTimeout(() => {
          clearInterval(statusInterval);
        }, 15 * 60 * 1000);

        PlatformAlert.success('Success', 'Training started successfully! Check the status below.');
      } else {
        const errorData = await response.json();
        console.error('Training failed to start:', errorData);
        setTrainingProgress({ isTraining: false, message: `Failed to start training: ${errorData.detail}` });
        PlatformAlert.error('Training Failed', errorData.detail || 'An error occurred while starting training');
      }
    } catch (error) {
      console.error('Error during training:', error);
      setTrainingProgress({ isTraining: false, message: 'Failed to start training due to network error' });
      PlatformAlert.error('Network Error', 'Failed to connect to the training service');
    }
  };

  // Fetch stats on component mount
  useEffect(() => {
    fetchTrainingStats();
    fetchTrainingResults();
    fetchTrainingStatus();

    // Set up interval to check training status (less frequent updates)
    const statusInterval = setInterval(fetchTrainingStatus, 5000); // 5 seconds instead of 2

    return () => clearInterval(statusInterval);
  }, []);

  const getTrainingStatusColor = () => {
    if (!trainingStats) return '#6B7280';
    if (trainingStats.total_images === 0) return '#EF4444';
    if (trainingStats.total_images < 10) return '#F59E0B';
    return '#10B981';
  };

  const getTrainingStatusText = () => {
    if (!trainingStats) return 'Loading...';
    if (trainingStats.total_images === 0) return 'No training data';
    if (trainingStats.total_images < 10) return 'Insufficient data';
    return 'Ready for training';
  };

  // Tab component
  const TabButton = ({ tab, label, isActive }: { tab: TabType; label: string; isActive: boolean }) => (
    <TouchableOpacity
      onPress={() => setActiveTab(tab)}
      className={`flex-1 py-3 px-4 rounded-t-lg border-b-2 ${
        isActive
          ? 'bg-blue-500 border-blue-600'
          : 'bg-gray-200 dark:bg-gray-700 border-gray-300 dark:border-gray-600'
      }`}
      activeOpacity={0.7}
    >
      <ThemedText
        className={`text-center font-semibold ${
          isActive ? 'text-white' : 'text-gray-700 dark:text-gray-300'
        }`}
      >
        {label}
      </ThemedText>
    </TouchableOpacity>
  );

  // Training Tab Content
  const TrainingTab = () => (
    <ScrollView className="flex-1" showsVerticalScrollIndicator={false}>
      <ThemedView className="p-4">
        <ThemedText className="text-xl font-bold mb-4 text-center">
          Start Training
        </ThemedText>

        {/* Training Data Statistics */}
        <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
          <ThemedText className="text-lg font-semibold mb-3">Training Data Statistics</ThemedText>

          {isLoading ? (
            <View className="items-center py-4">
              <ActivityIndicator size="large" color={isDark ? "#fff" : "#000"} />
              <ThemedText className="mt-2">Loading statistics...</ThemedText>
            </View>
          ) : trainingStats ? (
            <View>
              <View className="flex-row justify-around mb-4">
                <View className="items-center">
                  <ThemedText className="text-3xl font-bold text-blue-500">{trainingStats.total_images}</ThemedText>
                  <ThemedText className="text-sm text-gray-600 dark:text-gray-400">Images</ThemedText>
                </View>
                <View className="items-center">
                  <ThemedText className="text-3xl font-bold text-green-500">{trainingStats.total_labels}</ThemedText>
                  <ThemedText className="text-sm text-gray-600 dark:text-gray-400">Labels</ThemedText>
                </View>
                <View className="items-center">
                  <ThemedText className="text-3xl font-bold text-purple-500">{trainingStats.classes.length}</ThemedText>
                  <ThemedText className="text-sm text-gray-600 dark:text-gray-400">Classes</ThemedText>
                </View>
              </View>

              {Object.keys(trainingStats.class_counts).length > 0 && (
                <View className="mt-4">
                  <ThemedText className="font-medium mb-2">Class Distribution:</ThemedText>
                  <ScrollView className="max-h-24">
                    {Object.entries(trainingStats.class_counts).map(([className, count]) => (
                      <View key={className} className="flex-row justify-between items-center py-1">
                        <ThemedText className="text-sm">{className}</ThemedText>
                        <View className="bg-blue-100 dark:bg-blue-900 px-2 py-1 rounded-full">
                          <ThemedText className="text-blue-800 dark:text-blue-200 text-xs font-bold">
                            {count}
                          </ThemedText>
                        </View>
                      </View>
                    ))}
                  </ScrollView>
                </View>
              )}
            </View>
          ) : (
            <ThemedText className="text-gray-500 dark:text-gray-400 text-center py-4">
              No training data available
            </ThemedText>
          )}

          <View className="mt-3">
            <CameraButton
              onPress={fetchTrainingStats}
              variant="success"
            >
              Refresh Stats
            </CameraButton>
          </View>
        </ThemedView>

        {/* Training Configuration */}
        <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
          <ThemedText className="text-lg font-semibold mb-3">Training Configuration</ThemedText>

          <View className="mb-4">
            <ThemedText className="mb-2">Number of Epochs:</ThemedText>
            <StyledTextInput
              value={epochs}
              onChangeText={setEpochs}
              placeholder="50"
            />
            <ThemedText className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Recommended: 50-100 epochs for fine-tuning
            </ThemedText>
          </View>

          <CameraButton
            onPress={startTraining}
            variant="purple"
            disabled={trainingProgress.isTraining}
          >
            {trainingProgress.isTraining ? 'Training in Progress...' : 'Start Fine-Tuning'}
          </CameraButton>
        </ThemedView>
      </ThemedView>
    </ScrollView>
  );

  // Status Tab Content
  const StatusTab = () => (
    <ScrollView className="flex-1" showsVerticalScrollIndicator={false}>
      <ThemedView className="p-4">
        <ThemedText className="text-xl font-bold mb-4 text-center">
          Training Status
        </ThemedText>

        {/* Current Training Status */}
        <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
          <ThemedText className="text-lg font-semibold mb-3">Current Status</ThemedText>

          {trainingProgress.isTraining ? (
            <ThemedView className="p-4 rounded-lg border border-blue-300 dark:border-blue-600 bg-blue-50 dark:bg-blue-900/20">
              <View className="items-center py-4">
                <ActivityIndicator size="large" color="#3b82f6" />
                <ThemedText className="mt-2 text-blue-700 dark:text-blue-300 text-center">
                  {trainingProgress.message || 'Training in progress...'}
                </ThemedText>
                <ThemedText className="text-sm text-blue-600 dark:text-blue-400 mt-2 text-center">
                  This may take several minutes. You can check results in the Results tab.
                </ThemedText>
              </View>
            </ThemedView>
          ) : (
            <ThemedView className="p-4 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-900/20">
              <ThemedText className="text-center text-gray-600 dark:text-gray-400">
                {trainingProgress.message || 'No training in progress'}
              </ThemedText>
            </ThemedView>
          )}
        </ThemedView>

        {/* Status Controls */}
        <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
          <ThemedText className="text-lg font-semibold mb-3">Status Controls</ThemedText>

          <View className="space-y-3">
            <CameraButton
              onPress={fetchTrainingStatus}
              variant="primary"
            >
              Refresh Status
            </CameraButton>

            <CameraButton
              onPress={fetchTrainingResults}
              variant="success"
            >
              Check Results
            </CameraButton>
          </View>
        </ThemedView>
      </ThemedView>
    </ScrollView>
  );

  // Results Tab Content
  const ResultsTab = () => (
    <ScrollView className="flex-1" showsVerticalScrollIndicator={false}>
      <ThemedView className="p-4">
        <ThemedText className="text-xl font-bold mb-4 text-center">
          Training Results
        </ThemedText>

        {isLoadingResults ? (
          <View className="items-center py-8">
            <ActivityIndicator size="large" color={isDark ? "#fff" : "#000"} />
            <ThemedText className="mt-2">Loading training results...</ThemedText>
          </View>
        ) : trainingResults && trainingResults.training_runs.length > 0 ? (
          <View>
            {trainingResults.training_runs.map((run, index) => (
              <ThemedView key={run.name} className="mb-6 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <ThemedText className="font-semibold text-lg mb-2">{run.name}</ThemedText>
                <ThemedText className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                  Trained on: {run.modified_date}
                </ThemedText>

                {run.training_args && (
                  <ThemedView className="mb-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                    <ThemedText className="text-sm font-medium mb-2">Training Parameters:</ThemedText>
                    <View className="flex-row justify-between">
                      <ThemedText className="text-sm">• Epochs: {run.training_args.epochs}</ThemedText>
                      <ThemedText className="text-sm">• Batch Size: {run.training_args.batch_size}</ThemedText>
                      <ThemedText className="text-sm">• Image Size: {run.training_args.image_size}</ThemedText>
                    </View>
                  </ThemedView>
                )}

                {run.metrics && (
                  <ThemedView className="mb-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20">
                    <ThemedText className="text-sm font-medium mb-2">Final Metrics:</ThemedText>
                    <View className="flex-row justify-between flex-wrap">
                      <ThemedText className="text-sm">• Precision: {(run.metrics.final_precision * 100).toFixed(2)}%</ThemedText>
                      <ThemedText className="text-sm">• Recall: {(run.metrics.final_recall * 100).toFixed(2)}%</ThemedText>
                      <ThemedText className="text-sm">• mAP@0.5: {(run.metrics.final_map50 * 100).toFixed(2)}%</ThemedText>
                      <ThemedText className="text-sm">• mAP@0.5:0.95: {(run.metrics.final_map50_95 * 100).toFixed(2)}%</ThemedText>
                    </View>
                    <View className="flex-row justify-between mt-2">
                      <ThemedText className="text-sm">• Total Epochs: {run.metrics.total_epochs}</ThemedText>
                      <ThemedText className="text-sm">• Training Time: {run.metrics.training_time.toFixed(1)}s</ThemedText>
                    </View>
                  </ThemedView>
                )}

                {run.has_weights && (
                  <ThemedView className="mb-3 p-3 rounded-lg bg-green-50 dark:bg-green-900/20">
                    <ThemedText className="text-sm font-medium mb-2">Model Weights:</ThemedText>
                    <View className="flex-row justify-between">
                      <ThemedText className="text-sm">• Best Model: {run.best_model_size.toFixed(2)} MB</ThemedText>
                      <ThemedText className="text-sm">• Last Model: {run.last_model_size.toFixed(2)} MB</ThemedText>
                    </View>
                  </ThemedView>
                )}

                {run.has_plots && (
                  <ThemedView className="p-3 rounded-lg bg-purple-50 dark:bg-purple-900/20">
                    <ThemedText className="text-sm font-medium mb-2">Training Plots:</ThemedText>
                    <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                      {run.plot_files.map((plotFile, plotIndex) => (
                        <TouchableOpacity
                          key={plotIndex}
                          className="mr-3"
                          onPress={() => openImageModal(`${env?.API_ENDPOINT}/training/results/${run.name}/plots/${plotFile}`)}
                          activeOpacity={0.7}
                        >
                          <Image
                            source={{
                              uri: `${env?.API_ENDPOINT}/training/results/${run.name}/plots/${plotFile}`
                            }}
                            style={{
                              width: 150,
                              height: 120,
                              borderRadius: 8,
                              borderWidth: 1,
                              borderColor: isDark ? '#4B5563' : '#D1D5DB'
                            }}
                            resizeMode="contain"
                          />
                          <ThemedText className="text-xs text-center mt-1 text-gray-600 dark:text-gray-400">
                            {plotFile.replace('.png', '').replace(/_/g, ' ').toUpperCase()}
                          </ThemedText>
                        </TouchableOpacity>
                      ))}
                    </ScrollView>
                  </ThemedView>
                )}
              </ThemedView>
            ))}
          </View>
        ) : (
          <ThemedView className="p-8 rounded-lg border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-900/20">
            <ThemedText className="text-gray-500 dark:text-gray-400 text-center text-lg">
              No training runs found
            </ThemedText>
            <ThemedText className="text-gray-400 dark:text-gray-500 text-center mt-2">
              Start training to see results here
            </ThemedText>
          </ThemedView>
        )}

        <View className="mt-4">
          <CameraButton
            onPress={fetchTrainingResults}
            variant="primary"
          >
            Refresh Results
          </CameraButton>
        </View>
      </ThemedView>
    </ScrollView>
  );

  return (
    <ThemedView className="flex-1">
      {/* Tab Buttons */}
      <View className="flex-row border-b border-gray-300 dark:border-gray-600">
        <TabButton tab="training" label="Training" isActive={activeTab === 'training'} />
        <TabButton tab="status" label="Status" isActive={activeTab === 'status'} />
        <TabButton tab="results" label="Results" isActive={activeTab === 'results'} />
      </View>

      {/* Tab Content */}
      {activeTab === 'training' && <TrainingTab />}
      {activeTab === 'status' && <StatusTab />}
      {activeTab === 'results' && <ResultsTab />}

      {/* Image Modal */}
      {selectedImage && (
        <Modal
          visible={imageModalVisible}
          onRequestClose={closeImageModal}
          transparent={true}
          animationType="fade"
        >
          <View style={{
            flex: 1,
            backgroundColor: 'rgba(0,0,0,0.95)',
            justifyContent: 'center',
            alignItems: 'center'
          }}>
            {/* Close button */}
            <TouchableOpacity
              style={{
                position: 'absolute',
                top: 50,
                right: 20,
                zIndex: 1000,
                backgroundColor: 'rgba(255,255,255,0.2)',
                borderRadius: 20,
                width: 40,
                height: 40,
                justifyContent: 'center',
                alignItems: 'center'
              }}
              onPress={closeImageModal}
            >
              <ThemedText style={{ color: 'white', fontSize: 20, fontWeight: 'bold' }}>
                ✕
              </ThemedText>
            </TouchableOpacity>

            {/* Image */}
            <TouchableOpacity
              style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}
              onPress={closeImageModal}
              activeOpacity={1}
            >
              <Image
                source={{ uri: selectedImage }}
                style={{
                  width: Dimensions.get('window').width * 0.9,
                  height: Dimensions.get('window').height * 0.8,
                  resizeMode: 'contain'
                }}
              />
            </TouchableOpacity>

            {/* Instructions */}
            <ThemedText style={{
              position: 'absolute',
              bottom: 50,
              color: 'white',
              fontSize: 14,
              textAlign: 'center',
              opacity: 0.8
            }}>
              Tap anywhere to close
            </ThemedText>
          </View>
        </Modal>
      )}
    </ThemedView>
  );
};

export default ModelTrainingManager;