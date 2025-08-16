import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { CameraButton } from "@/components/ui/CameraButton";
import env from "@/env";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, Dimensions, Image, Modal, SafeAreaView, ScrollView, TouchableOpacity, View } from "react-native";

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

const TrainingResultsScreen = () => {
  const [trainingResults, setTrainingResults] = useState<TrainingResults | null>(null);
  const [isLoadingResults, setIsLoadingResults] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageModalVisible, setImageModalVisible] = useState(false);

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

  const openImageModal = (imageUrl: string) => {
    setSelectedImage(imageUrl);
    setImageModalVisible(true);
  };

  const closeImageModal = () => {
    setImageModalVisible(false);
    setSelectedImage(null);
  };

  useEffect(() => {
    fetchTrainingResults();
  }, []);

  return (
    <SafeAreaView className="flex-1">
      <ThemedView className="flex-1">
        <ScrollView className="flex-1 p-4" showsVerticalScrollIndicator={false}>
          <ThemedText className="text-2xl font-bold mb-6 text-center">
            Training Results
          </ThemedText>

          {isLoadingResults ? (
            <View className="items-center py-8">
              <ActivityIndicator size="large" />
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
                                borderColor: '#4B5563'
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
        </ScrollView>
      </ThemedView>

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
    </SafeAreaView>
  );
};

export default TrainingResultsScreen;


