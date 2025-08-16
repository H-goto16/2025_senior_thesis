import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { CameraButton } from "@/components/ui/CameraButton";
import { StyledTextInput } from "@/components/ui/StyledTextInput";
import env from "@/env";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, SafeAreaView, ScrollView, View } from "react-native";

interface TrainingStats {
  total_images: number;
  total_labels: number;
  classes: string[];
  class_counts: Record<string, number>;
  data_directory: string;
}

const TrainingScreen = () => {
  const [trainingStats, setTrainingStats] = useState<TrainingStats | null>(null);
  const [epochs, setEpochs] = useState('50');
  const [isLoading, setIsLoading] = useState(false);

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

  const startTraining = async () => {
    const epochCount = parseInt(epochs);
    if (isNaN(epochCount) || epochCount <= 0) {
      alert('Please enter a valid number of epochs');
      return;
    }

    if (epochCount > 500) {
      alert('Maximum epochs allowed is 500');
      return;
    }

    if (!trainingStats || trainingStats.total_images === 0) {
      alert('No training data available. Please add some labeled images first.');
      return;
    }

    try {
      const response = await fetch(`${env?.API_ENDPOINT}/training/start?epochs=${epochs}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        alert('Training started successfully! Check the Training Status tab for progress.');
      } else {
        const errorData = await response.json();
        alert(`Training failed: ${errorData.detail || 'An error occurred'}`);
      }
    } catch (error) {
      alert('Failed to start training. Please try again.');
    }
  };

  useEffect(() => {
    fetchTrainingStats();
  }, []);

  return (
    <SafeAreaView className="flex-1">
      <ThemedView className="flex-1">
        <ScrollView className="flex-1 p-4" showsVerticalScrollIndicator={false}>
          <ThemedText className="text-2xl font-bold mb-6 text-center">
            Start Training
          </ThemedText>

          {/* Training Data Statistics */}
          <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
            <ThemedText className="text-lg font-semibold mb-3">Training Data Statistics</ThemedText>

            {isLoading ? (
              <View className="items-center py-4">
                <ActivityIndicator size="large" />
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
            >
              Start Fine-Tuning
            </CameraButton>
          </ThemedView>

          {/* Training Information */}
          <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
            <ThemedText className="text-lg font-semibold mb-3">Training Information</ThemedText>

            <View className="space-y-2">
              <ThemedText className="text-sm">
                • Training runs in the background to avoid blocking the UI
              </ThemedText>
              <ThemedText className="text-sm">
                • Check the Training Status tab to monitor progress
              </ThemedText>
              <ThemedText className="text-sm">
                • View results and graphs in the Training Results tab
              </ThemedText>
              <ThemedText className="text-sm">
                • Training may take several minutes depending on epochs
              </ThemedText>
            </View>
          </ThemedView>
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
};

export default TrainingScreen;
