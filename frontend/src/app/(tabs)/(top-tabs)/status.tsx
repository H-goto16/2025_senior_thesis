import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { CameraButton } from "@/components/ui/CameraButton";
import env from "@/env";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, SafeAreaView, ScrollView, View } from "react-native";

interface TrainingProgress {
  isTraining: boolean;
  currentEpoch?: number;
  totalEpochs?: number;
  message?: string;
}

const TrainingStatusScreen = () => {
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress>({ isTraining: false });

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

  useEffect(() => {
    fetchTrainingStatus();
    // Set up interval to check training status
    const statusInterval = setInterval(fetchTrainingStatus, 5000);
    return () => clearInterval(statusInterval);
  }, []);

  return (
    <SafeAreaView className="flex-1">
      <ThemedView className="flex-1">
        <ScrollView className="flex-1 p-4" showsVerticalScrollIndicator={false}>
          <ThemedText className="text-2xl font-bold mb-6 text-center">
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
            </View>
          </ThemedView>

          {/* Training Information */}
          <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
            <ThemedText className="text-lg font-semibold mb-3">Training Information</ThemedText>

            <View className="space-y-2">
              <ThemedText className="text-sm">
                • Training runs in the background to avoid blocking the UI
              </ThemedText>
              <ThemedText className="text-sm">
                • Status is automatically updated every 5 seconds
              </ThemedText>
              <ThemedText className="text-sm">
                • Check the Results tab to see completed training runs
              </ThemedText>
              <ThemedText className="text-sm">
                • Training progress and completion status are shown here
              </ThemedText>
            </View>
          </ThemedView>
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
};

export default TrainingStatusScreen;


