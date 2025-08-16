import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { CameraButton } from "@/components/ui/CameraButton";
import env from "@/env";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, SafeAreaView, ScrollView, View } from "react-native";

const ModelManagementScreen = () => {
  const [modelInfo, setModelInfo] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchModelInfo = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(env?.API_ENDPOINT + "/model/info");
      if (response.ok) {
        const info = await response.json();
        setModelInfo(info);
      } else {
        console.error('Failed to fetch model info');
      }
    } catch (error) {
      console.error('Error fetching model info:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  return (
    <SafeAreaView className="flex-1">
      <ThemedView className="flex-1">
        <ScrollView className="flex-1 p-4" showsVerticalScrollIndicator={false}>
          <ThemedText className="text-2xl font-bold mb-6 text-center">
            Model Management
          </ThemedText>

          {/* Model Information */}
          <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
            <ThemedText className="text-lg font-semibold mb-3">Current Model</ThemedText>

            {isLoading ? (
              <View className="items-center py-4">
                <ActivityIndicator size="large" />
                <ThemedText className="mt-2">Loading model information...</ThemedText>
              </View>
            ) : modelInfo ? (
              <View>
                <ThemedText className="mb-2">Model: {modelInfo.model_name || 'Unknown'}</ThemedText>
                <ThemedText className="mb-2">Classes: {Array.isArray(modelInfo.classes) ? modelInfo.classes.join(', ') : 'None'}</ThemedText>
                <ThemedText className="mb-2">Status: {modelInfo.status || 'Unknown'}</ThemedText>
                {modelInfo.error && (
                  <ThemedText className="mb-2 text-red-500">Error: {modelInfo.error}</ThemedText>
                )}
              </View>
            ) : (
              <ThemedText className="text-gray-500 dark:text-gray-400 text-center py-4">
                No model information available
              </ThemedText>
            )}

            <View className="mt-3">
              <CameraButton
                onPress={fetchModelInfo}
                variant="primary"
              >
                Refresh Model Info
              </CameraButton>
            </View>
          </ThemedView>

          {/* Model Actions */}
          <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
            <ThemedText className="text-lg font-semibold mb-3">Model Actions</ThemedText>

            <View className="space-y-3">
              <CameraButton
                onPress={() => {/* TODO: Implement model reload */}}
                variant="success"
              >
                Reload Model
              </CameraButton>

              <CameraButton
                onPress={() => {/* TODO: Implement model backup */}}
                variant="primary"
              >
                Backup Model
              </CameraButton>

              <CameraButton
                onPress={() => {/* TODO: Implement model restore */}}
                variant="secondary"
              >
                Restore Model
              </CameraButton>
            </View>
          </ThemedView>

          {/* Model Statistics */}
          <ThemedView className="mb-6 p-4 rounded-lg border border-gray-300 dark:border-gray-600">
            <ThemedText className="text-lg font-semibold mb-3">Model Statistics</ThemedText>

            <View className="flex-row justify-around">
              <View className="items-center">
                <ThemedText className="text-3xl font-bold text-blue-500">
                  {modelInfo?.total_detections || 0}
                </ThemedText>
                <ThemedText className="text-sm text-gray-600 dark:text-gray-400">Detections</ThemedText>
              </View>

              <View className="items-center">
                <ThemedText className="text-3xl font-bold text-green-500">
                  {modelInfo?.accuracy || '0%'}
                </ThemedText>
                <ThemedText className="text-sm text-gray-600 dark:text-gray-400">Accuracy</ThemedText>
              </View>

              <View className="items-center">
                <ThemedText className="text-3xl font-bold text-purple-500">
                  {Array.isArray(modelInfo?.classes) ? modelInfo.classes.length : 0}
                </ThemedText>
                <ThemedText className="text-sm text-gray-600 dark:text-gray-400">Classes</ThemedText>
              </View>
            </View>
          </ThemedView>
        </ScrollView>
      </ThemedView>
    </SafeAreaView>
  );
};

export default ModelManagementScreen;
