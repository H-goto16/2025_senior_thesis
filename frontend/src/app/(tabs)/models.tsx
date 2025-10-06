import { Ionicons } from '@expo/vector-icons';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useState } from 'react';
import {
    Alert,
    RefreshControl,
    ScrollView,
    Share,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import apiClient from '../../lib/api';
import { useModels, useTheme } from '../../lib/store';
import { cn } from '../../lib/utils';

export default function ModelsScreen() {
  const { isDarkMode } = useTheme();
  const { availableModels, currentModel, setAvailableModels, setCurrentModel } = useModels();
  const [refreshing, setRefreshing] = useState(false);
  const queryClient = useQueryClient();

  // Queries
  const { data: modelsData, refetch: refetchModels } = useQuery({
    queryKey: ['available-models'],
    queryFn: () => apiClient.getAvailableModels(),
  });

  const { data: modelInfo } = useQuery({
    queryKey: ['model-info'],
    queryFn: () => apiClient.getModelInfo(),
  });

  const { data: comparisonData } = useQuery({
    queryKey: ['model-comparison'],
    queryFn: () => apiClient.getModelComparison(),
  });

  const { data: validationData } = useQuery({
    queryKey: ['model-validation'],
    queryFn: () => apiClient.validateModel(),
  });

  // Mutations
  const loadModelMutation = useMutation({
    mutationFn: (modelPath: string) => apiClient.loadModel(modelPath),
    onSuccess: (_, modelPath) => {
      setCurrentModel(modelPath);
      Alert.alert('Success', 'Model loaded successfully!');
      queryClient.invalidateQueries({ queryKey: ['model-info'] });
      queryClient.invalidateQueries({ queryKey: ['model-validation'] });
    },
    onError: (error: any) => {
      Alert.alert('Error', error.message || 'Failed to load model');
    },
  });

  const backupModelMutation = useMutation({
    mutationFn: () => apiClient.backupCurrentModel(),
    onSuccess: () => {
      Alert.alert('Success', 'Model backed up successfully!');
      refetchModels();
    },
    onError: (error: any) => {
      Alert.alert('Error', error.message || 'Failed to backup model');
    },
  });

  const exportDataMutation = useMutation({
    mutationFn: () => apiClient.exportTrainingData(),
    onSuccess: async (blob) => {
      try {
        // Create a temporary URL for the blob
        const url = URL.createObjectURL(blob);

        // Try to share the file
        await Share.share({
          url: url,
          title: 'Training Data Export',
        });
      } catch (error) {
        Alert.alert('Info', 'Export completed. File saved to downloads.');
      }
    },
    onError: (error: any) => {
      Alert.alert('Error', error.message || 'Failed to export data');
    },
  });

  // Update store when models data changes
  React.useEffect(() => {
    if (modelsData?.models) {
      setAvailableModels(modelsData.models);
    }
  }, [modelsData, setAvailableModels]);

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      refetchModels(),
      queryClient.invalidateQueries({ queryKey: ['model-info'] }),
      queryClient.invalidateQueries({ queryKey: ['model-comparison'] }),
      queryClient.invalidateQueries({ queryKey: ['model-validation'] }),
    ]);
    setRefreshing(false);
  };

  const handleLoadModel = (modelPath: string) => {
    Alert.alert(
      'Load Model',
      `Are you sure you want to load this model?\n\nPath: ${modelPath}`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Load', onPress: () => loadModelMutation.mutate(modelPath) },
      ]
    );
  };

  const handleBackupModel = () => {
    Alert.alert(
      'Backup Model',
      'Create a backup of the current model?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Backup', onPress: () => backupModelMutation.mutate() },
      ]
    );
  };

  const handleExportData = () => {
    Alert.alert(
      'Export Training Data',
      'Export all training data and models as a ZIP file?',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Export', onPress: () => exportDataMutation.mutate() },
      ]
    );
  };

  const renderCurrentModelInfo = () => (
    <View className={cn(
      'mx-4 mb-4 p-4 rounded-xl',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      <View className="flex-row items-center justify-between mb-3">
        <Text className={cn(
          'text-lg font-semibold',
          isDarkMode ? 'text-white' : 'text-gray-900'
        )}>
          Current Model
        </Text>
        <View className="flex-row space-x-2">
          <TouchableOpacity
            onPress={handleBackupModel}
            disabled={backupModelMutation.isPending}
            className="p-2 rounded-lg bg-blue-500 dark:bg-blue-600"
          >
            <Ionicons name="save" size={16} color="white" />
          </TouchableOpacity>
          <TouchableOpacity
            onPress={handleExportData}
            disabled={exportDataMutation.isPending}
            className="p-2 rounded-lg bg-green-500 dark:bg-green-600"
          >
            <Ionicons name="download" size={16} color="white" />
          </TouchableOpacity>
        </View>
      </View>

      {modelInfo ? (
        <View className="space-y-2">
          <View className="flex-row justify-between">
            <Text className={cn(
              'text-sm',
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            )}>
              Model Path:
            </Text>
            <Text className={cn(
              'text-sm font-mono flex-1 text-right',
              isDarkMode ? 'text-white' : 'text-gray-900'
            )} numberOfLines={1}>
              {modelInfo.model_path}
            </Text>
          </View>

          <View className="flex-row justify-between">
            <Text className={cn(
              'text-sm',
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            )}>
              Classes:
            </Text>
            <Text className={cn(
              'text-sm font-medium',
              isDarkMode ? 'text-white' : 'text-gray-900'
            )}>
              {modelInfo.total_classes}
            </Text>
          </View>

          {validationData && (
            <>
              {validationData.map50 && (
                <View className="flex-row justify-between">
                  <Text className={cn(
                    'text-sm',
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  )}>
                    mAP@0.5:
                  </Text>
                  <Text className={cn(
                    'text-sm font-medium',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {(validationData.map50 * 100).toFixed(1)}%
                  </Text>
                </View>
              )}

              {validationData.precision && (
                <View className="flex-row justify-between">
                  <Text className={cn(
                    'text-sm',
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  )}>
                    Precision:
                  </Text>
                  <Text className={cn(
                    'text-sm font-medium',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {(validationData.precision * 100).toFixed(1)}%
                  </Text>
                </View>
              )}
            </>
          )}
        </View>
      ) : (
        <Text className={cn(
          'text-center py-4',
          isDarkMode ? 'text-gray-400' : 'text-gray-500'
        )}>
          No model information available
        </Text>
      )}
    </View>
  );

  const renderAvailableModels = () => (
    <View className={cn(
      'mx-4 mb-4 p-4 rounded-xl',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      <Text className={cn(
        'text-lg font-semibold mb-3',
        isDarkMode ? 'text-white' : 'text-gray-900'
      )}>
        Available Models
      </Text>

      {modelsData?.models?.length === 0 ? (
        <Text className={cn(
          'text-center py-8',
          isDarkMode ? 'text-gray-400' : 'text-gray-500'
        )}>
          No trained models available
        </Text>
      ) : (
        <View className="space-y-3">
          {modelsData?.models?.map((model, index) => (
            <TouchableOpacity
              key={`${model.run_name}-${index}`}
              onPress={() => model.best_model && handleLoadModel(model.best_model)}
              disabled={!model.best_model || loadModelMutation.isPending}
              className={cn(
                'p-3 rounded-lg border',
                model.best_model === currentModel
                  ? 'bg-blue-100 dark:bg-blue-900 border-blue-500'
                  : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600'
              )}
            >
              <View className="flex-row items-center justify-between mb-2">
                <Text className={cn(
                  'font-medium',
                  isDarkMode ? 'text-white' : 'text-gray-900'
                )}>
                  {model.run_name}
                </Text>
                {model.best_model === currentModel && (
                  <View className="px-2 py-1 rounded-full bg-green-100 dark:bg-green-900">
                    <Text className="text-xs text-green-800 dark:text-green-200">
                      Current
                    </Text>
                  </View>
                )}
              </View>

              <Text className={cn(
                'text-xs mb-1',
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              )}>
                {new Date(model.timestamp).toLocaleString()}
              </Text>

              {model.best_model && (
                <Text className={cn(
                  'text-xs font-mono',
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                )} numberOfLines={1}>
                  {model.best_model}
                </Text>
              )}

              {model.args?.type !== 'backup' && (
                <View className="flex-row justify-between mt-2">
                  <Text className={cn(
                    'text-xs',
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  )}>
                    Type: Training Run
                  </Text>
                  <Ionicons
                    name="chevron-forward"
                    size={16}
                    color={isDarkMode ? '#9CA3AF' : '#6B7280'}
                  />
                </View>
              )}
            </TouchableOpacity>
          ))}
        </View>
      )}
    </View>
  );

  const renderModelComparison = () => {
    if (!comparisonData?.comparison_plot) return null;

    return (
      <View className={cn(
        'mx-4 mb-4 p-4 rounded-xl',
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      )}>
        <Text className={cn(
          'text-lg font-semibold mb-3',
          isDarkMode ? 'text-white' : 'text-gray-900'
        )}>
          Model Performance Comparison
        </Text>

        <View style={{ height: 300 }}>
          <WebView
            source={{
              html: `
                <!DOCTYPE html>
                <html>
                <head>
                  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                  <style>
                    body { margin: 0; padding: 0; background: ${isDarkMode ? '#1f2937' : '#ffffff'}; }
                  </style>
                </head>
                <body>
                  <div id="plot" style="width:100%;height:300px;"></div>
                  <script>
                    const plotData = ${comparisonData.comparison_plot};
                    Plotly.newPlot('plot', plotData.data, plotData.layout, {responsive: true});
                  </script>
                </body>
                </html>
              `,
            }}
            style={{ backgroundColor: 'transparent' }}
          />
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView className={cn(
      'flex-1',
      isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
    )}>
      <ScrollView
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        showsVerticalScrollIndicator={false}
      >
        <View className="pt-4">
          {renderCurrentModelInfo()}
          {renderAvailableModels()}
          {renderModelComparison()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}





