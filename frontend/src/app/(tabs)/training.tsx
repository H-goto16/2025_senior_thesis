import { Ionicons } from '@expo/vector-icons';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useEffect, useState } from 'react';
import {
    Dimensions,
    RefreshControl,
    ScrollView,
    Text,
    TouchableOpacity,
    View
} from 'react-native';
import * as Progress from 'react-native-progress';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import { ModelClassManager } from '../../components/model/ModelClassManager';
import ModelTrainingManager from '../../components/model/ModelTrainingManager';
import apiClient from '../../lib/api';
import { useTheme, useTraining } from '../../lib/store';
import { cn } from '../../lib/utils';

const { width: screenWidth } = Dimensions.get('window');

export default function TrainingScreen() {
  const { isDarkMode } = useTheme();
  const { trainingStatus, setTrainingStatus } = useTraining();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<'classes' | 'training' | 'analytics'>('training');
  const queryClient = useQueryClient();

  // Queries
  const { data: statusData, refetch: refetchStatus } = useQuery({
    queryKey: ['training-status'],
    queryFn: () => apiClient.getTrainingStatus(),
    refetchInterval: trainingStatus?.is_training ? 2000 : 30000,
  });

  const { data: historyData, refetch: refetchHistory } = useQuery({
    queryKey: ['training-history'],
    queryFn: () => apiClient.getTrainingHistory(),
  });

  const { data: metricsData } = useQuery({
    queryKey: ['training-metrics', selectedRun],
    queryFn: () => selectedRun ? apiClient.getTrainingMetrics(selectedRun) : null,
    enabled: !!selectedRun,
  });

  const { data: datasetData } = useQuery({
    queryKey: ['dataset-analysis'],
    queryFn: () => apiClient.getDatasetAnalysis(),
  });

  // Update training status
  useEffect(() => {
    if (statusData) {
      setTrainingStatus(statusData);
    }
  }, [statusData, setTrainingStatus]);

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      refetchStatus(),
      refetchHistory(),
      queryClient.invalidateQueries({ queryKey: ['dataset-analysis'] }),
    ]);
    setRefreshing(false);
  };

  const renderTabButton = (tab: 'classes' | 'training' | 'analytics', label: string, icon: string) => (
    <TouchableOpacity
      onPress={() => setActiveTab(tab)}
      className={cn(
        'flex-1 py-3 px-4 rounded-lg mx-1',
        activeTab === tab
          ? (isDarkMode ? 'bg-blue-600' : 'bg-blue-500')
          : (isDarkMode ? 'bg-gray-700' : 'bg-gray-200')
      )}
    >
      <View className="flex-row items-center justify-center">
        <Ionicons
          name={icon as any}
          size={18}
          color={activeTab === tab ? 'white' : (isDarkMode ? '#9CA3AF' : '#6B7280')}
        />
        <Text className={cn(
          'ml-2 font-medium',
          activeTab === tab ? 'text-white' : (isDarkMode ? 'text-gray-300' : 'text-gray-700')
        )}>
          {label}
        </Text>
      </View>
    </TouchableOpacity>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'classes':
        return <ModelClassManager />;
      case 'training':
        return <ModelTrainingManager />;
      case 'analytics':
        return (
          <View className="p-6">
            <Text className={cn(
              'text-xl font-bold mb-4',
              isDarkMode ? 'text-white' : 'text-gray-900'
            )}>
              📊 Training Analytics
            </Text>

            {/* Current Training Status */}
            {statusData && (
              <View className={cn(
                'mb-6 p-4 rounded-lg',
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              )}>
                <Text className={cn(
                  'text-lg font-semibold mb-3',
                  isDarkMode ? 'text-white' : 'text-gray-900'
                )}>
                  Current Status
                </Text>
                <View className="flex-row items-center mb-2">
                  <View className={cn(
                    'w-3 h-3 rounded-full mr-3',
                    statusData.is_training ? 'bg-green-500' : 'bg-gray-400'
                  )} />
                  <Text className={cn(
                    'font-medium',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {statusData.is_training ? 'Training in Progress' : 'Ready'}
                  </Text>
                </View>
                {statusData.is_training && (
                  <>
                    <Text className={cn(
                      'text-sm mb-2',
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    )}>
                      Epoch {statusData.current_epoch} of {statusData.total_epochs}
                    </Text>
                    <Progress.Bar
                      progress={statusData.progress}
                      width={screenWidth - 80}
                      color={isDarkMode ? '#60A5FA' : '#3B82F6'}
                      unfilledColor={isDarkMode ? '#374151' : '#E5E7EB'}
                      borderWidth={0}
                      height={8}
                    />
                  </>
                )}
                <Text className={cn(
                  'text-sm mt-2',
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                )}>
                  {statusData.status_message}
                </Text>
              </View>
            )}

            {/* Training History */}
            {historyData?.history && historyData.history.length > 0 && (
              <View className={cn(
                'mb-6 p-4 rounded-lg',
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              )}>
                <Text className={cn(
                  'text-lg font-semibold mb-3',
                  isDarkMode ? 'text-white' : 'text-gray-900'
                )}>
                  Training History
                </Text>
                {historyData.history.slice(0, 5).map((run: any, index: number) => (
                  <TouchableOpacity
                    key={run.run_name}
                    onPress={() => setSelectedRun(run.run_name)}
                    className={cn(
                      'p-3 mb-2 rounded-lg border',
                      selectedRun === run.run_name
                        ? (isDarkMode ? 'bg-blue-900 border-blue-600' : 'bg-blue-50 border-blue-300')
                        : (isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-300')
                    )}
                  >
                    <View className="flex-row justify-between items-center">
                      <Text className={cn(
                        'font-medium',
                        isDarkMode ? 'text-white' : 'text-gray-900'
                      )}>
                        {run.run_name}
                      </Text>
                      <Text className={cn(
                        'text-sm',
                        isDarkMode ? 'text-gray-300' : 'text-gray-600'
                      )}>
                        {run.epochs} epochs
                      </Text>
                    </View>
                    <Text className={cn(
                      'text-sm mt-1',
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    )}>
                      mAP50: {(run.final_metrics?.map50 * 100 || 0).toFixed(1)}%
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}

            {/* Training Metrics Visualization */}
            {selectedRun && metricsData && (
              <View className={cn(
                'mb-6 p-4 rounded-lg',
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              )}>
                <Text className={cn(
                  'text-lg font-semibold mb-3',
                  isDarkMode ? 'text-white' : 'text-gray-900'
                )}>
                  Training Metrics - {selectedRun}
                </Text>
                {metricsData.plots && Object.keys(metricsData.plots).length > 0 && (
                  <View className="h-80 rounded-lg overflow-hidden">
                    <WebView
                      source={{ html: `
                        <html>
                          <head>
                            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                            <style>
                              body { margin: 0; padding: 10px; background: ${isDarkMode ? '#1F2937' : '#FFFFFF'}; }
                            </style>
                          </head>
                          <body>
                            <div id="plot" style="width:100%;height:300px;"></div>
                            <script>
                              const plotData = ${JSON.stringify(metricsData.plots.loss_plot || '{}')};
                              if (plotData && typeof plotData === 'object') {
                                Plotly.newPlot('plot', plotData.data || [], plotData.layout || {}, {responsive: true});
                              }
                            </script>
                          </body>
                        </html>
                      ` }}
                      style={{ flex: 1 }}
                      javaScriptEnabled={true}
                      domStorageEnabled={true}
                    />
                  </View>
                )}
              </View>
            )}

            {/* Dataset Analysis */}
            {datasetData && (
              <View className={cn(
                'mb-6 p-4 rounded-lg',
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              )}>
                <Text className={cn(
                  'text-lg font-semibold mb-3',
                  isDarkMode ? 'text-white' : 'text-gray-900'
                )}>
                  Dataset Overview
                </Text>
                <View className="flex-row justify-between mb-2">
                  <Text className={cn(
                    'text-sm',
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  )}>
                    Total Images:
                  </Text>
                  <Text className={cn(
                    'text-sm font-medium',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {datasetData.analysis?.total_images || 0}
                  </Text>
                </View>
                <View className="flex-row justify-between mb-2">
                  <Text className={cn(
                    'text-sm',
                    isDarkMode ? 'text-gray-300' : 'text-gray-600'
                  )}>
                    Total Annotations:
                  </Text>
                  <Text className={cn(
                    'text-sm font-medium',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {datasetData.analysis?.total_annotations || 0}
                  </Text>
                </View>
                {datasetData.analysis?.class_distribution && Object.keys(datasetData.analysis.class_distribution).length > 0 && (
                  <View className="mt-3">
                    <Text className={cn(
                      'text-sm font-medium mb-2',
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    )}>
                      Class Distribution:
                    </Text>
                    {Object.entries(datasetData.analysis.class_distribution).map(([className, count]) => (
                      <View key={className} className="flex-row justify-between mb-1">
                        <Text className={cn(
                          'text-sm',
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        )}>
                          {className}:
                        </Text>
                        <Text className={cn(
                          'text-sm',
                          isDarkMode ? 'text-gray-300' : 'text-gray-600'
                        )}>
                          {count}
                        </Text>
                      </View>
                    ))}
                  </View>
                )}
              </View>
            )}
          </View>
        );
      default:
        return <ModelTrainingManager />;
    }
  };

  return (
    <SafeAreaView className="flex-1">
      <View className={cn(
        'flex-1',
        isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
      )}>
        {/* Header */}
        <View className={cn(
          'p-6 border-b',
          isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
        )}>
          <Text className={cn(
            'text-2xl font-bold mb-2',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            🚀 Training Center
          </Text>
          <Text className={cn(
            'text-base mb-4',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            Comprehensive model training management
          </Text>

          {/* Tab Navigation */}
          <View className="flex-row">
            {renderTabButton('classes', 'Classes', 'library-outline')}
            {renderTabButton('training', 'Training', 'fitness-outline')}
            {renderTabButton('analytics', 'Analytics', 'analytics-outline')}
          </View>
        </View>

        {/* Content */}
        <ScrollView
          className="flex-1"
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              tintColor={isDarkMode ? '#60A5FA' : '#3B82F6'}
            />
          }
        >
          {renderContent()}
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}