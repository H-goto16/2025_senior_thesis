import { Ionicons } from '@expo/vector-icons';
import { useQuery } from '@tanstack/react-query';
import React, { useState } from 'react';
import {
    Dimensions,
    RefreshControl,
    ScrollView,
    Text,
    TouchableOpacity,
    View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { WebView } from 'react-native-webview';
import apiClient from '../../lib/api';
import { useTheme } from '../../lib/store';
import { cn } from '../../lib/utils';

const { width: screenWidth } = Dimensions.get('window');

export default function AnalyticsScreen() {
  const { isDarkMode, toggleDarkMode } = useTheme();
  const [refreshing, setRefreshing] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<'overview' | 'dataset' | 'performance'>('overview');

  // Queries
  const { data: datasetData, refetch: refetchDataset } = useQuery({
    queryKey: ['dataset-analysis'],
    queryFn: () => apiClient.getDatasetAnalysis(),
  });

  const { data: comparisonData, refetch: refetchComparison } = useQuery({
    queryKey: ['model-comparison'],
    queryFn: () => apiClient.getModelComparison(),
  });

  const { data: trainingHistory, refetch: refetchHistory } = useQuery({
    queryKey: ['training-history'],
    queryFn: () => apiClient.getTrainingHistory(),
  });

  const { data: statsData, refetch: refetchStats } = useQuery({
    queryKey: ['training-stats'],
    queryFn: () => apiClient.getTrainingDataStats(),
  });

  const onRefresh = async () => {
    setRefreshing(true);
    await Promise.all([
      refetchDataset(),
      refetchComparison(),
      refetchHistory(),
      refetchStats(),
    ]);
    setRefreshing(false);
  };

  const renderHeader = () => (
    <View className={cn(
      'mx-4 mb-4 p-4 rounded-xl flex-row items-center justify-between',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      <Text className={cn(
        'text-xl font-bold',
        isDarkMode ? 'text-white' : 'text-gray-900'
      )}>
        Analytics Dashboard
      </Text>
      <TouchableOpacity
        onPress={toggleDarkMode}
        className={cn(
          'p-2 rounded-lg',
          isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
        )}
      >
        <Ionicons
          name={isDarkMode ? 'sunny' : 'moon'}
          size={20}
          color={isDarkMode ? '#FCD34D' : '#6B7280'}
        />
      </TouchableOpacity>
    </View>
  );

  const renderMetricSelector = () => (
    <View className={cn(
      'mx-4 mb-4 p-1 rounded-xl flex-row',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      {[
        { key: 'overview', label: 'Overview', icon: 'pie-chart' },
        { key: 'dataset', label: 'Dataset', icon: 'folder' },
        { key: 'performance', label: 'Performance', icon: 'trending-up' },
      ].map((tab) => (
        <TouchableOpacity
          key={tab.key}
          onPress={() => setSelectedMetric(tab.key as any)}
          className={cn(
            'flex-1 flex-row items-center justify-center py-3 px-4 rounded-lg',
            selectedMetric === tab.key
              ? 'bg-blue-500 dark:bg-blue-600'
              : 'bg-transparent'
          )}
        >
          <Ionicons
            name={tab.icon as any}
            size={16}
            color={selectedMetric === tab.key ? 'white' : (isDarkMode ? '#9CA3AF' : '#6B7280')}
            style={{ marginRight: 8 }}
          />
          <Text className={cn(
            'text-sm font-medium',
            selectedMetric === tab.key
              ? 'text-white'
              : isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderOverview = () => (
    <View className="space-y-4">
      {/* Summary Cards */}
      <View className="mx-4 flex-row space-x-3">
        <View className={cn(
          'flex-1 p-4 rounded-xl',
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        )}>
          <View className="flex-row items-center mb-2">
            <Ionicons name="images" size={20} color="#3B82F6" />
            <Text className={cn(
              'ml-2 text-sm font-medium',
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            )}>
              Images
            </Text>
          </View>
          <Text className={cn(
            'text-2xl font-bold',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            {datasetData?.analysis?.total_images || 0}
          </Text>
        </View>

        <View className={cn(
          'flex-1 p-4 rounded-xl',
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        )}>
          <View className="flex-row items-center mb-2">
            <Ionicons name="pricetag" size={20} color="#10B981" />
            <Text className={cn(
              'ml-2 text-sm font-medium',
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            )}>
              Labels
            </Text>
          </View>
          <Text className={cn(
            'text-2xl font-bold',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            {datasetData?.analysis?.total_annotations || 0}
          </Text>
        </View>
      </View>

      <View className="mx-4 flex-row space-x-3">
        <View className={cn(
          'flex-1 p-4 rounded-xl',
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        )}>
          <View className="flex-row items-center mb-2">
            <Ionicons name="library" size={20} color="#F59E0B" />
            <Text className={cn(
              'ml-2 text-sm font-medium',
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            )}>
              Models
            </Text>
          </View>
          <Text className={cn(
            'text-2xl font-bold',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            {comparisonData?.models?.length || 0}
          </Text>
        </View>

        <View className={cn(
          'flex-1 p-4 rounded-xl',
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        )}>
          <View className="flex-row items-center mb-2">
            <Ionicons name="fitness" size={20} color="#EF4444" />
            <Text className={cn(
              'ml-2 text-sm font-medium',
              isDarkMode ? 'text-gray-300' : 'text-gray-600'
            )}>
              Training Runs
            </Text>
          </View>
          <Text className={cn(
            'text-2xl font-bold',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            {trainingHistory?.history?.length || 0}
          </Text>
        </View>
      </View>

      {/* Recent Activity */}
      <View className={cn(
        'mx-4 p-4 rounded-xl',
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      )}>
        <Text className={cn(
          'text-lg font-semibold mb-3',
          isDarkMode ? 'text-white' : 'text-gray-900'
        )}>
          Recent Training Runs
        </Text>

        {trainingHistory?.history?.slice(0, 3).map((run, index) => (
          <View
            key={run.run_name}
            className={cn(
              'flex-row items-center justify-between py-3',
              index < 2 ? 'border-b border-gray-200 dark:border-gray-600' : ''
            )}
          >
            <View className="flex-1">
              <Text className={cn(
                'font-medium',
                isDarkMode ? 'text-white' : 'text-gray-900'
              )}>
                {run.run_name}
              </Text>
              <Text className={cn(
                'text-sm',
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              )}>
                {new Date(run.timestamp).toLocaleDateString()}
              </Text>
            </View>
            <View className="items-end">
              <Text className={cn(
                'text-sm font-medium',
                isDarkMode ? 'text-white' : 'text-gray-900'
              )}>
                mAP: {(run.final_metrics.map50 * 100).toFixed(1)}%
              </Text>
              <Text className={cn(
                'text-xs',
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              )}>
                {run.epochs} epochs
              </Text>
            </View>
          </View>
        )) || (
          <Text className={cn(
            'text-center py-4',
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          )}>
            No training runs yet
          </Text>
        )}
      </View>
    </View>
  );

  const renderDatasetAnalysis = () => (
    <View className="space-y-4">
      {/* Dataset Stats */}
      <View className={cn(
        'mx-4 p-4 rounded-xl',
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      )}>
        <Text className={cn(
          'text-lg font-semibold mb-3',
          isDarkMode ? 'text-white' : 'text-gray-900'
        )}>
          Dataset Statistics
        </Text>

        {datasetData?.analysis && (
          <View className="space-y-3">
            <View className="flex-row justify-between">
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
                {datasetData.analysis.total_images}
              </Text>
            </View>

            <View className="flex-row justify-between">
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
                {datasetData.analysis.total_annotations}
              </Text>
            </View>

            <View className="flex-row justify-between">
              <Text className={cn(
                'text-sm',
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              )}>
                Number of Classes:
              </Text>
              <Text className={cn(
                'text-sm font-medium',
                isDarkMode ? 'text-white' : 'text-gray-900'
              )}>
                {Object.keys(datasetData.analysis.class_distribution).length}
              </Text>
            </View>

            <View className="flex-row justify-between">
              <Text className={cn(
                'text-sm',
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              )}>
                Avg. Annotations per Image:
              </Text>
              <Text className={cn(
                'text-sm font-medium',
                isDarkMode ? 'text-white' : 'text-gray-900'
              )}>
                {datasetData.analysis.total_images > 0
                  ? (datasetData.analysis.total_annotations / datasetData.analysis.total_images).toFixed(1)
                  : '0'}
              </Text>
            </View>
          </View>
        )}
      </View>

      {/* Class Distribution Chart */}
      {datasetData?.analysis?.class_distribution_plot && (
        <View className={cn(
          'mx-4 p-4 rounded-xl',
          isDarkMode ? 'bg-gray-800' : 'bg-white'
        )}>
          <Text className={cn(
            'text-lg font-semibold mb-3',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            Class Distribution
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
                      const plotData = ${datasetData.analysis.class_distribution_plot};
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
      )}
    </View>
  );

  const renderPerformanceAnalysis = () => (
    <View className="space-y-4">
      {/* Model Comparison Chart */}
      {comparisonData?.comparison_plot && (
        <View className={cn(
          'mx-4 p-4 rounded-xl',
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
      )}

      {/* Performance Metrics Table */}
      <View className={cn(
        'mx-4 p-4 rounded-xl',
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      )}>
        <Text className={cn(
          'text-lg font-semibold mb-3',
          isDarkMode ? 'text-white' : 'text-gray-900'
        )}>
          Model Performance Metrics
        </Text>

        {comparisonData?.models?.length === 0 ? (
          <Text className={cn(
            'text-center py-4',
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          )}>
            No trained models available
          </Text>
        ) : (
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <View className="min-w-full">
              {/* Header */}
              <View className="flex-row border-b border-gray-200 dark:border-gray-600 pb-2 mb-2">
                <Text className={cn(
                  'w-32 text-xs font-medium',
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                )}>
                  Model
                </Text>
                <Text className={cn(
                  'w-20 text-xs font-medium text-center',
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                )}>
                  mAP@0.5
                </Text>
                <Text className={cn(
                  'w-20 text-xs font-medium text-center',
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                )}>
                  Epochs
                </Text>
                <Text className={cn(
                  'w-24 text-xs font-medium text-center',
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                )}>
                  Date
                </Text>
              </View>

              {/* Rows */}
              {comparisonData?.models?.map((model, index) => (
                <View key={index} className="flex-row py-2">
                  <Text className={cn(
                    'w-32 text-xs',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )} numberOfLines={1}>
                    {model.run_name}
                  </Text>
                  <Text className={cn(
                    'w-20 text-xs text-center',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {(model.map50 * 100).toFixed(1)}%
                  </Text>
                  <Text className={cn(
                    'w-20 text-xs text-center',
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  )}>
                    {model.epochs}
                  </Text>
                  <Text className={cn(
                    'w-24 text-xs text-center',
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  )}>
                    {new Date(model.timestamp).toLocaleDateString()}
                  </Text>
                </View>
              ))}
            </View>
          </ScrollView>
        )}
      </View>
    </View>
  );

  const renderContent = () => {
    switch (selectedMetric) {
      case 'dataset':
        return renderDatasetAnalysis();
      case 'performance':
        return renderPerformanceAnalysis();
      default:
        return renderOverview();
    }
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
          {renderHeader()}
          {renderMetricSelector()}
          {renderContent()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}


