import env from '@/env';
import { useEffect, useState } from 'react';
import { ScrollView, View } from 'react-native';
import { ThemedText } from '../ThemedText';
import { ThemedView } from '../ThemedView';
import { CameraButton } from '../ui/CameraButton';
import { PlatformAlert } from '../ui/PlatformAlert';
import { StyledTextInput } from '../ui/StyledTextInput';

interface ModelInfo {
  model_path: string;
  vocab_file: string;
  classes: string[];
  total_classes: number;
  model_name?: string;
  status?: string;
  error?: string;
}

interface ClassesResponse {
  classes: string[];
  message: string;
}

export const ModelClassManager = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [classes, setClasses] = useState<string[]>([]);
  const [newClass, setNewClass] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);


  const fetchModelInfo = async () => {
    try {
      setIsRefreshing(true);
      const response = await fetch(`${env?.API_ENDPOINT}/model/info`);
      if (response.ok) {
        const data: ModelInfo = await response.json();
        setModelInfo(data);
        // Use classes field instead of current_classes, with safe fallback
        const classesArray = Array.isArray(data.classes) ? data.classes : [];
        setClasses(classesArray);
      } else {
        throw new Error('Failed to fetch model info');
      }
    } catch (error) {
      console.error('Error fetching model info:', error);
      PlatformAlert.error('Error', 'Failed to fetch model information');
      // Set empty classes array on error
      setClasses([]);
    } finally {
      setIsRefreshing(false);
    }
  };

  const fetchClasses = async () => {
    try {
      const response = await fetch(`${env?.API_ENDPOINT}/model/classes`);
      if (response.ok) {
        const data: ClassesResponse = await response.json();
        const classesArray = Array.isArray(data.classes) ? data.classes : [];
        setClasses(classesArray);
      } else {
        throw new Error('Failed to fetch classes');
      }
    } catch (error) {
      console.error('Error fetching classes:', error);
      PlatformAlert.error('Error', 'Failed to fetch classes');
      // Set empty classes array on error
      setClasses([]);
    }
  };

  const addClass = async () => {
    if (!newClass.trim()) {
      PlatformAlert.error('Error', 'Please enter a class name');
      return;
    }

    try {
      setIsLoading(true);
      const response = await fetch(`${env?.API_ENDPOINT}/model/classes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          classes: [newClass.trim()]
        }),
      });

      if (response.ok) {
        const data: ClassesResponse = await response.json();
        const classesArray = Array.isArray(data.classes) ? data.classes : [];
        setClasses(classesArray);
        setNewClass('');
        PlatformAlert.success('Success', data.message);
        // Update model info as well
        fetchModelInfo();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to add class');
      }
    } catch (error) {
      console.error('Error adding class:', error);
      PlatformAlert.error('Error', error instanceof Error ? error.message : 'Failed to add class');
    } finally {
      setIsLoading(false);
    }
  };

  const clearAllClasses = async () => {
    PlatformAlert.confirm(
      'Confirm',
      'Are you sure you want to clear all detection classes? This will make object detection unavailable until new classes are added.',
      async () => {
        try {
          setIsLoading(true);
          const response = await fetch(`${env?.API_ENDPOINT}/model/classes`, {
            method: 'DELETE',
          });

          if (response.ok) {
            setClasses([]);
            PlatformAlert.success('Success', 'All classes cleared successfully');
            fetchModelInfo();
          } else {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to clear classes');
          }
        } catch (error) {
          console.error('Error clearing classes:', error);
          PlatformAlert.error('Error', error instanceof Error ? error.message : 'Failed to clear classes');
        } finally {
          setIsLoading(false);
        }
      }
    );
  };

  const removeClass = async (classToRemove: string) => {
    PlatformAlert.confirm(
      'Confirm',
      `Remove "${classToRemove}" from detection classes?`,
      async () => {
        try {
          setIsLoading(true);
          // Since backend doesn't have individual class removal, we'll clear all and re-add the remaining
          const remainingClasses = classes.filter(cls => cls !== classToRemove);

          // Clear all classes first
          await fetch(`${env?.API_ENDPOINT}/model/classes`, { method: 'DELETE' });

          // Re-add remaining classes if any
          if (remainingClasses.length > 0) {
            const response = await fetch(`${env?.API_ENDPOINT}/model/classes`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                classes: remainingClasses
              }),
            });

            if (response.ok) {
              const data: ClassesResponse = await response.json();
              const classesArray = Array.isArray(data.classes) ? data.classes : [];
              setClasses(classesArray);
            } else {
              throw new Error('Failed to restore classes');
            }
          } else {
            setClasses([]);
          }

          PlatformAlert.success('Success', `Class "${classToRemove}" removed successfully`);
          fetchModelInfo();
        } catch (error) {
          console.error('Error removing class:', error);
          PlatformAlert.error('Error', error instanceof Error ? error.message : 'Failed to remove class');
          // Refresh to get current state
          fetchClasses();
        } finally {
          setIsLoading(false);
        }
      }
    );
  };

  useEffect(() => {
    fetchModelInfo();
  }, []);

  return (
    <ThemedView className="flex-1 p-5">
      <ScrollView showsVerticalScrollIndicator={false}>
        {/* Header */}
        <ThemedView className="mb-6 items-center">
          <ThemedText className="text-3xl font-bold mb-2">Model Management</ThemedText>
          <ThemedText className="text-base opacity-70 text-center">
            Manage detection classes for object recognition
          </ThemedText>
        </ThemedView>

        {/* Model Information Card */}
        {modelInfo && (
          <ThemedView className="mb-6 p-4 rounded-xl border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-900/20 shadow-sm">
            <ThemedText className="text-lg font-bold mb-3">Model Information</ThemedText>
            <View className="mb-2">
              <ThemedText className="text-sm font-semibold mb-1">Model Path:</ThemedText>
              <ThemedText className="text-sm opacity-80 font-mono">{modelInfo.model_path}</ThemedText>
            </View>
            <View className="mb-2">
              <ThemedText className="text-sm font-semibold mb-1">Vocabulary File:</ThemedText>
              <ThemedText className="text-sm opacity-80 font-mono">{modelInfo.vocab_file}</ThemedText>
            </View>
            <View>
              <ThemedText className="text-sm font-semibold mb-1">Total Classes:</ThemedText>
              <ThemedText className="text-sm opacity-80 font-mono">{modelInfo.total_classes}</ThemedText>
            </View>
          </ThemedView>
        )}

        {/* Add New Class Section */}
        <ThemedView className="mb-6">
          <ThemedText className="text-xl font-bold mb-4">Add New Class</ThemedText>
          <StyledTextInput
            placeholder="Enter class name (e.g., 'apple', 'car', 'person')"
            value={newClass}
            onChangeText={setNewClass}
          />
          <CameraButton
            variant="success"
            icon="add-circle"
            onPress={addClass}
            disabled={!newClass.trim() || isLoading}
            size="medium"
          >
            {isLoading ? 'Adding...' : 'Add Class'}
          </CameraButton>
        </ThemedView>

        {/* Classes Section */}
        <ThemedView className="mb-6">
          <ThemedView className="flex-row justify-between items-center mb-4">
            <ThemedText className="text-xl font-bold">
              Detection Classes ({Array.isArray(classes) ? classes.length : 0})
            </ThemedText>
            <CameraButton
              variant="secondary"
              icon="refresh"
              onPress={fetchClasses}
              disabled={isRefreshing}
              size="small"
            >
              {isRefreshing ? 'Refreshing...' : 'Refresh'}
            </CameraButton>
          </ThemedView>

          {Array.isArray(classes) && classes.length > 0 ? (
            <ThemedView className="mb-4">
              {classes.map((className, index) => (
                <ThemedView
                  key={index}
                  className="flex-row justify-between items-center p-4 mb-2 rounded-xl border-l-4 border-green-500 bg-green-50 dark:bg-green-900/20 shadow-sm"
                >
                  <ThemedView className="flex-1">
                    <ThemedText className="text-base font-semibold mb-1">{className}</ThemedText>
                    <ThemedText className="text-xs opacity-60">#{index + 1}</ThemedText>
                  </ThemedView>
                  <CameraButton
                    variant="danger"
                    icon="trash"
                    onPress={() => removeClass(className)}
                    disabled={isLoading}
                    size="small"
                  >
                    Remove
                  </CameraButton>
                </ThemedView>
              ))}
            </ThemedView>
          ) : (
            <ThemedView className="items-center p-8 rounded-xl bg-gray-50 dark:bg-gray-800/20">
              <ThemedText className="text-base font-semibold mb-1">
                No detection classes configured
              </ThemedText>
              <ThemedText className="text-sm opacity-70 text-center">
                Add some classes to enable object detection
              </ThemedText>
            </ThemedView>
          )}

          {Array.isArray(classes) && classes.length > 0 && (
            <ThemedView className="items-center">
              <CameraButton
                variant="danger"
                icon="warning"
                onPress={clearAllClasses}
                disabled={isLoading}
                size="medium"
              >
                {isLoading ? 'Clearing...' : 'Clear All Classes'}
              </CameraButton>
            </ThemedView>
          )}
        </ThemedView>
      </ScrollView>
    </ThemedView>
  );
};