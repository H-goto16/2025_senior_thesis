import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useRef, useState } from 'react';
import {
  Alert,
  Dimensions,
  PanGestureHandler,
  RefreshControl,
  ScrollView,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { PanGestureHandlerGestureEvent } from 'react-native-gesture-handler';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, {
  runOnJS,
  useAnimatedGestureHandler,
  useAnimatedStyle,
  useSharedValue,
} from 'react-native-reanimated';
import { Image } from 'expo-image';
import apiClient from '../../lib/api';
import { useLabeling, useTheme } from '../../lib/store';
import { cn } from '../../lib/utils';

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

interface BoundingBox {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  color: string;
}

const LABEL_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FECA57', '#FF9FF3', '#A8E6CF', '#FFD93D',
  '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE'
];

export default function LabelingScreen() {
  const { isDarkMode } = useTheme();
  const { labels, setLabels, addLabel, removeLabel } = useLabeling();
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentLabel, setCurrentLabel] = useState('');
  const [availableLabels, setAvailableLabels] = useState<string[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  
  const queryClient = useQueryClient();
  
  // Drawing state
  const startX = useSharedValue(0);
  const startY = useSharedValue(0);
  const currentX = useSharedValue(0);
  const currentY = useSharedValue(0);
  const isActive = useSharedValue(false);

  // Queries
  const { data: classesData, refetch: refetchClasses } = useQuery({
    queryKey: ['model-classes'],
    queryFn: () => apiClient.getClasses(),
  });

  // Mutations
  const submitLabelingMutation = useMutation({
    mutationFn: ({ imageBlob, labelingData }: { imageBlob: Blob; labelingData: any }) =>
      apiClient.submitLabelingData(imageBlob, labelingData),
    onSuccess: () => {
      Alert.alert('Success', 'Labeling data submitted successfully!');
      setCurrentImage(null);
      setLabels([]);
      queryClient.invalidateQueries({ queryKey: ['dataset-analysis'] });
      queryClient.invalidateQueries({ queryKey: ['training-stats'] });
    },
    onError: (error: any) => {
      Alert.alert('Error', error.message || 'Failed to submit labeling data');
    },
  });

  React.useEffect(() => {
    if (classesData?.classes) {
      setAvailableLabels(classesData.classes);
      if (classesData.classes.length > 0 && !currentLabel) {
        setCurrentLabel(classesData.classes[0]);
      }
    }
  }, [classesData, currentLabel]);

  const onRefresh = async () => {
    setRefreshing(true);
    await refetchClasses();
    setRefreshing(false);
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Please grant camera roll permissions to use this feature.');
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets[0] && result.assets[0].uri) {
      const imageUri = result.assets[0].uri;
      setCurrentImage(imageUri);
      setLabels([]);
      
      // Get image dimensions with error handling
      Image.getSize(
        imageUri, 
        (width, height) => {
          setImageSize({ width, height });
        },
        (error) => {
          console.error('Error getting image size:', error);
          Alert.alert('Error', 'Failed to load image dimensions');
        }
      );
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Please grant camera permissions to use this feature.');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled && result.assets[0] && result.assets[0].uri) {
      const imageUri = result.assets[0].uri;
      setCurrentImage(imageUri);
      setLabels([]);
      
      // Get image dimensions with error handling
      Image.getSize(
        imageUri, 
        (width, height) => {
          setImageSize({ width, height });
        },
        (error) => {
          console.error('Error getting camera image size:', error);
          Alert.alert('Error', 'Failed to load camera image dimensions');
        }
      );
    }
  };

  const addBoundingBox = (x: number, y: number, width: number, height: number) => {
    if (!currentLabel || width < 20 || height < 20) return;

    const newLabel: BoundingBox = {
      id: Math.random().toString(36).substr(2, 9),
      x: Math.max(0, x),
      y: Math.max(0, y),
      width: Math.min(width, imageSize.width - x),
      height: Math.min(height, imageSize.height - y),
      label: currentLabel,
      color: LABEL_COLORS[labels.length % LABEL_COLORS.length],
    };

    addLabel(newLabel);
  };

  const gestureHandler = useAnimatedGestureHandler<PanGestureHandlerGestureEvent>({
    onStart: (event) => {
      startX.value = event.x;
      startY.value = event.y;
      currentX.value = event.x;
      currentY.value = event.y;
      isActive.value = true;
      runOnJS(setIsDrawing)(true);
    },
    onActive: (event) => {
      currentX.value = event.x;
      currentY.value = event.y;
    },
    onEnd: () => {
      const x = Math.min(startX.value, currentX.value);
      const y = Math.min(startY.value, currentY.value);
      const width = Math.abs(currentX.value - startX.value);
      const height = Math.abs(currentY.value - startY.value);
      
      runOnJS(addBoundingBox)(x, y, width, height);
      runOnJS(setIsDrawing)(false);
      isActive.value = false;
    },
  });

  const drawingBoxStyle = useAnimatedStyle(() => {
    if (!isActive.value) return { display: 'none' };
    
    const x = Math.min(startX.value, currentX.value);
    const y = Math.min(startY.value, currentY.value);
    const width = Math.abs(currentX.value - startX.value);
    const height = Math.abs(currentY.value - startY.value);

    return {
      position: 'absolute',
      left: x,
      top: y,
      width,
      height,
      borderWidth: 2,
      borderColor: '#3B82F6',
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
    };
  });

  const submitLabels = async () => {
    if (!currentImage || labels.length === 0) {
      Alert.alert('Error', 'Please add at least one label before submitting.');
      return;
    }

    try {
      // Convert image to blob
      const response = await fetch(currentImage);
      const imageBlob = await response.blob();

      // Prepare labeling data
      const labelingData = {
        boxes: labels.map((label) => ({
          x1: label.x,
          y1: label.y,
          x2: label.x + label.width,
          y2: label.y + label.height,
          label: label.label,
        })),
        image_width: imageSize.width,
        image_height: imageSize.height,
      };

      submitLabelingMutation.mutate({ imageBlob, labelingData });
    } catch (error) {
      Alert.alert('Error', 'Failed to process image');
    }
  };

  const renderImageSelector = () => (
    <View className={cn(
      'mx-4 mb-4 p-4 rounded-xl',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      <Text className={cn(
        'text-lg font-semibold mb-3',
        isDarkMode ? 'text-white' : 'text-gray-900'
      )}>
        Select Image
      </Text>

      <View className="flex-row space-x-3">
        <TouchableOpacity
          onPress={pickImage}
          className="flex-1 py-3 rounded-lg bg-blue-500 dark:bg-blue-600 flex-row items-center justify-center"
        >
          <Ionicons name="images" size={16} color="white" style={{ marginRight: 8 }} />
          <Text className="text-white font-medium">Gallery</Text>
        </TouchableOpacity>

        <TouchableOpacity
          onPress={takePhoto}
          className="flex-1 py-3 rounded-lg bg-green-500 dark:bg-green-600 flex-row items-center justify-center"
        >
          <Ionicons name="camera" size={16} color="white" style={{ marginRight: 8 }} />
          <Text className="text-white font-medium">Camera</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderLabelSelector = () => (
    <View className={cn(
      'mx-4 mb-4 p-4 rounded-xl',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      <Text className={cn(
        'text-lg font-semibold mb-3',
        isDarkMode ? 'text-white' : 'text-gray-900'
      )}>
        Current Label: {currentLabel || 'None'}
      </Text>

      {availableLabels.length === 0 ? (
        <Text className={cn(
          'text-center py-4',
          isDarkMode ? 'text-gray-400' : 'text-gray-500'
        )}>
          No labels available. Please add classes in the Detection tab first.
        </Text>
      ) : (
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <View className="flex-row space-x-2">
            {availableLabels.map((label, index) => (
              <TouchableOpacity
                key={label}
                onPress={() => setCurrentLabel(label)}
                className={cn(
                  'px-4 py-2 rounded-lg',
                  currentLabel === label
                    ? 'bg-blue-500 dark:bg-blue-600'
                    : 'bg-gray-200 dark:bg-gray-600'
                )}
              >
                <Text className={cn(
                  'font-medium',
                  currentLabel === label
                    ? 'text-white'
                    : isDarkMode ? 'text-gray-200' : 'text-gray-700'
                )}>
                  {label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </ScrollView>
      )}
    </View>
  );

  const renderImageEditor = () => {
    if (!currentImage) return null;

    const imageAspectRatio = imageSize.width / imageSize.height;
    const containerWidth = screenWidth - 32;
    const containerHeight = Math.min(containerWidth / imageAspectRatio, screenHeight * 0.5);
    const displayWidth = containerWidth;
    const displayHeight = containerHeight;

    return (
      <View className={cn(
        'mx-4 mb-4 p-4 rounded-xl',
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      )}>
        <View className="flex-row items-center justify-between mb-3">
          <Text className={cn(
            'text-lg font-semibold',
            isDarkMode ? 'text-white' : 'text-gray-900'
          )}>
            Label Image
          </Text>
          <Text className={cn(
            'text-sm',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            {labels.length} labels
          </Text>
        </View>

        <View style={{ width: displayWidth, height: displayHeight }}>
          <PanGestureHandler onGestureEvent={gestureHandler}>
            <Animated.View style={{ width: '100%', height: '100%' }}>
              <Image
                source={{ uri: currentImage }}
                style={{ width: '100%', height: '100%' }}
                contentFit="contain"
              />
              
              {/* Existing bounding boxes */}
              {labels.map((label, index) => (
                <View
                  key={label.id}
                  style={{
                    position: 'absolute',
                    left: (label.x / imageSize.width) * displayWidth,
                    top: (label.y / imageSize.height) * displayHeight,
                    width: (label.width / imageSize.width) * displayWidth,
                    height: (label.height / imageSize.height) * displayHeight,
                    borderWidth: 2,
                    borderColor: label.color,
                    backgroundColor: `${label.color}20`,
                  }}
                >
                  <View
                    style={{
                      position: 'absolute',
                      top: -24,
                      left: 0,
                      backgroundColor: label.color,
                      paddingHorizontal: 6,
                      paddingVertical: 2,
                      borderRadius: 4,
                    }}
                  >
                    <Text style={{ color: 'white', fontSize: 12, fontWeight: 'bold' }}>
                      {label.label}
                    </Text>
                  </View>
                  
                  <TouchableOpacity
                    onPress={() => removeLabel(index)}
                    style={{
                      position: 'absolute',
                      top: -12,
                      right: -12,
                      width: 24,
                      height: 24,
                      borderRadius: 12,
                      backgroundColor: '#EF4444',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Ionicons name="close" size={12} color="white" />
                  </TouchableOpacity>
                </View>
              ))}
              
              {/* Drawing box */}
              <Animated.View style={drawingBoxStyle} />
            </Animated.View>
          </PanGestureHandler>
        </View>

        <View className="flex-row mt-4 space-x-3">
          <TouchableOpacity
            onPress={() => setLabels([])}
            className="px-4 py-2 rounded-lg bg-red-500 dark:bg-red-600"
          >
            <Text className="text-white font-medium">Clear All</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={submitLabels}
            disabled={labels.length === 0 || submitLabelingMutation.isPending}
            className={cn(
              'flex-1 py-2 rounded-lg flex-row items-center justify-center',
              labels.length === 0 || submitLabelingMutation.isPending
                ? 'bg-gray-300 dark:bg-gray-600'
                : 'bg-green-500 dark:bg-green-600'
            )}
          >
            <Ionicons
              name="checkmark"
              size={16}
              color="white"
              style={{ marginRight: 8 }}
            />
            <Text className="text-white font-medium">
              {submitLabelingMutation.isPending ? 'Submitting...' : 'Submit Labels'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  };

  const renderInstructions = () => (
    <View className={cn(
      'mx-4 mb-4 p-4 rounded-xl',
      isDarkMode ? 'bg-gray-800' : 'bg-white'
    )}>
      <Text className={cn(
        'text-lg font-semibold mb-3',
        isDarkMode ? 'text-white' : 'text-gray-900'
      )}>
        How to Label
      </Text>

      <View className="space-y-2">
        <View className="flex-row items-start">
          <View className="w-6 h-6 rounded-full bg-blue-500 items-center justify-center mr-3 mt-0.5">
            <Text className="text-white text-xs font-bold">1</Text>
          </View>
          <Text className={cn(
            'flex-1 text-sm',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            Select an image from gallery or take a photo
          </Text>
        </View>

        <View className="flex-row items-start">
          <View className="w-6 h-6 rounded-full bg-blue-500 items-center justify-center mr-3 mt-0.5">
            <Text className="text-white text-xs font-bold">2</Text>
          </View>
          <Text className={cn(
            'flex-1 text-sm',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            Choose a label from the available classes
          </Text>
        </View>

        <View className="flex-row items-start">
          <View className="w-6 h-6 rounded-full bg-blue-500 items-center justify-center mr-3 mt-0.5">
            <Text className="text-white text-xs font-bold">3</Text>
          </View>
          <Text className={cn(
            'flex-1 text-sm',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            Drag to draw bounding boxes around objects
          </Text>
        </View>

        <View className="flex-row items-start">
          <View className="w-6 h-6 rounded-full bg-blue-500 items-center justify-center mr-3 mt-0.5">
            <Text className="text-white text-xs font-bold">4</Text>
          </View>
          <Text className={cn(
            'flex-1 text-sm',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            Tap the X button to remove unwanted labels
          </Text>
        </View>

        <View className="flex-row items-start">
          <View className="w-6 h-6 rounded-full bg-blue-500 items-center justify-center mr-3 mt-0.5">
            <Text className="text-white text-xs font-bold">5</Text>
          </View>
          <Text className={cn(
            'flex-1 text-sm',
            isDarkMode ? 'text-gray-300' : 'text-gray-600'
          )}>
            Submit labels to add them to the training dataset
          </Text>
        </View>
      </View>
    </View>
  );

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
          {renderImageSelector()}
          {renderLabelSelector()}
          {renderImageEditor()}
          {renderInstructions()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}


