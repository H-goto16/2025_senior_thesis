import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { CameraButton } from '@/components/ui/CameraButton';
import { PlatformAlert } from '@/components/ui/PlatformAlert';
import { StyledTextInput } from '@/components/ui/StyledTextInput';
import env from '@/env';
import { useLabeling } from '@/lib/store';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useRef, useState } from 'react';
import {
    ActivityIndicator,
    Dimensions,
    Image,
    PanResponder,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    useColorScheme,
    View
} from 'react-native';

const { width: screenWidth } = Dimensions.get('window');

interface BoundingBox {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  color: string;
}

const LabelingScreen = () => {
  const params = useLocalSearchParams();
  const router = useRouter();
  const { currentImage } = useLabeling();
  const imageUri = (params.imageUri as string) || currentImage || '';
  const existingClassesParam = (params.existingClasses as string) || '';
  const existingClasses = existingClassesParam ? JSON.parse(existingClassesParam) : [];
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';

  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState({ x: 0, y: 0 });
  const [currentBox, setCurrentBox] = useState<BoundingBox | null>(null);
  const [newLabelName, setNewLabelName] = useState('');
  const [selectedLabel, setSelectedLabel] = useState('');
  const [imageLayout, setImageLayout] = useState({ width: 0, height: 0, x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState(false);

  const imageRef = useRef<Image>(null);

  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#FECA57', '#FF9FF3', '#A8E6CF', '#FFD93D'
  ];

  const getRandomColor = () => colors[Math.floor(Math.random() * colors.length)];

  const panResponder = PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,

    onPanResponderGrant: (event) => {
      const { locationX, locationY } = event.nativeEvent;
      setIsDrawing(true);
      setStartPoint({ x: locationX, y: locationY });

      const newId = Date.now().toString();
      const newBox: BoundingBox = {
        id: newId,
        x1: locationX,
        y1: locationY,
        x2: locationX,
        y2: locationY,
        label: '',
        color: getRandomColor()
      };
      setCurrentBox(newBox);
    },

    onPanResponderMove: (event) => {
      if (!isDrawing || !currentBox) return;

      const { locationX, locationY } = event.nativeEvent;
      const updatedBox = {
        ...currentBox,
        x2: locationX,
        y2: locationY
      };
      setCurrentBox(updatedBox);
    },

    onPanResponderRelease: () => {
      if (currentBox && isDrawing) {
        const width = Math.abs(currentBox.x2 - currentBox.x1);
        const height = Math.abs(currentBox.y2 - currentBox.y1);

        if (width > 10 && height > 10) {
          setBoundingBoxes(prev => [...prev, currentBox]);
        }
      }
      setIsDrawing(false);
      setCurrentBox(null);
    }
  });

  const getCurrentLabel = () => {
    return selectedLabel || newLabelName.trim();
  };

  const addLabelToBoundingBox = (boxId: string) => {
    const label = getCurrentLabel();
    if (!label) {
      PlatformAlert.error("Error", "Please enter or select a label");
      return;
    }

    setBoundingBoxes(prev =>
      prev.map(box =>
        box.id === boxId ? { ...box, label } : box
      )
    );
    setNewLabelName('');
    setSelectedLabel('');
  };

  const removeBoundingBox = (boxId: string) => {
    setBoundingBoxes(prev => prev.filter(box => box.id !== boxId));
  };

  const submitLabeling = async () => {
    if (boundingBoxes.length === 0) {
      PlatformAlert.error("Error", "Please add at least one bounding box");
      return;
    }

    const unlabeledBoxes = boundingBoxes.filter(box => !box.label);
    if (unlabeledBoxes.length > 0) {
      PlatformAlert.error("Error", "Please label all bounding boxes");
      return;
    }

    setIsLoading(true);

    try {
      if (!imageUri || imageUri === 'unknown') {
        PlatformAlert.error("Error", "Invalid image URI");
        setIsLoading(false);
        return;
      }

      Image.getSize(
        imageUri,
        async (originalWidth, originalHeight) => {
          const scaleX = originalWidth / imageLayout.width;
          const scaleY = originalHeight / imageLayout.height;

          const normalizedBoxes = boundingBoxes.map(box => ({
            label: box.label,
            x1: Math.min(box.x1, box.x2) * scaleX,
            y1: Math.min(box.y1, box.y2) * scaleY,
            x2: Math.max(box.x1, box.x2) * scaleX,
            y2: Math.max(box.y1, box.y2) * scaleY
          }));

          const formData = new FormData();
          const response = await fetch(imageUri);
          const blob = await response.blob();
          formData.append('image', blob, 'labeled_image.jpg');
          formData.append('labeling_data', JSON.stringify({
            boxes: normalizedBoxes,
            image_width: originalWidth,
            image_height: originalHeight
          }));

          const uploadResponse = await fetch(env?.API_ENDPOINT + "/labeling/submit", {
            method: 'POST',
            body: formData,
          });

          if (!uploadResponse.ok) {
            throw new Error(`Upload failed with status ${uploadResponse.status}`);
          }

          await uploadResponse.json();
          PlatformAlert.success("Success", "Labeling data submitted successfully! The model will be retrained.");
          router.back();
        },
        () => {
          PlatformAlert.error("Error", "Failed to get image dimensions");
          setIsLoading(false);
        }
      );
    } catch (error) {
      PlatformAlert.error("Error", "Failed to submit labeling data");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ThemedView style={styles.container}>
      <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
        <ThemedText style={styles.title}>Image Labeling</ThemedText>
        <ThemedText style={styles.instruction}>
          Tap and drag to create bounding boxes around objects
        </ThemedText>

        <View style={styles.imageContainer}>
          <View
            style={[styles.imageWrapper, { backgroundColor: isDark ? '#374151' : '#F3F4F6' }]}
            {...panResponder.panHandlers}
          >
            <Image
              ref={imageRef}
              source={{ uri: imageUri }}
              style={styles.image}
              resizeMode="contain"
              onLayout={(event) => {
                const { width, height, x, y } = event.nativeEvent.layout;
                setImageLayout({ width, height, x, y });
              }}
            />

            {currentBox && (
              <View
                style={[
                  styles.boundingBox,
                  {
                    left: Math.min(currentBox.x1, currentBox.x2),
                    top: Math.min(currentBox.y1, currentBox.y2),
                    width: Math.abs(currentBox.x2 - currentBox.x1),
                    height: Math.abs(currentBox.y2 - currentBox.y1),
                    borderColor: currentBox.color,
                  }
                ]}
              />
            )}

            {boundingBoxes.map((box) => (
              <View key={box.id}>
                <View
                  style={[
                    styles.boundingBox,
                    {
                      left: Math.min(box.x1, box.x2),
                      top: Math.min(box.y1, box.y2),
                      width: Math.abs(box.x2 - box.x1),
                      height: Math.abs(box.y2 - box.y1),
                      borderColor: box.color,
                    }
                  ]}
                />
                {box.label && (
                  <View
                    style={[
                      styles.labelContainer,
                      {
                        left: Math.min(box.x1, box.x2),
                        top: Math.min(box.y1, box.y2) - 25,
                        backgroundColor: box.color,
                      }
                    ]}
                  >
                    <Text style={styles.labelText}>{box.label}</Text>
                  </View>
                )}
                <TouchableOpacity
                  style={[
                    styles.removeButton,
                    {
                      left: Math.max(box.x1, box.x2) - 20,
                      top: Math.min(box.y1, box.y2),
                    }
                  ]}
                  onPress={() => removeBoundingBox(box.id)}
                >
                  <Text style={styles.removeButtonText}>×</Text>
                </TouchableOpacity>
              </View>
            ))}
          </View>
        </View>

        <ThemedView style={styles.labelingSection}>
          <ThemedText style={styles.sectionTitle}>Add Label</ThemedText>

          {existingClasses.length > 0 && (
            <View style={styles.existingClassesContainer}>
              <ThemedText style={styles.subTitle}>Select existing class:</ThemedText>
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                <View style={styles.classButtonsContainer}>
                  {existingClasses.map((className: string, index: number) => (
                    <TouchableOpacity
                      key={index}
                      style={[
                        styles.classButton,
                        selectedLabel === className && styles.selectedClassButton
                      ]}
                      onPress={() => {
                        setSelectedLabel(className);
                        setNewLabelName('');
                      }}
                    >
                      <Text style={[
                        styles.classButtonText,
                        selectedLabel === className && styles.selectedClassButtonText
                      ]}>
                        {className}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </ScrollView>
            </View>
          )}

          <StyledTextInput
            label="Or enter new class name"
            placeholder="Enter object class name"
            value={newLabelName}
            onChangeText={(text) => {
              setNewLabelName(text);
              if (text) setSelectedLabel('');
            }}
          />

          {boundingBoxes.length > 0 && (
            <View style={styles.boundingBoxesList}>
              <ThemedText style={styles.subTitle}>Bounding Boxes ({boundingBoxes.length})</ThemedText>
              {boundingBoxes.map((box, index) => (
                <View key={box.id} style={styles.boxListItem}>
                  <View style={[styles.colorIndicator, { backgroundColor: box.color }]} />
                  <ThemedText style={styles.boxLabel}>
                    Box {index + 1}: {box.label || 'Unlabeled'}
                  </ThemedText>
                  {!box.label && (
                    <TouchableOpacity
                      style={styles.addLabelButton}
                      onPress={() => addLabelToBoundingBox(box.id)}
                    >
                      <Text style={styles.addLabelButtonText}>Add Label</Text>
                    </TouchableOpacity>
                  )}
                </View>
              ))}
            </View>
          )}
        </ThemedView>

        <View style={styles.buttonGroup}>
          <CameraButton
            variant="secondary"
            icon="arrow-back"
            onPress={() => router.back()}
            size="medium"
            style={styles.actionButton}
          >
            {"Cancel"}
          </CameraButton>
          <CameraButton
            variant="success"
            icon="checkmark"
            onPress={submitLabeling}
            size="medium"
            style={styles.actionButton}
            disabled={isLoading || boundingBoxes.length === 0}
          >
            {isLoading ? "Submitting..." : "Submit"}
          </CameraButton>
        </View>
      </ScrollView>

      {isLoading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#ffffff" />
          <Text style={styles.loadingText}>Submitting labeling data...</Text>
        </View>
      )}
    </ThemedView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 20,
    marginBottom: 10,
  },
  instruction: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
    opacity: 0.8,
    paddingHorizontal: 20,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 20,
    paddingHorizontal: 10,
  },
  imageWrapper: {
    position: 'relative',
    width: screenWidth - 20,
    height: 300,
    borderRadius: 16,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  boundingBox: {
    position: 'absolute',
    borderWidth: 3,
    borderStyle: 'solid',
  },
  labelContainer: {
    position: 'absolute',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  labelText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  removeButton: {
    position: 'absolute',
    width: 20,
    height: 20,
    backgroundColor: 'red',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  labelingSection: {
    margin: 20,
    padding: 20,
    borderRadius: 12,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  subTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 8,
  },
  existingClassesContainer: {
    marginBottom: 20,
  },
  classButtonsContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  classButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#E5E7EB',
    borderRadius: 20,
    marginRight: 8,
  },
  selectedClassButton: {
    backgroundColor: '#3B82F6',
  },
  classButtonText: {
    fontSize: 14,
    color: '#374151',
  },
  selectedClassButtonText: {
    color: 'white',
  },
  boundingBoxesList: {
    marginTop: 20,
  },
  boxListItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  colorIndicator: {
    width: 16,
    height: 16,
    borderRadius: 8,
    marginRight: 12,
  },
  boxLabel: {
    flex: 1,
    fontSize: 14,
  },
  addLabelButton: {
    backgroundColor: '#3B82F6',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  addLabelButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  buttonGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    margin: 20,
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
    fontSize: 18,
    color: 'white',
    fontWeight: '600',
  },
});

export default LabelingScreen;


