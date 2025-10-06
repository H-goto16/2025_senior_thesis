import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { CameraButton, CaptureButton, IconButton } from "@/components/ui/CameraButton";
import { DetectionCard } from "@/components/ui/DetectionCard";
import { PlatformAlert } from "@/components/ui/PlatformAlert";
import { StyledTextInput } from "@/components/ui/StyledTextInput";
import env from "@/env";
import { useLabeling } from "@/lib/store";
import { type CameraType, CameraView, useCameraPermissions } from "expo-camera";
import * as FileSystem from "expo-file-system";
import * as ImagePicker from "expo-image-picker";
import { useRouter } from "expo-router";
import { useRef, useState } from "react";
import { ActivityIndicator, Image, Platform, ScrollView, StyleSheet, View, useColorScheme } from "react-native";

interface Detection {
  class: string;
  confidence: number;
  bbox: number[];
}

interface DetectionResponse {
  detections: Detection[];
  message: string;
  processed_image: string;
}

const HomeScreen = () => {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [detectionResults, setDetectionResults] = useState<DetectionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [currentClasses, setCurrentClasses] = useState<string[]>([]);
  const cameraRef = useRef<CameraView>(null);
  const colorScheme = useColorScheme();
  const isDark = colorScheme === 'dark';
  const router = useRouter();
  const { setCurrentImage } = useLabeling();

  const fetchCurrentClasses = async () => {
    try {
      const response = await fetch(env?.API_ENDPOINT +"/model/classes");
      if (response.ok) {
        const data = await response.json();
        setCurrentClasses(data.classes || []);
        return data.classes || [];
      }
    } catch (error) {
      console.error('Error fetching classes:', error);
    }
    return [];
  };

  const addNewClass = async (className: string) => {
    try {
      const response = await fetch(env?.API_ENDPOINT +"/model/classes", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          classes: [className.trim()]
        }),
      });

      if (response.ok) {
        console.log(`Class "${className}" added successfully`);
        PlatformAlert.success("Success", `Class "${className}" has been added`);
        setNewClassName('');
        await fetchCurrentClasses();
        return true;
      } else {
        const errorData = await response.json();
        console.error('Failed to add class:', errorData);
        PlatformAlert.error("Error", errorData.detail || "Failed to add class");
        return false;
      }
    } catch (error) {
      console.error('Error adding class:', error);
      PlatformAlert.error("Error", "Failed to add class");
      return false;
    }
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: true,
          imageType: 'jpg',
        });

        if (photo) {
          setCapturedImage(photo.uri);
        }
      } catch (error) {
        PlatformAlert.error("Error", "Failed to take picture");
        console.error(error);
      }
    }
  };

  const selectFromGallery = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
        base64: true,
      });

      if (!result.canceled && result.assets[0]) {
        setCapturedImage(result.assets[0].uri);
      }
    } catch (error) {
      PlatformAlert.error("Error", "Failed to select image");
      console.error(error);
    }
  };

  const uploadImageToAPI = async (imageUri: string, base64?: string) => {
    console.log('Uploading image to API...', { imageUri });
    setIsLoading(true);
    setDetectionResults(null);

    try {
      let result: DetectionResponse;

      if (Platform.OS === 'web') {
        console.log('Using fetch for web platform');
        const formData = new FormData();

        if (base64) {
          const response = await fetch(`data:image/jpeg;base64,${base64}`);
          const blob = await response.blob();
          formData.append('image', blob, 'image.jpg');
        } else {
          const response = await fetch(imageUri);
          const blob = await response.blob();
          formData.append('image', blob, 'image.jpg');
        }

        const uploadResponse = await fetch(env?.API_ENDPOINT + "/detect", {
          method: 'POST',
          body: formData,
        });

        if (!uploadResponse.ok) {
          const errorText = await uploadResponse.text();
          throw new Error(`Upload failed with status ${uploadResponse.status}: ${errorText}`);
        }

        result = await uploadResponse.json();
      } else {
        console.log('Using FileSystem.uploadAsync for native platform');
        const uploadResult = await FileSystem.uploadAsync(
          env?.API_ENDPOINT + "/detect",
          imageUri,
          {
            fieldName: 'image',
            httpMethod: 'POST',
            uploadType: FileSystem.FileSystemUploadType.MULTIPART,
          }
        );

        console.log('Upload result:', uploadResult);

        if (uploadResult.status !== 200) {
          console.error('Upload failed:', uploadResult.body);
          throw new Error(`Upload failed with status ${uploadResult.status}: ${uploadResult.body}`);
        }

        result = JSON.parse(uploadResult.body);
      }

      console.log('Detection successful:', result);
      setDetectionResults(result);

      if (result.detections.length > 0) {
        PlatformAlert.success("Detection Complete", `${result.detections.length} objects detected`);
      } else {
        PlatformAlert.success("Detection Result", "No objects detected");
      }
    } catch (error) {
      console.error("Upload error:", error);
      let errorMessage = "Failed to analyze image";

      if (error instanceof Error) {
        errorMessage = error.message;
      }

      PlatformAlert.error("Error", errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddClass = async () => {
    if (newClassName.trim()) {
      const success = await addNewClass(newClassName);
      if (success && capturedImage) {
        await uploadImageToAPI(capturedImage);
      }
    } else {
      PlatformAlert.error("Error", "Please enter a class name");
    }
  };

  const navigateToLabeling = async () => {
    if (!capturedImage) return;

    const classes = await fetchCurrentClasses();

    // Store the current image in global store to avoid huge data URLs in query params
    setCurrentImage(capturedImage);

    router.push({
      pathname: '/manual-labeling',
      params: {
        existingClasses: JSON.stringify(classes)
      }
    });
  };

  const resetToCamera = () => {
    setCapturedImage(null);
    setDetectionResults(null);
    setNewClassName('');
  };

  const getNoDetectionCardStyle = () => ({
    backgroundColor: isDark ? 'rgba(239, 68, 68, 0.2)' : 'rgba(239, 68, 68, 0.1)',
    borderLeftColor: '#EF4444',
  });

  if (!permission) {
    return (
      <ThemedView style={styles.permissionContainer}>
        <ThemedText style={styles.permissionText}>
          Checking camera permission...
        </ThemedText>
      </ThemedView>
    );
  }

  if (!permission.granted) {
    return (
      <ThemedView style={styles.permissionContainer}>
        <ThemedText style={styles.permissionTitle}>Camera Access Required</ThemedText>
        <ThemedText style={styles.permissionText}>
          This app needs camera permission to take photos for object detection.
        </ThemedText>
        <CameraButton
          variant="primary"
          size="large"
          icon="camera"
          onPress={requestPermission}
          style={styles.permissionButton}
        >
          Grant Permission
        </CameraButton>
      </ThemedView>
    );
  }

  if (capturedImage) {
    console.log('Rendering captured image view:', {
      capturedImage: !!capturedImage,
      detectionResults: !!detectionResults,
      detectionsCount: detectionResults?.detections.length || 0,
      hasProcessedImage: !!detectionResults?.processed_image
    });

    return (
      <ThemedView style={styles.container}>
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <ThemedText style={styles.title}>
            {detectionResults ? 'Detection Results' : 'Captured Image'}
          </ThemedText>

          <View style={styles.imageContainer}>
            <Image
              source={{
                uri: detectionResults?.processed_image
                  ? `data:image/jpeg;base64,${detectionResults.processed_image}`
                  : capturedImage
              }}
              style={[styles.resultImage, { backgroundColor: isDark ? '#374151' : '#F3F4F6' }]}
              resizeMode="contain"
            />
          </View>

          {detectionResults && (
            <ThemedView style={styles.resultsSection}>
              <ThemedText style={styles.resultMessage}>
                {detectionResults.message}
              </ThemedText>

              {detectionResults.detections.length > 0 ? (
                <ThemedView style={styles.detectionsContainer}>
                  {detectionResults.detections.map((detection, index) => (
                    <DetectionCard
                      key={index}
                      detection={detection}
                      index={index}
                    />
                  ))}
                </ThemedView>
              ) : (
                <ThemedView style={[styles.noDetectionCard, getNoDetectionCardStyle()]}>
                  <ThemedText style={[styles.noDetectionText, { color: isDark ? '#F87171' : '#EF4444' }]}>
                    No objects detected in this image
                  </ThemedText>
                </ThemedView>
              )}
            </ThemedView>
          )}

          <ThemedView style={styles.buttonGroup}>
            <CameraButton
              variant="secondary"
              icon="camera"
              onPress={resetToCamera}
              size="medium"
              style={styles.actionButton}
            >
              New Photo
            </CameraButton>
            <CameraButton
              variant={detectionResults ? "primary" : "success"}
              icon="scan"
              onPress={() => uploadImageToAPI(capturedImage)}
              size="medium"
              style={styles.actionButton}
              disabled={isLoading}
            >
              {detectionResults ? 'Detect Again' : 'Detect'}
            </CameraButton>
          </ThemedView>

          <ThemedView style={styles.labelingButtonContainer}>
            <CameraButton
              variant="secondary"
              icon="create"
              onPress={navigateToLabeling}
              size="medium"
              disabled={isLoading}
            >
              {"Manual Labeling"}
            </CameraButton>
            <ThemedText style={styles.labelingHelpText}>
              Can't detect what you need? Create manual labels to improve the model
            </ThemedText>
          </ThemedView>

          <ThemedView style={styles.addClassSection}>
            <StyledTextInput
              label="Add New Detection Class"
              placeholder="Enter class name (e.g., 'apple', 'car')"
              value={newClassName}
              onChangeText={setNewClassName}
            />
            <CameraButton
              variant="purple"
              icon="add-circle"
              onPress={handleAddClass}
              disabled={!newClassName.trim() || isLoading}
              size="medium"
            >
              {"Add Class & Retrain"}
            </CameraButton>
          </ThemedView>
        </ScrollView>
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <CameraView
        style={styles.camera}
        facing={facing}
        ref={cameraRef}
      >
        <View style={styles.cameraControls}>
          <IconButton
            icon="camera-reverse"
            onPress={() => setFacing(current => (current === 'back' ? 'front' : 'back'))}
            style={styles.flipButton}
          />

          <CaptureButton
            onPress={takePicture}
            disabled={isLoading}
          />

          <IconButton
            icon="images"
            onPress={selectFromGallery}
            style={styles.galleryButton}
          />
        </View>
      </CameraView>

      {isLoading && (
        <ThemedView style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#ffffff" />
          <ThemedText style={styles.loadingText}>Analyzing image...</ThemedText>
        </ThemedView>
      )}

      {detectionResults && (
        <ThemedView style={styles.resultsOverlay}>
          <ScrollView style={styles.resultsScroll} showsVerticalScrollIndicator={false}>
            <ThemedText style={styles.overlayTitle}>Detection Results</ThemedText>
            <ThemedText style={styles.overlayMessage}>{detectionResults.message}</ThemedText>

            {detectionResults.detections.map((detection, index) => (
              <ThemedView key={index} style={styles.overlayCard}>
                <ThemedText style={styles.overlayClassName}>
                  {detection.class}
                </ThemedText>
                <ThemedText style={styles.overlayConfidence}>
                  Confidence: {(detection.confidence * 100).toFixed(1)}%
                </ThemedText>
                <ThemedText style={styles.overlayPosition}>
                  Position: [{detection.bbox.map(coord => Math.round(coord)).join(', ')}]
                </ThemedText>
              </ThemedView>
            ))}
          </ScrollView>
        </ThemedView>
      )}
    </ThemedView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  permissionText: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
    opacity: 0.8,
  },
  permissionButton: {
    minWidth: 200,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  resultImage: {
    width: '100%',
    height: 300,
    borderRadius: 16,
  },
  resultsSection: {
    marginBottom: 20,
  },
  resultMessage: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 16,
    fontStyle: 'italic',
    opacity: 0.8,
  },
  detectionsContainer: {
    marginBottom: 20,
  },
  noDetectionCard: {
    padding: 20,
    borderRadius: 12,
    borderLeftWidth: 4,
    alignItems: 'center',
  },
  noDetectionText: {
    fontSize: 16,
    fontWeight: '600',
  },
  buttonGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 30,
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
  addClassSection: {
    marginTop: 10,
  },
  camera: {
    flex: 1,
  },
  cameraControls: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    paddingBottom: 50,
    paddingHorizontal: 30,
  },
  flipButton: {
    position: 'absolute',
    left: 30,
  },
  galleryButton: {
    position: 'absolute',
    right: 30,
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
  resultsOverlay: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    maxHeight: '60%',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
  },
  resultsScroll: {
    flex: 1,
  },
  overlayTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: 'white',
    marginBottom: 8,
    textAlign: 'center',
  },
  overlayMessage: {
    fontSize: 16,
    color: 'white',
    marginBottom: 16,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  overlayCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    padding: 16,
    marginBottom: 12,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#3B82F6',
  },
  overlayClassName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#111827',
    marginBottom: 4,
  },
  overlayConfidence: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 4,
  },
  overlayPosition: {
    fontSize: 12,
    color: '#9CA3AF',
  },
  labelingButtonContainer: {
    marginHorizontal: 20,
    marginBottom: 20,
    alignItems: 'center',
  },
  labelingHelpText: {
    fontSize: 14,
    textAlign: 'center',
    marginTop: 8,
    opacity: 0.7,
    fontStyle: 'italic',
  },
});

export default HomeScreen;