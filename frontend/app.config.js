import 'dotenv/config';

export default {
  expo: {
    name: 'Dish Detection',
    slug: 'dish-detection',
    version: '1.0.0',
    orientation: 'portrait',
    icon: './src/assets/images/icon.png',
    scheme: 'dish-detection',
    userInterfaceStyle: 'automatic',
    newArchEnabled: true,
    ios: {
      supportsTablet: true,
      bundleIdentifier: 'com.dishdetection.app',
    },
    android: {
      adaptiveIcon: {
        foregroundImage: './src/assets/images/adaptive-icon.png',
        backgroundColor: '#ffffff',
      },
      package: 'com.dishdetection.app',
    },
    web: {
      bundler: 'metro',
      output: 'static',
      favicon: './src/assets/images/favicon.png',
    },
    plugins: [
      'expo-router',
      [
        'expo-camera',
        {
          cameraPermission: 'Allow $(PRODUCT_NAME) to access your camera',
          microphonePermission: 'Allow $(PRODUCT_NAME) to access your microphone',
          recordAudioAndroid: true,
        },
      ],
      [
        'expo-image-picker',
        {
          photosPermission: 'The app accesses your photos to let you select images for object detection and labeling.',
        },
      ],
      [
        'expo-file-system',
        {
          iCloudContainerEnvironment: 'Production',
        },
      ],
    ],
    experiments: {
      typedRoutes: true,
    },
    extra: {
      router: {
        origin: false,
      },
      eas: {
        projectId: 'your-project-id-here',
      },
      apiUrl: process.env.EXPO_PUBLIC_API_URL || 'http://localhost:8000',
    },
  },
};