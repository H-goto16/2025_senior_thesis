import { Ionicons } from '@expo/vector-icons';
import { Tabs } from 'expo-router';
import { Platform } from 'react-native';

import { HapticTab } from '@/components/HapticTab';
import TabBarBackground from '@/components/ui/TabBarBackground';
import { useTheme } from '@/lib/store';

export default function TabLayout() {
  const { isDarkMode } = useTheme();

  const tintColor = isDarkMode ? '#60A5FA' : '#3B82F6';
  const inactiveTintColor = isDarkMode ? '#9CA3AF' : '#6B7280';

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: tintColor,
        tabBarInactiveTintColor: inactiveTintColor,
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarBackground: TabBarBackground,
        tabBarStyle: {
          backgroundColor: isDarkMode ? '#1F2937' : '#FFFFFF',
          borderTopColor: isDarkMode ? '#374151' : '#E5E7EB',
          ...Platform.select({
            ios: {
              position: 'absolute',
            },
            default: {},
          }),
        },
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Detection',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="camera" size={size || 24} color={color} />
          ),
        }}
      />
      {/** Labeling tab removed per request */}
      <Tabs.Screen
        name="training"
        options={{
          title: 'Training',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="fitness" size={size || 24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="models"
        options={{
          title: 'Models',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="library" size={size || 24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="analytics"
        options={{
          title: 'Analytics',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="analytics" size={size || 24} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
