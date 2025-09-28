import Constants from 'expo-constants';
import { Platform } from 'react-native';

const extra = Constants.expoConfig?.extra ?? {};

const required = (key: string): string => {
  const value = extra[key];
  if (typeof value !== 'string' || value === '') {
    console.error(`Environment variable ${key} is not set!!`);
    return 'http://localhost:8000';
  }
  return value;
};

const webEnv =
  Platform.OS === 'web'
    ? {
        API_ENDPOINT: process.env.EXPO_PUBLIC_API_ENDPOINT || 'http://localhost:8000',
      }
    : null;

const env =
  Platform.OS === 'web'
    ? webEnv
    : {
        API_ENDPOINT: required('API_ENDPOINT'),
      };

export default env;
