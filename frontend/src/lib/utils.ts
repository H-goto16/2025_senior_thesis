import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number, decimals = 2) {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

export function formatDuration(seconds: number) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}

export function formatPercentage(value: number, decimals = 1) {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;

  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;

  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

export function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

export function validateEmail(email: string) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

export function validateImageFile(file: File) {
  const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
  const maxSize = 10 * 1024 * 1024; // 10MB

  if (!allowedTypes.includes(file.type)) {
    throw new Error('Invalid file type. Please select a JPEG, PNG, or WebP image.');
  }

  if (file.size > maxSize) {
    throw new Error('File size too large. Please select an image smaller than 10MB.');
  }

  return true;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function copyToClipboard(text: string) {
  if (navigator.clipboard) {
    return navigator.clipboard.writeText(text);
  } else {
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    document.execCommand('copy');
    document.body.removeChild(textArea);
    return Promise.resolve();
  }
}

export function isValidUrl(string: string) {
  try {
    new URL(string);
    return true;
  } catch (_) {
    return false;
  }
}

export function truncateText(text: string, maxLength: number) {
  if (text.length <= maxLength) return text;
  return text.substr(0, maxLength) + '...';
}

export function capitalizeFirst(str: string) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function camelToKebab(str: string) {
  return str.replace(/([a-z0-9]|(?=[A-Z]))([A-Z])/g, '$1-$2').toLowerCase();
}

export function kebabToCamel(str: string) {
  return str.replace(/-([a-z])/g, (g) => g[1].toUpperCase());
}

// Color utilities for charts and visualizations
export const chartColors = {
  primary: '#3B82F6',
  secondary: '#10B981',
  accent: '#F59E0B',
  danger: '#EF4444',
  warning: '#F59E0B',
  info: '#06B6D4',
  success: '#10B981',
  muted: '#6B7280',
};

export const darkChartColors = {
  primary: '#60A5FA',
  secondary: '#34D399',
  accent: '#FBBF24',
  danger: '#F87171',
  warning: '#FBBF24',
  info: '#22D3EE',
  success: '#34D399',
  muted: '#9CA3AF',
};

export function getChartColor(colorName: keyof typeof chartColors, isDark = false) {
  return isDark ? darkChartColors[colorName] : chartColors[colorName];
}

export function generateColorPalette(count: number, isDark = false) {
  const colors = isDark ? Object.values(darkChartColors) : Object.values(chartColors);
  const palette = [];

  for (let i = 0; i < count; i++) {
    palette.push(colors[i % colors.length]);
  }

  return palette;
}