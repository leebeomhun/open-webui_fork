import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.example.openwebui',
  appName: 'KCD Assistant',
  webDir: 'build',
  server: {
    url: 'http://192.168.25.33:8080',
    cleartext: false // HTTPS를 사용하므로 cleartext는 필요하지 않습니다.
  },
  ios: {
    contentInset: 'automatic'
  }
};

export default config;
