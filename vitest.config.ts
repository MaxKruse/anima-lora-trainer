import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    include: ['app/**/*.test.ts', 'app/**/*.test.tsx', 'lib/**/*.test.ts', 'lib/**/*.test.tsx'],
    exclude: ['**/node_modules/**', '**/tests/**'],
    setupFiles: ['./vitest.setup.ts'],
  },
});
