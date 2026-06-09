'use client';

import { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  errorMessage?: string;
  onRetry?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Global error boundary that catches render errors and shows a fallback UI.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleRetry = (): void => {
    this.setState({ hasError: false, error: null });
    this.props.onRetry?.();
  };

  render(): ReactNode {
    if (this.state.hasError || this.props.errorMessage) {
      const message = this.props.errorMessage || this.state.error?.message || 'An unexpected error occurred';

      return (
        <div className="flex items-center justify-center min-h-[200px] p-6">
          <div className="text-center border border-red-200 dark:border-red-800 rounded-lg bg-red-50 dark:bg-red-900/20 p-6 max-w-md">
            <h2 className="text-lg font-semibold text-red-800 dark:text-red-400 mb-2">
              Something went wrong
            </h2>
            <p className="text-sm text-red-700 dark:text-red-400 mb-4">{message}</p>
            {this.props.onRetry && (
              <button
                onClick={this.handleRetry}
                className="px-4 py-2 text-sm bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
