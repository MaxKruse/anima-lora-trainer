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
        <div className="error-boundary">
          <div className="error-content">
            <h2 className="error-title">Something went wrong</h2>
            <p className="error-message">{message}</p>
            {this.props.onRetry && (
              <button onClick={this.handleRetry} className="retry-button">
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
