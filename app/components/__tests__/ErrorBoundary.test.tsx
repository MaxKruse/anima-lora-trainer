import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';

async function importErrorBoundary() {
  const mod = await import('../ErrorBoundary');
  return mod.ErrorBoundary;
}

// Component that throws during render
function BuggyComponent() {
  throw new Error('Something went wrong!');
}

describe('ErrorBoundary', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  it('catches render errors and shows fallback UI', async () => {
    const ErrorBoundary = await importErrorBoundary();

    render(
      <ErrorBoundary>
        <BuggyComponent />
      </ErrorBoundary>
    );

    await waitFor(() => {
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });
  });

  it('shows error message from API responses', async () => {
    const ErrorBoundary = await importErrorBoundary();

    render(
      <ErrorBoundary errorMessage="Failed to connect to training server">
        <div>Child content</div>
      </ErrorBoundary>
    );

    await waitFor(() => {
      expect(screen.getByText('Failed to connect to training server')).toBeInTheDocument();
    });
  });

  it('provides "retry" action where applicable', async () => {
    const onRetry = vi.fn();

    const ErrorBoundary = await importErrorBoundary();

    render(
      <ErrorBoundary onRetry={onRetry}>
        <BuggyComponent />
      </ErrorBoundary>
    );

    await waitFor(() => {
      const retryBtn = screen.getByRole('button', { name: /retry/i });
      expect(retryBtn).toBeInTheDocument();
      fireEvent.click(retryBtn);
    });

    expect(onRetry).toHaveBeenCalled();
  });
});
