import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DirectoryPicker } from '../DirectoryPicker';

// Mock fetch for verify functionality
const mockFetch = vi.fn();
global.fetch = mockFetch;

function createDirectoryPickerProps() {
  return {
    label: 'Test Directory',
    value: '',
    onChange: vi.fn(),
    id: 'test-dir',
  };
}

describe('DirectoryPicker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders label and input', () => {
    const props = createDirectoryPickerProps();
    render(<DirectoryPicker {...props} />);

    expect(screen.getByLabelText('Test Directory')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('/path/to/directory')).toBeInTheDocument();
  });

  it('calls onChange when input value changes', async () => {
    const onChange = vi.fn();
    render(
      <DirectoryPicker
        label="Test"
        value=""
        onChange={onChange}
        id="test"
      />
    );

    const input = screen.getByLabelText('Test');
    fireEvent.change(input, { target: { value: '/new/path' } });
    expect(onChange).toHaveBeenCalledWith('/new/path');
  });

  it('displays custom placeholder', () => {
    render(
      <DirectoryPicker
        label="Test"
        value=""
        onChange={() => {}}
        placeholder="/custom/placeholder"
        id="test"
      />
    );

    expect(screen.getByPlaceholderText('/custom/placeholder')).toBeInTheDocument();
  });

  it('displays hint text', () => {
    render(
      <DirectoryPicker
        label="Test"
        value=""
        onChange={() => {}}
        placeholder="/path"
        hint="This is a helpful hint"
        id="test"
      />
    );

    expect(screen.getByText('This is a helpful hint')).toBeInTheDocument();
  });

  it('displays error message', () => {
    render(
      <DirectoryPicker
        label="Test"
        value=""
        onChange={() => {}}
        error="This field is required"
        id="test"
      />
    );

    expect(screen.getByText('This field is required')).toBeInTheDocument();
  });

  it('shows clear button when value is set', () => {
    const onChange = vi.fn();
    render(
      <DirectoryPicker
        label="Test"
        value="/some/path"
        onChange={onChange}
        id="test"
      />
    );

    const clearButton = screen.getByTitle('Clear');
    expect(clearButton).toBeInTheDocument();

    fireEvent.click(clearButton);
    expect(onChange).toHaveBeenCalledWith('');
  });

  it('auto-verifies when value changes (autoVerify default)', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ exists: true }),
    });

    render(
      <DirectoryPicker
        label="Test"
        value="/existing/path"
        onChange={() => {}}
        id="test"
      />
    );

    // Auto-verify should fire on mount with existing value
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/config/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: '/existing/path' }),
      });
    });
  });

  it('does not auto-verify when autoVerify is false', async () => {
    render(
      <DirectoryPicker
        label="Test"
        value="/some/path"
        onChange={() => {}}
        id="test"
        autoVerify={false}
      />
    );

    // Wait a bit to ensure no fetch is called
    await new Promise((resolve) => setTimeout(resolve, 500));
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it('shows success message when directory exists (auto-verify)', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ exists: true }),
    });

    render(
      <DirectoryPicker
        label="Test"
        value="/existing/path"
        onChange={() => {}}
        id="test"
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Directory exists/)).toBeInTheDocument();
    });
  });

  it('shows warning when directory does not exist (auto-verify)', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ exists: false }),
    });

    render(
      <DirectoryPicker
        label="Test"
        value="/nonexistent/path"
        onChange={() => {}}
        id="test"
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Directory not found/)).toBeInTheDocument();
    });
  });
});
