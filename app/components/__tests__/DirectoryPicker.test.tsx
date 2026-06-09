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
    // Reset picker support detection - delete the property entirely
    delete (window as any).showDirectoryPicker;
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

    // The clear button (✕) should be visible
    const clearButton = screen.getByTitle('Clear');
    expect(clearButton).toBeInTheDocument();

    fireEvent.click(clearButton);
    expect(onChange).toHaveBeenCalledWith('');
  });

  it('displays Check button', () => {
    render(
      <DirectoryPicker
        label="Test"
        value="/some/path"
        onChange={() => {}}
        id="test"
      />
    );

    const checkButton = screen.getByTitle('Check if directory exists');
    expect(checkButton).toBeInTheDocument();
  });

  it('Check button is disabled when value is empty', () => {
    render(
      <DirectoryPicker
        label="Test"
        value=""
        onChange={() => {}}
        id="test"
      />
    );

    const checkButton = screen.getByTitle('Check if directory exists');
    expect(checkButton).toBeDisabled();
  });

  it('calls verify API when Check is clicked', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ exists: true }),
    });

    const onChange = vi.fn();
    render(
      <DirectoryPicker
        label="Test"
        value="/existing/path"
        onChange={onChange}
        id="test"
      />
    );

    const checkButton = screen.getByTitle('Check if directory exists');
    fireEvent.click(checkButton);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/config/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: '/existing/path' }),
      });
    });
  });

  it('shows success message when directory exists', async () => {
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

    const checkButton = screen.getByTitle('Check if directory exists');
    fireEvent.click(checkButton);

    await waitFor(() => {
      expect(screen.getByText(/Directory exists/)).toBeInTheDocument();
    });
  });

  it('shows warning when directory does not exist', async () => {
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

    const checkButton = screen.getByTitle('Check if directory exists');
    fireEvent.click(checkButton);

    await waitFor(() => {
      expect(screen.getByText(/Directory not found/)).toBeInTheDocument();
    });
  });

  it('does not show Browse button when showDirectoryPicker is not supported', async () => {
    // Delete the property entirely so 'in window' check fails
    delete (window as any).showDirectoryPicker;

    render(
      <DirectoryPicker
        label="Test"
        value=""
        onChange={() => {}}
        id="test"
      />
    );

    // Wait for useEffect to run and detect support
    await waitFor(() => {
      const browseButtons = screen.queryAllByTitle('Open folder picker');
      expect(browseButtons).toHaveLength(0);
    });
  });
});
