import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

async function importMultiSelectDropdown() {
  const mod = await import('../MultiSelectDropdown');
  return mod.MultiSelectDropdown;
}

describe('MultiSelectDropdown', () => {
  const mockOnChange = vi.fn();
  const PRESETS = ['option-a', 'option-b', 'option-c'];

  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  afterEach(() => {
    cleanup();
  });

  it('renders with label and input field', async () => {
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={[]}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('Test Param')).toBeInTheDocument();
    const input = screen.getByPlaceholderText('Type to add...');
    expect(input).toBeInTheDocument();
  });

  it('shows selected values as tags', async () => {
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={['option-a', 'custom-val']}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    // Tags are rendered as spans with the value text
    const allTexts = screen.getAllByText('option-a');
    expect(allTexts.length).toBeGreaterThan(0);
    expect(screen.getByText('custom-val')).toBeInTheDocument();
  });

  it('shows preset options when focused', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={[]}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);

    await waitFor(() => {
      // Dropdown buttons appear for presets
      const dropdown = document.querySelector('[class*="absolute"]');
      expect(dropdown).toBeInTheDocument();
    });
  });

  it('filters presets by typed text', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={[]}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);
    await user.type(input, 'option-b');

    await waitFor(() => {
      // Only option-b should be in the dropdown (option-a and option-c filtered out)
      const dropdownButtons = document.querySelectorAll('button[type="button"]');
      const dropdownTexts = Array.from(dropdownButtons).map(b => b.textContent);
      expect(dropdownTexts.some(t => t?.includes('option-b'))).toBe(true);
    });
  });

  it('adds preset on click', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={[]}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);

    await waitFor(() => {
      const dropdown = document.querySelector('[class*="absolute"]');
      expect(dropdown).toBeInTheDocument();
    });

    // Click the first preset option button in the dropdown
    const dropdownButtons = document.querySelectorAll('button[type="button"]');
    await user.click(dropdownButtons[0]);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(['option-a']);
    });
  });

  it('adds custom value on enter key', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={[]}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);
    await user.type(input, 'my-custom-value{enter}');

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(['my-custom-value']);
    });
  });

  it('removes value on tag delete click', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    const initialValue = ['option-a', 'option-b'];
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={initialValue}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    // Click the delete button (×) on the first tag
    const deleteButtons = screen.getAllByText('×');
    await user.click(deleteButtons[0]);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(['option-b']);
    });
  });

  it('does not add duplicate values', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={['option-a']}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);
    await user.type(input, 'option-a{enter}');

    // Should not call onChange because it's a duplicate
    expect(mockOnChange).not.toHaveBeenCalled();
  });

  it('shows add-new option when typing custom text', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={[]}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);
    await user.type(input, 'brand-new');

    await waitFor(() => {
      expect(screen.getByText(/add/i)).toBeInTheDocument();
    });
  });

  it('handles backspace to remove last tag when input is empty', async () => {
    const user = userEvent.setup();
    const MultiSelectDropdown = await importMultiSelectDropdown();
    render(
      <MultiSelectDropdown
        label="Test Param"
        value={['option-a', 'option-b']}
        presets={PRESETS}
        onChange={mockOnChange}
      />
    );

    const input = screen.getByPlaceholderText('Type to add...');
    await user.click(input);
    await user.keyboard('{Backspace}');

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(['option-a']);
    });
  });
});
