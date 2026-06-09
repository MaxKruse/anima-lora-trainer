'use client';

import { useState, useMemo, useRef, useEffect } from 'react';

interface LogViewerProps {
  lines: string[];
  autoScroll?: boolean;
}

/**
 * Scrollable, searchable log viewer panel.
 */
export function LogViewer({ lines, autoScroll = true }: LogViewerProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter lines by search term
  const filteredLines = useMemo(() => {
    if (!searchTerm.trim()) return lines;
    const lower = searchTerm.toLowerCase();
    return lines.filter((line) => line.toLowerCase().includes(lower));
  }, [lines, searchTerm]);

  // Auto-scroll to latest line
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [filteredLines.length, autoScroll]);

  const getLineClass = (line: string): string => {
    const lower = line.toLowerCase();
    if (lower.includes('error') || lower.includes('failed') || lower.includes('fail')) {
      return 'log-line-error';
    }
    if (lower.includes('warning') || lower.includes('warn')) {
      return 'log-line-warning';
    }
    return 'log-line-info';
  };

  return (
    <div className="log-viewer">
      {/* Search input */}
      <div className="log-search">
        <input
          type="search"
          role="searchbox"
          placeholder="Search logs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <span className="log-count">
          {filteredLines.length} / {lines.length} lines
        </span>
      </div>

      {/* Log lines container */}
      <div
        ref={containerRef}
        role="log"
        className="log-container"
        data-auto-scroll={autoScroll ? 'true' : 'false'}
      >
        {filteredLines.map((line, index) => (
          <div key={index} className={`log-line ${getLineClass(line)}`}>
            {line || '\u00A0'}
          </div>
        ))}

        {filteredLines.length === 0 && (
          <div className="log-empty">No matching lines</div>
        )}
      </div>
    </div>
  );
}
