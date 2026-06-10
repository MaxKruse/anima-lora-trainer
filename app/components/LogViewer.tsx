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
  const userScrolledUpRef = useRef(false);

  // Filter lines by search term
  const filteredLines = useMemo(() => {
    if (!searchTerm.trim()) return lines;
    const lower = searchTerm.toLowerCase();
    return lines.filter((line) => line.toLowerCase().includes(lower));
  }, [lines, searchTerm]);

  // Check if user is near the bottom of the scroll container
  const isNearBottom = (el: HTMLDivElement): boolean => {
    const threshold = 50; // px from bottom
    return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  };

  // Track manual scroll position — if user scrolls up, stop auto-scrolling
  const handleScroll = () => {
    if (containerRef.current) {
      userScrolledUpRef.current = !isNearBottom(containerRef.current);
    }
  };

  // Auto-scroll to latest line only when user hasn't scrolled up
  useEffect(() => {
    if (autoScroll && containerRef.current && !userScrolledUpRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [filteredLines.length, autoScroll]);

  const getLineClass = (line: string): string => {
    const lower = line.toLowerCase();
    if (lower.includes('error') || lower.includes('failed') || lower.includes('fail')) {
      return 'text-red-600 dark:text-red-400';
    }
    if (lower.includes('warning') || lower.includes('warn')) {
      return 'text-yellow-600 dark:text-yellow-400';
    }
    return 'text-slate-700 dark:text-slate-300';
  };

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden">
      {/* Search input */}
      <div className="flex items-center justify-between gap-2 p-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900">
        <input
          type="search"
          role="searchbox"
          placeholder="Search logs..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="flex-1 px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded bg-white dark:bg-slate-800 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500 placeholder-slate-400 dark:placeholder-slate-500"
        />
        <span className="text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
          {filteredLines.length} / {lines.length} lines
        </span>
      </div>

      {/* Log lines container */}
      <div
        ref={containerRef}
        role="log"
        onScroll={handleScroll}
        className="h-96 overflow-y-auto p-3 font-mono text-xs bg-slate-50 dark:bg-slate-900"
        data-auto-scroll={autoScroll ? 'true' : 'false'}
      >
        {filteredLines.map((line, index) => (
          <div key={index} className={`leading-relaxed ${getLineClass(line)}`}>
            {line || '\u00A0'}
          </div>
        ))}

        {filteredLines.length === 0 && (
          <div className="text-slate-400 dark:text-slate-500 py-4 text-center">No matching lines</div>
        )}
      </div>
    </div>
  );
}
