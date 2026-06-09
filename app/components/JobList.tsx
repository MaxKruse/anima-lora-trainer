'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { LogViewer } from './LogViewer';

interface JobPermutation {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  params: Record<string, any>;
}

interface JobData {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  params: Record<string, any>;
  createdAt: string;
  updatedAt?: string;
  error?: string;
  permutations?: JobPermutation[];
}

interface JobProgress {
  status: string;
  currentEpoch: number;
  totalEpochs: number;
  currentStep: number;
  totalSteps: number;
  avgLoss: number | null;
  error: string | null;
  exitCode: number | null;
  outputFiles?: string[];
}

export function JobList() {
  const [jobs, setJobs] = useState<JobData[]>([]);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const [logsExpanded, setLogsExpanded] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [progressMap, setProgressMap] = useState<Record<string, JobProgress>>({});
  const [logLinesMap, setLogLinesMap] = useState<Record<string, string[]>>({});
  const [logsLoading, setLogsLoading] = useState<Record<string, boolean>>({});
  const [cancellingJobs, setCancellingJobs] = useState<Record<string, boolean>>({});
  const [deletingJobs, setDeletingJobs] = useState<Record<string, boolean>>({});

  // Refs for SSE EventSource connections
  const sseRefs = useRef<Record<string, EventSource | null>>({});

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch('/api/jobs');
      const data = await res.json();
      if (res.ok && data.jobs) {
        setJobs(data.jobs);
      }
    } catch {
      // Ignore fetch errors
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchProgress = useCallback(async (jobId: string) => {
    try {
      const res = await fetch(`/api/progress/${jobId}`);
      if (res.ok) {
        const data = await res.json();
        setProgressMap((prev) => ({ ...prev, [jobId]: data }));
      }
    } catch {
      // Ignore progress fetch errors
    }
  }, []);

  const fetchLogs = useCallback(async (jobId: string) => {
    setLogsLoading((prev) => ({ ...prev, [jobId]: true }));
    try {
      const res = await fetch(`/api/logs/${jobId}`);
      if (res.ok) {
        const data = await res.json();
        setLogLinesMap((prev) => ({ ...prev, [jobId]: data.lines || [] }));
      }
    } catch {
      // Ignore log fetch errors
    } finally {
      setLogsLoading((prev) => ({ ...prev, [jobId]: false }));
    }
  }, []);

  // SSE log streaming for expanded running jobs
  const connectSSE = useCallback((jobId: string) => {
    // Close existing connection if any
    if (sseRefs.current[jobId]) {
      sseRefs.current[jobId].close();
      sseRefs.current[jobId] = null;
    }

    // Load initial logs first
    fetchLogs(jobId);

    const es = new EventSource(`/api/logs/${jobId}?follow`);

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.chunk) {
          setLogLinesMap((prev) => {
            const existing = prev[jobId] || [];
            const chunkLines = data.chunk.split('\n');
            // If existing lines end with a partial line, merge it
            if (existing.length > 0 && chunkLines.length > 0) {
              chunkLines[0] = existing[existing.length - 1] + chunkLines[0];
              existing.pop();
            }
            // Remove trailing empty string from split
            if (chunkLines[chunkLines.length - 1] === '') {
              chunkLines.pop();
            }
            return { ...prev, [jobId]: [...existing, ...chunkLines] };
          });
        }
      } catch {
        // Ignore parse errors
      }
    };

    es.onerror = () => {
      // SSE connection closed or errored — don't reconnect automatically
      es.close();
      sseRefs.current[jobId] = null;
    };

    sseRefs.current[jobId] = es;
  }, [fetchLogs]);

  const disconnectSSE = useCallback((jobId: string) => {
    if (sseRefs.current[jobId]) {
      sseRefs.current[jobId].close();
      sseRefs.current[jobId] = null;
    }
  }, []);

  // Fetch jobs on mount
  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Poll for running jobs' progress
  useEffect(() => {
    const runningJobs = jobs.filter((j) => j.status === 'running');
    if (runningJobs.length === 0) return;

    // Fetch progress immediately
    runningJobs.forEach((j) => fetchProgress(j.id));

    const interval = setInterval(() => {
      fetchJobs();
      // Re-check running jobs from current state
      jobs
        .filter((j) => j.status === 'running')
        .forEach((j) => fetchProgress(j.id));
    }, 3000);

    return () => clearInterval(interval);
  }, [jobs, fetchJobs, fetchProgress]);

  // Connect SSE when logs are expanded for a running job
  useEffect(() => {
    if (!logsExpanded) return;

    const job = jobs.find((j) => j.id === logsExpanded);
    if (job?.status === 'running') {
      connectSSE(logsExpanded);
    } else {
      // For non-running jobs, just fetch logs once
      fetchLogs(logsExpanded);
    }

    // Cleanup: disconnect SSE when logs are collapsed or job changes
    return () => {
      disconnectSSE(logsExpanded);
    };
  }, [logsExpanded, jobs, connectSSE, disconnectSSE, fetchLogs]);

  function toggleExpand(jobId: string) {
    setExpandedJob(expandedJob === jobId ? null : jobId);
  }

  function toggleLogs(jobId: string) {
    if (logsExpanded === jobId) {
      setLogsExpanded(null);
      disconnectSSE(jobId);
    } else {
      setLogsExpanded(jobId);
    }
  }

  async function cancelJob(jobId: string) {
    if (!confirm('Cancel this training job? Any progress made so far will be lost.')) return;
    setCancellingJobs((prev) => ({ ...prev, [jobId]: true }));

    try {
      const res = await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
      const data = await res.json();

      if (!res.ok) {
        alert(data.error || 'Failed to cancel job');
      } else {
        // Refresh jobs list
        fetchJobs();
        // Disconnect SSE if this job's logs were expanded
        disconnectSSE(jobId);
        // Close logs if they were expanded for this job
        if (logsExpanded === jobId) {
          setLogsExpanded(null);
        }
      }
    } catch {
      alert('Failed to cancel job');
    } finally {
      setCancellingJobs((prev) => ({ ...prev, [jobId]: false }));
    }
  }

  async function deleteJob(jobId: string) {
    if (!confirm('Delete this job and all its output files?')) return;
    setDeletingJobs((prev) => ({ ...prev, [jobId]: true }));

    try {
      const res = await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
      if (res.ok) {
        setJobs((prev) => prev.filter((j) => j.id !== jobId));
        setProgressMap((prev) => {
          const next = { ...prev };
          delete next[jobId];
          return next;
        });
        setLogLinesMap((prev) => {
          const next = { ...prev };
          delete next[jobId];
          return next;
        });
        if (logsExpanded === jobId) {
          setLogsExpanded(null);
          disconnectSSE(jobId);
        }
        if (expandedJob === jobId) setExpandedJob(null);
      } else {
        const data = await res.json();
        alert(data.error || 'Failed to delete job');
      }
    } catch {
      alert('Failed to delete job');
    } finally {
      setDeletingJobs((prev) => ({ ...prev, [jobId]: false }));
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'running':
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400';
      case 'completed':
        return 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400';
      case 'failed':
        return 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-400';
      default:
        return 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300';
    }
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  function getProgressPercent(progress: JobProgress | undefined): number {
    if (!progress || !progress.totalSteps || progress.totalSteps === 0) return 0;
    return Math.round((progress.currentStep / progress.totalSteps) * 100);
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 mb-4">Training Jobs</h2>
        <p className="text-slate-500 dark:text-slate-400">Loading jobs...</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">Training Jobs</h2>
        <button
          onClick={fetchJobs}
          className="px-3 py-1 text-sm bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded hover:bg-slate-300 dark:hover:bg-slate-600 transition-colors"
        >
          Refresh
        </button>
      </div>

      {jobs.length === 0 && (
        <p className="text-slate-500 dark:text-slate-400">No training jobs found.</p>
      )}

      <div className="space-y-3">
        {jobs.map((job) => {
          const progress = progressMap[job.id];
          const percent = getProgressPercent(progress);
          const jobLogLines = logLinesMap[job.id] || [];
          const isSSEConnected = job.status === 'running' && logsExpanded === job.id;

          return (
            <div
              key={job.id}
              className="border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden"
            >
              {/* Job header */}
              <div className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span className="font-medium text-slate-900 dark:text-slate-100">{job.params.loraName || job.id}</span>
                      <span
                        className={`px-2 py-0.5 text-xs font-medium rounded-full ${getStatusColor(
                          job.status
                        )}`}
                      >
                        {job.status}
                      </span>
                      {isSSEConnected && (
                        <span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400">
                          <span className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                          Live
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                      {job.params.networkDim && `Dim: ${job.params.networkDim} `}
                      {job.params.networkAlpha && `Alpha: ${job.params.networkAlpha} `}
                      {job.params.epochs && `Epochs: ${job.params.epochs}`}
                      {job.params.resolution && ` @ ${job.params.resolution}px`}
                    </p>
                    <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                      Started: {formatDate(job.createdAt)}
                    </p>
                  </div>
                </div>

                {/* Progress bar for running jobs */}
                {job.status === 'running' && progress && (
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 mb-1">
                      <span>
                        Step {progress.currentStep}/{progress.totalSteps}
                        {progress.totalEpochs > 0 && ` · Epoch ${progress.currentEpoch}/${progress.totalEpochs}`}
                      </span>
                      {progress.avgLoss != null && typeof progress.avgLoss === 'number' && (
                        <span>Loss: {progress.avgLoss.toFixed(4)}</span>
                      )}
                    </div>
                    <div className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 dark:bg-blue-400 transition-all duration-500"
                        style={{ width: `${percent}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Error message */}
                {job.error && (
                  <p className="text-red-600 dark:text-red-400 text-sm mt-2">{job.error}</p>
                )}
                {progress?.error && job.status === 'failed' && !job.error && (
                  <p className="text-red-600 dark:text-red-400 text-sm mt-2">{progress.error}</p>
                )}

                {/* Output files for completed jobs */}
                {job.status === 'completed' && progress?.outputFiles && progress.outputFiles.length > 0 && (
                  <div className="mt-2">
                    <p className="text-xs text-slate-500 dark:text-slate-400">Output files:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {progress.outputFiles.map((file) => (
                        <a
                          key={file}
                          href={`/api/download?runId=${encodeURIComponent(job.id)}&file=${encodeURIComponent(file)}`}
                          download={file}
                          className="text-xs font-mono bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 px-2 py-0.5 rounded hover:underline"
                        >
                          {file}
                        </a>
                      ))}
                    </div>
                  </div>
                )}

                {/* Action buttons */}
                <div className="flex gap-2 mt-3 flex-wrap">
                  <button
                    onClick={() => toggleLogs(job.id)}
                    className="px-3 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
                  >
                    {logsExpanded === job.id ? 'Hide Logs' : 'View Logs'}
                  </button>

                  {job.status === 'running' && (
                    <button
                      onClick={() => cancelJob(job.id)}
                      disabled={cancellingJobs[job.id]}
                      className="px-3 py-1 text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors disabled:opacity-50"
                    >
                      {cancellingJobs[job.id] ? 'Cancelling...' : 'Cancel'}
                    </button>
                  )}

                  {job.permutations && job.permutations.length > 0 && (
                    <button
                      onClick={() => toggleExpand(job.id)}
                      className="px-3 py-1 text-xs bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-200 rounded hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
                    >
                      {expandedJob === job.id ? 'Collapse' : 'Expand'}
                    </button>
                  )}

                  {job.status !== 'running' && (
                    <button
                      onClick={() => deleteJob(job.id)}
                      disabled={deletingJobs[job.id]}
                      className="px-3 py-1 text-xs bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 rounded hover:bg-red-200 dark:hover:bg-red-900/50 transition-colors disabled:opacity-50 ml-auto"
                    >
                      {deletingJobs[job.id] ? 'Deleting...' : 'Delete'}
                    </button>
                  )}
                </div>
              </div>

              {/* Log viewer */}
              {logsExpanded === job.id && (
                <div className="border-t border-slate-200 dark:border-slate-700">
                  <LogViewer
                    lines={jobLogLines}
                    autoScroll={true}
                  />
                </div>
              )}

              {/* Expanded permutation details */}
              {expandedJob === job.id && job.permutations && (
                <div className="border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-4">
                  <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Permutations ({job.permutations.length})
                  </h4>
                  <div className="space-y-2">
                    {job.permutations.map((perm) => (
                      <div
                        key={perm.id}
                        className="flex items-center justify-between text-sm"
                      >
                        <span className="text-slate-700 dark:text-slate-300">
                          {perm.params.networkDim && `Dim: ${perm.params.networkDim} `}
                          {perm.params.networkAlpha && `Alpha: ${perm.params.networkAlpha}`}
                        </span>
                        <span
                          className={`px-2 py-0.5 text-xs rounded-full ${getStatusColor(
                            perm.status
                          )}`}
                        >
                          {perm.status}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
