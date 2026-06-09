'use client';

import { useState, useEffect, useCallback } from 'react';

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

export function JobList() {
  const [jobs, setJobs] = useState<JobData[]>([]);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Poll for running jobs
  useEffect(() => {
    const hasRunning = jobs.some((j) => j.status === 'running');
    if (!hasRunning) return;

    const interval = setInterval(fetchJobs, 3000);
    return () => clearInterval(interval);
  }, [jobs, fetchJobs]);

  function toggleExpand(jobId: string) {
    setExpandedJob(expandedJob === jobId ? null : jobId);
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'running':
        return 'bg-yellow-100 text-yellow-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <h2 className="text-xl font-bold mb-4">Training Jobs</h2>
        <p className="text-gray-500">Loading jobs...</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">Training Jobs</h2>
        <button
          onClick={fetchJobs}
          className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300"
        >
          Refresh
        </button>
      </div>

      {jobs.length === 0 && (
        <p className="text-gray-500">No training jobs found.</p>
      )}

      <div className="space-y-3">
        {jobs.map((job) => (
          <div
            key={job.id}
            className="border rounded-lg bg-white overflow-hidden"
          >
            {/* Job header */}
            <div className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <span className="font-medium">{job.params.loraName || job.id}</span>
                    <span
                      className={`px-2 py-0.5 text-xs font-medium rounded-full ${getStatusColor(
                        job.status
                      )}`}
                    >
                      {job.status}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">
                    {job.params.networkDim && `Dim: ${job.params.networkDim} `}
                    {job.params.networkAlpha && `Alpha: ${job.params.networkAlpha} `}
                    {job.params.epochs && `Epochs: ${job.params.epochs}`}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Started: {formatDate(job.createdAt)}
                  </p>
                </div>

                {/* Expand button for matrix jobs */}
                {job.permutations && job.permutations.length > 0 && (
                  <button
                    onClick={() => toggleExpand(job.id)}
                    className="ml-3 px-2 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200"
                    aria-label="expand"
                  >
                    {expandedJob === job.id ? 'Collapse' : 'Expand'}
                  </button>
                )}
              </div>

              {/* Error message */}
              {job.error && (
                <p className="text-red-600 text-sm mt-2">{job.error}</p>
              )}
            </div>

            {/* Expanded permutation details */}
            {expandedJob === job.id && job.permutations && (
              <div className="border-t bg-gray-50 p-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Permutations ({job.permutations.length})
                </h4>
                <div className="space-y-2">
                  {job.permutations.map((perm) => (
                    <div
                      key={perm.id}
                      className="flex items-center justify-between text-sm"
                    >
                      <span>
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
        ))}
      </div>
    </div>
  );
}
