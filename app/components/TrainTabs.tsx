'use client';

import { useState } from 'react';
import { type TrainingParams } from '../lib/training-schema';
import { AnimaTab } from './AnimaTab';

export type ModelType = 'anima' | 'flux' | 'sd3' | 'sdxl' | 'sd15' | 'hunyuan' | 'lumina';

interface TrainTabsProps {
  onSubmit: (params: TrainingParams) => void;
  trainingImagesPath?: string;
  matrixMode?: boolean;
}

const MODEL_TABS: { key: ModelType; label: string }[] = [
  { key: 'anima', label: 'Anima' },
  { key: 'flux', label: 'FLUX' },
  { key: 'sd3', label: 'SD3' },
  { key: 'sdxl', label: 'SDXL' },
  { key: 'sd15', label: 'SD 1.5' },
  { key: 'hunyuan', label: 'Hunyuan' },
  { key: 'lumina', label: 'Lumina' },
];

const IMPLEMENTED_MODELS: ModelType[] = ['anima'];

function ComingSoonTab({ modelLabel }: { modelLabel: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="text-4xl mb-4">🚧</div>
      <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-2">
        {modelLabel} Training
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md">
        Training support for {modelLabel} is coming soon. Check back later or switch to the Anima
        tab to start training now.
      </p>
    </div>
  );
}

export function TrainTabs({ onSubmit, trainingImagesPath, matrixMode = false }: TrainTabsProps) {
  const [activeTab, setActiveTab] = useState<ModelType>('anima');

  return (
    <div className="max-w-screen-xl mx-auto px-16 py-8">
      {/* Model type tabs */}
      <div className="mb-6 border-b border-slate-200 dark:border-slate-700">
        <nav className="flex gap-1 -mb-px overflow-x-auto" aria-label="Model types">
          {MODEL_TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                activeTab === tab.key
                  ? 'border-slate-900 dark:border-slate-100 text-slate-900 dark:text-slate-100 bg-slate-200 dark:bg-slate-700'
                  : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:border-slate-300 dark:hover:border-slate-600'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      <div>
        {IMPLEMENTED_MODELS.includes(activeTab) ? (
          <AnimaTab
            onSubmit={onSubmit}
            trainingImagesPath={trainingImagesPath}
            matrixMode={matrixMode}
          />
        ) : (
          <ComingSoonTab
            modelLabel={MODEL_TABS.find((t) => t.key === activeTab)?.label ?? activeTab}
          />
        )}
      </div>
    </div>
  );
}
