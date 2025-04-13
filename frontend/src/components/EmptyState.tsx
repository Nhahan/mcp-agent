/** @jsxImportSource react */
import React from 'react';
import { TFunction } from 'i18next';
import profileImage from '../assets/profile.png';

// Define the props for the EmptyState component
interface EmptyStateProps {
  t: TFunction;
  setInputMessage: (input: string) => void;
}

// EmptyState Component: Shown when there are no messages
const EmptyState: React.FC<EmptyStateProps> = ({ t, setInputMessage }) => {
  const examples = [
    { icon: '💡', title: t('example1Title'), desc: t('example1Desc') },
    { icon: '📝', title: t('example2Title'), desc: t('example2Desc') },
    { icon: '📘', title: t('example3Title'), desc: t('example3Desc') },
    { icon: '✨', title: t('example4Title'), desc: t('example4Desc') },
  ];

  return (
    <div className="flex flex-col items-center text-center px-4 py-12 md:pb-16">
      <div className="mb-6 flex flex-col items-center">
        <img 
          src={profileImage}
          alt="AI Profile"
          className="w-16 h-16 rounded-lg shadow-lg mb-4" 
        />
        <h1 className="text-2xl font-semibold text-gray-800 dark:text-white">{t('emptyStateTitle')}</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">{t('emptyStateSubtitle')}</p>
      </div>

      <div className="w-full max-w-md mb-8">
        <h2 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-3">{t('emptyStateExamples')}</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {examples.map((ex, index) => (
            <div 
              key={index} 
              className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors cursor-pointer"
              onClick={() => setInputMessage(ex.desc)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setInputMessage(ex.desc); }}
            >
              <p className="text-sm font-medium text-gray-800 dark:text-white"><span className="mr-2">{ex.icon}</span>{ex.title}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{ex.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default EmptyState; 