/** @jsxImportSource react */
import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { TFunction } from 'i18next';

// Props for the Thinking Indicator
interface ThinkingIndicatorProps {
  content: string | null; // Content can be null initially
  t: TFunction; // Add t function prop
}

// Thinking Indicator Component: Shows while the AI is processing
const ThinkingIndicator: React.FC<ThinkingIndicatorProps> = ({ content, t }) => {
  const [dots, setDots] = useState('.');
  const [displayedThought, setDisplayedThought] = useState('');
  const typingTimeoutRef = useRef<number | null>(null);
  const dotsIntervalRef = useRef<number | null>(null);

  // Effect for the initial "추론 중..." dot animation
  useEffect(() => {
    if (content === null) {
      // Start dot animation only when content is null
      setDisplayedThought(''); // Clear any previous thought
      dotsIntervalRef.current = window.setInterval(() => {
        setDots((prevDots) => {
          if (prevDots.length >= 6) return '.';
          return prevDots + '.';
        });
      }, 400); // Adjust speed as needed
    } else {
      // If content arrives, stop dot animation
      if (dotsIntervalRef.current) {
        window.clearInterval(dotsIntervalRef.current);
        dotsIntervalRef.current = null;
      }
      setDots('.'); // Reset dots
    }

    // Cleanup dot animation interval
    return () => {
      if (dotsIntervalRef.current) {
        window.clearInterval(dotsIntervalRef.current);
        dotsIntervalRef.current = null;
      }
    };
  }, [content]); // Run when content changes (null -> value or value -> null)

  // Effect for the thought content typing animation
  useEffect(() => {
    if (content !== null) {
      // Start typing animation only when content is not null
      setDisplayedThought(''); // Reset display for new thought
      let index = 0;

      if (typingTimeoutRef.current) {
        window.clearTimeout(typingTimeoutRef.current);
      }

      const typeCharacter = () => {
        if (index < content.length) {
          const charToAdd = content[index];
          if (charToAdd !== undefined) {
            setDisplayedThought((prev) => prev + charToAdd);
          }
          index++;
          typingTimeoutRef.current = window.setTimeout(typeCharacter, 25); // Adjust speed
        } else {
          typingTimeoutRef.current = null;
        }
      };

      // Small delay before starting typing
      typingTimeoutRef.current = window.setTimeout(typeCharacter, 50); 

    } else {
       // If content becomes null, clear the displayed thought
       setDisplayedThought('');
       if (typingTimeoutRef.current) {
         window.clearTimeout(typingTimeoutRef.current);
         typingTimeoutRef.current = null;
       }
    }

    // Cleanup typing timeout
    return () => {
      if (typingTimeoutRef.current) {
        window.clearTimeout(typingTimeoutRef.current);
        typingTimeoutRef.current = null;
      }
    };
  }, [content]); // Run when content changes

  // Markdown components (keep code highlighting)
  const markdownComponents = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <div className="relative rounded my-1 code-block-container bg-gray-500/10 dark:bg-gray-400/10">
          <div className="flex items-center justify-between px-2 py-0.5 bg-gray-500/20 dark:bg-gray-400/20 rounded-t">
            <span className="text-xs font-semibold text-gray-600 dark:text-gray-400">
              {match[1] || 'code'}
            </span>
          </div>
          <SyntaxHighlighter
            style={oneDark}
            language={((match && match[1]) ? match[1] : 'plaintext') as string}
            PreTag="div"
            customStyle={{ padding: '0.5rem', fontSize: '0.8rem' }}
            {...props}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        </div>
      ) : (
        <code className={`${className || ''} inline-code bg-gray-500/20 dark:bg-gray-400/20 px-1 py-0.5 rounded text-xs font-mono`} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <div className="relative pl-3 py-1 border-l-4 border-gray-400 dark:border-gray-600 w-full animate-blinking min-h-[2.5rem] flex items-center">
      <div className="prose prose-sm dark:prose-invert max-w-none break-words text-gray-500/80 dark:text-gray-400/80 w-full text-left">
        {content === null ? (
          <span>{t('thinking')}{dots}</span>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {String(displayedThought || '')}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
};

export default ThinkingIndicator; 