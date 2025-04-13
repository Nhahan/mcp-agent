import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ThoughtBubbleProps {
  content: string | null;
}

const ThoughtBubble: React.FC<ThoughtBubbleProps> = ({ content }) => {
  const [displayedContent, setDisplayedContent] = useState('');
  const [showCursor, setShowCursor] = useState(true);
  const contentRef = useRef(content); // Ref to track content changes
  const typingTimeoutRef = useRef<number | null>(null); // FIX: Use number type for browser setTimeout ID

  // Reset and start typing effect when content prop changes
  useEffect(() => {
    // Clear previous typing timeout if content changes
    if (typingTimeoutRef.current) {
      window.clearTimeout(typingTimeoutRef.current); // Use window.clearTimeout
    }

    setDisplayedContent(''); // Reset displayed content
    setShowCursor(true); // Show cursor immediately
    contentRef.current = content; // Update ref

    let currentTypingIndex = 0;
    const targetContent = content ?? ""; // Use empty string if content is null initially
    const initialMessage = content === null ? "생각 중..." : ""; // Show initial message if content starts as null
    
    if (initialMessage) {
        setDisplayedContent(initialMessage);
        // Optionally hide cursor after initial message if content is null
        // setShowCursor(false); 
    }

    // Only start typing effect if there is actual content to type
    if (targetContent) {
        const typeCharacter = () => {
            if (currentTypingIndex < targetContent.length) {
                // If content changed during typing, stop this instance
                if (contentRef.current !== targetContent) return; 

                setDisplayedContent((prev) => {
                    // Use substring for safer updates and prevent potential 'undefined'
                    // Ensure index doesn't exceed target length
                    const nextIndex = Math.min(currentTypingIndex + 1, targetContent.length);
                    return targetContent.substring(0, nextIndex);
                });
                currentTypingIndex++;
                typingTimeoutRef.current = window.setTimeout(typeCharacter, 20); 
            } else {
                // Typing finished
                setShowCursor(false); // Hide cursor when done
            }
        };

        // Start typing after a brief delay if showing initial message
        const startDelay = initialMessage ? 300 : 50; 
        // Use window.setTimeout
        typingTimeoutRef.current = window.setTimeout(() => {
            if (initialMessage) setDisplayedContent(''); // Clear initial message before typing starts
            setShowCursor(true); // Ensure cursor is visible when typing starts
            typeCharacter();
        }, startDelay); 
    }

    // Cleanup function to clear timeout on unmount or content change
    return () => {
      if (typingTimeoutRef.current) {
        window.clearTimeout(typingTimeoutRef.current); // Use window.clearTimeout
      }
    };
  }, [content]); // Rerun effect only when content prop changes

  // If no content is provided initially and nothing is displayed yet, don't render
  // Allow rendering if displayedContent has the initial message
  if (content === null && displayedContent === '') { 
     return null;
  }

  // Define components for ReactMarkdown
  const markdownComponents = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={oneDark}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className ? className : 'bg-gray-200 dark:bg-gray-600 px-1 py-0.5 rounded text-sm font-mono'.trim()} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <div className="flex justify-start mb-4 w-full"> {/* Ensure full width */}
      <div className="flex max-w-2xl w-full"> {/* Ensure bubble itself takes full available width up to max-w-2xl */}
        <div className="relative px-4 py-2 rounded-lg shadow bg-purple-100 dark:bg-purple-900/50 border border-purple-200 dark:border-purple-800 text-purple-700 dark:text-purple-300 w-full animate-blinking"> {/* Added w-full and animate-blinking */}
          <div className="absolute -top-1.5 left-3 w-3 h-3 bg-purple-100 dark:bg-purple-900/50 transform rotate-45 border-l border-t border-purple-200 dark:border-purple-800" style={{ content: '' }}></div>
          <strong className="font-medium text-xs block mb-1">Thinking...</strong>
          {/* Render the thought content using ReactMarkdown */} 
          <div className="prose prose-sm dark:prose-invert max-w-none break-words text-purple-700 dark:text-purple-300">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {displayedContent}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThoughtBubble;

// Ensure animations (fade-in-slow, blink, spin-slow) are defined in tailwind.config.js
/* Example:
 theme: {
   extend: {
     animation: {
       'fade-in-slow': 'fadeIn 0.8s ease-out forwards',
       'blink': 'blink 1s step-end infinite',
       'spin-slow': 'spin 3s linear infinite',
     },
     keyframes: {
       fadeIn: {
         '0%': { opacity: 0 },
         '100%': { opacity: 0.9 }, // Target opacity for thought bubble
       },
       // ... other keyframes
     }
   }
 }
*/ 