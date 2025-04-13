/** @jsxImportSource react */
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faClipboard, faCheck, faUser } from '@fortawesome/free-solid-svg-icons';
import profileImage from '../assets/profile.png';

// Define the props for the Message component using any type
interface MessageProps {
  message: any; // Use any type after removing types.ts
}

// Message Component: Renders a single chat message
const Message: React.FC<MessageProps> = ({ message }) => {
  const [copied, setCopied] = useState(false);
  const [displayedContent, setDisplayedContent] = useState('');
  const typingTimeoutRef = useRef<number | null>(null);

  const isUser = message.sender === 'user';
  const bgColor = isUser ? 'bg-gray-100 dark:bg-gray-800' : 'bg-white dark:bg-gray-700';
  const align = isUser ? 'justify-end' : 'justify-start';
  const icon = isUser ? faUser : null;
  const textColor = isUser ? 'text-gray-800 dark:text-gray-200' : 'text-gray-800 dark:text-gray-200';

  // Typing effect for assistant messages (improved cleanup and undefined check)
  useEffect(() => {
    let intervalId: number | null = null;

    if (!isUser && message.content) {
      setDisplayedContent(''); // Reset content when message changes
      let index = 0;

      // Function to start the interval
      const startTyping = () => {
        intervalId = setInterval(() => {
          // Strict check to prevent accessing index out of bounds
          if (index < message.content.length) {
            const charToAdd = message.content[index];
            // Check if charToAdd is valid before updating state
            if (charToAdd !== undefined) { 
              setDisplayedContent((prev) => prev + charToAdd);
            }
            index++;
          } else {
            if (intervalId) clearInterval(intervalId);
            intervalId = null;
          }
        }, 15) as unknown as number;
      };

      // Clear previous interval if running
      if (typingTimeoutRef.current) {
        clearInterval(typingTimeoutRef.current);
      }

      startTyping();
      typingTimeoutRef.current = intervalId;

    } else if (isUser) {
      // Ensure user messages display immediately and clear any running timeouts
      if (typingTimeoutRef.current) {
        clearInterval(typingTimeoutRef.current);
        typingTimeoutRef.current = null;
      }
      setDisplayedContent(message.content);
    }

    // Cleanup function
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
      if (typingTimeoutRef.current) {
          clearInterval(typingTimeoutRef.current);
          typingTimeoutRef.current = null;
      }
    };
  }, [message.content, isUser]);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(err => {
      console.error('Failed to copy text: ', err);
    });
  };

  // Custom code component for react-markdown
  const CodeBlock = ({ node, inline, className, children, ...props }: any) => {
    const match = /language-(\w+)/.exec(className || '');
    const codeContent = String(children).replace(/\n$/, '');
    const [codeCopied, setCodeCopied] = useState(false);

    return !inline && match ? (
      <div className="relative rounded-md my-2 code-block-container bg-gray-800">
        <div className="flex items-center justify-between px-3 py-1 bg-gray-700 rounded-t-md">
          <span className="text-xs font-semibold text-gray-400">
            {match[1] || 'code'} {/* Fallback language display */}
          </span>
          <button
            onClick={() => {
              navigator.clipboard.writeText(codeContent).then(() => {
                setCodeCopied(true);
                setTimeout(() => setCodeCopied(false), 2000);
              });
            }}
            className="text-xs text-gray-400 hover:text-gray-200 transition-colors duration-150"
            aria-label="Copy code"
          >
            <FontAwesomeIcon icon={codeCopied ? faCheck : faClipboard} className="mr-1" />
            {codeCopied ? 'Copied!' : 'Copy code'}
          </button>
        </div>
        <SyntaxHighlighter
          style={oneDark}
          // Provide fallback language 'text' or 'plaintext' if match fails
          language={match && match[1] ? match[1] : 'plaintext'}
          PreTag="div"
          customStyle={{ padding: '0.5rem', fontSize: '0.8rem' }}
          {...props}
        >
          {codeContent}
        </SyntaxHighlighter>
      </div>
    ) : (
      <code className={`${className || ''} inline-code bg-gray-200 dark:bg-gray-600 px-1 py-0.5 rounded text-sm font-mono`} {...props}>
        {children}
      </code>
    );
  };

  // 로봇 아이콘 대신 사용할 프로필 이미지 컴포넌트
  const ProfileIcon = () => (
    <img 
      src={profileImage}
      alt="AI Profile"
      className="w-full h-full object-cover rounded-full"
    />
  );

  return (
    <div className={`flex ${align} mb-4 message-container group`}>
      <div className={`flex max-w-2xl ${align}`}>
        <div className={`flex items-center justify-center h-8 w-8 rounded-lg flex-shrink-0 mr-3 ml-1 overflow-hidden ${isUser ? 'bg-blue-500 text-white' : ''}`}>
          {isUser ? (
            <FontAwesomeIcon icon={icon!} />
          ) : (
            <ProfileIcon />
          )}
        </div>
        <div className={`relative px-4 py-2 rounded-lg shadow ${bgColor} ${textColor} message-content min-h-[2.5rem] flex items-end`}>
          <div className={`prose prose-sm dark:prose-invert max-w-none break-words w-full text-left`}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code: CodeBlock,
              }}
            >
              {displayedContent}
            </ReactMarkdown>
          </div>
          {!isUser && displayedContent.length === message.content.length && (
            <button
              onClick={handleCopy}
              className="absolute top-1 right-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 opacity-0 group-hover:opacity-100 transition-opacity duration-150 p-1 bg-gray-100 dark:bg-gray-600 rounded"
              aria-label="Copy message"
              style={{ transition: 'opacity 0.15s ease-in-out' }}
            >
              <FontAwesomeIcon icon={copied ? faCheck : faClipboard} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message; 