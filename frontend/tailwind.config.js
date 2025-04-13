/** @type {import('tailwindcss').Config} */
import typography from '@tailwindcss/typography'; // Import typography plugin

export default {
  darkMode: 'class', // Enable class-based dark mode
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Define a more modern color palette
        gray: {
          50: '#f8fafc',  // Lightest gray
          100: '#f1f5f9', 
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b', // Darker gray for UI elements
          900: '#0f172a', // Dark background component
          950: '#020617'  // Darkest background
        },
        blue: { // Adjust blue shades
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6', // Primary blue
          600: '#2563eb', // Darker blue
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          950: '#172554'
        },
        purple: { // Add purple shades for gradients
          500: '#8b5cf6',
          600: '#7c3aed'
        }
      },
      // Add custom animations or keyframes if needed beyond index.css
      animation: {
        // Cleaned up animations
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'fade-in-slow': 'fadeInSlow 0.8s ease-out forwards', // Use unique name for keyframe
        'blink': 'blink 1s step-end infinite',
        'spin-slow': 'spin 3s linear infinite',
        'pulse-subtle': 'pulseSubtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        typing: 'typing 2s steps(40, end) infinite',
        // Add blinking animation
        blinking: 'blinking 1.5s ease-in-out infinite',
      },
      keyframes: {
        // Combined and cleaned up keyframes
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }, // Default fade-in target opacity
        },
        fadeInSlow: { // Keyframe for slower fade-in with specific opacity
          '0%': { opacity: '0' },
          '100%': { opacity: '0.9' }, // Target opacity for thought bubble
        },
        blink: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
        pulseSubtle: { // Use matching name from animation definition
          '0%, 100%': { opacity: '0.85' }, 
          '50%': { opacity: '0.95' }, 
        },
        typing: {
          'from': { width: '0' },
          'to': { width: '100%' },
        },
        // Add blinking keyframes
        blinking: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.5 },
        },
        // spin keyframes are built-in
      },
      // Extend typography styles if needed
      typography: (theme) => ({
        DEFAULT: {
          css: {
            '--tw-prose-body': theme('colors.gray.700'),
            '--tw-prose-headings': theme('colors.gray.900'),
            '--tw-prose-links': theme('colors.blue.600'),
            // ... other prose styles
            '--tw-prose-invert-body': theme('colors.gray.300'),
            '--tw-prose-invert-headings': theme('colors.white'),
            '--tw-prose-invert-links': theme('colors.blue.400'),
            // Define code block styles
             '--tw-prose-pre-code': theme('colors.gray.300'),
             '--tw-prose-pre-bg': theme('colors.gray.900'), // Dark bg for code in light mode
             '--tw-prose-invert-pre-code': theme('colors.gray.300'),
             '--tw-prose-invert-pre-bg': theme('colors.gray.800/60'), // Slightly transparent dark bg
             code: { 
                fontWeight: 'normal',
                padding: '0.1em 0.3em', 
                borderRadius: '0.25em',
                backgroundColor: 'var(--tw-prose-pre-bg)', // Use var for consistency
                color: 'var(--tw-prose-code)'
             },
             'code::before': { content: 'none' }, // Remove backticks display
             'code::after': { content: 'none' },
             pre: {
                padding: theme('padding.3'),
                borderRadius: theme('borderRadius.md'),
             }
          },
        },
      }),
    },
  },
  plugins: [
    typography, // Enable the typography plugin
    // Add other plugins like forms if needed
  ],
} 