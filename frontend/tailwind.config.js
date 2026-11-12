/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ws: {
          bg: '#111315',
          panel: '#1B1F23',
          panel2: '#22272C',
          border: '#343B42',
          text: '#E7E9EB',
          muted: '#929AA3',
          accent: '#789DC4',
          success: '#78B89A',
          warn: '#D6A65D',
          error: '#CF7777',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
    },
  },
  plugins: [],
}

