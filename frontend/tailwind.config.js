/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#111111',
        paper: '#f5f5f3',
        cream: '#ffffff',
        cobalt: '#b0000c',
        coral: '#e50010',
        acid: '#f7a8ae',
        mist: '#e8e8e5',
      },
      boxShadow: {
        offset: '5px 5px 0 #111111',
        'offset-sm': '3px 3px 0 #111111',
      },
      fontFamily: {
        sans: ['Manrope', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"DM Mono"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
