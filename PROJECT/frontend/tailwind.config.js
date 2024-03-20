/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/flowbite/**/*.js",
    'node_modules/flowbite-react/lib/esm/**/*.js'
  ],
  theme: {
    fontFamily: {
      // sans: "DM Sans",
      sans:"Preahvihear",
    },
    extend: {
      colors: {
        "dark-purple":"#081A51",
        "light-white":'rgba(255, 255, 255, 0.18)',
        primary: {"50":"#f5f3ff","100":"#ede9fe","200":"#ddd6fe","300":"#c4b5fd","400":"#a78bfa","500":"#8b5cf6","600":"#7c3aed","700":"#6d28d9","800":"#5b21b6","900":"#4c1d95","950":"#2e1065"}
      },
      
    },
  },
  plugins: [
    require('flowbite-typography'),
    require('flowbite/plugin')
  ],
}

