import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const textVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      delay: 0.5,
      staggerChildren: 0.1, // Adjust the delay between characters
    },
  },
};

const charVariants = {
  hidden: { opacity: 0, y: -50, x: 20 }, // Initial jump positions
  visible: { opacity: 1, y: 0, x: 0 },
};

const TextAnimate = () => {
  const [displayText, setDisplayText] = useState('');
  const fullText = 'MedSecure';
  let charIndex = 0;

  useEffect(() => {
    const intervalId = setInterval(() => {
      if (charIndex <= fullText.length) {
        setDisplayText(fullText.substring(0, charIndex));
        charIndex++;
      } else {
        clearInterval(intervalId);
      }
    }, 100); // Typewriter speed

    return () => clearInterval(intervalId);
  }, []);

  return (
    <div>
      <motion.div
        initial="hidden"
        animate="visible"
        variants={textVariants}
      >
        {displayText.split('').map((char, index) => (
          <motion.span
            key={index}
            variants={charVariants}
            style={{
              display: 'inline-block',
              overflow: 'hidden',
            }}
          >
            {char}
          </motion.span>
        ))}
      </motion.div>
      </div>
  );
};

export default TextAnimate;
