import React from 'react';
import { useInView } from 'react-intersection-observer';

function RevealOnScroll({ children, className = '', delay = 0, direction = 'up' }) {
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.15,
  });

  const directionStyles = {
    up: { transform: 'translateY(40px)' },
    down: { transform: 'translateY(-40px)' },
    left: { transform: 'translateX(-40px)' },
    right: { transform: 'translateX(40px)' },
  };

  return (
    <div
      ref={ref}
      className={`reveal ${inView ? 'visible' : ''} ${className}`}
      style={{
        opacity: inView ? 1 : 0,
        transform: inView ? 'translate(0, 0)' : directionStyles[direction]?.transform,
        transition: `opacity 0.8s ease ${delay}ms, transform 0.8s ease ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}

export default RevealOnScroll;
