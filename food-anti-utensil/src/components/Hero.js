import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { getLatestEpisode } from '../data/episodes';
import './Hero.css';

function Hero() {
  const latestEp = getLatestEpisode();
  const [glitchActive, setGlitchActive] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setGlitchActive(true);
      setTimeout(() => setGlitchActive(false), 200);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <section className="hero" id="home">
      {/* Video/image background placeholder */}
      <div className="hero__bg">
        <div className="hero__bg-gradient" />
        <div className="hero__scanlines" />
      </div>

      <div className="hero__content container">
        <motion.div
          className="hero__badge"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          <span className="hero__badge-dot" />
          NOW STREAMING — SEASON 3
        </motion.div>

        <motion.h1
          className={`hero__title ${glitchActive ? 'hero__title--glitch' : ''}`}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
        >
          <span className="hero__title-line">COOKING</span>
          <span className="hero__title-line hero__title-line--accent">WITHOUT</span>
          <span className="hero__title-line">RULES</span>
        </motion.h1>

        <motion.p
          className="hero__tagline"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.6 }}
        >
          No forks. No spoons. No mercy.<br />
          A cooking show that respects nothing but flavor.
        </motion.p>

        <motion.div
          className="hero__actions"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1, duration: 0.6 }}
        >
          <Link to={`/episode/${latestEp.id}`} className="btn btn--primary btn--large">
            <span className="hero__play-icon">&#9654;</span>
            Watch Latest Episode
          </Link>
          <Link to="/season/3" className="btn btn--outline btn--large">
            Browse Seasons
          </Link>
        </motion.div>

        <motion.div
          className="hero__latest"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 0.6 }}
        >
          <span className="hero__latest-label">Latest:</span>
          <span className="hero__latest-title">
            S{latestEp.season}E{latestEp.episode} — {latestEp.title}
          </span>
          <span className="hero__latest-views">{latestEp.views} views</span>
        </motion.div>
      </div>

      <motion.div
        className="hero__scroll-indicator"
        animate={{ y: [0, 10, 0] }}
        transition={{ repeat: Infinity, duration: 2 }}
      >
        <span>Scroll</span>
        <div className="hero__scroll-line" />
      </motion.div>
    </section>
  );
}

export default Hero;
