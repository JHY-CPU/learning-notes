import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import RevealOnScroll from './RevealOnScroll';
import EpisodeCard from './EpisodeCard';
import { seasons, getEpisodesBySeason } from '../data/episodes';
import './SeasonPage.css';

function SeasonPage() {
  const { number } = useParams();
  const seasonNum = parseInt(number);
  const season = seasons.find(s => s.number === seasonNum);
  const episodes = getEpisodesBySeason(seasonNum);

  const seasonColors = {
    1: '#ff6b35',
    2: '#cc0000',
    3: '#c4a035',
  };
  const accentColor = seasonColors[seasonNum] || '#ff6b35';

  if (!season) {
    return (
      <div className="season-page season-page--404">
        <div className="container">
          <h1 className="season-page__404-title">SEASON NOT FOUND</h1>
          <p className="season-page__404-text">This season hasn't been filmed yet. Or maybe it was too dangerous.</p>
          <Link to="/" className="btn btn--primary">Go Home</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="season-page">
      {/* Season hero */}
      <section className="season-page__hero" style={{ '--accent': accentColor }}>
        <div className="season-page__hero-bg" />
        <div className="container season-page__hero-content">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="season-page__breadcrumb">
              <Link to="/">Home</Link>
              <span>/</span>
              <span>Season {season.number}</span>
            </div>

            <span className="season-page__season-tag">
              SEASON {season.number} &mdash; {season.year}
            </span>

            <h1 className="season-page__title">{season.name}</h1>
            <p className="season-page__desc">{season.description}</p>

            <div className="season-page__stats">
              <div className="season-page__stat">
                <span className="season-page__stat-value">{episodes.length}</span>
                <span className="season-page__stat-label">Episodes</span>
              </div>
              <div className="season-page__stat">
                <span className="season-page__stat-value">
                  {episodes.reduce((acc, ep) => {
                    const mins = parseInt(ep.duration.split(':')[0]);
                    return acc + mins;
                  }, 0)}m
                </span>
                <span className="season-page__stat-label">Total Runtime</span>
              </div>
              <div className="season-page__stat">
                <span className="season-page__stat-value">
                  {new Set(episodes.flatMap(ep => ep.guests)).size}
                </span>
                <span className="season-page__stat-label">Guests</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Season tabs */}
      <section className="season-page__tabs section section--dark">
        <div className="container">
          <div className="season-page__season-nav">
            {seasons.map(s => (
              <Link
                key={s.number}
                to={`/season/${s.number}`}
                className={`season-page__season-tab ${
                  s.number === seasonNum ? 'season-page__season-tab--active' : ''
                }`}
                style={{
                  '--tab-color': seasonColors[s.number] || '#ff6b35',
                }}
              >
                <span className="season-page__season-tab-num">S{s.number}</span>
                <span className="season-page__season-tab-name">{s.name}</span>
                <span className="season-page__season-tab-year">{s.year}</span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Episodes list */}
      <section className="season-page__episodes section section--darker">
        <div className="container">
          <RevealOnScroll>
            <h2 className="section-title">
              Season {season.number}: {season.name}
            </h2>
          </RevealOnScroll>

          <div className="season-page__episode-list">
            {episodes.map((ep, i) => (
              <RevealOnScroll key={ep.id} delay={i * 80}>
                <div className="season-page__episode-row">
                  <span className="season-page__episode-num">
                    {String(ep.episode).padStart(2, '0')}
                  </span>
                  <EpisodeCard episode={ep} index={i} />
                </div>
              </RevealOnScroll>
            ))}
          </div>

          {episodes.length === 0 && (
            <div className="season-page__empty">
              <p>No episodes available yet. We're probably filming something dangerous.</p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default SeasonPage;
