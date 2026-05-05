import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { getEpisodeById, getEpisodesBySeason } from '../data/episodes';
import EpisodeCard from './EpisodeCard';
import './EpisodePage.css';

function EpisodePage() {
  const { id } = useParams();
  const episode = getEpisodeById(id);
  const [showTranscript, setShowTranscript] = useState(false);

  if (!episode) {
    return (
      <div className="episode-page episode-page--404">
        <div className="container">
          <h1 className="episode-page__404-title">EPISODE NOT FOUND</h1>
          <p className="episode-page__404-text">
            This episode was either too chaotic to archive or doesn't exist.
          </p>
          <Link to="/" className="btn btn--primary">Go Home</Link>
        </div>
      </div>
    );
  }

  const seasonEpisodes = getEpisodesBySeason(episode.season);
  const currentIndex = seasonEpisodes.findIndex(ep => ep.id === episode.id);
  const prevEp = currentIndex > 0 ? seasonEpisodes[currentIndex - 1] : null;
  const nextEp = currentIndex < seasonEpisodes.length - 1 ? seasonEpisodes[currentIndex + 1] : null;
  const relatedEps = seasonEpisodes.filter(ep => ep.id !== episode.id).slice(0, 3);

  const hue = (episode.season * 60 + episode.episode * 30) % 360;
  const placeholderBg = `hsl(${hue}, 15%, 12%)`;
  const placeholderAccent = `hsl(${hue}, 60%, 45%)`;

  return (
    <div className="episode-page">
      {/* Hero area */}
      <section className="episode-page__hero" style={{ background: placeholderBg }}>
        <div
          className="episode-page__hero-pattern"
          style={{
            background: `
              radial-gradient(ellipse at 30% 40%, ${placeholderAccent}18, transparent 60%),
              radial-gradient(ellipse at 70% 60%, ${placeholderAccent}0d, transparent 50%)
            `,
          }}
        />
        <div className="episode-page__hero-overlay" />
        <div className="container episode-page__hero-content">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="episode-page__breadcrumb">
              <Link to="/">Home</Link>
              <span>/</span>
              <Link to={`/season/${episode.season}`}>Season {episode.season}</Link>
              <span>/</span>
              <span>Episode {episode.episode}</span>
            </div>

            <span className="episode-page__ep-badge">
              S{episode.season}E{episode.episode}
            </span>

            <h1 className="episode-page__title">{episode.title}</h1>
            <p className="episode-page__subtitle">{episode.subtitle}</p>

            <div className="episode-page__meta">
              <span className="episode-page__meta-item">
                <span className="episode-page__meta-icon">&#9201;</span> {episode.duration}
              </span>
              <span className="episode-page__meta-item">
                <span className="episode-page__meta-icon">&#9733;</span> {episode.views} views
              </span>
              <span className="episode-page__meta-item">
                {episode.date}
              </span>
            </div>

            <div className="episode-page__actions">
              <a
                href={episode.videoUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn--primary btn--large"
              >
                <span>&#9654;</span> Watch Now
              </a>
              <button
                className="btn btn--outline"
                onClick={() => setShowTranscript(!showTranscript)}
              >
                {showTranscript ? 'Hide' : 'Show'} Description
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Content area */}
      <section className="episode-page__content section section--dark">
        <div className="container">
          <div className="episode-page__grid">
            <div className="episode-page__main">
              <AnimatePresence>
                {showTranscript && (
                  <motion.div
                    className="episode-page__description"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.4 }}
                  >
                    <h3>The Story</h3>
                    <p>{episode.description}</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Challenges */}
              {episode.challenges.length > 0 && (
                <div className="episode-page__challenges">
                  <h3 className="episode-page__section-title">Challenges</h3>
                  <div className="episode-page__challenge-list">
                    {episode.challenges.map((c, i) => (
                      <motion.div
                        key={i}
                        className="episode-page__challenge"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                      >
                        <span className="episode-page__challenge-num">
                          {String(i + 1).padStart(2, '0')}
                        </span>
                        <span className="episode-page__challenge-name">{c}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {/* Guests */}
              {episode.guests.length > 0 && (
                <div className="episode-page__guests">
                  <h3 className="episode-page__section-title">Guest Appearances</h3>
                  <div className="episode-page__guest-list">
                    {episode.guests.map((g, i) => (
                      <div key={i} className="episode-page__guest">
                        <div className="episode-page__guest-avatar">
                          {g.charAt(0)}
                        </div>
                        <span>{g}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Sidebar */}
            <aside className="episode-page__sidebar">
              {/* Navigation */}
              <div className="episode-page__nav">
                {prevEp ? (
                  <Link to={`/episode/${prevEp.id}`} className="episode-page__nav-link episode-page__nav-link--prev">
                    <span className="episode-page__nav-label">&#8592; Previous</span>
                    <span className="episode-page__nav-title">{prevEp.title}</span>
                  </Link>
                ) : (
                  <div className="episode-page__nav-link episode-page__nav-link--disabled">
                    <span className="episode-page__nav-label">&#8592; Previous</span>
                    <span className="episode-page__nav-title">Start of Season</span>
                  </div>
                )}
                {nextEp ? (
                  <Link to={`/episode/${nextEp.id}`} className="episode-page__nav-link episode-page__nav-link--next">
                    <span className="episode-page__nav-label">Next &#8594;</span>
                    <span className="episode-page__nav-title">{nextEp.title}</span>
                  </Link>
                ) : (
                  <div className="episode-page__nav-link episode-page__nav-link--disabled">
                    <span className="episode-page__nav-label">Next &#8594;</span>
                    <span className="episode-page__nav-title">Latest Episode</span>
                  </div>
                )}
              </div>

              {/* More from season */}
              {relatedEps.length > 0 && (
                <div className="episode-page__related">
                  <h3 className="episode-page__section-title">
                    More from Season {episode.season}
                  </h3>
                  {relatedEps.map((ep, i) => (
                    <EpisodeCard key={ep.id} episode={ep} variant="compact" index={i} />
                  ))}
                </div>
              )}
            </aside>
          </div>
        </div>
      </section>
    </div>
  );
}

export default EpisodePage;
