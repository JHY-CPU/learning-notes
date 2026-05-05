import React from 'react';
import { Link } from 'react-router-dom';
import RevealOnScroll from './RevealOnScroll';
import { featuredEpisodes } from '../data/episodes';
import './FeaturedEpisode.css';

function FeaturedEpisode() {
  const featured = featuredEpisodes[featuredEpisodes.length - 1]; // latest featured

  return (
    <section className="featured section section--darker">
      <div className="container">
        <RevealOnScroll>
          <span className="section-subtitle">Now Playing</span>
          <h2 className="section-title">Featured Episode</h2>
        </RevealOnScroll>

        <RevealOnScroll delay={200}>
          <Link to={`/episode/${featured.id}`} className="featured__card">
            <div className="featured__thumb">
              <div className="featured__thumb-bg" />
              <div className="featured__thumb-overlay">
                <div className="featured__play">
                  <span className="featured__play-icon">&#9654;</span>
                  <span className="featured__play-text">Play Episode</span>
                </div>
              </div>
              <div className="featured__badge-row">
                <span className="featured__badge">FEATURED</span>
                <span className="featured__badge featured__badge--season">
                  SEASON {featured.season}
                </span>
              </div>
            </div>
            <div className="featured__content">
              <span className="featured__ep-label">
                S{featured.season}E{featured.episode}
              </span>
              <h3 className="featured__title">{featured.title}</h3>
              <p className="featured__subtitle">{featured.subtitle}</p>
              <p className="featured__desc">{featured.description}</p>
              <div className="featured__meta">
                <span>{featured.duration}</span>
                <span>{featured.views} views</span>
                {featured.guests.length > 0 && (
                  <span className="featured__guest-tag">
                    {featured.guests.join(', ')}
                  </span>
                )}
              </div>
              {featured.challenges.length > 0 && (
                <div className="featured__challenges">
                  {featured.challenges.map((c, i) => (
                    <span key={i} className="featured__challenge">{c}</span>
                  ))}
                </div>
              )}
            </div>
          </Link>
        </RevealOnScroll>
      </div>
    </section>
  );
}

export default FeaturedEpisode;
