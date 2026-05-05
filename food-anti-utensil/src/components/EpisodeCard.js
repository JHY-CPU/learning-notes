import React from 'react';
import { Link } from 'react-router-dom';
import './EpisodeCard.css';

function EpisodeCard({ episode, index = 0, variant = 'default' }) {
  const seasonLabel = `S${episode.season}E${episode.episode}`;

  // Generate a deterministic color based on season and episode
  const hue = (episode.season * 60 + episode.episode * 30) % 360;
  const placeholderBg = `hsl(${hue}, 15%, 15%)`;
  const placeholderAccent = `hsl(${hue}, 60%, 45%)`;

  return (
    <Link
      to={`/episode/${episode.id}`}
      className={`episode-card episode-card--${variant}`}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <div className="episode-card__thumb" style={{ background: placeholderBg }}>
        <div
          className="episode-card__thumb-pattern"
          style={{
            background: `radial-gradient(circle at 30% 40%, ${placeholderAccent}22, transparent 60%),
                         linear-gradient(135deg, transparent 40%, ${placeholderAccent}11 100%)`,
          }}
        />
        <div className="episode-card__thumb-overlay">
          <span className="episode-card__play-btn">&#9654;</span>
        </div>
        <span className="episode-card__duration">{episode.duration}</span>
        <span className="episode-card__season-tag">{seasonLabel}</span>
      </div>

      <div className="episode-card__info">
        <h3 className="episode-card__title">{episode.title}</h3>
        {episode.subtitle && (
          <p className="episode-card__subtitle">{episode.subtitle}</p>
        )}
        {variant !== 'compact' && (
          <p className="episode-card__desc">{episode.description}</p>
        )}
        <div className="episode-card__meta">
          <span className="episode-card__views">{episode.views} views</span>
          {episode.guests.length > 0 && (
            <span className="episode-card__guests">
              +{episode.guests.length} guest{episode.guests.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>
    </Link>
  );
}

export default EpisodeCard;
