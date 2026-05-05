import React, { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import RevealOnScroll from './RevealOnScroll';
import EpisodeCard from './EpisodeCard';
import { episodes, seasons } from '../data/episodes';
import './EpisodeGrid.css';

function EpisodeGrid() {
  const [activeSeason, setActiveSeason] = useState('all');

  const filteredEpisodes = useMemo(() => {
    if (activeSeason === 'all') return episodes;
    return episodes.filter(ep => ep.season === parseInt(activeSeason));
  }, [activeSeason]);

  // Show only first 6 on homepage
  const displayedEpisodes = filteredEpisodes.slice(0, 6);

  return (
    <section className="episode-grid section section--dark" id="episodes">
      <div className="container">
        <RevealOnScroll>
          <span className="section-subtitle">The Archive</span>
          <h2 className="section-title">All Episodes</h2>
        </RevealOnScroll>

        <RevealOnScroll delay={100}>
          <div className="episode-grid__filters">
            <button
              className={`episode-grid__filter ${activeSeason === 'all' ? 'episode-grid__filter--active' : ''}`}
              onClick={() => setActiveSeason('all')}
            >
              All
            </button>
            {seasons.map(s => (
              <button
                key={s.number}
                className={`episode-grid__filter ${activeSeason === String(s.number) ? 'episode-grid__filter--active' : ''}`}
                onClick={() => setActiveSeason(String(s.number))}
              >
                Season {s.number}
              </button>
            ))}
          </div>
        </RevealOnScroll>

        <div className="episode-grid__grid">
          {displayedEpisodes.map((ep, i) => (
            <RevealOnScroll key={ep.id} delay={i * 80}>
              <EpisodeCard episode={ep} index={i} />
            </RevealOnScroll>
          ))}
        </div>

        {filteredEpisodes.length > 6 && (
          <RevealOnScroll delay={200}>
            <div className="episode-grid__more">
              <Link
                to={activeSeason === 'all' ? '/season/3' : `/season/${activeSeason}`}
                className="btn btn--rust-outline"
              >
                View All Episodes
              </Link>
            </div>
          </RevealOnScroll>
        )}
      </div>
    </section>
  );
}

export default EpisodeGrid;
