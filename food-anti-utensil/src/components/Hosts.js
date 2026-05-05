import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import RevealOnScroll from './RevealOnScroll';
import { hosts } from '../data/hosts';
import './Hosts.css';

function Hosts() {
  const [activeHost, setActiveHost] = useState(hosts[0]);

  return (
    <section className="hosts section section--darker" id="hosts">
      <div className="container">
        <RevealOnScroll>
          <span className="section-subtitle">The Maniacs</span>
          <h2 className="section-title section-title--center">Meet The Hosts</h2>
        </RevealOnScroll>

        <div className="hosts__layout">
          {/* Host selector */}
          <div className="hosts__selector">
            {hosts.map((host) => (
              <button
                key={host.id}
                className={`hosts__tab ${activeHost.id === host.id ? 'hosts__tab--active' : ''}`}
                onClick={() => setActiveHost(host)}
                style={{
                  '--host-color': host.color,
                }}
              >
                <span className="hosts__tab-name">{host.name}</span>
                <span className="hosts__tab-role">{host.role}</span>
              </button>
            ))}
          </div>

          {/* Host detail */}
          <AnimatePresence mode="wait">
            <motion.div
              key={activeHost.id}
              className="hosts__detail"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -30 }}
              transition={{ duration: 0.4 }}
            >
              <div
                className="hosts__portrait"
                style={{
                  background: `linear-gradient(135deg, ${activeHost.color}33, var(--smoke))`,
                  borderColor: activeHost.color,
                }}
              >
                <div className="hosts__portrait-placeholder">
                  <span className="hosts__portrait-initial">
                    {activeHost.name.charAt(0)}
                  </span>
                </div>
              </div>

              <div className="hosts__info">
                <h3 className="hosts__name" style={{ color: activeHost.color }}>
                  {activeHost.name}
                </h3>
                <span className="hosts__role">{activeHost.role}</span>
                <p className="hosts__bio">{activeHost.bio}</p>

                <div className="hosts__specialties">
                  <span className="hosts__specialties-label">Specialties:</span>
                  {activeHost.specialties.map((s, i) => (
                    <span key={i} className="hosts__specialty">
                      {s}
                    </span>
                  ))}
                </div>

                <blockquote className="hosts__catchphrase">
                  "{activeHost.catchphrase}"
                </blockquote>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  );
}

export default Hosts;
