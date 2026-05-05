import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import RevealOnScroll from './RevealOnScroll';
import { hosts, crew } from '../data/hosts';
import './AboutPage.css';

function AboutPage() {
  return (
    <div className="about-page">
      {/* Hero */}
      <section className="about-page__hero">
        <div className="about-page__hero-bg" />
        <div className="container about-page__hero-content">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="about-page__breadcrumb">
              <Link to="/">Home</Link>
              <span>/</span>
              <span>About</span>
            </div>
            <h1 className="about-page__title">
              ABOUT<br />
              <span className="about-page__title-accent">ANTI-UTENSIL</span>
            </h1>
            <p className="about-page__lead">
              We are not a cooking show. We are an anti-cooking show. A rebellion against
              everything the Food Network taught you. No scripts. No safety nets. No forks.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Manifesto */}
      <section className="about-page__manifesto section section--dark">
        <div className="container">
          <RevealOnScroll>
            <div className="about-page__manifesto-inner">
              <span className="section-subtitle">Our Manifesto</span>
              <h2 className="about-page__manifesto-title">
                THE UTENSIL IS THE ENEMY
              </h2>
              <div className="about-page__manifesto-text">
                <p>
                  Somewhere along the way, cooking became sterile. Clinical. Safe. Television taught us
                  that food should be precise, measured, and plated with tweezers. That every dish needs
                  a backstory and every chef needs a catchphrase.
                </p>
                <p>
                  We said no.
                </p>
                <p>
                  Anti-Utensil started in a kitchen that had no right to exist—a basement apartment with
                  a hot plate, a blowtorch, and three people who believed that the best food comes from
                  chaos. From instinct. From the primal urge to tear, burn, and devour.
                </p>
                <p>
                  We don't use forks because forks are a barrier between you and your food. We don't use
                  recipes because recipes are someone else's rules. We don't use timers because hunger
                  doesn't wait.
                </p>
                <p>
                  This is cooking at its most raw. Its most honest. Its most dangerous.
                </p>
                <p>
                  <strong>Welcome to the chaos.</strong>
                </p>
              </div>
            </div>
          </RevealOnScroll>
        </div>
      </section>

      {/* Hosts detailed */}
      <section className="about-page__hosts section section--darker">
        <div className="container">
          <RevealOnScroll>
            <span className="section-subtitle">The Anarchists</span>
            <h2 className="section-title">The Hosts</h2>
          </RevealOnScroll>

          {hosts.map((host, i) => (
            <RevealOnScroll key={host.id} delay={i * 150}>
              <div className="about-page__host-card" style={{ '--host-color': host.color }}>
                <div className="about-page__host-portrait">
                  <div className="about-page__host-initial">
                    {host.name.charAt(0)}
                  </div>
                </div>
                <div className="about-page__host-info">
                  <h3 className="about-page__host-name" style={{ color: host.color }}>
                    {host.name}
                  </h3>
                  <span className="about-page__host-role">{host.role}</span>
                  <p className="about-page__host-bio">{host.bio}</p>
                  <div className="about-page__host-specialties">
                    {host.specialties.map((s, j) => (
                      <span key={j} className="about-page__host-specialty">{s}</span>
                    ))}
                  </div>
                  <blockquote className="about-page__host-quote">
                    "{host.catchphrase}"
                  </blockquote>
                </div>
              </div>
            </RevealOnScroll>
          ))}
        </div>
      </section>

      {/* Crew */}
      <section className="about-page__crew section section--dark">
        <div className="container">
          <RevealOnScroll>
            <span className="section-subtitle">The Unsung Heroes</span>
            <h2 className="section-title">The Crew</h2>
          </RevealOnScroll>

          <div className="about-page__crew-grid">
            {crew.map((member, i) => (
              <RevealOnScroll key={i} delay={i * 100}>
                <div className="about-page__crew-card">
                  <h4 className="about-page__crew-name">{member.name}</h4>
                  <span className="about-page__crew-role">{member.role}</span>
                  <p className="about-page__crew-note">{member.note}</p>
                </div>
              </RevealOnScroll>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="about-page__faq section section--darker">
        <div className="container">
          <RevealOnScroll>
            <span className="section-subtitle">Questions We're Tired Of</span>
            <h2 className="section-title section-title--center">FAQ</h2>
          </RevealOnScroll>

          <div className="about-page__faq-list">
            {[
              {
                q: "Do you really not use utensils?",
                a: "We really, truly, genuinely do not use utensils. Unless the episode specifically calls for their destruction. Our hands are our primary tools. Secondary tools include gravity, fire, and sheer willpower."
              },
              {
                q: "Is the food actually good?",
                a: "Surprisingly, yes. About 70% of the time. The other 30% ranges from 'edible' to 'we need to see a doctor.' Chaos is a spectrum."
              },
              {
                q: "How many interns have you gone through?",
                a: "We've lost count. The current intern has survived 3 episodes, which is a show record. Their identity is classified for their own protection."
              },
              {
                q: "Is the sourdough starter from Season 1 still alive?",
                a: "Her name is Gertrude. She lives in Jax's apartment. She has her own Instagram. She is thriving and honestly doing better than all of us."
              },
              {
                q: "Can I be a guest on the show?",
                a: "Send us a video of you cooking your worst dish with no utensils. We'll judge you. If you survive the judgement, we'll talk."
              },
            ].map((item, i) => (
              <RevealOnScroll key={i} delay={i * 80}>
                <div className="about-page__faq-item">
                  <h4 className="about-page__faq-q">{item.q}</h4>
                  <p className="about-page__faq-a">{item.a}</p>
                </div>
              </RevealOnScroll>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="about-page__cta section section--grime">
        <div className="container">
          <RevealOnScroll>
            <div className="about-page__cta-inner">
              <h2 className="about-page__cta-title">
                READY TO<br />
                <span className="about-page__cta-accent">JOIN THE CHAOS?</span>
              </h2>
              <div className="about-page__cta-actions">
                <a
                  href="https://www.youtube.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn--primary btn--large"
                >
                  Subscribe on YouTube
                </a>
                <Link to="/season/3" className="btn btn--outline btn--large">
                  Start Watching
                </Link>
              </div>
            </div>
          </RevealOnScroll>
        </div>
      </section>
    </div>
  );
}

export default AboutPage;
