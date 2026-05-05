import React, { useState } from 'react';
import { motion } from 'framer-motion';
import RevealOnScroll from './RevealOnScroll';
import './Newsletter.css';

function Newsletter() {
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState('idle'); // idle | submitting | success

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!email) return;
    setStatus('submitting');
    setTimeout(() => {
      setStatus('success');
      setEmail('');
    }, 1500);
  };

  return (
    <section className="newsletter section section--grime">
      <div className="container">
        <RevealOnScroll>
          <div className="newsletter__inner">
            <div className="newsletter__content">
              <span className="section-subtitle">Join The Anarchy</span>
              <h2 className="newsletter__title">
                GET THE<br />
                <span className="newsletter__title-accent">DISPATCH</span>
              </h2>
              <p className="newsletter__desc">
                New episodes. Behind-the-scenes disasters. Unreleased footage of things going horribly wrong.
                Delivered to your inbox like a flaming bag of content.
              </p>
            </div>

            <div className="newsletter__form-wrap">
              {status === 'success' ? (
                <motion.div
                  className="newsletter__success"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <span className="newsletter__success-icon">&#10003;</span>
                  <p>You're in. Check your inbox for confirmation (and probably hot sauce stains).</p>
                </motion.div>
              ) : (
                <form className="newsletter__form" onSubmit={handleSubmit}>
                  <div className="newsletter__input-wrap">
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="your@email.com"
                      className="newsletter__input"
                      required
                      aria-label="Email address"
                    />
                    <button
                      type="submit"
                      className="newsletter__submit"
                      disabled={status === 'submitting'}
                    >
                      {status === 'submitting' ? 'SENDING...' : 'SUBSCRIBE'}
                    </button>
                  </div>
                  <p className="newsletter__disclaimer">
                    No spam. Unsubscribe anytime. We respect your inbox more than we respect utensils.
                  </p>
                </form>
              )}
            </div>
          </div>
        </RevealOnScroll>
      </div>
    </section>
  );
}

export default Newsletter;
