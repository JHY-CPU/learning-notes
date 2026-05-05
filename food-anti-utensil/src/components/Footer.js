import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__top">
          <div className="footer__brand">
            <Link to="/" className="footer__logo">
              <span className="footer__logo-icon">&#9746;</span>
              <span className="footer__logo-text">
                ANTI<br />UTENSIL
              </span>
            </Link>
            <p className="footer__tagline">
              Cooking without rules since 2024.<br />
              No forks were used in the making of this show.
            </p>
          </div>

          <div className="footer__links">
            <div className="footer__col">
              <h4 className="footer__col-title">Show</h4>
              <Link to="/season/3" className="footer__link">Seasons</Link>
              <Link to="/about" className="footer__link">About</Link>
              <a href="#episodes" className="footer__link">Episodes</a>
            </div>
            <div className="footer__col">
              <h4 className="footer__col-title">Connect</h4>
              <a href="https://youtube.com" target="_blank" rel="noopener noreferrer" className="footer__link">YouTube</a>
              <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="footer__link">Instagram</a>
              <a href="https://tiktok.com" target="_blank" rel="noopener noreferrer" className="footer__link">TikTok</a>
            </div>
            <div className="footer__col">
              <h4 className="footer__col-title">Legal</h4>
              <a href="#privacy" className="footer__link">Privacy</a>
              <a href="#terms" className="footer__link">Terms</a>
              <a href="#contact" className="footer__link">Contact</a>
            </div>
          </div>
        </div>

        <div className="footer__bottom">
          <p className="footer__copy">
            &copy; {new Date().getFullYear()} Anti-Utensil. All rights reserved. No utensils were harmed.
          </p>
          <p className="footer__credit">
            Designed with <span className="footer__heart">&#9829;</span> and hot sauce.
          </p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
