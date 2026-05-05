import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import FeaturedEpisode from './components/FeaturedEpisode';
import EpisodeGrid from './components/EpisodeGrid';
import Hosts from './components/Hosts';
import Merch from './components/Merch';
import Newsletter from './components/Newsletter';
import Footer from './components/Footer';
import EpisodePage from './components/EpisodePage';
import SeasonPage from './components/SeasonPage';
import AboutPage from './components/AboutPage';
import ScrollToTop from './components/ScrollToTop';
import './App.css';

function HomePage() {
  return (
    <>
      <Hero />
      <FeaturedEpisode />
      <EpisodeGrid />
      <Hosts />
      <Merch />
      <Newsletter />
    </>
  );
}

function App() {
  return (
    <Router>
      <div className="App">
        <div className="grain-overlay" aria-hidden="true" />
        <Navbar />
        <ScrollToTop />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/episode/:id" element={<EpisodePage />} />
            <Route path="/season/:number" element={<SeasonPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
