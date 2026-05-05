import React from 'react';
import { motion } from 'framer-motion';
import RevealOnScroll from './RevealOnScroll';
import './Merch.css';

const merchItems = [
  {
    id: 1,
    name: "BURN THE SPATULA Tee",
    price: "$32",
    color: "#ff6b35",
    tag: "BEST SELLER",
    desc: "Heavy cotton. Pre-shrunk. Covered in sauce (painted, not real)."
  },
  {
    id: 2,
    name: "CHOPSTICKS ARE FOR COWARDS Hoodie",
    price: "$58",
    color: "#cc0000",
    tag: "NEW",
    desc: "Thick fleece. Hood that actually fits over a human head. Anti-utensil manifesto on the back."
  },
  {
    id: 3,
    name: "HANDS ONLY Apron",
    price: "$28",
    color: "#c4a035",
    tag: null,
    desc: "Waxed canvas. Adjustable. Has pockets for things you shouldn't put in pockets."
  },
  {
    id: 4,
    name: "THE MELTDOWN Poster",
    price: "$18",
    color: "#cc0000",
    tag: "SEASON 2",
    desc: "18x24 screen print. Signed by no one. Looks incredible on a wall."
  }
];

function Merch() {
  return (
    <section className="merch section section--dark" id="merch">
      <div className="container">
        <RevealOnScroll>
          <span className="section-subtitle">Rep The Chaos</span>
          <h2 className="section-title">Merch</h2>
        </RevealOnScroll>

        <div className="merch__grid">
          {merchItems.map((item, i) => (
            <RevealOnScroll key={item.id} delay={i * 100}>
              <motion.div
                className="merch__card"
                whileHover={{ y: -8 }}
                transition={{ duration: 0.3 }}
              >
                <div
                  className="merch__image"
                  style={{
                    background: `linear-gradient(135deg, ${item.color}22, ${item.color}08)`,
                  }}
                >
                  <div className="merch__image-placeholder">
                    <span
                      className="merch__image-text"
                      style={{ color: item.color }}
                    >
                      {item.name.split(' ')[0]}
                    </span>
                  </div>
                  {item.tag && (
                    <span
                      className="merch__tag"
                      style={{ background: item.color }}
                    >
                      {item.tag}
                    </span>
                  )}
                </div>
                <div className="merch__info">
                  <h3 className="merch__name">{item.name}</h3>
                  <p className="merch__desc">{item.desc}</p>
                  <div className="merch__bottom">
                    <span className="merch__price">{item.price}</span>
                    <button
                      className="merch__buy"
                      style={{ borderColor: item.color, color: item.color }}
                    >
                      Add to Cart
                    </button>
                  </div>
                </div>
              </motion.div>
            </RevealOnScroll>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Merch;
