// Episodes data organized by season
export const seasons = [
  {
    number: 1,
    name: "The Reckoning",
    year: "2024",
    description: "Where it all began. No rules. No utensils. Just food."
  },
  {
    number: 2,
    name: "Into the Fire",
    year: "2024",
    description: "We pushed harder. Hotter. Messier. Nothing was sacred."
  },
  {
    number: 3,
    name: "Raw & Uncut",
    year: "2025",
    description: "Current season. We stopped pretending this was normal."
  }
];

export const episodes = [
  // Season 1
  {
    id: "s1e1-the-scramble",
    season: 1,
    episode: 1,
    title: "The Scramble",
    subtitle: "Eggs Like You've Never Seen Them",
    description: "We opened our first episode by declaring war on the spatula. Four dozen eggs, a cast iron pan heated to 500 degrees, and absolutely zero regard for kitchen safety. The scramble that resulted was simultaneously the best and worst thing we've ever made.",
    duration: "22:14",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Blindfold Egg Crack Challenge", "No-Utensil Scramble", "Floor Seasoning"],
    featured: true,
    date: "2024-01-15",
    views: "2.4M"
  },
  {
    id: "s1e2-meat-mayhem",
    season: 1,
    episode: 2,
    title: "Meat Mayhem",
    subtitle: "The Butcher Weeps",
    description: "We brought in a whole ribeye and treated it with the disrespect it didn't deserve. Dry-aged for 45 days, dropped on the counter, seasoned with whatever was within arm's reach, and cooked by being held over a gas burner with our bare hands (sort of).",
    duration: "25:37",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Chef Marco 'The Animal' Venti"],
    challenges: ["Blindfold Butchery", "Hand-Cooked Steak", "Hot Sauce Roulette"],
    featured: false,
    date: "2024-01-29",
    views: "1.8M"
  },
  {
    id: "s1e3-pasta-anarchy",
    season: 1,
    episode: 3,
    title: "Pasta Anarchy",
    subtitle: "Italy Would Disown Us",
    description: "Fresh pasta made on a dirty table, sauce improvised from whatever we found in the back of the fridge. We used a wine bottle as a rolling pin and a guitar string to cut the noodles. Nonnas everywhere felt a disturbance in the force.",
    duration: "28:02",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Wine Bottle Pasta", "Guitar String Noodles", "Mystery Sauce"],
    featured: false,
    date: "2024-02-12",
    views: "3.1M"
  },
  {
    id: "s1e4-soup-chaos",
    season: 1,
    episode: 4,
    title: "Soup Chaos",
    subtitle: "Liquid Disaster",
    description: "A soup episode that started with good intentions and ended with the fire department being called (not really, but close). We made a bone broth from scratch using only a hammer and a car battery charger (don't ask).",
    duration: "19:45",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Fire Marshal Dan"],
    challenges: ["Hammer Bone Broth", "Blindfold Seasoning", "Soup Pong"],
    featured: false,
    date: "2024-02-26",
    views: "1.2M"
  },
  {
    id: "s1e5-the-ferment",
    season: 1,
    episode: 5,
    title: "The Ferment",
    subtitle: "Controlled Decay",
    description: "We attempted fermentation with zero knowledge and maximum confidence. Kimchi that could strip paint, hot sauce that made our cameraman cry, and sourdough that somehow turned sentient.",
    duration: "31:20",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Speed Fermentation", "Hot Sauce Gauntlet", "Sourdough Naming Ceremony"],
    featured: false,
    date: "2024-03-11",
    views: "980K"
  },
  {
    id: "s1e6-finale-burn",
    season: 1,
    episode: 6,
    title: "The Finale Burn",
    subtitle: "Everything Goes Up In Flames",
    description: "Season 1 finale. We invited every guest back, gave them each a mystery basket, one utensil (which they had to destroy before cooking), and 30 minutes. The kitchen looked like a war zone. The food was transcendent.",
    duration: "45:12",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Chef Marco 'The Animal' Venti", "Fire Marshal Dan"],
    challenges: ["Utensil Destruction Ceremony", "Mystery Basket Mayhem", "Final Plating Without Hands"],
    featured: true,
    date: "2024-03-25",
    views: "5.7M"
  },
  // Season 2
  {
    id: "s2e1-grill-or-die",
    season: 2,
    episode: 1,
    title: "Grill Or Die",
    subtitle: "Fire Is The Only Utensil",
    description: "Season 2 opens with fire. Literally. We built a grill out of shopping cart parts and cooked everything directly on the coals. Chicken, corn, fish, shoes (okay not shoes). The grill collapsed mid-cook and we kept going.",
    duration: "24:33",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Shopping Cart Grill Build", "Coal-Wrapped Fish", "Survival Cooking"],
    featured: false,
    date: "2024-06-03",
    views: "2.1M"
  },
  {
    id: "s2e2-raw-day",
    season: 2,
    episode: 2,
    title: "Raw Day",
    subtitle: "No Heat. No Mercy.",
    description: "We banned all forms of heat. No fire, no oven, no microwave, no warm thoughts. Everything was raw, cold, and honest. Ceviche made in a kiddie pool. Tartare mixed with bare hands. The crew wore gloves. We did not.",
    duration: "21:18",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Sashimi Master Tanaka"],
    challenges: ["Kiddie Pool Ceviche", "Blindfold Tartare", "Ice Carving Plate"],
    featured: true,
    date: "2024-06-17",
    views: "3.4M"
  },
  {
    id: "s2e3-bread-wars",
    season: 2,
    episode: 3,
    title: "Bread Wars",
    subtitle: "The Staff Of Life Gets Staffed",
    description: "We challenged ourselves to make bread using only ancient methods. Hands for kneading, a hot rock for baking, and tears for salt (okay, we used actual salt). The sourdough starter from Season 1 made a surprise return and it was ANGRY.",
    duration: "27:45",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Ancient Bread Method", "Hot Rock Baking", "Sourdough Revenge Arc"],
    featured: false,
    date: "2024-07-01",
    views: "1.5M"
  },
  {
    id: "s2e4-dessert-disaster",
    season: 2,
    episode: 4,
    title: "Dessert Disaster",
    subtitle: "Sugar Is A Weapon",
    description: "Desserts made with aggression. Caramel heated to the temperature of the sun. Cake batter mixed in a paint bucket. Ice cream churned by running in a circle. Everything was on fire at some point, including the sugar.",
    duration: "23:58",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Pastry Rebel 'Sugar Vandal'"],
    challenges: ["Lava Caramel", "Paint Bucket Cake", "Human Ice Cream Churn"],
    featured: false,
    date: "2024-07-15",
    views: "2.8M"
  },
  {
    id: "s2e5-global-assault",
    season: 2,
    episode: 5,
    title: "Global Assault",
    subtitle: "Offending Every Cuisine Simultaneously",
    description: "We attempted dishes from 6 different countries in 30 minutes with no recipes and no respect for tradition. Somehow the tacos were good. The sushi was a crime. The curry made someone hallucinate (mildly).",
    duration: "32:10",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["6 Countries, 30 Minutes", "No Recipe Roulette", "Cultural Sensitivity Training (Failed)"],
    featured: false,
    date: "2024-07-29",
    views: "4.2M"
  },
  {
    id: "s2e6-season-2-finale",
    season: 2,
    episode: 6,
    title: "The Meltdown",
    subtitle: "Season 2 Ends In Tears",
    description: "The finale that broke us. Each host had to cook their nemesis dish—the one thing they've never been able to make. No utensils, no help, no safety net. One person cried. One person walked out. The food was incredible.",
    duration: "48:30",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Chef Marco 'The Animal' Venti", "Sashimi Master Tanaka", "Pastry Rebel 'Sugar Vandal'"],
    challenges: ["Nemesis Dish Showdown", "Emotional Breakdown Cooking", "Final Judgement"],
    featured: true,
    date: "2024-08-12",
    views: "7.1M"
  },
  // Season 3 (current)
  {
    id: "s3e1-deconstruction",
    season: 3,
    episode: 1,
    title: "Deconstruction",
    subtitle: "Take Everything Apart",
    description: "Season 3 opener. We deconstructed a five-course fine dining menu and rebuilt it using only hands, fire, and gravity. The amuse-bouche was launched from a catapult. We're not explaining further.",
    duration: "26:44",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Catapult Amuse-Bouche", "Gravity Plating", "Deconstructed Everything"],
    featured: true,
    date: "2025-01-20",
    views: "1.9M"
  },
  {
    id: "s3e2-the-forage",
    season: 3,
    episode: 2,
    title: "The Forage",
    subtitle: "Nature's Grocery Store",
    description: "We left the kitchen entirely. Foraging in the woods, cooking over a campfire, eating with sticks. The most primal episode we've ever made. A raccoon stole our main ingredient and we had to improvise.",
    duration: "29:15",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: ["Wilderness Chef 'Moss'"],
    challenges: ["Forest Foraging", "Campfire Cooking", "Raccoon Improv"],
    featured: false,
    date: "2025-02-03",
    views: "2.3M"
  },
  {
    id: "s3e3-living-ingredient",
    season: 3,
    episode: 3,
    title: "The Living Ingredient",
    subtitle: "Alive Until The Last Second",
    description: "Everything in this episode was alive when we started. Lobster, clams, the sourdough starter (still alive from S1), and a head of lettuce that we swear had a personality. The most honest cooking we've ever done.",
    duration: "25:00",
    thumbnail: null,
    videoUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    guests: [],
    challenges: ["Live Lobster Takedown", "Sourdough Check-In", "Lettuce Eulogy"],
    featured: false,
    date: "2025-02-17",
    views: "1.7M"
  }
];

export const featuredEpisodes = episodes.filter(ep => ep.featured);

export const getEpisodesBySeason = (seasonNumber) =>
  episodes.filter(ep => ep.season === seasonNumber);

export const getEpisodeById = (id) =>
  episodes.find(ep => ep.id === id);

export const getLatestEpisode = () =>
  episodes[episodes.length - 1];
