# DSCI 641 — Music Recommender System
**Course:** DSCI 641 Recommender Systems, Spring 2026
**Option:** 2 — Live, Web-Based Recommender System

---

## The Problem We're Solving

Streaming platforms like Spotify are great at recommending popular artists — but that's exactly the problem. Their algorithms are optimized for engagement, which means they keep pushing the same top 1% of artists to everyone. Users get stuck in a loop of the familiar.

**Our research question:** Can we recommend lesser-known, diverse artists that a user would genuinely enjoy — artists they never would have found on their own?

Our EDA on the Last.fm dataset confirms this is worth solving: the top 10% of artists capture 85% of all plays. The other 14,000+ artists in the dataset are virtually invisible despite being genuinely good music.

---

## What We're Building

A web application where a user connects their Spotify account and receives personalized recommendations for artists they've never heard of. The emphasis is on **novelty and diversity** — surfacing real music from the long tail, not just more of what's already popular.

**User story:** Maya opens the app and clicks "Connect with Spotify." The app reads her listening history and generates 10 artist recommendations. Each card shows the artist's name, their tags/genre description, a novelty score, and a link to listen on Spotify. The artists are ones she's genuinely never heard — not underground for the sake of it, but meaningfully matched to her taste. She saves two of them. Next week, the recommendations refresh.

---

## Dataset

### What We Have (Experimental Data)
**HetRec 2011 Last.fm Dataset** — sourced from GroupLens Research

| File | Contents | Size |
|------|----------|------|
| `artists.dat` | Artist ID, name, Last.fm URL | 17,632 artists |
| `user_artists.dat` | User → Artist play counts (implicit feedback) | 92,834 interactions |
| `tags.dat` | Tag vocabulary | 11,946 tags |
| `user_taggedartists.dat` | User → Artist → Tag annotations | — |
| `user_friends.dat` | Social graph (who follows whom) | 12,717 edges |

Key properties:
- **Sparsity:** ~97% — most users have heard a tiny fraction of available artists
- **Power law distribution:** top 10% of artists = 85% of all plays
- **Implicit feedback:** play counts, not explicit ratings — requires different modeling

### The 2011 Problem and How We're Addressing It
The dataset ends in 2011 — 15 years of music, artists, and entire genres are missing. We're addressing this by **enriching the dataset with the Spotify API**:

- For each of the 17,632 artists, we'll query Spotify's Search API by name
- Pull back: current `popularity` score (0–100), `followers`, and `genres`
- This gives us 2026-accurate definitions of "obscure" instead of relying on 2011 play counts
- Expected match rate: ~80–90% of artists (some name ambiguity/mismatches expected)

This means our interaction graph (who listened to what) stays historical, but our content features and popularity definitions are current.

### What the Real App Would Use (Ideal Data)
- Spotify streaming history via API (real play counts per user)
- Spotify artist metadata (genres, popularity, follower count, related artists)
- Potentially Last.fm scrobbles for users who have linked accounts

The Last.fm dataset is an appropriate proxy because it's the same domain (music listening), same platform, same implicit feedback structure. We use it to validate that our algorithms work before building on live data.

---

## Algorithms

We're testing four algorithms, progressing from simple to sophisticated. This progression is intentional — each model adds more information than the last.

### 1. Popularity Baseline
**What it does:** Recommends the most globally popular artists the user hasn't heard yet.
**Why:** Required course baseline. Also makes the point that a naive popularity recommender is exactly what we're trying to beat on novelty metrics.
**Implementation:** pandas — rank artists by total plays, filter out ones the user has heard.

### 2. User-Based Collaborative Filtering
**What it does:** Finds users with similar listening histories, recommends what they listened to.
**Why:** Classic CF — good benchmark for whether taste similarity alone can surface novel artists.
**Implementation:** LensKit

### 3. Matrix Factorization (ALS/BPR)
**What it does:** Learns latent taste representations from implicit feedback (play counts). Finds a lower-dimensional space where similar users and artists are close together.
**Why:** Strong standard baseline for implicit feedback datasets. ALS (Alternating Least Squares) or BPR (Bayesian Personalized Ranking) are both appropriate here.
**Implementation:** `implicit` library

### 4. LightGCN with Social Augmentation (Graph Neural Network)
**What it does:** Models the entire system as a heterogeneous graph — users and artists as nodes, listens and friendships as edges. Learns embeddings by propagating information across the graph.
**Why:** Our dataset has a social graph (`user_friends.dat`) that the other algorithms completely ignore. GNNs can capture the idea that your friends' taste influences your recommendations — and that artists in the same "subculture cluster" belong together even if no single user has heard both.

**Graph schema:**
```
Nodes:  User nodes (1,892)
        Artist nodes (17,632)

Edges:  [user] --listens--> [artist]   (from user_artists.dat, weighted by log play count)
        [user] --friend-->  [user]      (from user_friends.dat, made undirected)

Features:
        Artists: bag-of-tags multi-hot vectors (from user_taggedartists.dat)
        Users:   learnable embeddings (no user metadata available)
```

**Architecture:** Modified LightGCN
- Layer 1: Each artist embedding updated by users who listen to it
- Layer 2: Each user embedding updated by their friends' taste
- Layer 3: Artist similarity propagated through shared listeners and tags
- Final embedding: sum across all layers (standard LightGCN approach)
- Training: BPR loss with negative sampling

**Implementation:** PyTorch Geometric (PyG)

**Important implementation note:** Play counts need log-normalization before scaling to 0–1. The raw counts have extreme outliers (power law) that would drown out the signal otherwise:
```python
weight_normalized = log1p(weight) / max(log1p(weight))
```

---

## Evaluation

### Train/Test Split
Per-user 80/20 random split — we hold out 20% of each user's listened artists as the test set. We cannot do a temporal split because `user_artists.dat` has no timestamps.

### Accuracy Metrics
- **NDCG@10** — measures whether relevant artists appear near the top of the recommendation list
- **Recall@10** — what fraction of held-out artists appear in the top 10 recommendations

### Beyond-Accuracy Metrics (novelty focus)
These are central to our research question:

| Metric | Definition |
|--------|-----------|
| **Expected Popularity Complement (EPC)** | Average inverse popularity of recommended artists. Higher = more novel. |
| **Intra-list Diversity** | How different are the 10 recommended artists from each other (using tag vector cosine distance) |
| **Tail Coverage** | % of recommendations drawn from the bottom 80% of artists by play count |

We'll report all metrics for all 4 algorithms in a comparison table. The goal is to find a model that doesn't sacrifice too much accuracy in exchange for meaningful novelty gains.

---

## Web Application

### Stack
- **Backend:** FastAPI (Python) — recommendation endpoint, Spotify OAuth handling
- **Frontend:** Simple React or plain HTML/JS — artist cards with name, genre tags, novelty score, Spotify preview
- **Spotify Integration:** OAuth 2.0 to read user's top artists/recent plays, map to Last.fm ID space by artist name

### The Matching Problem
The live app needs to bridge Spotify data (what a real user listens to) with the Last.fm ID space (what our model knows). We match by artist name with fuzzy matching to handle slight spelling differences. Unmatched artists are dropped. This is the messiest engineering part of the project.

### What the App Does, Step by Step
1. User clicks "Connect Spotify"
2. App pulls their top 50 artists via Spotify API
3. Names are matched to Last.fm artist IDs
4. Matched artists are treated as the user's "listening history" input to the model
5. Model generates top-N recommendations from the Last.fm artist space
6. App displays artist cards enriched with Spotify metadata (image, genres, popularity)

---

## Timeline

| Dates | Milestone | Deliverable |
|-------|-----------|-------------|
| Now → Apr 12 | Finish EDA, confirm team | EDA notebook |
| Apr 12 → Apr 24 | Write proposal | Proposal document |
| Apr 24 → May 9 | Implement baselines (CF, MF) + evaluation framework | Working models + metrics |
| May 9 → May 23 | Implement LightGCN, Spotify API enrichment | GNN model, enriched dataset |
| May 23 → Jun 1 | Build web app, integrate best model | Working demo |
| Jun 1 → Jun 3 | Polish slides, rehearse demo | Presentation |
| Jun 3 → Jun 10 | Write final report | Final submission |

---

## Division of Work (Proposed)

| Area | Description |
|------|-------------|
| **Data & EDA** | Data loading, cleaning, Spotify API enrichment, EDA notebook |
| **Algorithms** | CF, MF, LightGCN implementation and evaluation |
| **Web App Backend** | FastAPI, Spotify OAuth, recommendation endpoint |
| **Web App Frontend** | UI, artist cards, Spotify integration |

These can be split 2+2 or mixed depending on strengths. The algorithm work and web app work can run largely in parallel once the evaluation framework is in place.

---

## Key Risks

| Risk | Mitigation |
|------|-----------|
| Spotify ↔ Last.fm name matching is noisy | Build matching pipeline early, manually verify edge cases, accept ~80% match rate |
| GNN training time / debugging | Start LightGCN after baselines are working; 9 weeks is sufficient buffer |
| Option 2 full requirements not yet posted | Monitor course page — framework from Option 1 gives enough to start |
| 2011 dataset missing modern artists | Spotify API enrichment addresses metadata; clearly acknowledged in proposal as a justified limitation |

---

## What Success Looks Like

A working web app where a real user can log in with Spotify and receive artist recommendations that are:
- Genuinely relevant to their taste (competitive NDCG vs. popularity baseline)
- Meaningfully more novel than what Spotify would recommend (higher EPC, higher tail coverage)
- Explainable — the user can see *why* an artist was recommended (shared tags, similar listeners)

And a report that clearly justifies every algorithmic and data decision in terms of the application — which is exactly what the rubric asks for.
