from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import hashlib


# ============================================================
# Configs immuables (invariants explicites)
# ============================================================

@dataclass(frozen=True)
class EncoderConfig:
    dim: int = 64
    ngram: int = 3
    hash_algorithm: str = "blake2b"
    normalize: bool = True

    def __post_init__(self):
        assert self.dim > 0
        assert self.ngram >= 1
        assert self.hash_algorithm in hashlib.algorithms_available


@dataclass(frozen=True)
class GeometryConfig:
    strength: float = 0.15
    spherical: bool = True

    def __post_init__(self):
        assert 0.0 <= self.strength <= 1.0


@dataclass(frozen=True)
class KMeansConfig:
    k: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    spherical: bool = True
    min_cluster_size: int = 1

    def __post_init__(self):
        assert self.k >= 1
        assert self.max_iter >= 1
        assert self.tol > 0.0
        assert self.min_cluster_size >= 1


@dataclass(frozen=True)
class ScoringConfig:
    sigma: float = 0.75
    size_weight: float = 0.3
    compact_weight: float = 0.2
    separation_weight: float = 0.2
    balance_weight: float = 0.3

    def __post_init__(self):
        assert self.sigma > 0.0
        for w in (self.size_weight, self.compact_weight,
                  self.separation_weight, self.balance_weight):
            assert 0.0 <= w <= 1.0


@dataclass(frozen=True)
class PipelineConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    geom: GeometryConfig = field(default_factory=GeometryConfig)
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    global_seed: int = 123456789

    def __post_init__(self):
        assert isinstance(self.global_seed, int)


# ============================================================
# Utils déterministes
# ============================================================

def l2(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < eps:
            return np.zeros_like(v)
        return v / n
    elif v.ndim == 2:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n = np.where((n < eps) | ~np.isfinite(n), 1.0, n)
        return v / n
    else:
        raise ValueError("l2 expects 1D or 2D array")


def deterministic_hash_int(s: str, algorithm: str = "blake2b") -> int:
    h = getattr(hashlib, algorithm)(s.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def make_rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def deterministic_normal_vector(dim: int, seed: int) -> np.ndarray:
    rng = make_rng_from_seed(seed)
    v = rng.normal(size=dim)
    return l2(v)


def deterministic_orthonormal_basis_qr(dim: int, seed: int) -> np.ndarray:
    """
    Base orthonormale déterministe via QR (plus stable que MGS maison).
    Invariant : Q approx orthonormale, signe fixé sur la première coordonnée.
    """
    rng = make_rng_from_seed(seed)
    M = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(M)
    # Fixer le signe pour déterminisme strict
    signs = np.where(Q[:, 0] >= 0, 1.0, -1.0)
    Q = Q * signs[:, None]
    return Q.astype(np.float64)


# ============================================================
# Pipeline monolithique 10/10 (disruptif, mais testable)
# ============================================================

class IdeaPipeline10X:
    """
    Pipeline monolithique déterministe, vectorisé, avec invariants explicites.
    API publique : `run(query, ideas)`.
    Tout est encapsulé dans une seule classe, mais chaque bloc est pur.
    """

    def __init__(self, cfg: PipelineConfig = PipelineConfig()):
        self.cfg = cfg
        self._projection = deterministic_orthonormal_basis_qr(
            self.cfg.encoder.dim,
            seed=self.cfg.global_seed,
        )
        self._check_projection_invariants()

    # --------------------------------------------------------
    # Invariants internes
    # --------------------------------------------------------

    def _check_projection_invariants(self, tol: float = 1e-6) -> None:
        Q = self._projection
        assert Q.shape == (self.cfg.encoder.dim, self.cfg.encoder.dim)
        I = Q @ Q.T
        err = np.linalg.norm(I - np.eye(Q.shape[0]))
        assert err < tol, f"Projection not orthonormal enough, err={err}"

    # ========================================================
    # Bloc 1 : Encodage déterministe (n-grams + hashing)
    # ========================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        n = self.cfg.encoder.ngram
        if not text:
            return []
        if len(text) <= n:
            return [text]
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def _encode_token(self, token: str) -> np.ndarray:
        seed = deterministic_hash_int(token, self.cfg.encoder.hash_algorithm)
        return deterministic_normal_vector(self.cfg.encoder.dim, seed)

    def _encode_text(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            empty_seed = deterministic_hash_int("<EMPTY>", self.cfg.encoder.hash_algorithm)
            v = deterministic_normal_vector(self.cfg.encoder.dim, empty_seed)
        else:
            embs = np.stack([self._encode_token(t) for t in tokens], axis=0)
            # pondération simple par fréquence (TF-like)
            v = embs.mean(axis=0)

        v_proj = v @ self._projection
        if self.cfg.encoder.normalize:
            v_proj = l2(v_proj)
        assert v_proj.shape == (self.cfg.encoder.dim,)
        return v_proj.astype(np.float64)

    def _encode_ideas(self, ideas: List[Dict[str, Any]]) -> np.ndarray:
        texts = [str(idea["text"]) for idea in ideas]
        embs = np.stack([self._encode_text(t) for t in texts], axis=0)
        assert embs.ndim == 2 and embs.shape[1] == self.cfg.encoder.dim
        return embs

    # ========================================================
    # Bloc 2 : Géométrie query-aware (SLERP / interpolation)
    # ========================================================

    @staticmethod
    def _slerp_batch(X: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
        X = l2(X)
        q = l2(q)
        dot = np.clip(X @ q, -1.0, 1.0)
        close = np.abs(dot - 1.0) < 1e-8
        far = ~close

        Y = np.empty_like(X)

        if np.any(close):
            Xc = X[close]
            Yc = (1.0 - alpha) * Xc + alpha * q
            Y[close] = l2(Yc)

        if np.any(far):
            Xf = X[far]
            dot_f = dot[far]
            theta = np.arccos(dot_f)
            sin_theta = np.sin(theta)
            sin_theta = np.where(np.abs(sin_theta) < 1e-12, 1e-12, sin_theta)
            a = np.sin((1.0 - alpha) * theta) / sin_theta
            b = np.sin(alpha * theta) / sin_theta
            Yf = a[:, None] * Xf + b[:, None] * q
            Y[far] = l2(Yf)

        return Y

    def _deform_embeddings(self, embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
        if embeddings.size == 0 or self.cfg.geom.strength <= 0.0:
            return embeddings
        query = query.reshape(-1)
        if not self.cfg.geom.spherical:
            Y = (1.0 - self.cfg.geom.strength) * embeddings + self.cfg.geom.strength * query
            return l2(Y) if self.cfg.encoder.normalize else Y
        return self._slerp_batch(embeddings, query, self.cfg.geom.strength)

    # ========================================================
    # Bloc 3 : KMeans déterministe renforcé
    # ========================================================

    @staticmethod
    def _kmeans_init_centers(X: np.ndarray, k: int) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        k = min(k, n)
        if k == 0:
            return np.zeros((0, d), dtype=np.float64)

        # tri lexicographique pour déterminisme
        order = np.lexsort([X[:, i] for i in reversed(range(d))])
        X_sorted = X[order]

        centers = []
        centers.append(X_sorted[n // 2])

        if k == 1:
            return np.stack(centers, axis=0)

        dists = np.linalg.norm(X_sorted - centers[0], axis=1)
        for _ in range(1, k):
            idx = int(np.argmax(dists))
            centers.append(X_sorted[idx])
            C = np.stack(centers, axis=0)
            d_new = np.linalg.norm(X_sorted[:, None, :] - C[None, :, :], axis=2).min(axis=1)
            dists = np.minimum(dists, d_new)

        centers = np.stack(centers, axis=0)
        return centers

    @staticmethod
    def _kmeans_assign(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if centers.size == 0:
            n = X.shape[0]
            return np.zeros(n, dtype=int), np.full(n, np.inf, dtype=np.float64)

        diff = X[:, None, :] - centers[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        labels = dists.argmin(axis=1)
        d_min = dists[np.arange(X.shape[0]), labels]
        return labels, d_min

    def _kmeans_update_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        n, d = X.shape
        k = centers.shape[0]
        new_centers = np.zeros((k, d), dtype=np.float64)

        for i in range(k):
            mask = labels == i
            count = int(mask.sum())
            if count >= self.cfg.kmeans.min_cluster_size:
                new_centers[i] = X[mask].mean(axis=0)
            else:
                # cluster vide ou trop petit -> point le plus éloigné du barycentre global
                global_center = X.mean(axis=0, keepdims=True)
                dists = np.linalg.norm(X - global_center, axis=1)
                farthest_idx = int(np.argmax(dists))
                new_centers[i] = X[farthest_idx]

        return new_centers

    def _kmeans_run(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        assert X.ndim == 2
        n, d = X.shape
        k = min(self.cfg.kmeans.k, n)
        if k == 0:
            return np.zeros((0, d), dtype=np.float64), np.zeros(n, dtype=int)

        centers = self._kmeans_init_centers(X, k)
        labels = np.zeros(n, dtype=int)
        prev_labels: Optional[np.ndarray] = None

        for _ in range(self.cfg.kmeans.max_iter):
            labels, _ = self._kmeans_assign(X, centers)

            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()

            new_centers = self._kmeans_update_centers(X, labels, centers)

            if self.cfg.kmeans.spherical:
                new_centers = l2(new_centers)

            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            if shift < self.cfg.kmeans.tol:
                break

        assert centers.shape == (k, d)
        assert labels.shape == (n, )
        return centers, labels

    # ========================================================
    # Bloc 4 : Scoring multi-critères (plus “propre”)
    # ========================================================

    def _score_clusters(
        self,
        centers: np.ndarray,
        query: np.ndarray,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        if centers.size == 0:
            return np.zeros(0, dtype=np.float64)

        cfg = self.cfg.scoring
        k = centers.shape[0]
        query = query.reshape(-1)

        # 1. Distance à la requête (gaussienne)
        dists = np.linalg.norm(centers - query[None, :], axis=1)
        base_scores = np.exp(-(dists**2) / (2.0 * cfg.sigma**2))

        # 2. Taille des clusters
        sizes = np.bincount(labels, minlength=k).astype(np.float64)
        size_scores = np.log1p(sizes)
        if size_scores.max() > 0:
            size_scores /= size_scores.max()

        # 3. Compacité (variance intra-cluster)
        compact_scores = np.zeros(k, dtype=np.float64)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                cluster_embs = embeddings[mask]
                center = centers[i]
                diff = cluster_embs - center[None, :]
                var = np.mean(np.sum(diff**2, axis=1))
                compact_scores[i] = 1.0 / (1.0 + var)
        if compact_scores.max() > 0:
            compact_scores /= compact_scores.max()

        # 4. Séparation inter-centres
        if k > 1:
            diff_cc = centers[:, None, :] - centers[None, :, :]
            dist_cc = np.linalg.norm(diff_cc, axis=2)
            separation = dist_cc.sum(axis=1) / (k - 1)
            separation_scores = separation / (separation.max() + 1e-12)
        else:
            separation_scores = np.ones(k, dtype=np.float64)

        # 5. Balance des tailles
        mean_size = sizes.mean() if k > 0 else 1.0
        balance_scores = 1.0 / (1.0 + np.abs(sizes - mean_size))
        if balance_scores.max() > 0:
            balance_scores /= balance_scores.max()

        scores = base_scores * (
            1.0
            + cfg.size_weight * size_scores
            + cfg.compact_weight * compact_scores
            + cfg.separation_weight * separation_scores
            + cfg.balance_weight * balance_scores
        )
        if scores.max() > 0:
            scores /= scores.max()
        return scores

    # ========================================================
    # Bloc 5 : Synthèse pivot-aware
    # ========================================================

    @staticmethod
    def _build_cluster_ids(labels: np.ndarray, k: int) -> List[List[int]]:
        return [np.where(labels == i)[0].tolist() for i in range(k)]

    def _cluster_keywords(
        self,
        ids: List[int],
        ideas: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[str]:
        from collections import Counter

        counter = Counter()
        for idx in ids:
            text = self._normalize_text(str(ideas[idx]["text"]))
            tokens = self._tokenize(text)
            counter.update(tokens)
        return [t for t, _ in counter.most_common(top_k)]

    def _synthesize(
        self,
        cluster_ids: List[List[int]],
        ideas: List[Dict[str, Any]],
        scores: np.ndarray,
        centers: np.ndarray,
        embeddings: np.ndarray,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for cid, ids in enumerate(cluster_ids):
            if not ids:
                pivot_idx = -1
                pivot_text = ""
                keywords: List[str] = []
            else:
                cluster_embs = embeddings[ids]
                center = centers[cid]
                diff = cluster_embs - center[None, :]
                dists = np.linalg.norm(diff, axis=1)
                local_idx = int(np.argmin(dists))
                pivot_idx = ids[local_idx]
                pivot_text = str(ideas[pivot_idx]["text"])
                keywords = self._cluster_keywords(ids, ideas, top_k=5)

            summary = (
                f"[Cluster {cid}] "
                f"size={len(ids)} "
                f"score={scores[cid]:.3f} "
                f"pivot='{pivot_text[:120].replace('\\n', ' ').strip()}' "
                f"keywords={keywords}"
            )

            results.append(
                {
                    "cluster_id": cid,
                    "size": len(ids),
                    "score": float(scores[cid]),
                    "pivot_index": pivot_idx,
                    "pivot": pivot_text,
                    "keywords": keywords,
                    "summary": summary,
                }
            )
        return results

    # ========================================================
    # API publique
    # ========================================================

    def run(self, query: str, ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Monolithe logique :
          - encodage
          - déformation query-aware
          - clustering KMeans déterministe
          - scoring multi-critères
          - synthèse pivot-aware
        """
        if not ideas:
            return []

        # 1. Encodage
        query_embedding = self._encode_text(query)
        idea_embeddings = self._encode_ideas(ideas)

        assert idea_embeddings.ndim == 2
        assert query_embedding.ndim == 1
        assert idea_embeddings.shape[1] == query_embedding.shape[0]

        # 2. Déformation géométrique
        deformed_embeddings = self._deform_embeddings(idea_embeddings, query_embedding)
        assert deformed_embeddings.shape == idea_embeddings.shape

        # 3. Clustering
        centers, labels = self._kmeans_run(deformed_embeddings)
        k = centers.shape[0]
        cluster_ids = self._build_cluster_ids(labels, k)

        # 4. Scoring
        scores = self._score_clusters(centers, query_embedding, deformed_embeddings, labels)
        assert scores.shape == (k,)

        # 5. Synthèse
        return self._synthesize(cluster_ids, ideas, scores, centers, deformed_embeddings)


# ============================================================
# Exemple d’utilisation rapide
# ============================================================

if __name__ == "__main__":
    ideas = [
        {"text": "Build a micro-SaaS for deterministic scientific pipelines."},
        {"text": "Create a plugin to visualize photon-limited imaging reconstructions."},
        {"text": "Design a robust KMeans implementation for high-dimensional embeddings."},
        {"text": "Write a blog post about deterministic initialization in machine learning."},
        {"text": "Develop an API for clustering and scoring research ideas."},
        {"text": "Optimize numpy operations for real-time embedding generation."},
        {"text": "Implement a caching system for deterministic token embeddings."},
    ]

    pipeline = IdeaPipeline10X()
    query = "How to monetize deterministic scientific code?"
    results = pipeline.run(query, ideas)

    for r in results:
        print(r["summary"])
