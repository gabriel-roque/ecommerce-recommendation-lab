import numpy as np

from app.models.features import ARTICLE_FEATURES, CUSTOMER_FEATURES, HybridFeatures
from app.models.mappings import DenseMappings
from app.recommendation.popularity import PopularityRecommender
from app.recommendation.service import RecommendationService


class TrackingModel:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, features, **kwargs):
        del kwargs
        items = np.asarray(features["article_index"])
        self.batch_sizes.append(items.size)
        return (items / 10).reshape(-1, 1)


def hybrid_features(mappings):
    customers = [
        {"customer_key": key, **{name: "known" for name in CUSTOMER_FEATURES}}
        for key in mappings.customer_keys
    ]
    articles = [
        {"article_key": key, **{name: f"value-{key}" for name in ARTICLE_FEATURES}}
        for key in mappings.article_keys
    ]
    return HybridFeatures.build(mappings, customers, articles)


def test_popularity_counts_purchases_and_uses_deterministic_ties():
    purchases = [
        {"article_key": 20},
        {"article_key": 10},
        {"article_key": 20},
        {"article_key": 30},
    ]
    popularity = PopularityRecommender.fit(purchases)
    assert [item for item, _ in popularity.recommend(3)] == [20, 10, 30]
    assert popularity.recommend(1, exclude_item_ids={10})[0][0] == 20


def test_known_user_scores_catalog_in_batches_and_excludes_seen_items():
    mappings = DenseMappings.build([100], [10, 20, 30, 40, 50])
    popularity = PopularityRecommender(np.asarray([10, 20]), np.asarray([1.0, 0.5]))
    model = TrackingModel()
    service = RecommendationService(
        model,
        mappings,
        hybrid_features(mappings),
        popularity,
        {100: {40, 50}},
        batch_size=2,
    )

    result = service.recommend(100, k=2)

    assert result.strategy == "neural"
    assert [item.article_key for item in result.recommendations] == [30, 20]
    assert all(size <= 2 for size in model.batch_sizes)
    assert not {40, 50}.intersection(item.article_key for item in result.recommendations)


def test_unknown_user_gets_popularity_cold_start():
    mappings = DenseMappings.build([100], [10, 20, 30])
    popularity = PopularityRecommender(
        np.asarray([30, 20, 10]), np.asarray([1.0, 0.5, 0.25])
    )
    service = RecommendationService(
        TrackingModel(), mappings, hybrid_features(mappings), popularity, {}
    )

    result = service.recommend(999, k=2)

    assert result.strategy == "popularity"
    assert result.cold_start is True
    assert result.reason == "customer_not_in_hm_artifact"
    assert [item.article_key for item in result.recommendations] == [30, 20]


def test_known_user_reranks_only_configured_candidates():
    mappings = DenseMappings.build([100], [10, 20, 30, 40, 50])
    popularity = PopularityRecommender(
        np.asarray([20, 40, 50]), np.asarray([1.0, 0.8, 0.6])
    )
    model = TrackingModel()
    service = RecommendationService(
        model,
        mappings,
        hybrid_features(mappings),
        popularity,
        {},
        batch_size=8,
        candidate_article_keys=popularity.article_keys,
        candidate_limit=2,
    )

    result = service.recommend(100, k=1)

    assert [item.article_key for item in result.recommendations] == [40]
    assert model.batch_sizes == [2]
