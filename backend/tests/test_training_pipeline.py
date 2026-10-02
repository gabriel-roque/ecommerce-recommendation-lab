from types import SimpleNamespace

from app.training.config import TrainingConfig
from app.training.metrics import SampledRankingResult
from app.training.pipeline import TrainingObserver, TrainingPipeline
from app.training.split import TemporalSplit


class FakeModel:
    def fit(self, dataset, **kwargs):
        del dataset, kwargs
        return SimpleNamespace(history={"loss": [0.1]})


def test_pipeline_excludes_every_known_positive_from_negative_sampling(monkeypatch, tmp_path):
    purchases = [
        {"customer_key": 1, "article_key": 10},
        {"customer_key": 1, "article_key": 20},
        {"customer_key": 1, "article_key": 30},
    ]
    split = TemporalSplit(purchases[:1], purchases[1:2], purchases[2:])
    customers = [{"customer_key": 1, "age_band": "25-34"}]
    articles = [
        {
            "article_key": key,
            "department_name": "Casa",
            "category_name": "Cozinha",
            "subcategory_name": "Utensílio",
            "brand": "Aurora",
            "price_band": "econômico",
        }
        for key in (10, 20, 30)
    ]
    datasets = []
    evaluated = []
    seeds = []
    statuses = []

    def fake_dataset(users, items, interactions, **kwargs):
        datasets.append((interactions, kwargs))
        assert all(
            interactions.contains(int(user), int(item))
            for user, item in zip(users, items, strict=True)
        )
        return object()

    def fake_evaluation(model, users, items, interactions, features, **kwargs):
        del model, users, items, features, kwargs
        evaluated.append(interactions)
        return SampledRankingResult(1.0, 1.0, 1.0, 10, 1, 0, 10)

    tensorflow = SimpleNamespace(
        keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: None),
            utils=SimpleNamespace(set_random_seed=seeds.append),
        )
    )
    monkeypatch.setattr("app.training.pipeline.temporal_split", lambda _: split)
    monkeypatch.setattr("app.training.pipeline.build_tf_dataset", fake_dataset)
    monkeypatch.setattr("app.training.pipeline.evaluate_sampled_ranking", fake_evaluation)
    monkeypatch.setattr("app.training.pipeline.require_tensorflow", lambda: tensorflow)
    monkeypatch.setattr("app.training.pipeline.save_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.training.pipeline.validate_artifact", lambda _: object())
    monkeypatch.setattr("app.training.pipeline.hardware_info", lambda: {})
    pipeline = TrainingPipeline(
        tmp_path,
        config=TrainingConfig(epochs=1, negative_ratio=1, evaluation_negatives=10),
        observer=TrainingObserver(status_sink=statuses.append),
    )
    model = FakeModel()
    monkeypatch.setattr(pipeline, "_model", lambda *_: model)

    pipeline.run(
        purchases,
        customers=customers,
        articles=articles,
        model_version="test",
        dataset_manifest={"dataset": "ecommerce_sintetico_br", "semantics": "purchase_only", "source_fingerprint": "abc"},
    )

    train_index, validation_index = [entry[0] for entry in datasets]
    assert train_index.seen(0).tolist() == [0]
    assert validation_index.seen(0).tolist() == [0, 1]
    assert datasets[1][1]["resample_each_iteration"] is False
    assert evaluated[0].seen(0).tolist() == [0, 1]
    assert seeds == [42]
    assert statuses == ["PREPARING_DATA", "TRAINING", "EVALUATING", "SAVING_MODEL"]
