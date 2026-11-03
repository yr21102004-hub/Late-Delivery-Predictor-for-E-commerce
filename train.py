"""
train.py
---------
Master training script for the E-commerce Late Delivery Prediction System.

Usage:
    python train.py

This script orchestrates the full ML pipeline:
    1. Load raw data
    2. Clean data
    3. Merge tables
    4. Feature engineering
    5. Split into Train / Dev / Test
    6. Train all models
    7. Evaluate on Dev set
    8. Hyperparameter tuning (Random Forest + Gradient Boosting)
    9. Compare and select best model
   10. Refit on Train + Dev
   11. Evaluate on Test set (final evaluation)
   12. Save model as Pickle
   13. Save comparison report

Restrictions:
    - Test set is NEVER used before step 11.
    - Model selection is based exclusively on Dev set performance.
    - All metrics are computed on actual experimental results.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

# ── Add project root to PYTHONPATH ──────────────────────────────────────── #
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import CFG, get_logger
from src.data_processing import DataCleaner, DataLoader, DataMerger, FeatureEngineer
from src.model_evaluation import ModelComparator, ModelEvaluator, save_metrics_json
from src.model_training import DataSplitter, ModelTrainer

logger = get_logger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("E-commerce Late Delivery Prediction — Training Pipeline")
    logger.info("NextGen Institute | Random State: %d", CFG.random_state)
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # Phase 1: Data pipeline                                              #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 1: Data Loading & Cleaning ---")
    loader  = DataLoader()
    tables  = loader.load_all()

    cleaner = DataCleaner()
    cleaned = cleaner.clean_all(tables)

    merger = DataMerger()
    merged = merger.merge_all(cleaned)
    merger.save(merged)

    fe = FeatureEngineer()
    featured = fe.create_features(merged)
    df_model = fe.select_features(featured)
    fe.save(df_model)

    # ------------------------------------------------------------------ #
    # Phase 2: Data splitting                                             #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 2: Stratified Train / Dev / Test Split ---")
    splitter = DataSplitter()
    X_train, X_dev, X_test, y_train, y_dev, y_test = splitter.split(df_model)

    logger.info(
        "Split sizes — Train: %d | Dev: %d | Test: %d",
        len(X_train), len(X_dev), len(X_test),
    )

    # ------------------------------------------------------------------ #
    # Phase 3: Train all models                                           #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 3: Training All Models ---")
    trainer = ModelTrainer()
    fitted_pipelines = trainer.train_all(X_train, y_train)

    # ------------------------------------------------------------------ #
    # Phase 4: Evaluate on Dev set                                        #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 4: Dev Set Evaluation ---")
    evaluator = ModelEvaluator()
    for name, pipeline in fitted_pipelines.items():
        metrics = evaluator.evaluate(pipeline, X_dev, y_dev,
                                     model_name=name, split="dev")
        save_metrics_json(metrics, f"{name.replace(' ', '_').lower()}_dev.json")

        # Generate visualisations for each model
        evaluator.plot_confusion_matrix(pipeline, X_dev, y_dev,
                                        model_name=name, split="dev")
        evaluator.plot_roc_curve(pipeline, X_dev, y_dev,
                                 model_name=name, split="dev")
        evaluator.plot_precision_recall_curve(pipeline, X_dev, y_dev,
                                              model_name=name, split="dev")

    # ------------------------------------------------------------------ #
    # Phase 5: Cross-validation on training set                           #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 5: Cross-Validation (Training Set) ---")
    for name, pipeline in fitted_pipelines.items():
        cv_result = trainer.cross_validate_model(pipeline, X_train, y_train,
                                                 scoring="f1")
        logger.info("%s | CV F1: %.4f ± %.4f", name,
                    cv_result["mean_cv_score"], cv_result["std_cv_score"])

    # ------------------------------------------------------------------ #
    # Phase 6: Hyperparameter tuning (RF + GB)                           #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 6: Hyperparameter Tuning ---")
    logger.info("Tuning Random Forest …")
    best_rf = trainer.tune_random_forest(X_train, y_train, n_iter=15)
    fitted_pipelines["Random Forest (Tuned)"] = best_rf

    logger.info("Tuning Gradient Boosting …")
    best_gb = trainer.tune_gradient_boosting(X_train, y_train, n_iter=10)
    fitted_pipelines["Gradient Boosting (Tuned)"] = best_gb

    # Evaluate tuned models on Dev
    for name in ["Random Forest (Tuned)", "Gradient Boosting (Tuned)"]:
        pipeline = fitted_pipelines[name]
        metrics = evaluator.evaluate(pipeline, X_dev, y_dev,
                                     model_name=name, split="dev")
        save_metrics_json(metrics, f"{name.replace(' ', '_').lower()}_dev.json")
        evaluator.plot_confusion_matrix(pipeline, X_dev, y_dev,
                                        model_name=name, split="dev")
        evaluator.plot_roc_curve(pipeline, X_dev, y_dev,
                                 model_name=name, split="dev")
        evaluator.plot_precision_recall_curve(pipeline, X_dev, y_dev,
                                              model_name=name, split="dev")

    # ------------------------------------------------------------------ #
    # Phase 7: Model comparison and selection                             #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 7: Model Comparison & Selection ---")
    comparator = ModelComparator()
    comparison_df, best_name, best_pipeline = comparator.compare(
        fitted_pipelines, X_dev, y_dev
    )
    logger.info("\nModel Comparison Table:")
    logger.info("\n%s", comparison_df[
        ["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]
    ].to_string(index=False))

    # Feature importance for best model
    comparator.plot_feature_importance(best_pipeline, best_name)

    # Learning curve for best model
    logger.info("Generating learning curve for %s …", best_name)
    evaluator.plot_learning_curve(best_pipeline, X_train, y_train,
                                  model_name=best_name)

    # ------------------------------------------------------------------ #
    # Phase 8: Refit on Train + Dev & evaluate on Test                   #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 8: Final Model Training on Train+Dev ---")
    final_pipeline = trainer.refit_on_train_dev(
        best_pipeline, X_train, X_dev, y_train, y_dev
    )

    logger.info("\n--- PHASE 8b: FINAL TEST SET EVALUATION ---")
    logger.info("(This is the first and only time the test set is used)")
    final_metrics = evaluator.evaluate(
        final_pipeline, X_test, y_test,
        model_name=best_name, split="test"
    )
    save_metrics_json(final_metrics, "final_model_test_metrics.json")

    # Final visualisations
    evaluator.plot_confusion_matrix(final_pipeline, X_test, y_test,
                                    model_name=f"{best_name}_FINAL", split="test")
    evaluator.plot_roc_curve(final_pipeline, X_test, y_test,
                             model_name=f"{best_name}_FINAL", split="test")
    evaluator.plot_precision_recall_curve(final_pipeline, X_test, y_test,
                                          model_name=f"{best_name}_FINAL", split="test")

    logger.info("\n=== FINAL TEST SET METRICS ===")
    logger.info("Model:     %s", best_name)
    logger.info("Accuracy:  %.4f", final_metrics["accuracy"])
    logger.info("Precision: %.4f", final_metrics["precision"])
    logger.info("Recall:    %.4f", final_metrics["recall"])
    logger.info("F1-Score:  %.4f", final_metrics["f1_score"])
    logger.info("ROC-AUC:   %.4f", final_metrics["roc_auc"])
    logger.info("Confusion Matrix:\n%s", final_metrics["classification_report"])

    # ------------------------------------------------------------------ #
    # Phase 9: Save the final pipeline as Pickle                         #
    # ------------------------------------------------------------------ #
    logger.info("\n--- PHASE 9: Saving Final Pipeline ---")
    model_path = CFG.model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as fh:
        pickle.dump(final_pipeline, fh)
    logger.info("Final pipeline saved → %s", model_path)

    # Verification: reload and re-predict on first 5 test rows
    logger.info("Verifying saved model …")
    with open(model_path, "rb") as fh:
        loaded_pipeline = pickle.load(fh)

    original_preds = final_pipeline.predict(X_test[:5])
    loaded_preds   = loaded_pipeline.predict(X_test[:5])
    assert (original_preds == loaded_preds).all(), "Pickle verification failed!"
    logger.info("Model verification passed. Predictions match.")

    # ------------------------------------------------------------------ #
    # Done                                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("Best model: %s", best_name)
    logger.info("Saved to:   %s", model_path)
    logger.info("Figures:    %s", CFG.figures_dir)
    logger.info("Reports:    %s", CFG.reports_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

