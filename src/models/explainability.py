"""
Model explainability module using SHAP values and feature importance.
Provides interpretable explanations for predictions.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import structlog

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("SHAP not available - explainability features limited")

logger = structlog.get_logger(__name__)


class ModelExplainer:
    """
    Provides explainability for model predictions.
    
    Uses SHAP values, feature importance, and natural language explanations.
    """
    
    def __init__(self):
        """Initialize model explainer."""
        self.logger = logger
        self.shap_available = SHAP_AVAILABLE
    
    def explain_prediction(
        self,
        model: Any,
        features: pd.DataFrame,
        prediction: float,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain a model prediction using SHAP values.
        
        Args:
            model: Trained model (must have predict method)
            features: Feature DataFrame
            prediction: Model prediction value
            feature_names: Optional feature names
            
        Returns:
            Dictionary with explanation and feature importance
        """
        if feature_names is None:
            feature_names = list(features.columns)
        
        explanations = {
            "prediction": float(prediction),
            "feature_importance": {},
            "top_contributors": [],
            "explanation": "",
        }
        
        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            explanations["feature_importance"] = importance_dict
            
            # Sort by importance
            sorted_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )
            explanations["top_contributors"] = [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in sorted_features[:10]
            ]
        
        # Use SHAP if available
        if self.shap_available and len(features) > 0:
            try:
                # Use TreeExplainer for tree-based models
                if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(features.iloc[-1:])
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]  # For binary classification
                    
                    shap_dict = dict(zip(feature_names, shap_values.flatten()))
                    explanations["shap_values"] = shap_dict
                    
                    # Top contributors by SHAP
                    sorted_shap = sorted(
                        shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
                    )
                    explanations["top_shap_contributors"] = [
                        {"feature": feat, "shap_value": float(val)}
                        for feat, val in sorted_shap[:10]
                    ]
            
            except Exception as e:
                logger.warning("SHAP explanation failed", error=str(e))
        
        # Generate natural language explanation
        explanations["explanation"] = self._generate_explanation(explanations)
        
        return explanations
    
    def _generate_explanation(self, explanations: Dict[str, Any]) -> str:
        """
        Generate natural language explanation from feature importance.
        
        Args:
            explanations: Explanation dictionary
            
        Returns:
            Natural language explanation string
        """
        parts = []
        
        # Top contributors
        if "top_contributors" in explanations and explanations["top_contributors"]:
            top_feat = explanations["top_contributors"][0]
            parts.append(
                f"Primary driver: {top_feat['feature']} "
                f"(importance: {top_feat['importance']:.3f})"
            )
        
        # SHAP contributors
        if "top_shap_contributors" in explanations:
            shap_contributors = explanations["top_shap_contributors"][:3]
            shap_parts = []
            for contrib in shap_contributors:
                direction = "increases" if contrib["shap_value"] > 0 else "decreases"
                shap_parts.append(
                    f"{contrib['feature']} {direction} prediction"
                )
            
            if shap_parts:
                parts.append("Key factors: " + ", ".join(shap_parts))
        
        # Prediction context
        prediction = explanations.get("prediction", 0)
        if prediction > 0.2:
            parts.append("High volatility predicted")
        elif prediction > 0.1:
            parts.append("Moderate volatility predicted")
        else:
            parts.append("Low volatility predicted")
        
        return ". ".join(parts) + "."
    
    def calculate_prediction_confidence(
        self,
        model: Any,
        features: pd.DataFrame,
        historical_accuracy: Optional[float] = None,
    ) -> float:
        """
        Calculate confidence score for prediction.
        
        Args:
            model: Trained model
            features: Feature DataFrame
            historical_accuracy: Historical model accuracy (optional)
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence
        
        # Adjust based on feature quality
        missing_pct = features.isnull().sum().sum() / (len(features) * len(features.columns))
        confidence *= (1 - missing_pct)
        
        # Adjust based on historical accuracy
        if historical_accuracy is not None:
            confidence = (confidence + historical_accuracy) / 2
        
        return max(0.0, min(1.0, confidence))
    
    def explain_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Get feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(model, 'feature_importances_'):
            return dict(zip(feature_names, model.feature_importances_))
        else:
            logger.warning("Model does not support feature importance")
            return {}

