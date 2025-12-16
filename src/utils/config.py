"""
Configuration management for Market Activity Prediction Agent.
Handles environment variables, API keys, and system settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields in .env file
        protected_namespaces=("settings_",),  # Fix model_ prefix warning
    )
    
    # Google Cloud & BigQuery
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    bigquery_project_id: str = Field(default="", env="BIGQUERY_PROJECT_ID")
    bigquery_dataset: str = Field(default="market_data", env="BIGQUERY_DATASET")
    bigquery_table_name: str = Field(default="sample_ticker_data", env="BIGQUERY_TABLE_NAME")
    # Support for BQ_DATASET_ID format: project.dataset.table
    bq_dataset_id: Optional[str] = Field(default=None, env="BQ_DATASET_ID")
    google_application_credentials: Optional[str] = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    
    # Model Parameters
    volatility_lookback_days: int = Field(default=30, env="VOLATILITY_LOOKBACK_DAYS")
    event_correlation_window: int = Field(default=90, env="EVENT_CORRELATION_WINDOW")
    confidence_threshold: float = Field(default=0.70, env="CONFIDENCE_THRESHOLD")
    alert_volatility_threshold: float = Field(
        default=1.5, env="ALERT_VOLATILITY_THRESHOLD"
    )
    
    # API Settings
    api_rate_limit: str = Field(default="100/minute", env="API_RATE_LIMIT")
    cache_ttl_seconds: int = Field(default=60, env="CACHE_TTL_SECONDS")
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # JWT Authentication
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Twilio (for SMS alerts)
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: Optional[str] = Field(
        default=None, env="TWILIO_PHONE_NUMBER"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="json", env="LOG_FORMAT"
    )  # json or text
    
    # Model Training
    model_retrain_frequency_days: int = Field(
        default=7, env="MODEL_RETRAIN_FREQUENCY_DAYS"
    )
    min_training_samples: int = Field(default=100, env="MIN_TRAINING_SAMPLES")
    
    # Prediction Settings
    default_forecast_horizon_days: int = Field(
        default=5, env="DEFAULT_FORECAST_HORIZON_DAYS"
    )
    max_forecast_horizon_days: int = Field(
        default=30, env="MAX_FORECAST_HORIZON_DAYS"
    )


# Global settings instance
settings = Settings()

