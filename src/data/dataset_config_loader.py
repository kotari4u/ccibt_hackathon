"""
Utility to load and validate dataset configuration from JSON.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class DatasetConfigLoader:
    """Load and manage dataset configuration from JSON file."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to data_set_config.json (defaults to project root)
        """
        if config_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "data_set_config.json"
        
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            if not self.config_path.exists():
                logger.warning(
                    "Dataset config file not found",
                    path=str(self.config_path)
                )
                return
            
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            logger.info(
                "Dataset config loaded",
                path=str(self.config_path),
                dataset=self.get_dataset_name(),
            )
        except Exception as e:
            logger.error("Failed to load dataset config", error=str(e))
            raise
    
    def get_dataset_name(self) -> str:
        """Get dataset name from config."""
        if self.config and "dataset_config" in self.config:
            return self.config["dataset_config"].get("dataset_name", "market_data")
        return "market_data"
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table schema dictionary or None if not found
        """
        if not self.config or "dataset_config" not in self.config:
            return None
        
        tables = self.config["dataset_config"].get("tables", {})
        return tables.get(table_name)
    
    def get_all_tables(self) -> Dict[str, Any]:
        """Get all table configurations."""
        if not self.config or "dataset_config" not in self.config:
            return {}
        
        return self.config["dataset_config"].get("tables", {})
    
    def get_required_tables(self) -> list[str]:
        """Get list of required table names."""
        tables = self.get_all_tables()
        return [
            name for name, config in tables.items()
            if config.get("required", False)
        ]
    
    def validate_table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in configuration.
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists in config
        """
        return table_name in self.get_all_tables()
    
    def get_query_config(self) -> Dict[str, Any]:
        """Get query configuration."""
        if not self.config or "dataset_config" not in self.config:
            return {}
        
        return self.config["dataset_config"].get("query_config", {})
    
    def get_retention_config(self) -> Dict[str, Any]:
        """Get data retention configuration."""
        if not self.config or "dataset_config" not in self.config:
            return {}
        
        return self.config["dataset_config"].get("data_retention", {})


# Global instance
_config_loader: Optional[DatasetConfigLoader] = None


def get_dataset_config() -> DatasetConfigLoader:
    """Get global dataset config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = DatasetConfigLoader()
    return _config_loader

