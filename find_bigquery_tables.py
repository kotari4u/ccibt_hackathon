#!/usr/bin/env python3
"""
Script to find BigQuery tables and their schemas.
Helps identify the correct dataset and table names.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.data.bigquery_client import BigQueryClient
    from src.utils.config import settings
    
    print("=" * 60)
    print("BigQuery Table Discovery")
    print("=" * 60)
    print(f"\nProject ID: {settings.bigquery_project_id}")
    print(f"Dataset (from config): {settings.bigquery_dataset}")
    print("\n" + "=" * 60)
    
    # Initialize client
    bq = BigQueryClient()
    
    # List all datasets
    print("\n1. Listing all datasets in project...")
    datasets = bq.list_datasets()
    print(f"Found {len(datasets)} datasets:")
    for ds in datasets:
        print(f"  - {ds}")
    
    # For each dataset, list tables
    print("\n2. Listing tables in each dataset...")
    target_table = "sample_ticker_data"
    found_tables = []
    
    for dataset in datasets[:10]:  # Check first 10 datasets
        print(f"\nDataset: {dataset}")
        try:
            tables = bq.list_tables(dataset)
            print(f"  Tables ({len(tables)}): {', '.join(tables[:10])}")
            
            # Check if target table exists
            if target_table in tables:
                print(f"  ✓ Found '{target_table}' in dataset '{dataset}'")
                found_tables.append((dataset, target_table))
                
                # Get schema
                schema = bq.get_table_schema(target_table, dataset)
                if schema:
                    print(f"  Columns: {[f['name'] for f in schema]}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if found_tables:
        print(f"\nFound '{target_table}' in:")
        for dataset, table in found_tables:
            print(f"  - {settings.bigquery_project_id}.{dataset}.{table}")
        print(f"\nUpdate your .env file:")
        print(f"  BIGQUERY_DATASET={found_tables[0][0]}")
    else:
        print(f"\n'{target_table}' not found in any dataset.")
        print("Please check the table name in your BigQuery console.")
    
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

