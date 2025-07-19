#!/bin/bash
set -e

echo "Waiting for OpenSearch at ${OPENSEARCH_HOST}..."
until curl -s -X GET "${OPENSEARCH_HOST}/_cluster/health?wait_for_status=yellow&timeout=30s" >/dev/null; do
  sleep 5
done
echo "OpenSearch is available!"

echo "Starting data insertion..."

# Article insertions
python scripts/opensearch/articles/insert_arxiv_articles.py
python scripts/opensearch/articles/insert_gov_articles.py
python scripts/opensearch/articles/insert_news_articles.py
python scripts/opensearch/articles/insert_wiki_articles.py

# Numerical data insertions
python scripts/opensearch/numerical/insert_building_stock.py
python scripts/opensearch/numerical/insert_energy_performance.py
python scripts/opensearch/numerical/insert_financial_performance.py
python scripts/opensearch/numerical/insert_reference_buildings.py
python scripts/opensearch/numerical/insert_social_performance.py
python scripts/opensearch/numerical/insert_electricity_prices.py
python scripts/opensearch/numerical/insert_energy_efficiency_indicators.py
python scripts/opensearch/numerical/insert_energy_import_dependency.py
python scripts/opensearch/numerical/insert_energy_intensity_of_economy.py
python scripts/opensearch/numerical/insert_final_energy_consumption_households_per_capita.py
python scripts/opensearch/numerical/insert_gas_prices.py
python scripts/opensearch/numerical/insert_gdp.py
python scripts/opensearch/numerical/insert_households_number.py
python scripts/opensearch/numerical/insert_inability_to_keep_home_warm.py
python scripts/opensearch/numerical/insert_population.py
python scripts/opensearch/numerical/insert_renewable_energy_share.py

echo "Opensearch data insertion completed successfully!"