#!/bin/bash

echo "Waiting for OpenSearch at $OPENSEARCH_HOST..."
until curl -s -X GET "$OPENSEARCH_HOST/_cluster/health?wait_for_status=yellow&timeout=30s" >/dev/null; do
  sleep 5
done
echo "OpenSearch is available!"

echo "Starting data insertion..."

# Article insertions
python scripts/articles/insert_arxiv_articles.py
python scripts/articles/insert_gov_articles.py
python scripts/articles/insert_news_articles.py
python scripts/articles/insert_wiki_articles.py

# Numerical data insertions
python scripts/numerical/insert_building_stock.py
python scripts/numerical/insert_energy_performance.py
python scripts/numerical/insert_financial_performance.py
python scripts/numerical/insert_reference_buildings.py
python scripts/numerical/insert_social_performance.py
python scripts/numerical/insert_annual_energy_balances.py
python scripts/numerical/insert_electricity_prices.py
python scripts/numerical/insert_energy_efficiency_indicators.py
python scripts/numerical/insert_energy_import_dependency.py
python scripts/numerical/insert_energy_intensity_of_economy.py
python scripts/numerical/insert_final_energy_consumption_households_per_capita.py
python scripts/numerical/insert_gas_prices.py
python scripts/numerical/insert_gdp.py
python scripts/numerical/insert_ghg_emissions_energy.py
python scripts/numerical/insert_households_number.py
python scripts/numerical/insert_inability_to_keep_home_warm.py
python scripts/numerical/insert_population.py
python scripts/numerical/insert_renewable_energy_share.py

echo "Data insertion completed successfully!"