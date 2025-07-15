// Import TSO Nodes explicitly
LOAD CSV WITH HEADERS FROM 'file:///tso_network/tso_nodes.csv' AS row
WITH row WHERE row.country IS NOT NULL
MERGE (tso:TSO {country: trim(row.country)})
SET tso.area_code = row.area_code;

// Import INTERCONNECTED_WITH relationships explicitly
LOAD CSV WITH HEADERS FROM 'file:///tso_network/interconnected_with_relationships.csv' AS row
WITH row WHERE row.source_country IS NOT NULL AND row.target_country IS NOT NULL
MATCH (source:TSO {country: trim(row.source_country)})
MATCH (target:TSO {country: trim(row.target_country)})
MERGE (source)-[r:INTERCONNECTED_WITH]->(target)
SET r.status = row.status;

// Explicit Indexes for optimized querying
CREATE INDEX tso_country_index IF NOT EXISTS FOR (t:TSO) ON (t.country);
CREATE INDEX tso_area_code_index IF NOT EXISTS FOR (t:TSO) ON (t.area_code);
