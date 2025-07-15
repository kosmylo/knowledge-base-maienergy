// Import PowerPlant nodes
LOAD CSV WITH HEADERS FROM 'file:///powerplants/powerplants_nodes.csv' AS row
WITH row WHERE row.plant_name IS NOT NULL
MERGE (p:PowerPlant {plant_name: trim(row.plant_name)})
SET p.capacity_mw = toFloat(row.capacity_mw),
    p.latitude = toFloat(row.latitude),
    p.longitude = toFloat(row.longitude),
    p.commissioning_year = toInteger(row.commissioning_year),
    p.source = row.source;

// Import Country nodes
LOAD CSV WITH HEADERS FROM 'file:///powerplants/countries_nodes.csv' AS row
WITH row WHERE row.country_iso IS NOT NULL
MERGE (c:Country {country_iso: trim(row.country_iso)});

// Import Owner nodes
LOAD CSV WITH HEADERS FROM 'file:///powerplants/owners_nodes.csv' AS row
WITH row WHERE row.name IS NOT NULL
MERGE (o:Owner {name: trim(row.name)});

// Import FuelType nodes
LOAD CSV WITH HEADERS FROM 'file:///powerplants/fuel_types_nodes.csv' AS row
WITH row WHERE row.type IS NOT NULL
MERGE (f:FuelType {type: trim(row.type)});

// Import LOCATED_IN relationships
LOAD CSV WITH HEADERS FROM 'file:///powerplants/located_in_relationships.csv' AS row
WITH row WHERE row.plant_name IS NOT NULL AND row.country_iso IS NOT NULL
MATCH (p:PowerPlant {plant_name: trim(row.plant_name)})
MATCH (c:Country {country_iso: trim(row.country_iso)})
MERGE (p)-[:LOCATED_IN]->(c);

// Import OWNED_BY relationships
LOAD CSV WITH HEADERS FROM 'file:///powerplants/owned_by_relationships.csv' AS row
WITH row WHERE row.plant_name IS NOT NULL AND row.owner IS NOT NULL
MATCH (p:PowerPlant {plant_name: trim(row.plant_name)})
MATCH (o:Owner {name: trim(row.owner)})
MERGE (p)-[:OWNED_BY]->(o);

// Import USES_FUEL relationships
LOAD CSV WITH HEADERS FROM 'file:///powerplants/uses_fuel_relationships.csv' AS row
WITH row WHERE row.plant_name IS NOT NULL AND row.fuel_type IS NOT NULL
MATCH (p:PowerPlant {plant_name: trim(row.plant_name)})
MATCH (f:FuelType {type: trim(row.fuel_type)})
MERGE (p)-[:USES_FUEL]->(f);

// Indexes for efficient querying
CREATE INDEX powerplant_name_index IF NOT EXISTS FOR (p:PowerPlant) ON (p.plant_name);
CREATE INDEX country_iso_index IF NOT EXISTS FOR (c:Country) ON (c.country_iso);
CREATE INDEX owner_name_index IF NOT EXISTS FOR (o:Owner) ON (o.name);
CREATE INDEX fuel_type_index IF NOT EXISTS FOR (f:FuelType) ON (f.type);