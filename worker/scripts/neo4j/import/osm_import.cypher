// --------------------- CONSTRAINTS & INDEXES ---------------------
CREATE CONSTRAINT chargingstation_osm_id_unique IF NOT EXISTS FOR (n:ChargingStation) REQUIRE n.osm_id IS UNIQUE;
CREATE CONSTRAINT powerplant_osm_id_unique IF NOT EXISTS FOR (n:PowerPlant) REQUIRE n.osm_id IS UNIQUE;
CREATE CONSTRAINT solarfarm_osm_id_unique IF NOT EXISTS FOR (n:SolarFarm) REQUIRE n.osm_id IS UNIQUE;
CREATE CONSTRAINT windturbine_osm_id_unique IF NOT EXISTS FOR (n:WindTurbine) REQUIRE n.osm_id IS UNIQUE;
CREATE CONSTRAINT substation_osm_id_unique IF NOT EXISTS FOR (n:Substation) REQUIRE n.osm_id IS UNIQUE;
CREATE CONSTRAINT transmissionline_osm_id_unique IF NOT EXISTS FOR (n:TransmissionLine) REQUIRE n.osm_id IS UNIQUE;
CREATE CONSTRAINT country_name_unique IF NOT EXISTS FOR (n:Country) REQUIRE n.name IS UNIQUE;

// --------------------- IMPORT NODES ---------------------

// Charging Stations
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/chargingstations_nodes.csv' AS row
    WITH row WHERE row.osm_id IS NOT NULL
    MERGE (cs:ChargingStation {osm_id: trim(row.osm_id)})
    SET cs.name = row.name,
        cs.operator = row.operator,
        cs.capacity = row.capacity,
        cs.opening_hours = row.opening_hours,
        cs.phone = row.phone,
        cs.website = row.website,
        cs.socket_types = row.socket_types,
        cs.latitude = toFloat(row.latitude),
        cs.longitude = toFloat(row.longitude),
        cs.country = row.country
} IN TRANSACTIONS OF 500 ROWS;

// Power Plants
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/powerplants_nodes.csv' AS row
    WITH row WHERE row.osm_id IS NOT NULL
    MERGE (pp:PowerPlant {osm_id: trim(row.osm_id)})
    SET pp.name = row.name,
        pp.operator = row.operator,
        pp.source = row.source,
        pp.method = row.method,
        pp.capacity = row.capacity,
        pp.latitude = toFloat(row.latitude),
        pp.longitude = toFloat(row.longitude),
        pp.country = row.country
} IN TRANSACTIONS OF 500 ROWS;

// Solar Farms
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/solarfarms_nodes.csv' AS row
    WITH row WHERE row.osm_id IS NOT NULL
    MERGE (sf:SolarFarm {osm_id: trim(row.osm_id)})
    SET sf.operator = row.operator,
        sf.source = row.source,
        sf.method = row.method,
        sf.capacity = row.capacity,
        sf.latitude = toFloat(row.latitude),
        sf.longitude = toFloat(row.longitude),
        sf.country = row.country
} IN TRANSACTIONS OF 500 ROWS;

// Wind Turbines
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/windturbines_nodes.csv' AS row
    WITH row WHERE row.osm_id IS NOT NULL
    MERGE (wt:WindTurbine {osm_id: trim(row.osm_id)})
    SET wt.operator = row.operator,
        wt.manufacturer = row.manufacturer,
        wt.model = row.model,
        wt.source = row.source,
        wt.method = row.method,
        wt.capacity = row.capacity,
        wt.rotor_diameter = row.rotor_diameter,
        wt.latitude = toFloat(row.latitude),
        wt.longitude = toFloat(row.longitude),
        wt.country = row.country
} IN TRANSACTIONS OF 500 ROWS;

// Substations
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/substations_nodes.csv' AS row
    WITH row WHERE row.osm_id IS NOT NULL
    MERGE (ss:Substation {osm_id: trim(row.osm_id)})
    SET ss.latitude = toFloat(row.latitude),
        ss.longitude = toFloat(row.longitude),
        ss.country = row.country
} IN TRANSACTIONS OF 500 ROWS;

// Transmission Lines
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/transmissionlines_nodes.csv' AS row
    WITH row WHERE row.osm_id IS NOT NULL
    MERGE (tl:TransmissionLine {osm_id: trim(row.osm_id)})
    SET tl.name = row.name,
        tl.operator = row.operator,
        tl.voltage = row.voltage,
        tl.circuits = row.circuits,
        tl.cables = row.cables,
        tl.wires = row.wires,
        tl.start_date = row.start_date,
        tl.latitude = toFloat(row.latitude),
        tl.longitude = toFloat(row.longitude),
        tl.country = row.country
} IN TRANSACTIONS OF 500 ROWS;

// Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/countries_nodes.csv' AS row
    WITH row WHERE row.country_name IS NOT NULL
    MERGE (c:Country {name: trim(row.country_name)})
} IN TRANSACTIONS OF 100 ROWS;

// --------------------- RELATIONSHIPS ---------------------

// Charging Stations LOCATED_IN Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/chargingstations_located_in_relationships.csv' AS row
    MATCH (cs:ChargingStation {osm_id: trim(row.source_id)})
    MATCH (c:Country {name: trim(row.target_country)})
    MERGE (cs)-[:LOCATED_IN]->(c)
} IN TRANSACTIONS OF 500 ROWS;

// Power Plants LOCATED_IN Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/powerplants_located_in_relationships.csv' AS row
    MATCH (pp:PowerPlant {osm_id: trim(row.source_id)})
    MATCH (c:Country {name: trim(row.target_country)})
    MERGE (pp)-[:LOCATED_IN]->(c)
} IN TRANSACTIONS OF 500 ROWS;

// Solar Farms LOCATED_IN Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/solarfarms_located_in_relationships.csv' AS row
    MATCH (sf:SolarFarm {osm_id: trim(row.source_id)})
    MATCH (c:Country {name: trim(row.target_country)})
    MERGE (sf)-[:LOCATED_IN]->(c)
} IN TRANSACTIONS OF 500 ROWS;

// Wind Turbines LOCATED_IN Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/windturbines_located_in_relationships.csv' AS row
    MATCH (wt:WindTurbine {osm_id: trim(row.source_id)})
    MATCH (c:Country {name: trim(row.target_country)})
    MERGE (wt)-[:LOCATED_IN]->(c)
} IN TRANSACTIONS OF 500 ROWS;

// Substations LOCATED_IN Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/substations_located_in_relationships.csv' AS row
    MATCH (ss:Substation {osm_id: trim(row.source_id)})
    MATCH (c:Country {name: trim(row.target_country)})
    MERGE (ss)-[:LOCATED_IN]->(c)
} IN TRANSACTIONS OF 500 ROWS;

// Transmission Lines LOCATED_IN Countries
CALL {
    LOAD CSV WITH HEADERS FROM 'file:///osm/neo4j_import/transmissionlines_located_in_relationships.csv' AS row
    MATCH (tl:TransmissionLine {osm_id: trim(row.source_id)})
    MATCH (c:Country {name: trim(row.target_country)})
    MERGE (tl)-[:LOCATED_IN]->(c)
} IN TRANSACTIONS OF 500 ROWS;