// Import Grid Nodes explicitly
LOAD CSV WITH HEADERS FROM 'file:///gridkit/grid_nodes.csv' AS row
WITH row WHERE row.id IS NOT NULL
MERGE (n:GridNode {id: trim(row.id)})
SET n.longitude = toFloat(row.longitude),
    n.latitude = toFloat(row.latitude),
    n.type = row.type,
    n.frequency = row.frequency,
    n.voltage = row.voltage,
    n.operator = row.operator,
    n.name = row.name,
    n.country = row.country,
    n.label = row.label;

// Import CONNECTED_TO relationships explicitly
LOAD CSV WITH HEADERS FROM 'file:///gridkit/connected_to_relationships.csv' AS row
WITH row WHERE row.source IS NOT NULL AND row.target IS NOT NULL
MATCH (source:GridNode {id: trim(row.source)})
MATCH (target:GridNode {id: trim(row.target)})
MERGE (source)-[r:CONNECTED_TO]->(target)
SET r.cables = row.cables,
    r.voltage = row.voltage,
    r.wires = row.wires;

// Explicit Indexes for efficient querying
CREATE INDEX gridnode_id_index IF NOT EXISTS FOR (n:GridNode) ON (n.id);
CREATE INDEX gridnode_type_index IF NOT EXISTS FOR (n:GridNode) ON (n.type);
CREATE INDEX gridnode_country_index IF NOT EXISTS FOR (n:GridNode) ON (n.country);