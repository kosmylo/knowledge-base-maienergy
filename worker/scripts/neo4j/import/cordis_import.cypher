// Import Projects
LOAD CSV WITH HEADERS FROM 'file:///cordis/projects_nodes.csv' AS row
WITH row WHERE row.projectId IS NOT NULL
MERGE (p:Project {projectId: trim(row.projectId)})
SET p.acronym = row.acronym,
    p.title = row.title,
    p.objective = row.objective,
    p.startDate = date(row.startDate),
    p.endDate = date(row.endDate),
    p.ecMaxContribution = toFloat(row.ecMaxContribution),
    p.totalCost = toFloat(row.totalCost),
    p.frameworkProgramme = row.frameworkProgramme;

// Import Organizations
LOAD CSV WITH HEADERS FROM 'file:///cordis/organizations_nodes.csv' AS row
WITH row WHERE row.organizationId IS NOT NULL
MERGE (o:Organization {organizationId: trim(row.organizationId)})
SET o.name = row.name,
    o.shortName = row.shortName,
    o.country = row.country,
    o.vatNumber = row.vatNumber,
    o.city = row.city,
    o.activityType = row.activityType;

// Import Topics
LOAD CSV WITH HEADERS FROM 'file:///cordis/topics_nodes.csv' AS row
WITH row WHERE row.topicId IS NOT NULL
MERGE (t:Topic {topicId: trim(row.topicId)})
SET t.title = row.topicTitle;

// Import Legal Bases
LOAD CSV WITH HEADERS FROM 'file:///cordis/legal_basis_nodes.csv' AS row
WITH row WHERE row.legalBasisId IS NOT NULL
MERGE (l:LegalBasis {legalBasisId: trim(row.legalBasisId)})
SET l.title = row.legalBasisTitle;

// Import PARTICIPATED_IN relationships
LOAD CSV WITH HEADERS FROM 'file:///cordis/participated_in_relationships.csv' AS row
WITH row WHERE row.projectId IS NOT NULL AND row.organizationId IS NOT NULL
MATCH (p:Project {projectId: trim(row.projectId)})
MATCH (o:Organization {organizationId: trim(row.organizationId)})
MERGE (o)-[r:PARTICIPATED_IN]->(p)
SET r.role = row.role,
    r.ecContribution = toFloat(row.ecContribution);

// Import HAS_TOPIC relationships
LOAD CSV WITH HEADERS FROM 'file:///cordis/has_topic_relationships.csv' AS row
WITH row WHERE row.projectId IS NOT NULL AND row.topicId IS NOT NULL
MATCH (p:Project {projectId: trim(row.projectId)})
MATCH (t:Topic {topicId: trim(row.topicId)})
MERGE (p)-[:HAS_TOPIC]->(t);

// Import HAS_LEGAL_BASIS relationships
LOAD CSV WITH HEADERS FROM 'file:///cordis/has_legalbasis_relationships.csv' AS row
WITH row WHERE row.projectId IS NOT NULL AND row.legalBasisId IS NOT NULL
MATCH (p:Project {projectId: trim(row.projectId)})
MATCH (l:LegalBasis {legalBasisId: trim(row.legalBasisId)})
MERGE (p)-[:HAS_LEGAL_BASIS]->(l);

// Create Indexes for improved query performance
CREATE INDEX project_id_index IF NOT EXISTS FOR (p:Project) ON (p.projectId);
CREATE INDEX organization_id_index IF NOT EXISTS FOR (o:Organization) ON (o.organizationId);
CREATE INDEX topic_id_index IF NOT EXISTS FOR (t:Topic) ON (t.topicId);
CREATE INDEX legalbasis_id_index IF NOT EXISTS FOR (l:LegalBasis) ON (l.legalBasisId);