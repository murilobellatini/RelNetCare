// Delete all nodes and relationships
MATCH (n)
DETACH DELETE n;

// Creating a node for each type with the name property as the class name
CREATE (:PER { name: 'Person' })
CREATE (:PET { name: 'Pet' })
CREATE (:HEALTH { name: 'Health' })
CREATE (:GPE { name: 'GPE' })
CREATE (:ORG { name: 'ORG' });

// Connect nodes with appropriate relationships
MATCH (per:PER), (pet:PET), (health:HEALTH), (gpe:GPE), (org:ORG)
MERGE (per)-[:PET]->(pet)
MERGE (per)-[:HEALTH_STATUS]->(health)
MERGE (per)-[:PLACE_OF_RESIDENCE]->(gpe)
MERGE (per)-[:MEMBER_OF]->(org)
MERGE (per)-[:PARENT]->(per)
MERGE (per)-[:VISITED]->(gpe)
MERGE (per)-[:SPOUSE]->(per)
MERGE (per)-[:ACQUAINTANCE]->(per)
MERGE (per)-[:PLACE_OF_BIRTH]->(gpe)
MERGE (per)-[:SIBLING]->(per);

// Return the full graph
MATCH (n)
RETURN n;
