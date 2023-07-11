// Creating nodes for Hilda, Max the cat, Heidenheim and "Rollator Benutzer Verein"
MERGE (per:PER { name: 'Hilda Schmidt', dob: '1940-03-15' })
MERGE (pet:PET { name: 'Max', species: 'Cat' })
MERGE (health:HEALTH { status: 'Rollator User' })
MERGE (gpe:GPE { name: 'Heidenheim an der Brenz', region: 'Baden-Württemberg', country: 'Germany' }) // Hilda's birthplace
MERGE (org:ORG { name: 'Lesekreis Heidenheim', founded: '2000' })

// Creating relations for Hilda
MERGE (per)-[:PET { since: '2018' }]->(pet)
MERGE (per)-[:HEALTH_STATUS]->(health)
MERGE (per)-[:PLACE_OF_RESIDENCE]->(gpe)
MERGE (per)-[:MEMBER_OF]->(org)

// Adding more nodes and relationships
MERGE (son:PER { name: 'Johannes Schmidt', dob: '1970-05-01' }) // Hilda's son
MERGE (munich:GPE { name: 'Munich', country: 'Germany' }) // A place Hilda visited
MERGE (per)-[:PARENT]->(son)
MERGE (per)-[:VISITED]->(munich)

// Adding even more nodes
MERGE (husband:PER { name: 'Karl Schmidt', dob: '1940-02-10', death: '2018-04-18' }) // Hilda's late husband
MERGE (daughter:PER { name: 'Maria Schmidt', dob: '1968-08-12' }) // Hilda's daughter
MERGE (friend:PER { name: 'Ingrid Fischer', dob: '1940-03-30' }) // Hilda's friend

// Adding siblings for Hilda
MERGE (brother:PER { name: 'Friedrich Schmidt', dob: '1938-07-07' }) // Hilda's brother
MERGE (sister:PER { name: 'Brigitte Schmidt', dob: '1942-11-20' }) // Hilda's sister
MERGE (per)-[:SIBLING]->(brother)
MERGE (per)-[:SIBLING]->(sister)

// Adding remaining relationships
MERGE (per)-[:SPOUSE { status: 'Widowed' }]->(husband)
MERGE (per)-[:PARENT]->(daughter)
MERGE (per)-[:ACQUAINTANCE]->(friend)
MERGE (per)-[:PLACE_OF_BIRTH]->(gpe);

// return full graph
MATCH (n)
RETURN n;
