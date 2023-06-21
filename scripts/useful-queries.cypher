// extracts all entities and relations from dialogue 42
MATCH (d:Dialogue {id: 42})-[:CONTAINS|RELATION]->(n)
RETURN d,
       [(d)-[r]->(n) | {source: d.id, target: n, type: type(r)}] AS relationships,
       n.name AS entity_name;


// delete all nodes in graph
MATCH (n)
DETACH DELETE n;

// get unique queries and dialogue ids
MATCH (:Entity)-[r:RELATION]->(:Entity)
WHERE size(r.trigger) > 0
RETURN distinct r.trigger, r.dialogue_id
