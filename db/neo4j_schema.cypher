// Academic Graph RAG schema constraints and indexes for Neo4j.

CREATE CONSTRAINT paper_id IF NOT EXISTS
FOR (p:Paper) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT author_id IF NOT EXISTS
FOR (a:Author) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT concept_id IF NOT EXISTS
FOR (c:Concept) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT venue_id IF NOT EXISTS
FOR (v:Venue) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT dataset_id IF NOT EXISTS
FOR (d:Dataset) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT institution_id IF NOT EXISTS
FOR (i:Institution) REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT community_id IF NOT EXISTS
FOR (c:Community) REQUIRE c.id IS UNIQUE;

CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title);
CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year);
CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi);
CREATE INDEX paper_arxiv IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id);
CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name);
CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name);
CREATE INDEX concept_type IF NOT EXISTS FOR (c:Concept) ON (c.type);
CREATE INDEX chunk_section IF NOT EXISTS FOR (c:Chunk) ON (c.section_type);
CREATE INDEX dataset_name IF NOT EXISTS FOR (d:Dataset) ON (d.name);
CREATE INDEX venue_name IF NOT EXISTS FOR (v:Venue) ON (v.name);
CREATE INDEX community_leiden_id IF NOT EXISTS FOR (c:Community) ON (c.leidenId);
CREATE INDEX concept_leiden_community IF NOT EXISTS FOR (c:Concept) ON (c.leidenCommunity);
