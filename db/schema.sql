-- Academic paper schema for Graph RAG (Postgres as structured source of truth).

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS venues (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT CHECK (type IN ('conference', 'journal', 'workshop', 'preprint', 'other')),
    year INTEGER,
    UNIQUE (name, year)
);

CREATE TABLE IF NOT EXISTS institutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    country TEXT
);

CREATE TABLE IF NOT EXISTS authors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    affiliation TEXT,
    orcid TEXT UNIQUE,
    embedding vector(768)
);

CREATE TABLE IF NOT EXISTS papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    abstract TEXT,
    year INTEGER,
    venue_id UUID REFERENCES venues(id) ON DELETE SET NULL,
    doi TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    url TEXT,
    citation_count INTEGER DEFAULT 0,
    source_filename TEXT,
    embedding vector(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id UUID NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    author_order INTEGER DEFAULT 0,
    PRIMARY KEY (paper_id, author_id)
);

CREATE TABLE IF NOT EXISTS author_institutions (
    author_id UUID NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    institution_id UUID NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
    PRIMARY KEY (author_id, institution_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    section_type TEXT DEFAULT 'body',
    position INTEGER NOT NULL DEFAULT 0,
    embedding vector(768),
    UNIQUE (paper_id, position)
);

CREATE TABLE IF NOT EXISTS concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (
        type IN ('architecture', 'dataset', 'task', 'metric', 'method', 'other')
    ),
    aliases TEXT[] DEFAULT '{}',
    embedding vector(768),
    UNIQUE (name, type)
);

CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    domain TEXT,
    size TEXT,
    url TEXT
);

CREATE TABLE IF NOT EXISTS paper_citations (
    citing_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    cited_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    PRIMARY KEY (citing_paper_id, cited_paper_id),
    CHECK (citing_paper_id <> cited_paper_id)
);

CREATE TABLE IF NOT EXISTS paper_extends (
    derived_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    base_paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    PRIMARY KEY (derived_paper_id, base_paper_id),
    CHECK (derived_paper_id <> base_paper_id)
);

CREATE TABLE IF NOT EXISTS paper_methods (
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE TABLE IF NOT EXISTS paper_datasets (
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_id, dataset_id)
);

CREATE TABLE IF NOT EXISTS paper_metrics (
    paper_id UUID NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    value TEXT,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE TABLE IF NOT EXISTS chunk_concepts (
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, concept_id)
);

CREATE TABLE IF NOT EXISTS concept_relations (
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    related_concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    relation TEXT DEFAULT 'RELATED_TO',
    PRIMARY KEY (concept_id, related_concept_id),
    CHECK (concept_id <> related_concept_id)
);

CREATE INDEX IF NOT EXISTS idx_papers_title ON papers (title);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers (year);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks (paper_id);
CREATE INDEX IF NOT EXISTS idx_authors_name ON authors (name);
CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts (type);
