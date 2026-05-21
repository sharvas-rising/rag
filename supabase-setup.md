 create extension if not exists vector;

  create table lessons_chunks (
    id text primary key,
    lesson_number text,
    subject text,
    level text,
    title text,
    topic text,
    objective text,
    section_name text,
    duration text,
    content text,
    embedding vector(1536),
    created_at timestamp default now()
  );

  create index on lessons_chunks using ivfflat (embedding
  vector_cosine_ops) with (lists = 100);


  create or replace function search_lessons_vector(
    query_embedding vector(1536),
    match_count int default 4
  )
  returns table(
    id text,
    lesson_number text,
    topic text,
    section_name text,
    content text,
    duration text,
    similarity float8
  ) as $$
  begin
    return query
    select
      lessons_chunks.id,
      lessons_chunks.lesson_number,
      lessons_chunks.topic,
      lessons_chunks.section_name,
      lessons_chunks.content,
      lessons_chunks.duration,
      (1 - (lessons_chunks.embedding <=> query_embedding)) as similarity
    from lessons_chunks
    order by lessons_chunks.embedding <=> query_embedding
    limit match_count;
  end;
  $$ language plpgsql;