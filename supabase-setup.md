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
    match_count int default 4,
    filter_subject text default null,
    filter_level text default null,
    filter_lesson_number text default null,
    filter_section_name text default null
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
      (1 - (lessons_chunks.embedding <=> query_embedding)) as
  similarity
    from lessons_chunks
    where
     filter_subject is not null
    and filter_level is not null
    and lessons_chunks.subject = filter_subject
    and lessons_chunks.level = filter_level
    and   (filter_lesson_number is null or lessons_chunks.lesson_number
   = filter_lesson_number)
      and (filter_section_name is null or
  lessons_chunks.section_name = filter_section_name)
    order by lessons_chunks.embedding <=> query_embedding
    limit match_count;
  end;
  $$ language plpgsql;


    CREATE TABLE user_access_log (
    wa_id TEXT PRIMARY KEY,
    access_time TIMESTAMP NOT NULL DEFAULT NOW(),
    subject TEXT,
    level TEXT
  );

  CREATE INDEX idx_user_access_time ON
  user_access_log(access_time);