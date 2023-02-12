drop table if exists user;

create table user (
    id integer primary key autoincrement,
    name text,
    password text not null,
    username text unique not null
);
