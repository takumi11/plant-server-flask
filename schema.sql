drop table if exists results;
create table results (
  `id` integer primary key autoincrement,
  `inputname` text not null,
  `disease` text not null,
  `savename` text not null,
  `created` datetime default CURRENT_TIMESTAMP
);
