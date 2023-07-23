-- 创建表结构
create table dl_hash
(
    datetime    datetime    null,
    user_id     varchar(50) null,
    resource_id varchar(50) null
);
create table resources_hashed
(
    resource_id      varchar(50)   null,
    title            varchar(1000) null,
    meta_title       varchar(1000) null,
    mata_description varchar(2000) null,
    date_added       datetime      null,
    date_updated     datetime      null
);
create table user_hashed
(
    user_id    varchar(50) null,
    country_id integer     null,
    career_id  integer     null
);
-- 首先确定训练集和测试集的比例,例如训练数据占80%,测试数据占20%。
-- 使用SQL的RAND()函数生成一个0到1之间的随机数
select *, rand() as r
from dl_hash;

-- 训练集 80%
select *
from
(
  select *, rand() as r
  from dl_hash
) t
where r < 0.8;

-- 测试集 20%
select *
from
(
  select *, rand() as r
  from dl_hash
) t
where r >= 0.8;

-- 存入新的训练集表中
create table train_set as select *
from
(
  select *, rand() as r
  from dl_hash
) t
where r < 0.8;

create table test_set as
select *
from
(
  select *, rand() as r
  from dl_hash
) t
where r >= 0.8;

-- 创建视图
select `dl`.`datetime`          AS `datetime`,
       `dl`.`user_id`           AS `user_id`,
       `user`.`country_id`      AS `country_id`,
       `user`.`career_id`       AS `career_id`,
       `dl`.`resource_id`       AS `resource_id`,
       `res`.`title`            AS `title`,
       `res`.`meta_title`       AS `meta_title`,
       `res`.`mata_description` AS `mata_description`,
       `res`.`date_added`       AS `date_added`,
       `res`.`date_updated`     AS `date_updated`
from ((`final`.`test_set` `dl` join `final`.`resources_hashed` `res`
       on ((`dl`.`resource_id` = `res`.`resource_id`))) join `final`.`user_hashed` `user`
      on ((`dl`.`user_id` = `user`.`user_id`)));

select `dl`.`datetime`          AS `datetime`,
       `dl`.`user_id`           AS `user_id`,
       `user`.`country_id`      AS `country_id`,
       `user`.`career_id`       AS `career_id`,
       `dl`.`resource_id`       AS `resource_id`,
       `res`.`title`            AS `title`,
       `res`.`meta_title`       AS `meta_title`,
       `res`.`mata_description` AS `mata_description`,
       `res`.`date_added`       AS `date_added`,
       `res`.`date_updated`     AS `date_updated`
from ((`final`.`train_set` `dl` join `final`.`resources_hashed` `res`
       on ((`dl`.`resource_id` = `res`.`resource_id`))) join `final`.`user_hashed` `user`
      on ((`dl`.`user_id` = `user`.`user_id`)));