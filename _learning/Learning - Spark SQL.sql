-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Spark SQL for Data Engineers – Platinum Notebook
-- MAGIC
-- MAGIC **Layer:** Platinum (companion to *Learning - Python* – same structure, pure SQL) **Updated:** 2026-11-17 · **Target:** Databricks Runtime 17+ (Apache Spark 4.x) – classic or serverless compute
-- MAGIC
-- MAGIC Every chapter has:
-- MAGIC
-- MAGIC * 📘 **Markdown cells** – definitions, rules and pitfalls
-- MAGIC * 🧮 **SQL cells** – runnable samples on tiny in-memory views (nothing is written to a catalog)
-- MAGIC * 🔁 **Elsewhere** – how the same thing is spelled in T-SQL / PostgreSQL / ANSI
-- MAGIC * 🆕 **What's new** – features added in Spark 3.4 – 4.x and Databricks
-- MAGIC
-- MAGIC | # | Chapter |
-- MAGIC |---|---------|
-- MAGIC | 0 | Environment check & sample data |
-- MAGIC | 1 | Self-joins (and Spark-only join types) |
-- MAGIC | 2 | Recursive CTEs |
-- MAGIC | 3 | Lateral joins |
-- MAGIC | 4 | Complex types & higher-order functions |
-- MAGIC | 5 | JSON & VARIANT |
-- MAGIC | 6 | Function toolbox – nulls, safe casts, strings, dates, aggregates |
-- MAGIC | 7 | Windows, QUALIFY, GROUP BY ALL, PIVOT & friends |
-- MAGIC | 8 | SQL UDFs, table functions, variables & dynamic SQL |
-- MAGIC | 9 | What's new in Spark 4.x – pipe syntax, SQL scripting, collations |
-- MAGIC | 10 | Capstone – org & sales report in one statement |
-- MAGIC | 11 | Cheat sheet – Spark SQL vs other dialects |
-- MAGIC
-- MAGIC > **Legend:** 🟠 = Databricks-only syntax (not in open-source Apache Spark). Everything else runs on plain Spark 4.x as well.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 0. Environment check & sample data
-- MAGIC
-- MAGIC * A SQL cell may hold several statements separated by `;` – only the **last** result is displayed.
-- MAGIC * All samples use **temporary views** built from `VALUES` – they live in the Spark session and vanish when it ends.
-- MAGIC * Spark 4 runs with **ANSI mode on** by default: invalid casts, overflows and division by zero raise errors instead of returning `NULL` (see the `try_*` family in chapter 6).

-- COMMAND ----------

SELECT version()          AS spark_version,
       current_catalog()  AS catalog,
       current_schema()   AS schema,
       current_timezone() AS session_tz,
       current_date()     AS today;

-- COMMAND ----------

-- Org chart: manager_id points back to emp_id (a self-reference). Two roots: Alice and Judy.
CREATE OR REPLACE TEMPORARY VIEW employees AS
SELECT * FROM VALUES
  ( 1, 'Alice', NULL, 'Board',       300000, DATE'2015-01-10'),
  ( 2, 'Bob',      1, 'Engineering', 200000, DATE'2016-03-01'),
  ( 3, 'Carol',    1, 'Sales',       190000, DATE'2016-09-15'),
  ( 4, 'Dan',      2, 'Engineering', 150000, DATE'2018-02-01'),
  ( 5, 'Eve',      2, 'Engineering', 150000, DATE'2019-07-22'),
  ( 6, 'Frank',    3, 'Sales',       120000, DATE'2019-11-04'),
  ( 7, 'Grace',    4, 'Engineering', 160000, DATE'2021-05-17'),
  ( 8, 'Heidi',    4, 'Engineering', 105000, DATE'2022-01-03'),
  ( 9, 'Ivan',     6, 'Sales',        90000, DATE'2023-06-12'),
  (10, 'Judy',  NULL, 'Advisory',     80000, DATE'2024-10-01')
AS t(emp_id, name, manager_id, dept, salary, hire_date);

-- Orders: emp_id = sales rep, items = ARRAY<STRUCT>, payload = raw JSON string.
CREATE OR REPLACE TEMPORARY VIEW orders AS
SELECT * FROM VALUES
  (101, 6, 'ACME',    TIMESTAMP'2026-01-05 09:15:00',  65.50,
   array(named_struct('sku', 'A1', 'qty', 2, 'price', 10.0D), named_struct('sku', 'B2', 'qty', 1, 'price', 45.5D)),
   '{"channel":"web","tags":["new","promo"],"ship":{"city":"Wrocław","days":2}}'),
  (102, 6, 'ACME',    TIMESTAMP'2026-01-20 14:02:00', 100.00,
   array(named_struct('sku', 'C3', 'qty', 5, 'price', 20.0D)),
   '{"channel":"phone","tags":[],"ship":{"city":"Wrocław","days":1}}'),
  (103, 9, 'Globex',  TIMESTAMP'2026-02-02 08:40:00', 149.90,
   array(named_struct('sku', 'A1', 'qty', 1, 'price', 10.0D), named_struct('sku', 'C3', 'qty', 2, 'price', 20.0D), named_struct('sku', 'D4', 'qty', 1, 'price', 99.9D)),
   '{"channel":"web","tags":["bulk"],"ship":{"city":"Gdańsk","days":4}}'),
  (104, 3, 'Initech', TIMESTAMP'2026-02-14 16:30:00', 299.70,
   array(named_struct('sku', 'D4', 'qty', 3, 'price', 99.9D)),
   '{"channel":"partner","ship":{"city":"Kraków","days":3}}'),
  (105, 9, 'ACME',    TIMESTAMP'2026-03-01 11:05:00',  91.00,
   array(named_struct('sku', 'B2', 'qty', 2, 'price', 45.5D)),
   '{"channel":"web","tags":["promo"],"ship":{"city":"Wrocław","days":"2"}}'),
  (106, 6, 'Globex',  TIMESTAMP'2026-03-09 10:00:00',  20.00,
   array(named_struct('sku', 'C3', 'qty', 1, 'price', 20.0D)),
   '{"channel":"web", broken json')
AS t(order_id, emp_id, customer, order_ts, amount, items, payload);

-- Directed graph with a cycle (WRO → WAW → GDN → WRO).
CREATE OR REPLACE TEMPORARY VIEW routes AS
SELECT * FROM VALUES
  ('WRO', 'WAW', 350), ('WRO', 'KRK', 270), ('KRK', 'WAW', 295), ('WAW', 'KRK', 295),
  ('WAW', 'GDN', 340), ('GDN', 'WRO', 480), ('GDN', 'SZZ', 370)
AS t(src, dst, km);

SELECT * FROM employees ORDER BY emp_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 1. Self-joins (and Spark-only join types)
-- MAGIC
-- MAGIC * A **self-join** joins a table to itself. It is an ordinary join – the only requirement is that **each copy gets its own alias**.
-- MAGIC * Typical uses: **hierarchies** (employee → manager), **pairs** inside a group, **comparing a row with another row** (previous / next / peer), **duplicate detection**.
-- MAGIC * Pair pattern: `a.id < b.id` returns every unordered pair once; `a.id <> b.id` returns both directions.
-- MAGIC * Non-equi self-joins (`<`, `BETWEEN`, …) can not use a hash join – Spark falls back to a sort-merge or nested-loop join, which is quadratic inside each group. Always keep an **equality key** in the `ON` clause when you can.
-- MAGIC * If you only need the *previous/next* row, a **window function** (`lag` / `lead`) is cheaper than a self-join – one scan instead of two.
-- MAGIC * `NULL = NULL` is `NULL` (not true). Use the **null-safe** operator `<=>` (ANSI: `IS NOT DISTINCT FROM`) when NULL keys should match.
-- MAGIC
-- MAGIC 🔁 **Elsewhere:** `LEFT SEMI JOIN` / `LEFT ANTI JOIN` are Spark/Hive spellings of `WHERE EXISTS (…)` / `WHERE NOT EXISTS (…)`. They never duplicate left rows and expose only left-side columns.

-- COMMAND ----------

-- 1.1 Hierarchy: employee -> manager. LEFT JOIN keeps the roots (manager_id IS NULL).
SELECT e.emp_id,
       e.name                     AS employee,
       m.name                     AS manager,
       e.salary - m.salary        AS diff_vs_manager,
       e.salary > m.salary        AS earns_more_than_manager
FROM employees e
LEFT JOIN employees m ON m.emp_id = e.manager_id
ORDER BY e.emp_id;

-- COMMAND ----------

-- 1.2 Pairs inside a group: colleagues in the same department, each pair once.
SELECT a.dept,
       a.name                         AS person_a,
       b.name                         AS person_b,
       abs(a.salary - b.salary)       AS salary_gap,
       a.salary = b.salary            AS same_salary
FROM employees a
JOIN employees b
  ON  a.dept   = b.dept          -- equality key -> hash / sort-merge join
  AND a.emp_id < b.emp_id        -- '<' = every pair once, no self-pairs
ORDER BY a.dept, salary_gap;

-- COMMAND ----------

-- 1.3 Previous row: self-join on a row number ...
WITH o AS (
  SELECT *, row_number() OVER (PARTITION BY customer ORDER BY order_ts) AS rn
  FROM orders
)
SELECT cur.customer, cur.order_id, cur.order_ts,
       prev.order_id                              AS prev_order_id,
       datediff(cur.order_ts, prev.order_ts)      AS days_since_prev
FROM o cur
LEFT JOIN o prev
  ON prev.customer = cur.customer AND prev.rn = cur.rn - 1
ORDER BY cur.customer, cur.order_ts;

-- COMMAND ----------

-- ... and the cheaper equivalent with a window function (one scan, no join).
SELECT customer, order_id, order_ts,
       lag(order_id) OVER w                       AS prev_order_id,
       datediff(order_ts, lag(order_ts) OVER w)   AS days_since_prev
FROM orders
WINDOW w AS (PARTITION BY customer ORDER BY order_ts)
ORDER BY customer, order_ts;

-- COMMAND ----------

-- 1.4 NULL keys: '=' never matches NULL, '<=>' does. Alice and Judy both have manager_id NULL.
SELECT a.name AS person_a, b.name AS person_b,
       a.manager_id,
       a.manager_id =   b.manager_id AS eq,            -- NULL for the roots
       a.manager_id <=> b.manager_id AS null_safe_eq   -- true for the roots
FROM employees a
JOIN employees b
  ON a.manager_id <=> b.manager_id AND a.emp_id < b.emp_id
ORDER BY a.manager_id NULLS FIRST, a.emp_id;

-- COMMAND ----------

-- 1.5 SEMI / ANTI self-joins: who manages somebody, and who does not?
SELECT 'manager' AS kind, array_sort(collect_list(m.name)) AS people
FROM employees m LEFT SEMI JOIN employees e ON e.manager_id = m.emp_id
UNION ALL
SELECT 'individual contributor', array_sort(collect_list(m.name))
FROM employees m LEFT ANTI JOIN employees e ON e.manager_id = m.emp_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 2. Recursive CTEs
-- MAGIC
-- MAGIC 🆕 **Spark 4.1 / Databricks Runtime 17.0** – before that, hierarchies needed a Python loop, GraphFrames or a fixed number of self-joins.
-- MAGIC
-- MAGIC ```sql
-- MAGIC WITH RECURSIVE cte_name (col1, col2, …) [MAX RECURSION LEVEL n] AS (
-- MAGIC     anchor_query                -- runs once, defines column types
-- MAGIC     UNION ALL
-- MAGIC     recursive_step              -- references cte_name; runs on the rows produced by the previous iteration
-- MAGIC )
-- MAGIC SELECT … FROM cte_name;
-- MAGIC ```
-- MAGIC
-- MAGIC * **Anchor** = starting rows. **Step** = joins the *previous iteration's* rows to find the next level. Recursion stops when the step returns **no rows**.
-- MAGIC * The **anchor defines the column types** – cast there if the step produces something wider (e.g. `CAST(name AS STRING)`, `0L`, `array(...)`).
-- MAGIC * **Safety limits:** 100 iterations and 1,000,000 rows by default; exceeding them **raises an error** (it does not silently truncate). Raise the level with `MAX RECURSION LEVEL n`; a `LIMIT` on the outer query also stops an otherwise endless recursion.
-- MAGIC * **Cycles** in the data = endless recursion. Carry the visited **path in an array** and filter with `NOT array_contains(path, next)`.
-- MAGIC * The recursive reference is allowed only in the step, typically once in `FROM`/`JOIN` – not inside aggregates, window functions or correlated subqueries. Aggregate **after** the recursion, in the outer query.
-- MAGIC * Only use recursion when the depth is unknown. For plain series Spark has cheaper built-ins: `range()`, `sequence()` + `explode()`.
-- MAGIC
-- MAGIC 🔁 **Elsewhere:** T-SQL omits the keyword `RECURSIVE` and limits with `OPTION (MAXRECURSION n)`; PostgreSQL has `CYCLE … SET …` and `SEARCH DEPTH FIRST` clauses (not in Spark – build the path yourself); Oracle's `CONNECT BY` has no Spark equivalent.

-- COMMAND ----------

-- 2.1 Hello, recursion: numbers 1..10
WITH RECURSIVE nums (n) AS (
  SELECT 1                             -- anchor
  UNION ALL
  SELECT n + 1 FROM nums WHERE n < 10  -- step; stops when no row passes the WHERE
)
SELECT n, n * n AS square FROM nums;

-- COMMAND ----------

-- The idiomatic Spark way for series: no recursion needed.
SELECT 'range()'    AS how, array_agg(id)  AS vals FROM range(1, 11)
UNION ALL
SELECT 'sequence()', sequence(1, 10)
UNION ALL
SELECT 'date spine', transform(sequence(DATE'2026-01-01', DATE'2026-03-01', INTERVAL 1 MONTH), d -> month(d));

-- COMMAND ----------

-- 2.2 Top-down hierarchy: level and full path for every employee
WITH RECURSIVE org (emp_id, name, manager_id, level, path) AS (
  SELECT emp_id, name, manager_id, 0, name
  FROM employees
  WHERE manager_id IS NULL                              -- anchor: the roots
  UNION ALL
  SELECT e.emp_id, e.name, e.manager_id, o.level + 1, o.path || ' > ' || e.name
  FROM org o
  JOIN employees e ON e.manager_id = o.emp_id           -- step: direct reports of the previous level
)
SELECT repeat('— ', level) || name AS org_chart, level, path
FROM org
ORDER BY path;

-- COMMAND ----------

-- 2.3 Bottom-up: chain of command for one person (walk towards the root)
WITH RECURSIVE chain (emp_id, name, manager_id, steps_up) AS (
  SELECT emp_id, name, manager_id, 0 FROM employees WHERE name = 'Heidi'
  UNION ALL
  SELECT m.emp_id, m.name, m.manager_id, c.steps_up + 1
  FROM chain c
  JOIN employees m ON m.emp_id = c.manager_id
)
SELECT steps_up, name FROM chain ORDER BY steps_up;

-- COMMAND ----------

-- 2.4 Transitive closure + aggregation AFTER the recursion:
--     headcount and payroll of everybody below each manager (not only direct reports)
WITH RECURSIVE closure (ancestor_id, emp_id, depth) AS (
  SELECT emp_id, emp_id, 0 FROM employees                -- everyone is their own ancestor at depth 0
  UNION ALL
  SELECT c.ancestor_id, e.emp_id, c.depth + 1
  FROM closure c
  JOIN employees e ON e.manager_id = c.emp_id
)
SELECT m.name                                   AS manager,
       count_if(c.depth = 1)                    AS direct_reports,
       count_if(c.depth > 0)                    AS all_reports,
       sum(e.salary) FILTER (WHERE c.depth > 0) AS payroll_below,
       max(c.depth)                             AS levels_below
FROM closure c
JOIN employees m ON m.emp_id = c.ancestor_id
JOIN employees e ON e.emp_id = c.emp_id
GROUP BY m.name
HAVING count_if(c.depth > 0) > 0
ORDER BY all_reports DESC, manager;

-- COMMAND ----------

-- 2.5 Graph traversal with cycle protection: every route from WRO that never visits a city twice
WITH RECURSIVE trips (city, hops, km, path) AS (
  SELECT dst, 1, km, array(src, dst)
  FROM routes
  WHERE src = 'WRO'
  UNION ALL
  SELECT r.dst, t.hops + 1, t.km + r.km, array_append(t.path, r.dst)
  FROM trips t
  JOIN routes r ON r.src = t.city
  WHERE NOT array_contains(t.path, r.dst)               -- cycle guard: WRO → WAW → GDN → WRO is cut here
)
SELECT city, hops, km, array_join(path, ' → ') AS route
FROM trips
ORDER BY city, km;

-- COMMAND ----------

-- Shortest route per destination – aggregate in the outer query (min_by is Spark's "argmin")
WITH RECURSIVE trips (city, km, path) AS (
  SELECT dst, km, array(src, dst) FROM routes WHERE src = 'WRO'
  UNION ALL
  SELECT r.dst, t.km + r.km, array_append(t.path, r.dst)
  FROM trips t JOIN routes r ON r.src = t.city
  WHERE NOT array_contains(t.path, r.dst)
)
SELECT city,
       min(km)                              AS shortest_km,
       array_join(min_by(path, km), ' → ')  AS shortest_route,
       count(*)                             AS routes_found
FROM trips
GROUP BY city
ORDER BY shortest_km;

-- COMMAND ----------

-- 2.6 Limits: 300 iterations need MAX RECURSION LEVEL (default 100 -> error without it)
WITH RECURSIVE deep (n) MAX RECURSION LEVEL 500 AS (
  SELECT 1
  UNION ALL
  SELECT n + 1 FROM deep WHERE n < 300
)
SELECT count(*) AS iterations, sum(n) AS total FROM deep;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 3. Lateral joins
-- MAGIC
-- MAGIC * Normally the items in `FROM` can not see each other. **`LATERAL`** lets a subquery (or table function) on the right **reference columns of the tables to its left** – it is evaluated *per left row*, like a `for each` loop.
-- MAGIC * Forms: `FROM a, LATERAL (…)` · `a CROSS JOIN LATERAL (…)` · `a LEFT JOIN LATERAL (…) ON true` (keeps left rows with no match).
-- MAGIC * Typical uses: **top-N per group**, **latest row per key**, **several aggregates per row** in one subquery (a scalar subquery can return only one column), **reusing a computed expression**, and **exploding arrays / calling table functions per row**.
-- MAGIC * Spark rewrites (de-correlates) a lateral subquery into ordinary joins and windows – it is not executed row by row. A few correlated shapes are not supported and raise `UNSUPPORTED_SUBQUERY_EXPRESSION_CATEGORY`.
-- MAGIC * **`LATERAL VIEW`** is the older Hive syntax and works **only with generator functions** (`explode`, `posexplode`, `inline`, `json_tuple`, `stack`). `LATERAL VIEW OUTER` keeps rows whose array is empty/NULL.
-- MAGIC * 🆕 **Lateral column alias** (Spark 3.4): a `SELECT` item may reuse an alias defined earlier in the same `SELECT` list.
-- MAGIC
-- MAGIC 🔁 **Elsewhere:** T-SQL / Oracle `CROSS APPLY` = `CROSS JOIN LATERAL`, `OUTER APPLY` = `LEFT JOIN LATERAL … ON true`. PostgreSQL uses the same `LATERAL` keyword.

-- COMMAND ----------

-- 3.1 Top-N per group: the two best-paid people of every department
SELECT d.dept, t.name, t.salary
FROM (SELECT DISTINCT dept FROM employees) d,
     LATERAL (
       SELECT e.name, e.salary
       FROM employees e
       WHERE e.dept = d.dept            -- correlation: reference to the left side
       ORDER BY e.salary DESC, e.name
       LIMIT 2
     ) t
ORDER BY d.dept, t.salary DESC;

-- COMMAND ----------

-- 3.2 LEFT JOIN LATERAL: latest order per sales person – people without orders are kept (NULLs)
SELECT e.name, e.dept, o.order_id, o.customer, o.order_ts, o.amount
FROM employees e
LEFT JOIN LATERAL (
  SELECT order_id, customer, order_ts, amount
  FROM orders
  WHERE orders.emp_id = e.emp_id
  ORDER BY order_ts DESC
  LIMIT 1
) o ON true
WHERE e.dept = 'Sales'
ORDER BY e.name;

-- COMMAND ----------

-- 3.3 Several aggregates per row in one lateral subquery (a scalar subquery returns only one column)
SELECT e.name, s.orders, s.revenue, s.biggest, s.customers
FROM employees e,
     LATERAL (
       SELECT count(*)                           AS orders,
              sum(o.amount)                      AS revenue,
              max(o.amount)                      AS biggest,
              array_sort(collect_set(o.customer)) AS customers
       FROM orders o
       WHERE o.emp_id = e.emp_id
     ) s
WHERE s.orders > 0
ORDER BY s.revenue DESC;

-- COMMAND ----------

-- 3.4 Chained LATERAL: each step can use the columns computed by the previous one
SELECT o.order_id, o.amount, v.vat, g.gross, g.gross > 150 AS big_order
FROM orders o,
     LATERAL (SELECT round(o.amount * 0.23, 2) AS vat) v,
     LATERAL (SELECT o.amount + v.vat          AS gross) g
ORDER BY o.order_id;

-- COMMAND ----------

-- 🆕 The same with a lateral column alias (Spark 3.4+): reuse an alias inside one SELECT list
SELECT order_id,
       amount,
       round(amount * 0.23, 2) AS vat,
       amount + vat            AS gross,       -- 'vat' was defined one line above
       gross > 150             AS big_order
FROM orders
ORDER BY order_id;

-- COMMAND ----------

-- 3.5 LATERAL + table-valued generator: one row per order line.
--     inline() explodes ARRAY<STRUCT> straight into columns.
SELECT o.order_id, o.customer, i.sku, i.qty, i.price, i.qty * i.price AS line_total
FROM orders o,
     LATERAL inline(o.items) AS i
ORDER BY o.order_id, i.sku;

-- COMMAND ----------

-- The legacy Hive spelling: LATERAL VIEW (generators only). posexplode also returns the position.
SELECT o.order_id, v.pos, v.item.sku, v.item.qty
FROM orders o
LATERAL VIEW posexplode(o.items) v AS pos, item
ORDER BY o.order_id, v.pos;

-- COMMAND ----------

-- 3.6 Lateral against a derived table: for every month, the best order of that month
SELECT m.month, b.order_id, b.customer, b.amount
FROM (SELECT DISTINCT date_trunc('MONTH', order_ts)::date AS month FROM orders) m
CROSS JOIN LATERAL (
  SELECT order_id, customer, amount
  FROM orders
  WHERE date_trunc('MONTH', order_ts)::date = m.month
  ORDER BY amount DESC
  LIMIT 1
) b
ORDER BY m.month;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 4. Complex types & higher-order functions
-- MAGIC
-- MAGIC Spark SQL has first-class **`ARRAY`**, **`MAP`** and **`STRUCT`** columns – most warehouses make you unnest and re-aggregate instead.
-- MAGIC
-- MAGIC * **Higher-order functions** take a **lambda** `x -> expr` and run it per element *inside the row* (no explode / shuffle / group by): `transform`, `filter`, `exists`, `forall`, `aggregate` (alias `reduce`), `zip_with`, `array_sort(arr, (l, r) -> …)`, `map_filter`, `transform_keys`, `transform_values`, `map_zip_with`.
-- MAGIC * `array_of_structs.field` returns an **array of that field** – no lambda needed.
-- MAGIC * **Generators** turn one row into many: `explode`, `explode_outer`, `posexplode`, `inline`, `stack`. Spark allows **one generator directly in the `SELECT` list** – for more than one use `LATERAL`.
-- MAGIC * Arrays are **0-based with `[]`**, but **1-based with `element_at()`** (negative index = from the end). Under ANSI mode `arr[99]` raises; `try_element_at` / `get()` return `NULL`.
-- MAGIC
-- MAGIC 🔁 **Elsewhere:** BigQuery `UNNEST`, Snowflake `FLATTEN`, PostgreSQL `unnest` / `jsonb_array_elements`. Lambdas over arrays exist in Trino/Presto and DuckDB, not in T-SQL.

-- COMMAND ----------

-- 4.1 Higher-order functions on ARRAY<STRUCT<sku, qty, price>>
SELECT order_id,
       items.sku                                               AS skus,           -- field of every element
       transform(items, i -> round(i.qty * i.price, 2))         AS line_totals,
       aggregate(items, 0D, (acc, i) -> acc + i.qty * i.price)  AS total,          -- fold / reduce
       filter(items, i -> i.price > 40).sku                     AS pricey_skus,
       exists(items, i -> i.sku = 'D4')                         AS has_d4,
       forall(items, i -> i.qty >= 2)                           AS all_qty_2plus,
       transform(items, (i, idx) -> concat(idx + 1, ':', i.sku)) AS numbered        -- 2-arg lambda: element, index
FROM orders
ORDER BY order_id;

-- COMMAND ----------

-- 4.2 Array toolbox
SELECT array(3, 1, 2, 3, NULL)                                    AS a,
       array_distinct(a)                                          AS distinct_,
       array_compact(a)                                           AS no_nulls,
       sort_array(a, false)                                       AS sorted_desc,
       array_sort(array('bb', 'a', 'ccc'), (l, r) -> length(r) - length(l)) AS by_length_desc,
       a[0]                                                       AS first_0_based,
       element_at(a, 1)                                           AS first_1_based,
       element_at(a, -2)                                          AS second_from_end,
       try_element_at(a, 99)                                      AS out_of_range,
       slice(a, 2, 2)                                             AS slice_,
       array_union(array(1, 2), array(2, 3))                      AS union_,
       array_intersect(array(1, 2), array(2, 3))                  AS intersect_,
       array_except(array(1, 2), array(2, 3))                     AS except_,
       arrays_overlap(array(1, 2), array(2, 3))                   AS overlap_,
       flatten(array(array(1, 2), array(3)))                      AS flat,
       zip_with(array(1, 2, 3), array(10, 20, 30), (x, y) -> x * y) AS zipped,
       arrays_zip(array('a', 'b'), array(1, 2))                   AS zipped_structs,
       array_join(array('a', NULL, 'c'), '-', '?')                AS joined,
       size(a)                                                    AS size_;

-- COMMAND ----------

-- 4.3 Maps and structs
SELECT map('web', 3, 'phone', 1, 'partner', 1)                     AS m,
       m['web']                                                    AS lookup,
       try_element_at(m, 'fax')                                    AS missing_key,
       map_keys(m)                                                 AS keys,
       map_filter(m, (k, v) -> v > 1)                              AS busy,
       transform_values(m, (k, v) -> v * 100)                      AS scaled,
       map_from_arrays(array('a', 'b'), array(1, 2))               AS from_arrays,
       map_from_entries(array(struct('x', 1), struct('y', 2)))     AS from_entries,
       map_concat(map('a', 1), map('b', 2))                        AS merged,
       named_struct('city', 'Wrocław', 'zip', '50-001')            AS s,
       s.city                                                      AS struct_field,
       to_json(s)                                                  AS as_json;

-- COMMAND ----------

-- 4.4 Generator in the SELECT list (Spark-specific) and the way back with collect_list / array_agg
WITH lines AS (
  SELECT order_id, customer, inline(items)      -- one row per element, struct fields become columns
  FROM orders
)
SELECT sku,
       sum(qty)                                 AS units,
       round(sum(qty * price), 2)               AS revenue,
       array_sort(collect_set(customer))        AS customers,
       array_agg(order_id)                      AS order_ids
FROM lines
GROUP BY sku
ORDER BY revenue DESC;

-- COMMAND ----------

-- stack(n, v1, v2, ...) – a quick manual unpivot / inline table
SELECT name, stack(2, 'monthly', int(salary / 12), 'yearly', salary) AS (period, pay)
FROM employees
WHERE dept = 'Sales'
ORDER BY name, period;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 5. JSON & VARIANT
-- MAGIC
-- MAGIC | Approach | Parse cost | Schema | Use when |
-- MAGIC |---|---|---|---|
-- MAGIC | `get_json_object(str, '$.a.b')` / 🟠 `str:a.b` | re-parses the string on **every** call | none, always returns `STRING` | ad-hoc peeks into raw (bronze) data |
-- MAGIC | `from_json(str, 'schema')` → `STRUCT` | parsed once | fixed, declared up front | stable, known structure (silver) |
-- MAGIC | 🆕 `parse_json(str)` → **`VARIANT`** (Spark 4.0) | parsed once into a binary format | schema-on-read, keeps types | drifting / unknown structure |
-- MAGIC
-- MAGIC * `from_json` returns `NULL` fields for malformed input (PERMISSIVE); `parse_json` **raises** – use `try_parse_json`.
-- MAGIC * VARIANT paths are **case-sensitive**; values are extracted with `variant_get(v, '$.path', 'type')` (raises on a bad cast) or `try_variant_get` (returns `NULL`).
-- MAGIC * 🟠 Databricks shorthand: `col:path.to[0].field` works on JSON strings **and** VARIANT; cast with `::type`.

-- COMMAND ----------

-- 5.1 JSON strings: path extraction and from_json with a DDL schema
SELECT order_id,
       get_json_object(payload, '$.channel')       AS channel,
       get_json_object(payload, '$.ship.city')     AS city,
       get_json_object(payload, '$.tags[0]')       AS first_tag,
       json_array_length(get_json_object(payload, '$.tags')) AS tag_count,
       from_json(payload, 'channel STRING, tags ARRAY<STRING>, ship STRUCT<city: STRING, days: INT>') AS parsed,
       parsed.ship.days                            AS ship_days       -- lateral column alias again
FROM orders
ORDER BY order_id;

-- COMMAND ----------

-- Infer a schema from a sample document, and go back to JSON
SELECT schema_of_json('{"channel":"web","tags":["new"],"ship":{"city":"Wrocław","days":2}}') AS inferred_schema,
       to_json(named_struct('id', 1, 'tags', array('a', 'b')))                                AS struct_to_json,
       to_json(map('k', 1))                                                                   AS map_to_json;

-- COMMAND ----------

-- 5.2 🟠 Databricks colon operator: <json string or variant> : path  [:: type]
SELECT order_id,
       payload:channel              AS channel,
       payload:ship.city            AS city,
       payload:tags[0]              AS first_tag,
       payload:ship.days::int       AS ship_days
FROM orders
WHERE order_id <> 106
ORDER BY order_id;

-- COMMAND ----------

-- 5.3 🆕 VARIANT (Spark 4.0): parse once, query many times, types are preserved
WITH v AS (
  SELECT order_id, try_parse_json(payload) AS doc     -- NULL for the broken document 106
  FROM orders
)
SELECT order_id,
       doc,
       variant_get(doc, '$.channel', 'string')        AS channel,
       variant_get(doc, '$.ship.city', 'string')      AS city,
       try_variant_get(doc, '$.ship.days', 'int')     AS ship_days,      -- 105 stores "2" as a string -> still cast
       schema_of_variant(doc)                         AS shape,
       doc IS NULL                                    AS unparseable
FROM v
ORDER BY order_id;

-- COMMAND ----------

-- 5.4 VARIANT + LATERAL: variant_explode() turns an array/object into rows (pos, key, value)
SELECT o.order_id, t.pos, t.value::string AS tag
FROM orders o,
     LATERAL variant_explode(variant_get(try_parse_json(o.payload), '$.tags')) AS t
ORDER BY o.order_id, t.pos;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 6. Function toolbox – nulls, safe casts, strings, dates, aggregates
-- MAGIC
-- MAGIC **NULL handling & safe evaluation**
-- MAGIC
-- MAGIC * `coalesce(a, b, …)` first non-NULL · `nvl(a, b)` / `ifnull(a, b)` two-argument form · `nvl2(x, if_not_null, if_null)` · `nullif(a, b)` · 🆕 `nullifzero(x)` / `zeroifnull(x)` (4.0).
-- MAGIC * Spark 4 = **ANSI mode**: `CAST('abc' AS INT)`, `1/0`, integer overflow and `array[99]` **raise errors**. The **`try_*`** family returns `NULL` instead: `try_cast`, `try_divide`, `try_add`, `try_subtract`, `try_multiply`, `try_sum`, `try_avg`, `try_to_number`, `try_to_timestamp`, `try_element_at`, `try_parse_json`, `try_variant_get`.
-- MAGIC * Cast shorthand: `expr::type` (Spark 4.0; same as `CAST`). 🟠 Databricks also has `expr?::type` = `try_cast`.
-- MAGIC * `typeof(expr)` shows the resulting data type – handy for checking implicit casts.
-- MAGIC
-- MAGIC 🔁 **Elsewhere:** `TRY_CAST` exists in T-SQL / Snowflake; `SAFE_CAST` in BigQuery; `NVL` comes from Oracle, `ISNULL(a, b)` is the T-SQL spelling (in Spark `isnull(a)` takes **one** argument and returns a boolean!).

-- COMMAND ----------

SELECT coalesce(NULL, NULL, 'c')                   AS coalesce_,
       nvl(NULL, 'fallback')                       AS nvl_,
       nvl2(NULL, 'has value', 'is null')          AS nvl2_,
       nullif(0, 0)                                AS nullif_,
       zeroifnull(NULL)                            AS zeroifnull_,
       isnull(NULL)                                AS isnull_one_arg,
       NULL <=> NULL                               AS null_safe_eq,
       1 IS DISTINCT FROM NULL                     AS is_distinct,
       try_cast('abc' AS INT)                      AS bad_int,
       try_cast('42' AS INT)                       AS good_int,
       '2026-11-17'::date                          AS colon_cast,
       try_divide(10, 0)                           AS div_by_zero,
       try_add(2147483647, 1)                      AS int_overflow,
       try_to_number('1,234.50', '9,999.99')       AS parsed_number,
       try_to_timestamp('17/11/2026 08:30', 'dd/MM/yyyy HH:mm') AS parsed_ts,
       try_to_timestamp('not a date')              AS bad_ts,
       typeof(1 + 1.5)                             AS type_int_plus_dec,
       typeof(1 / 2)                               AS type_of_division;   -- '/' always returns DOUBLE; use 'div' for integer division

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Strings**
-- MAGIC
-- MAGIC * Concatenate with `||` or `concat()` (NULL if any part is NULL) vs `concat_ws(sep, …)` (**skips** NULLs, accepts arrays).
-- MAGIC * Regex uses **Java** syntax: `regexp_extract(str, pattern, group)`, `regexp_extract_all`, `regexp_replace`, `regexp_like` / `RLIKE`, `regexp_count`. In a normal string literal backslashes must be doubled (`'\\d+'`) – or use a **raw literal** `r'\d+'`.
-- MAGIC * `LIKE` is case-sensitive; `ILIKE` is not. `LIKE ANY ('a%', 'b%')` / `LIKE ALL (…)` test several patterns at once.
-- MAGIC * `split(str, regex)` returns an **array** (0-based `[]`); `split_part(str, delim, n)` returns the n-th part (1-based, literal delimiter, negative = from the end).

-- COMMAND ----------

SELECT 'Bartosz' || ' ' || 'Malec'                              AS piped,
       concat_ws('-', 'a', NULL, 'c')                           AS concat_ws_skips_null,
       split('2026-11-17', '-')                                 AS split_arr,
       split_part('alice@example.com', '@', 2)                  AS domain,
       substring_index('a.b.c.d', '.', 2)                       AS first_two_parts,
       regexp_extract('INV-2026-00042', r'INV-(\d{4})-(\d+)', 2) AS invoice_no,
       regexp_extract_all('a1b22c333', r'\d+', 0)                  AS all_numbers,
       regexp_replace('  too    many   spaces ', r'\s+', ' ')   AS squeezed,
       'Wrocław' ILIKE 'WROC%'                                  AS ilike_,
       'data engineer' LIKE ANY ('%analyst%', '%engineer%')     AS like_any,
       initcap('spark sql')                                     AS initcap_,
       lpad('42', 6, '0')                                       AS lpad_,
       format_string('%s has %d rows (%.1f%%)', 'orders', 6, 12.345D) AS printf_style,
       format_number(1234567.891, 2)                            AS format_number_,
       translate('ąćęłńóśźż', 'ąćęłńóśźż', 'acelnoszz')         AS no_diacritics,
       levenshtein('kitten', 'sitting')                         AS edit_distance,
       left('Databricks', 4) || right('Databricks', 6)          AS left_right,
       startswith('Databricks', 'Data')                         AS startswith_,
       sha2('secret', 256)                                      AS sha256_hex,
       mask('AbC-1234')                                         AS masked;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Dates & timestamps**
-- MAGIC
-- MAGIC * `DATE` has no time, `TIMESTAMP` is an instant shown in the **session time zone**, `TIMESTAMP_NTZ` is a wall-clock value without zone.
-- MAGIC * Format patterns are **Java `DateTimeFormatter`** (`yyyy-MM-dd HH:mm:ss`), not `strftime` and not Oracle/Postgres `YYYY-MM-DD HH24:MI`. `MM` = month, `mm` = minute.
-- MAGIC * `datediff(end, start)` = days; `date_diff(unit, start, end)` / `timestampdiff` = any unit (note the **reversed argument order**).
-- MAGIC * `date_trunc(unit, ts)` returns a **timestamp**, `trunc(date, unit)` returns a **date** (and the arguments are swapped).
-- MAGIC * Interval arithmetic: `ts + INTERVAL 1 DAY`, `d + INTERVAL '1-6' YEAR TO MONTH`, `make_interval(...)`.

-- COMMAND ----------

SELECT TIMESTAMP'2026-11-17 08:30:45'                           AS ts,
       ts::date                                                AS d,
       date_trunc('MONTH', ts)                                 AS month_start_ts,
       trunc(d, 'YEAR')                                        AS year_start_date,
       last_day(d)                                             AS month_end,
       next_day(d, 'FRI')                                      AS next_friday,
       date_add(d, 10)                                         AS plus_10_days,
       add_months(d, -3)                                       AS minus_3_months,
       ts + INTERVAL 36 HOURS                                  AS plus_36h,
       dateadd(WEEK, 2, d)                                     AS plus_2_weeks,
       datediff(DATE'2026-12-24', d)                           AS days_to_xmas_eve,
       date_diff(HOUR, ts, TIMESTAMP'2026-11-18 00:00:00')     AS hours_to_midnight,
       round(months_between(DATE'2026-12-31', d), 2)           AS months_to_year_end,
       date_format(ts, 'EEEE, dd MMM yyyy HH:mm')              AS pretty,
       dayname(d) || ' / ' || monthname(d)                     AS names_4_0,
       extract(DOW FROM d)                                     AS dow_sunday_is_1,
       weekofyear(d)                                           AS iso_week,
       quarter(d)                                              AS qtr,
       to_date('17.11.2026', 'dd.MM.yyyy')                     AS parsed_date,
       make_date(2026, 2, 28) + 1                              AS day_after,
       unix_timestamp(ts)                                      AS epoch_seconds,
       from_unixtime(0)                                        AS epoch_zero_in_session_tz,
       convert_timezone('Europe/Warsaw', 'America/New_York', TIMESTAMP_NTZ'2026-11-17 08:30:00') AS in_new_york;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Aggregates you rarely find elsewhere**
-- MAGIC
-- MAGIC * `max_by(x, y)` / `min_by(x, y)` – value of `x` on the row where `y` is max / min ("argmax").
-- MAGIC * `count_if(cond)`, `any_value(x)`, `bool_and` / `every`, `bool_or` / `some`, `first(x, true)` / `last(x, true)` (ignore NULLs).
-- MAGIC * `collect_list` / `collect_set` / `array_agg` → arrays · 🆕 `listagg` / `string_agg` (4.0) → delimited string with `WITHIN GROUP (ORDER BY …)`.
-- MAGIC * `agg(...) FILTER (WHERE cond)` – conditional aggregation without `CASE`.
-- MAGIC * Approximate (fast on big data): `approx_count_distinct`, `percentile_approx` / `approx_percentile`, HyperLogLog sketches `hll_sketch_agg` + `hll_sketch_estimate`. Exact: `median`, `mode`, `percentile_cont(p) WITHIN GROUP (ORDER BY x)`.

-- COMMAND ----------

SELECT dept,
       count(*)                                          AS headcount,
       count_if(salary >= 150000)                        AS high_earners,
       max_by(name, salary)                              AS best_paid,
       min_by(name, hire_date)                           AS most_senior,
       sum(salary) FILTER (WHERE hire_date >= DATE'2020-01-01') AS payroll_hired_2020_plus,
       median(salary)                                    AS median_salary,
       percentile_approx(salary, 0.9)                    AS p90_approx,
       percentile_cont(0.9) WITHIN GROUP (ORDER BY salary) AS p90_exact,
       mode(salary)                                      AS most_common_salary,
       bool_and(salary > 100000)                         AS all_above_100k,
       approx_count_distinct(manager_id)                 AS managers_approx,
       listagg(name, ', ') WITHIN GROUP (ORDER BY name)  AS names,
       collect_set(year(hire_date))                      AS hire_years
FROM employees
GROUP BY dept
ORDER BY headcount DESC, dept;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Misc Spark-only helpers**
-- MAGIC
-- MAGIC `greatest` / `least` (row-wise max / min) · `hash`, `xxhash64`, `md5`, `sha2`, `crc32` (surrogate keys, change detection) · `uuid()` · `rand()`, 🆕 `uniform(lo, hi)`, 🆕 `randstr(n)` · `monotonically_increasing_id()` (unique, **not** consecutive) · `spark_partition_id()` · `width_bucket` · `assert_true(cond)` / `raise_error(msg)` (data-quality checks) · `decode(x, k1, v1, …, default)` and `if(cond, a, b)` (compact `CASE`).

-- COMMAND ----------

SELECT name,
       greatest(salary, 150000)                              AS floor_150k,
       least(salary, 150000)                                 AS cap_150k,
       xxhash64(emp_id, name)                                AS row_hash,
       sha2(concat_ws('|', emp_id, name, dept), 256)         AS business_key_hash,
       width_bucket(salary, 50000, 350000, 6)                AS salary_bucket,
       decode(dept, 'Engineering', 'ENG', 'Sales', 'SAL', 'OTHER') AS dept_code,
       if(manager_id IS NULL, 'root', 'report')              AS node_type,
       monotonically_increasing_id()                         AS mono_id,
       length(randstr(8))                                    AS random_string_len
FROM employees
ORDER BY emp_id;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 7. Windows, QUALIFY, GROUP BY ALL, PIVOT & friends
-- MAGIC
-- MAGIC * **Named windows** – define once in `WINDOW w AS (…)`, reuse with `OVER w`.
-- MAGIC * **Frames:** `ROWS BETWEEN …` counts physical rows; `RANGE BETWEEN …` works on the ORDER BY **value** (works with `INTERVAL` for time-based windows). Default frame with `ORDER BY` = `RANGE UNBOUNDED PRECEDING → CURRENT ROW`, which is why `last_value()` "does not work" without an explicit frame.
-- MAGIC * `IGNORE NULLS` in `lag`, `lead`, `first_value`, `last_value`, `nth_value` – forward-fill in one line.
-- MAGIC * **`QUALIFY`** filters on window functions (like `HAVING` for aggregates) – no subquery needed. Long available on Databricks, only recently in open-source Spark 4.x; also in Snowflake, BigQuery, Teradata.
-- MAGIC * 🆕 **`GROUP BY ALL`** / **`ORDER BY ALL`** (3.4) – group by every non-aggregated select item.
-- MAGIC * **`SELECT * EXCEPT (cols)`** · `GROUPING SETS` / `ROLLUP` / `CUBE` · `PIVOT` / `UNPIVOT` · `TABLESAMPLE` · set operations with `ALL` (`EXCEPT ALL`, `INTERSECT ALL`).
-- MAGIC * **Hints:** `/*+ BROADCAST(d) */`, `/*+ REPARTITION(8, key) */`, `/*+ MERGE(a, b) */` – physical advice no other engine understands.

-- COMMAND ----------

-- 7.1 Named window, ranking, running total, time-based RANGE frame, forward fill
SELECT customer, order_id, order_ts::date AS order_date, amount,
       row_number()  OVER w                                   AS seq,
       sum(amount)   OVER w                                   AS running_total,
       round(amount / sum(amount) OVER (PARTITION BY customer) * 100, 1) AS pct_of_customer,
       sum(amount)   OVER (ORDER BY order_ts RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW) AS last_30_days_all_customers,
       lag(amount)   OVER w                                   AS prev_amount,
       last_value(if(amount > 90, order_id, NULL)) IGNORE NULLS OVER w AS last_big_order_so_far
FROM orders
WINDOW w AS (PARTITION BY customer ORDER BY order_ts)
ORDER BY customer, order_ts;

-- COMMAND ----------

-- 7.2 QUALIFY: top-2 salaries per department (compare with the LATERAL version in 3.1)
SELECT dept, name, salary,
       dense_rank() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
FROM employees
QUALIFY rnk <= 2
ORDER BY dept, rnk, name;

-- COMMAND ----------

-- 7.3 🆕 GROUP BY ALL + SELECT * EXCEPT
SELECT e.dept,
       year(o.order_ts)                AS yr,
       date_format(o.order_ts, 'MMM')  AS mon,
       count(*)                        AS orders,
       sum(o.amount)                   AS revenue
FROM orders o
JOIN (SELECT * EXCEPT (salary, hire_date) FROM employees) e USING (emp_id)
GROUP BY ALL
ORDER BY ALL;

-- COMMAND ----------

-- 7.4 ROLLUP with grouping_id(): subtotals and a grand total in one pass
SELECT coalesce(customer, '— all customers —')                         AS customer,
       coalesce(get_json_object(payload, '$.channel'), '— all —')     AS channel,
       count(*)                                                       AS orders,
       sum(amount)                                                    AS revenue,
       grouping_id()                                                  AS gid       -- 0 = detail, 1 = customer subtotal, 3 = grand total
FROM orders
WHERE order_id <> 106
GROUP BY ROLLUP (customer, get_json_object(payload, '$.channel'))
ORDER BY gid, customer, channel;

-- COMMAND ----------

-- 7.5 PIVOT: rows -> columns
SELECT *
FROM (SELECT customer, date_format(order_ts, 'MMM') AS mon, amount FROM orders)
PIVOT (sum(amount) FOR mon IN ('Jan', 'Feb', 'Mar'))
ORDER BY customer;

-- COMMAND ----------

-- ... and UNPIVOT: columns -> rows
WITH wide AS (
  SELECT *
  FROM (SELECT customer, date_format(order_ts, 'MMM') AS mon, amount FROM orders)
  PIVOT (sum(amount) FOR mon IN ('Jan', 'Feb', 'Mar'))
)
SELECT *
FROM wide
UNPIVOT INCLUDE NULLS (revenue FOR mon IN (Jan, Feb, Mar))
ORDER BY customer, mon;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 8. SQL UDFs, table functions, variables & dynamic SQL
-- MAGIC
-- MAGIC * **SQL UDF** – `CREATE [TEMPORARY] FUNCTION f(args) RETURNS type RETURN expr`. It is **inlined** into the query plan, so it is as fast as writing the expression by hand (unlike Python UDFs). Persist it in Unity Catalog (`catalog.schema.f`) to share and govern it.
-- MAGIC * **SQL table function** – `RETURNS TABLE (…) RETURN SELECT …`. A parameterised view; call it in `FROM`, per row with `LATERAL`.
-- MAGIC * 🆕 Parameters may have `DEFAULT`s and be passed **by name**: `f(vat => 0.08)` (3.5).
-- MAGIC * 🆕 **Session variables** (4.0): `DECLARE VARIABLE`, `SET VAR`. They are typed, scoped to the session, and usable anywhere a constant is.
-- MAGIC * 🆕 **`IDENTIFIER(expr)`** – turn a string / variable into a table, column or function name safely (no string concatenation → no SQL injection).
-- MAGIC * 🆕 **`EXECUTE IMMEDIATE`** (4.0) – run a statement built at run time, with `USING` arguments for `?` / `:name` markers and optional `INTO variable`.
-- MAGIC
-- MAGIC 🔁 **Elsewhere:** T-SQL inline table-valued functions + `CROSS APPLY`, `DECLARE @v`, `sp_executesql`; PostgreSQL `EXECUTE … USING` in PL/pgSQL.

-- COMMAND ----------

-- 8.1 Scalar SQL UDF with a default and named arguments
CREATE OR REPLACE TEMPORARY FUNCTION net_to_gross(net DOUBLE, vat DOUBLE DEFAULT 0.23)
RETURNS DOUBLE
RETURN round(net * (1 + vat), 2);

SELECT net_to_gross(100)                      AS default_vat,
       net_to_gross(100, 0.08)                AS positional,
       net_to_gross(vat => 0.05, net => 100)  AS named;

-- COMMAND ----------

-- 8.2 SQL table function, called once per manager with LATERAL
CREATE OR REPLACE TEMPORARY FUNCTION reports_of(mgr INT)
RETURNS TABLE (emp_id INT, name STRING, salary INT)
RETURN SELECT emp_id, name, salary FROM employees WHERE manager_id = mgr;

SELECT m.name AS manager, r.name AS report, r.salary
FROM employees m,
     LATERAL reports_of(m.emp_id) r
ORDER BY manager, report;

-- COMMAND ----------

-- 8.3 Session variables + IDENTIFIER()
DECLARE OR REPLACE VARIABLE min_salary INT DEFAULT 0;
DECLARE OR REPLACE VARIABLE src_view   STRING DEFAULT 'employees';

SET VAR min_salary = (SELECT avg(salary) FROM employees);

SELECT min_salary AS threshold, name, salary
FROM IDENTIFIER(src_view)
WHERE salary >= min_salary
ORDER BY salary DESC;

-- COMMAND ----------

-- 8.4 EXECUTE IMMEDIATE: dynamic SQL with bound parameters and INTO
DECLARE OR REPLACE VARIABLE dept_payroll BIGINT;
DECLARE OR REPLACE VARIABLE stmt STRING DEFAULT 'SELECT sum(salary) FROM employees WHERE dept = ? AND salary > ?';

EXECUTE IMMEDIATE stmt INTO dept_payroll USING 'Engineering', 120000;

SELECT dept_payroll AS engineering_payroll_above_120k;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ⚡ **From PySpark:** the same parameter markers work in `spark.sql()` – values are bound, never concatenated into the SQL text.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = spark.sql(
-- MAGIC     "SELECT name, dept, salary FROM employees WHERE dept = :dept AND salary >= :min_salary ORDER BY salary DESC",
-- MAGIC     args={"dept": "Engineering", "min_salary": 150000},
-- MAGIC )
-- MAGIC display(df)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 9. What's new in Spark 4.x – pipe syntax, SQL scripting, collations
-- MAGIC
-- MAGIC | Feature | Since | Where in this notebook |
-- MAGIC |---|---|---|
-- MAGIC | ANSI mode on by default, `try_*` functions | 4.0 | ch. 6 |
-- MAGIC | `VARIANT` type, `parse_json`, `variant_get` | 4.0 | ch. 5 |
-- MAGIC | **SQL pipe syntax** `\|>` | 4.0 | below |
-- MAGIC | **SQL scripting** – `BEGIN … END`, `IF`, `WHILE`, `FOR`, handlers | 4.0 | below |
-- MAGIC | **Collations** – `COLLATE UTF8_LCASE`, `UNICODE_CI_AI` | 4.0 | below |
-- MAGIC | Session variables, `EXECUTE IMMEDIATE`, `IDENTIFIER()` | 4.0 | ch. 8 |
-- MAGIC | `listagg` / `string_agg`, `dayname`, `zeroifnull`, `randstr`, `uniform`, `::` cast | 4.0 | ch. 6 |
-- MAGIC | **Recursive CTEs** | 4.1 | ch. 2 |
-- MAGIC | Lateral column alias, `GROUP BY ALL` | 3.4 | ch. 3, 7 |
-- MAGIC | Named arguments, `DEFAULT` parameters in SQL UDFs | 3.5 | ch. 8 |
-- MAGIC
-- MAGIC **Pipe syntax** – a query written top-down in the order it executes. Start with `FROM`, then chain operators with `|>`: `WHERE`, `SELECT`, `EXTEND` (add columns), `SET` (replace), `DROP`, `AS`, `JOIN`, `AGGREGATE … GROUP BY`, `ORDER BY`, `LIMIT`, `PIVOT`, `UNION ALL` … It mixes freely with classic SQL and compiles to the same plan.

-- COMMAND ----------

-- 9.1 Pipe syntax: no nested subqueries, no HAVING – just another WHERE after the aggregate
FROM employees
|> WHERE manager_id IS NOT NULL
|> EXTEND salary / 12 AS monthly
|> AGGREGATE count(*) AS headcount, round(avg(monthly)) AS avg_monthly, max_by(name, salary) AS best_paid
   GROUP BY dept
|> WHERE headcount >= 2
|> ORDER BY avg_monthly DESC;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **SQL scripting** – procedural logic in pure SQL: local variables, `IF / CASE`, `WHILE / REPEAT / LOOP / FOR`, `LEAVE / ITERATE`, condition handlers. The whole `BEGIN … END` block is one statement; the result of the last query is returned. Good for small orchestration (loop over partitions, conditional DDL) – keep the heavy lifting set-based.

-- COMMAND ----------

-- 9.2 SQL scripting: loop over a query result and build a summary
BEGIN
  DECLARE summary STRING DEFAULT '';
  DECLARE big_depts INT DEFAULT 0;

  FOR r AS SELECT dept, count(*) AS n FROM employees GROUP BY dept ORDER BY dept DO
    SET summary = summary || r.dept || '=' || r.n || '; ';
    IF r.n >= 3 THEN
      SET big_depts = big_depts + 1;
    END IF;
  END FOR;

  SELECT summary AS headcount_by_dept, big_depts AS depts_with_3_plus;
END

-- COMMAND ----------

-- MAGIC %md
-- MAGIC **Collations** – rules for comparing and sorting strings. Default is `UTF8_BINARY` (byte-wise, case-sensitive). `UTF8_LCASE` = case-insensitive; ICU collations such as `UNICODE_CI` (case-insensitive), `UNICODE_CI_AI` (also accent-insensitive) or `pl` (Polish alphabet order) can be set per expression, column, or as a table/schema default – no more `lower()` on both sides of a join.

-- COMMAND ----------

-- 9.3 Collations: comparison and sort order
SELECT 'Łódź' = 'ŁÓDŹ'                                   AS binary_eq,
       'Łódź' COLLATE UTF8_LCASE = 'ŁÓDŹ'                AS case_insensitive_eq,
       'résumé' COLLATE UNICODE_CI_AI = 'RESUME'         AS accent_insensitive_eq,
       collation('abc' COLLATE UNICODE_CI)               AS collation_name,
       array_sort(array('Łódź', 'Zabrze', 'Lublin', 'Ćmielów', 'Cieszyn'))                        AS binary_order,
       sort_array(transform(array('Łódź', 'Zabrze', 'Lublin', 'Ćmielów', 'Cieszyn'), c -> c COLLATE pl)) AS polish_order;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 10. Capstone – org & sales report in one statement
-- MAGIC
-- MAGIC One query that combines everything: a **recursive CTE** builds the transitive closure of the org chart, a **self-join** resolves names, a **LATERAL** subquery picks each team's best order, **higher-order functions** recompute the order total from its line items, and a **window function** ranks the teams.
-- MAGIC
-- MAGIC > A *team* = a manager plus everybody below them, at any depth.

-- COMMAND ----------

WITH RECURSIVE closure (manager_id, emp_id, depth) AS (
  SELECT emp_id, emp_id, 0 FROM employees
  UNION ALL
  SELECT c.manager_id, e.emp_id, c.depth + 1
  FROM closure c
  JOIN employees e ON e.manager_id = c.emp_id
),
team_orders AS (                                   -- every order attributed to every manager above the seller
  SELECT c.manager_id, o.order_id, o.customer, o.amount,
         aggregate(o.items, 0D, (acc, i) -> acc + i.qty * i.price) AS recomputed
  FROM closure c
  JOIN orders o ON o.emp_id = c.emp_id
),
teams AS (
  SELECT c.manager_id,
         count_if(c.depth > 0)                     AS team_size,
         sum(e.salary)                             AS team_payroll,
         max(c.depth)                              AS levels
  FROM closure c
  JOIN employees e ON e.emp_id = c.emp_id
  GROUP BY c.manager_id
  HAVING count_if(c.depth > 0) > 0
)
SELECT m.name                                      AS manager,
       boss.name                                   AS reports_to,
       t.team_size, t.levels, t.team_payroll,
       s.orders, s.revenue,
       round(s.revenue / t.team_payroll * 100, 3)  AS revenue_pct_of_payroll,
       b.order_id                                  AS best_order,
       b.customer                                  AS best_customer,
       b.amount = b.recomputed                     AS best_order_total_checks_out,
       dense_rank() OVER (ORDER BY s.revenue DESC NULLS LAST) AS revenue_rank
FROM teams t
JOIN employees m         ON m.emp_id = t.manager_id
LEFT JOIN employees boss ON boss.emp_id = m.manager_id                 -- self-join
LEFT JOIN LATERAL (
  SELECT count(*) AS orders, sum(amount) AS revenue
  FROM team_orders x WHERE x.manager_id = t.manager_id
) s ON true
LEFT JOIN LATERAL (
  SELECT order_id, customer, amount, round(recomputed, 2) AS recomputed
  FROM team_orders x WHERE x.manager_id = t.manager_id
  ORDER BY amount DESC LIMIT 1
) b ON true
ORDER BY revenue_rank, manager;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## 11. Cheat sheet – Spark SQL vs other dialects
-- MAGIC
-- MAGIC | Need | Spark SQL / Databricks | T-SQL | PostgreSQL |
-- MAGIC |---|---|---|---|
-- MAGIC | Recursive CTE | `WITH RECURSIVE … MAX RECURSION LEVEL n` | `WITH …` + `OPTION (MAXRECURSION n)` | `WITH RECURSIVE …` (+ `CYCLE`, `SEARCH`) |
-- MAGIC | Per-row subquery | `LATERAL (…)`, `LEFT JOIN LATERAL … ON true` | `CROSS APPLY`, `OUTER APPLY` | `LATERAL` |
-- MAGIC | Unnest an array | `explode` / `inline` / `LATERAL VIEW` | `OPENJSON`, `STRING_SPLIT` | `unnest` |
-- MAGIC | Exists / not exists join | `LEFT SEMI JOIN` / `LEFT ANTI JOIN` | `EXISTS` / `NOT EXISTS` | `EXISTS` / `NOT EXISTS` |
-- MAGIC | Null-safe equality | `<=>` | `IS NOT DISTINCT FROM` (2022+) | `IS NOT DISTINCT FROM` |
-- MAGIC | Filter on a window function | `QUALIFY` | subquery / CTE | subquery / CTE |
-- MAGIC | Arg-max | `max_by(x, y)` | `TOP 1 … ORDER BY` | `DISTINCT ON` / `ORDER BY … LIMIT 1` |
-- MAGIC | String aggregation | `listagg` / `string_agg`, `collect_list` + `array_join` | `STRING_AGG` | `string_agg` |
-- MAGIC | Safe cast | `try_cast`, 🟠 `?::` | `TRY_CAST` | – (custom function) |
-- MAGIC | Semi-structured | `VARIANT`, `from_json`, 🟠 `col:path` | `JSON_VALUE`, `OPENJSON` | `jsonb`, `->`, `->>` |
-- MAGIC | Lambdas over arrays | `transform`, `filter`, `aggregate` | – | – |
-- MAGIC | Group by everything | `GROUP BY ALL` | – | – |
-- MAGIC | Procedural SQL | SQL scripting `BEGIN … END` | T-SQL batches | PL/pgSQL |
-- MAGIC | Top-down query | pipe syntax `\|>` | – | – |
-- MAGIC
-- MAGIC **Rules of thumb**
-- MAGIC
-- MAGIC 1. Depth unknown → **recursive CTE**; depth known or just a series → `sequence` / `range` / fixed self-joins.
-- MAGIC 2. "For each row, the top-N / latest / several aggregates" → **LATERAL** (or `QUALIFY` + window when you scan the same table anyway).
-- MAGIC 3. "Compare with the previous row" → **window function**; "compare with *any other* row" → **self-join** with an equality key.
-- MAGIC 4. Array work inside a row → **higher-order functions**; across rows → `explode` then aggregate.
-- MAGIC 5. Dirty input → **`try_*`** functions; never switch ANSI mode off to hide errors.