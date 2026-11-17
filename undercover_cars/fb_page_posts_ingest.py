# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Facebook Page → Delta: posts, publication timestamps & hashtags
# MAGIC
# MAGIC Ingests **all posts** of a public Facebook Page via the official **Graph API**,
# MAGIC lands them in a **bronze** Delta table (raw JSON, idempotent MERGE), and derives a
# MAGIC **silver** table with one row per post plus a parsed `hashtags` array and
# MAGIC `published_ts` (UTC).
# MAGIC
# MAGIC Target page: `nieoznakowaneradiowozy`
# MAGIC
# MAGIC ### Access requirements (read this first)
# MAGIC The Graph API only returns another page's posts if your token has the right reach:
# MAGIC - **You administer the page** → a **Page access token** (or a User token with
# MAGIC   `pages_read_engagement`) works directly.
# MAGIC - **You do NOT own the page** (likely here) → your App needs the
# MAGIC   **`page_public_content_access`** feature, granted through **Meta App Review**.
# MAGIC   Without it, the `/{page}/posts` edge returns an empty list or a permissions error.
# MAGIC
# MAGIC The token is read from a **Databricks secret** — never hard-code it.
# MAGIC
# MAGIC ### Notes on fields
# MAGIC - Facebook posts have **no native "tags"**. Here `hashtags` = `#...` tokens parsed
# MAGIC   from the post `message`. (Tagged entities would be the `to` / `message_tags`
# MAGIC   fields and need extra permissions — not included.)
# MAGIC - `/posts` returns posts published by the Page. Use `/feed` if you also want
# MAGIC   visitor posts (usually not what you want for an archive).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("page",            "nieoznakowaneradiowozy", "Page username or numeric ID")
dbutils.widgets.text("api_version",     "v21.0",                  "Graph API version")
dbutils.widgets.text("secret_scope",    "facebook",               "Databricks secret scope")
dbutils.widgets.text("secret_key",      "graph_access_token",     "Secret key holding the token")
dbutils.widgets.text("catalog",         "main",                   "Target catalog")
dbutils.widgets.text("schema",          "social_fb",              "Target schema")
dbutils.widgets.text("since",           "",                       "Only posts after (YYYY-MM-DD, optional)")
dbutils.widgets.text("until",           "",                       "Only posts before (YYYY-MM-DD, optional)")

PAGE          = dbutils.widgets.get("page").strip()
API_VERSION   = dbutils.widgets.get("api_version").strip()
SECRET_SCOPE  = dbutils.widgets.get("secret_scope").strip()
SECRET_KEY    = dbutils.widgets.get("secret_key").strip()
CATALOG       = dbutils.widgets.get("catalog").strip()
SCHEMA        = dbutils.widgets.get("schema").strip()
SINCE         = dbutils.widgets.get("since").strip()
UNTIL         = dbutils.widgets.get("until").strip()

BRONZE_TABLE  = f"{CATALOG}.{SCHEMA}.fb_posts_bronze"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.fb_posts_silver"

# Token comes from the secret store. Create it once with the Databricks CLI:
#   databricks secrets create-scope facebook
#   databricks secrets put-secret facebook graph_access_token
ACCESS_TOKEN  = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)

# Fields requested per post. created_time is ISO8601 with timezone.
POST_FIELDS   = "id,created_time,updated_time,message,story,permalink_url,status_type"
PAGE_SIZE     = 100  # max the edge reliably returns

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA  IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Graph API client (pagination + retry/backoff)

# COMMAND ----------

import time
import requests
from datetime import datetime, timezone

BASE = f"https://graph.facebook.com/{API_VERSION}"
session = requests.Session()


def _get(url, params=None, max_retries=5):
    """GET with exponential backoff on throttling / transient errors."""
    for attempt in range(max_retries):
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()

        # Rate limiting / transient server errors → back off and retry.
        if resp.status_code in (429, 500, 502, 503) or \
           (resp.status_code == 400 and "rate limit" in resp.text.lower()):
            wait = min(2 ** attempt * 5, 300)
            print(f"  [{resp.status_code}] backing off {wait}s (attempt {attempt+1})")
            time.sleep(wait)
            continue

        # Non-retryable: surface the Graph API error body.
        raise RuntimeError(f"Graph API {resp.status_code}: {resp.text}")
    raise RuntimeError(f"Exhausted retries for {url}")


def resolve_page_id(page):
    """Accept a numeric id as-is; otherwise resolve the vanity username."""
    if page.isdigit():
        return page
    data = _get(f"{BASE}/{page}", params={"fields": "id,name", "access_token": ACCESS_TOKEN})
    print(f"Resolved '{page}' -> id={data['id']} name={data.get('name')!r}")
    return data["id"]


def fetch_all_posts(page_id):
    """Yield every post object, following next-page cursors."""
    params = {
        "fields": POST_FIELDS,
        "limit": PAGE_SIZE,
        "access_token": ACCESS_TOKEN,
    }
    if SINCE:
        params["since"] = SINCE
    if UNTIL:
        params["until"] = UNTIL

    url = f"{BASE}/{page_id}/posts"
    total = 0
    while url:
        payload = _get(url, params=params)
        batch = payload.get("data", [])
        total += len(batch)
        print(f"  fetched {len(batch)} (running total {total})")
        for post in batch:
            yield post
        # After the first call, the cursor URL already carries all params.
        url = payload.get("paging", {}).get("next")
        params = None
    print(f"Done. {total} posts total.")


page_id = resolve_page_id(PAGE)
posts   = list(fetch_all_posts(page_id))
ingested_at = datetime.now(timezone.utc).isoformat()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Bronze — raw posts (idempotent MERGE on post id)

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

bronze_schema = StructType([
    StructField("id",           StringType(), False),
    StructField("created_time", StringType(), True),
    StructField("raw_json",     StringType(), True),
    StructField("ingested_at",  StringType(), True),
])

if posts:
    bronze_rows = [
        (p["id"], p.get("created_time"), json.dumps(p, ensure_ascii=False), ingested_at)
        for p in posts
    ]
    bronze_df = spark.createDataFrame(bronze_rows, schema=bronze_schema)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
            id           STRING,
            created_time STRING,
            raw_json     STRING,
            ingested_at  STRING
        ) USING DELTA
    """)

    bronze_df.createOrReplaceTempView("_bronze_updates")
    spark.sql(f"""
        MERGE INTO {BRONZE_TABLE} t
        USING _bronze_updates s ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET
            t.created_time = s.created_time,
            t.raw_json     = s.raw_json,
            t.ingested_at  = s.ingested_at
        WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"Bronze MERGE complete: {bronze_df.count()} rows processed into {BRONZE_TABLE}")
else:
    print("No posts returned — check token reach / page_public_content_access. Bronze unchanged.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Silver — parsed posts with hashtags array + UTC timestamp

# COMMAND ----------

# Unicode-aware hashtag regex: '#' + word chars (letters incl. Polish diacritics,
# digits, underscore). Lowercased for consistent grouping.
HASHTAG_RE = r"#(\w+)"

silver_df = (
    spark.table(BRONZE_TABLE)
         .select(
             F.col("id").alias("post_id"),
             F.from_json(
                 F.col("raw_json"),
                 "id STRING, created_time STRING, updated_time STRING, "
                 "message STRING, story STRING, permalink_url STRING, status_type STRING"
             ).alias("p"),
         )
         .select(
             "post_id",
             F.col("p.message").alias("message"),
             F.col("p.permalink_url").alias("permalink_url"),
             F.col("p.status_type").alias("status_type"),
             F.to_timestamp("p.created_time").alias("published_ts"),       # UTC
             F.to_timestamp("p.updated_time").alias("updated_ts"),
         )
         # Extract all #hashtags from the message, lowercase, as an array.
         .withColumn(
             "hashtags",
             F.expr(f"transform(regexp_extract_all(coalesce(message, ''), '{HASHTAG_RE}', 1), x -> lower(x))")
         )
         .withColumn("hashtag_count", F.size("hashtags"))
         .withColumn("published_date", F.to_date("published_ts"))
)

(silver_df.write
          .format("delta")
          .mode("overwrite")
          .option("overwriteSchema", "true")
          .saveAsTable(SILVER_TABLE))

print(f"Silver written: {silver_df.count()} rows -> {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quick checks

# COMMAND ----------

display(
    spark.table(SILVER_TABLE)
         .select("post_id", "published_ts", "hashtag_count", "hashtags", "permalink_url")
         .orderBy(F.col("published_ts").desc())
         .limit(25)
)

# COMMAND ----------

# Top hashtags across the whole archive
display(
    spark.table(SILVER_TABLE)
         .select(F.explode("hashtags").alias("hashtag"))
         .groupBy("hashtag")
         .count()
         .orderBy(F.col("count").desc())
         .limit(30)
)