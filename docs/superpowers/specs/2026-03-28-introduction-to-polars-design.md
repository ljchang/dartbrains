# Introduction to Polars Tutorial -- Design Spec

## Context

DartBrains has an existing Introduction to Pandas tutorial. We want to add a parallel Introduction to Polars tutorial as a standalone notebook that teaches polars on its own merits. Both tutorials will coexist — students can choose which to learn or use both.

## Decisions

| Decision | Choice |
|----------|--------|
| Framing | Standalone — no pandas comparisons or callouts |
| Dataset | Same salary.csv used by pandas tutorial |
| File | `content/Introduction_to_Polars.py` (marimo notebook) |
| Dependency | Add `polars` to pyproject.toml |
| TOC placement | After Introduction_to_Pandas in Getting Started section |

## Tutorial Structure

### 1. Introduction
- What polars is (Rust-powered DataFrame library for Python)
- Why it exists: speed, memory efficiency, expressive API, lazy evaluation
- When to use it (tabular data wrangling, ETL, data science)

### 2. Polars Objects
- **Series**: 1D typed array, `pl.Series("name", [1, 2, 3])`
- **DataFrame**: 2D typed table, no index (rows are accessed positionally or by filtering)
- Creating from dicts, lists, numpy arrays

### 3. Loading Data
- `pl.read_csv()` for eager loading
- `pl.scan_csv()` for lazy loading (brief mention, full section later)
- Load the salary dataset from the same GitHub raw URL

### 4. Inspecting Data
- `head()`, `tail()`, `sample()`
- `describe()` for summary statistics
- `schema` for column types
- `shape`, `columns`, `null_count()`

### 5. Dealing with Missing Values
- `null` (polars' missing value) vs `NaN` (only for floats)
- `is_null()`, `drop_nulls()`, `fill_null()`

### 6. The Expression API
- Core concept: `pl.col("name")` as the building block
- Chaining expressions: `pl.col("salary").mean()`, `pl.col("name").str.to_uppercase()`
- Conditional expressions: `pl.when(...).then(...).otherwise(...)`
- Why expressions matter: they're optimizable, composable, and the heart of polars

### 7. Creating New Columns
- `with_columns()` — add or transform columns
- Expressions inside `with_columns`
- `.alias()` to name computed columns
- Immutability: `with_columns` returns a new DataFrame

### 8. Selecting and Filtering
- `select()` — choose columns
- `filter()` — choose rows by condition
- Combining filters with `&`, `|`
- `pl.col("x").is_in([...])` for membership tests

### 9. Renaming
- `rename({"old": "new"})` for columns

### 10. Operations
- Arithmetic on columns
- String operations via `.str` namespace
- Date operations via `.dt` namespace
- `map_elements()` for custom Python functions (with performance caveat)

### 11. Joining Data
- `join()` — inner, left, outer, cross
- `concat()` — vertical (stacking rows) and horizontal (stacking columns)

### 12. Grouping and Aggregation
- `group_by().agg()` — group and compute multiple aggregations in one call
- Multiple aggregations: `[pl.col("salary").mean(), pl.col("salary").max()]`
- `.sort()` after grouping

### 13. Window Functions
- `over()` — compute group-level values without collapsing rows
- Example: salary relative to department mean using `pl.col("salary").mean().over("department")`
- Comparison to groupby+transform pattern

### 14. Reshaping
- `unpivot()` (equivalent to melt)
- `pivot()` (wide format)

### 15. Lazy Evaluation
- `pl.scan_csv()` creates a `LazyFrame`
- Chain operations without executing
- `.collect()` to execute
- `.explain()` to see the query plan
- Why lazy: query optimization, predicate pushdown, projection pushdown

### 16. Exercises
- Mirror the pandas tutorial exercises adapted for polars
- Exercise 1: Load data, inspect, filter
- Exercise 2: Group by, aggregate, sort
- Exercise 3: Join datasets, reshape

## Changes to Other Files

- `pyproject.toml`: add `polars` dependency
- `myst.yml`: add `content/Introduction_to_Polars.ipynb` after Introduction_to_Pandas in TOC
