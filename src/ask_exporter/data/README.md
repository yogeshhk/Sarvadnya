# Data Directory

This directory stores all data files for the Ask Exporter system.

## Structure

- `raw/` - Raw scraped data from export control websites
- `processed/` - Cleaned and processed export control data
- `cache/` - Cached query results and API responses
- `chroma_db/` - ChromaDB vector store (created automatically)
- `export_control.db` - SQLite database (created automatically)

## Notes

- All directories are git-ignored except this README
- Data files are generated during runtime
- Raw data is periodically updated from official sources
