#!/usr/bin/env python3
"""
Utility script to manage persisted indices for Linear RAG and Graph RAG backends.

Usage:
    python manage_indices.py list              # List all persisted indices
    python manage_indices.py info [type]       # Show detailed info about indices
    python manage_indices.py clear [type]      # Clear persisted indices
    python manage_indices.py size              # Show storage usage

Where [type] can be: linearrag, graphrag, or all (default)
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

MODELS_DIR = Path("models")
LINEAR_RAG_PREFIX = "linearrag_"
GRAPH_RAG_PREFIX = "graphrag_"


def get_dir_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def list_indices():
    """List all persisted indices."""
    print("\n📊 Persisted Indices Status\n" + "="*60)

    # Find all index directories
    linear_indices = []
    graph_indices = []

    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                if item.name.startswith(LINEAR_RAG_PREFIX):
                    data_source = item.name[len(LINEAR_RAG_PREFIX):]
                    linear_indices.append((data_source, item))
                elif item.name.startswith(GRAPH_RAG_PREFIX):
                    data_source = item.name[len(GRAPH_RAG_PREFIX):]
                    graph_indices.append((data_source, item))

    # Display Linear RAG indices
    if linear_indices:
        print("\n🔍 Linear RAG Indices:")
        for data_source, path in sorted(linear_indices):
            size = get_dir_size(path)
            metadata_path = path / "metadata.json"
            status = "✓ Active" if metadata_path.exists() else "⚠️  Incomplete"

            print(f"  • {data_source}")
            print(f"    Location: {path}")
            print(f"    Size: {format_size(size)}")
            print(f"    Status: {status}")

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"    Data Hash: {metadata.get('data_hash', 'N/A')[:16]}...")
                print(f"    Embedding Model: {metadata.get('embedding_model', 'N/A')}")
                print(f"    LLM Model: {metadata.get('llm_model', 'N/A')}")
    else:
        print("\n🔍 Linear RAG Indices: None found")

    # Display Graph RAG indices
    if graph_indices:
        print("\n🕸️  Graph RAG Indices:")
        for data_source, path in sorted(graph_indices):
            size = get_dir_size(path)
            metadata_path = path / "metadata.json"
            status = "✓ Active" if metadata_path.exists() else "⚠️  Incomplete"

            print(f"  • {data_source}")
            print(f"    Location: {path}")
            print(f"    Size: {format_size(size)}")
            print(f"    Status: {status}")

            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"    Data Hash: {metadata.get('data_hash', 'N/A')[:16]}...")
                print(f"    Embedding Model: {metadata.get('embedding_model', 'N/A')}")
                print(f"    LLM Model: {metadata.get('llm_model', 'N/A')}")
    else:
        print("\n🕸️  Graph RAG Indices: None found")

    total_size = get_dir_size(MODELS_DIR)
    total_indices = len(linear_indices) + len(graph_indices)
    print(f"\n{'='*60}")
    print(f"Total Indices: {total_indices}")
    print(f"Total Storage: {format_size(total_size)}\n")


def show_info(index_type: str = "all", data_source: str = None):
    """Show detailed information about indices.

    Args:
        index_type: 'linearrag', 'graphrag', or 'all'
        data_source: Specific data source to show (optional)
    """
    print(f"\n📋 Detailed Index Information\n" + "="*60)

    indices_to_show = []

    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                if item.name.startswith(LINEAR_RAG_PREFIX) and index_type in ["linearrag", "all"]:
                    data_source_name = item.name[len(LINEAR_RAG_PREFIX):]
                    if data_source is None or data_source == data_source_name:
                        indices_to_show.append(("Linear RAG", data_source_name, item))
                elif item.name.startswith(GRAPH_RAG_PREFIX) and index_type in ["graphrag", "all"]:
                    data_source_name = item.name[len(GRAPH_RAG_PREFIX):]
                    if data_source is None or data_source == data_source_name:
                        indices_to_show.append(("Graph RAG", data_source_name, item))

    if not indices_to_show:
        filter_desc = f" (type: {index_type}" + (f", data_source: {data_source}" if data_source else "") + ")"
        print(f"No indices found{filter_desc}")
        return

    for backend_type, data_source_name, path in sorted(indices_to_show):
        print(f"\n{backend_type} Index - {data_source_name}:")
        print("-" * 60)

        if not path.exists():
            print("  Status: Not found")
            continue

        print(f"  Status: ✓ Found")
        print(f"  Location: {path}")

        # List all files
        files = list(path.rglob('*'))
        file_list = [f for f in files if f.is_file()]

        print(f"  Files ({len(file_list)}):")
        for file in sorted(file_list):
            rel_path = file.relative_to(path)
            size = file.stat().st_size
            print(f"    - {rel_path} ({format_size(size)})")

        # Show metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            print("\n  Metadata:")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            for key, value in metadata.items():
                if key == "data_hash":
                    print(f"    {key}: {value[:16]}... (truncated)")
                else:
                    print(f"    {key}: {value}")

    print(f"\n{'='*60}")
    print(f"Total indices shown: {len(indices_to_show)}\n")


def clear_indices(index_type: str = "all", data_source: str = None):
    """Clear persisted indices.

    Args:
        index_type: 'linearrag', 'graphrag', or 'all'
        data_source: Specific data source to clear (optional)
    """
    indices_to_remove = []

    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                if item.name.startswith(LINEAR_RAG_PREFIX) and index_type in ["linearrag", "all"]:
                    data_source_name = item.name[len(LINEAR_RAG_PREFIX):]
                    if data_source is None or data_source == data_source_name:
                        indices_to_remove.append(("Linear RAG", data_source_name, item))
                elif item.name.startswith(GRAPH_RAG_PREFIX) and index_type in ["graphrag", "all"]:
                    data_source_name = item.name[len(GRAPH_RAG_PREFIX):]
                    if data_source is None or data_source == data_source_name:
                        indices_to_remove.append(("Graph RAG", data_source_name, item))

    if not indices_to_remove:
        print(f"\n🗑️  No indices found to clear (type: {index_type}" + (f", data_source: {data_source}" if data_source else "") + ")")
        return

    print(f"\n🗑️  Clearing Indices\n" + "="*60)
    print(f"Type: {index_type}" + (f" | Data Source: {data_source}" if data_source else ""))

    total_removed = 0
    total_size_freed = 0

    for backend_type, data_source_name, path in indices_to_remove:
        size = get_dir_size(path)
        print(f"\nRemoving {backend_type} index for '{data_source_name}'...")
        print(f"  Location: {path}")
        print(f"  Size: {format_size(size)}")

        try:
            shutil.rmtree(path)
            print(f"  Status: ✓ Removed successfully")
            total_removed += 1
            total_size_freed += size
        except Exception as e:
            print(f"  Status: ✗ Error: {e}")

    print(f"\n{'='*60}")
    print(f"Summary: Removed {total_removed} indices, freed {format_size(total_size_freed)}")
    print()


def show_size():
    """Show storage usage of indices."""
    print("\n💾 Storage Usage\n" + "="*60)

    # Calculate sizes for different index types
    linear_indices = []
    graph_indices = []

    if MODELS_DIR.exists():
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                if item.name.startswith(LINEAR_RAG_PREFIX):
                    data_source = item.name[len(LINEAR_RAG_PREFIX):]
                    size = get_dir_size(item)
                    linear_indices.append((data_source, size))
                elif item.name.startswith(GRAPH_RAG_PREFIX):
                    data_source = item.name[len(GRAPH_RAG_PREFIX):]
                    size = get_dir_size(item)
                    graph_indices.append((data_source, size))

    total_linear_size = sum(size for _, size in linear_indices)
    total_graph_size = sum(size for _, size in graph_indices)
    total_size = get_dir_size(MODELS_DIR)

    print(f"\nBy Index Type:")
    print(f"Linear RAG Indices: {format_size(total_linear_size)} ({len(linear_indices)} indices)")
    print(f"Graph RAG Indices:  {format_size(total_graph_size)} ({len(graph_indices)} indices)")
    print(f"{'-'*60}")
    print(f"Total Storage:      {format_size(total_size)}")

    if total_size > 0:
        print(f"\nBreakdown by Type:")
        if total_linear_size > 0:
            pct = (total_linear_size / total_size) * 100
            print(f"  Linear RAG: {pct:.1f}%")
        if total_graph_size > 0:
            pct = (total_graph_size / total_size) * 100
            print(f"  Graph RAG:  {pct:.1f}%")

    # Show individual index sizes if there are multiple
    if len(linear_indices) > 1 or len(graph_indices) > 1:
        print(f"\nIndividual Index Sizes:")

        if linear_indices:
            print("  Linear RAG:")
            for data_source, size in sorted(linear_indices):
                print(f"    • {data_source}: {format_size(size)}")

        if graph_indices:
            print("  Graph RAG:")
            for data_source, size in sorted(graph_indices):
                print(f"    • {data_source}: {format_size(size)}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Manage persisted indices for RAG backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_indices.py list
  python manage_indices.py info linearrag
  python manage_indices.py info linearrag --data-source graph_small
  python manage_indices.py clear graphrag
  python manage_indices.py clear all --data-source graph_small
  python manage_indices.py size
        """
    )
    
    parser.add_argument(
        'command',
        choices=['list', 'info', 'clear', 'size'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'type',
        nargs='?',
        choices=['linearrag', 'graphrag', 'all'],
        default='all',
        help='Index type (default: all)'
    )

    parser.add_argument(
        '--data-source',
        help='Specific data source to manage (optional)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_indices()
    elif args.command == 'info':
        show_info(args.type, args.data_source)
    elif args.command == 'clear':
        # Ask for confirmation
        target_desc = f"{args.type} indices" + (f" for data source '{args.data_source}'" if args.data_source else "")
        response = input(f"\n⚠️  Are you sure you want to clear {target_desc}? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            clear_indices(args.type, args.data_source)
        else:
            print("Operation cancelled.")
    elif args.command == 'size':
        show_size()


if __name__ == "__main__":
    main()


