"""
Create complete folder structure for Health Insurance Copilot
Run this in your project root directory
"""
import os
from pathlib import Path
from typing import Dict

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_section(title: str):
    print(f"\n{Colors.BOLD}{'='*60}")
    print(f"📁 {title}")
    print(f"{'='*60}{Colors.END}\n")


# Complete folder structure
FOLDER_STRUCTURE = {
    "config": [],
    
    "data": {
        "raw": {
            "policies": {
                "bajaj_allianz": [],
                "hdfc_ergo": [],
                "tata_aig": [],
                "care_health": [],
            },
            "company_info": [],
            "knowledge_base": [],
        },
        "processed": {
            "chunks": {
                "policies": [],
                "knowledge": [],
            },
            "embeddings": {
                "cache": [],
            },
        },
        "vector_store": {
            "collections": [],
        },
        "cache": {
            "query_cache": [],
            "semantic_cache": [],
            "response_cache": [],
            "retrieval_cache": [],
        },
        "fine_tuning": {
            "datasets": [],
            "checkpoints": [],
            "adapters": {
                "lora_weights": [],
            },
            "models": [],
            "logs": {
                "tensorboard": [],
            },
            "configs": [],
        },
        "evaluation": {
            "benchmarks": [],
            "results": {
                "base_model": [],
                "fine_tuned": [],
            },
            "human_eval": [],
            "reports": [],
        },
        "monitoring": {
            "traces": {
                "langsmith_traces": [],
            },
            "metrics": [],
            "guardrails": [],
        },
    },
    
    "src": {
        "core": [],
        "models": [],
        "services": {
            "llm": [],
            "rag": [],
            "cache": [],
            "guardrails": [],
            "evaluation": [],
        },
        "data_processing": [],
        "fine_tuning": [],
        "api": {
            "routes": [],
            "middleware": [],
        },
        "monitoring": [],
        "utils": [],
    },
    
    "scripts": {
        "data_collection": [],
        "data_processing": [],
        "fine_tuning": [],
        "evaluation": [],
        "deployment": [],
    },
    
    "frontend": {
        "pages": [],
        "components": [],
        "utils": [],
    },
    
    "tests": {
        "unit": [],
        "integration": [],
        "e2e": [],
        "performance": [],
    },
    
    "notebooks": [],
    "docs": [],
    "logs": [],
}


def create_folder_structure(base_path: Path, structure: Dict, parent_path="", level=0):
    """Recursively create folder structure and __init__.py files"""
    folder_count = 0
    init_count = 0
    
    for name, substructure in structure.items():
        folder_path = base_path / name
        folder_path.mkdir(parents=True, exist_ok=True)
        folder_count += 1
        
        # Current path for display
        current_path = f"{parent_path}/{name}" if parent_path else name
        
        # Print folder creation with indentation
        indent = "  " * level
        print(f"{indent}📁 {name}/")
        
        # Create __init__.py in Python package directories
        should_have_init = False
        
        # Check if this is a Python package directory
        if any(x in current_path for x in [
            "src/", "frontend/", "tests/", "scripts/"
        ]):
            should_have_init = True
        
        if should_have_init:
            init_file = folder_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                init_count += 1
                print(f"{indent}  └─ __init__.py")
        
        # Recursively create subfolders
        if isinstance(substructure, dict):
            sub_counts = create_folder_structure(
                folder_path, 
                substructure, 
                current_path, 
                level + 1
            )
            folder_count += sub_counts[0]
            init_count += sub_counts[1]
    
    return folder_count, init_count


def create_placeholder_files(base_path: Path):
    """Create .gitkeep files in empty data directories to preserve structure"""
    print_info("Creating .gitkeep files in data directories...")
    
    data_dirs = [
        "data/raw/policies/bajaj_allianz",
        "data/raw/policies/hdfc_ergo",
        "data/raw/policies/tata_aig",
        "data/raw/policies/care_health",
        "data/raw/company_info",
        "data/raw/knowledge_base",
        "data/processed/chunks/policies",
        "data/processed/chunks/knowledge",
        "data/processed/embeddings/cache",
        "data/vector_store/collections",
        "data/cache/query_cache",
        "data/cache/semantic_cache",
        "data/cache/response_cache",
        "data/cache/retrieval_cache",
        "data/fine_tuning/datasets",
        "data/fine_tuning/checkpoints",
        "data/fine_tuning/adapters/lora_weights",
        "data/fine_tuning/models",
        "data/fine_tuning/logs/tensorboard",
        "data/fine_tuning/configs",
        "data/evaluation/benchmarks",
        "data/evaluation/results/base_model",
        "data/evaluation/results/fine_tuned",
        "data/evaluation/human_eval",
        "data/evaluation/reports",
        "data/monitoring/traces/langsmith_traces",
        "data/monitoring/metrics",
        "data/monitoring/guardrails",
        "logs",
    ]
    
    gitkeep_count = 0
    for dir_path in data_dirs:
        gitkeep_file = base_path / dir_path / ".gitkeep"
        gitkeep_file.touch()
        gitkeep_count += 1
    
    print_success(f"Created {gitkeep_count} .gitkeep files")


def print_tree_summary():
    """Print a summary tree view"""
    tree = """
health-insurance-copilot/
├── config/                      # Configuration files (YAML)
├── data/                        # All data storage (gitignored)
│   ├── raw/                     # Raw collected data
│   │   ├── policies/            # Policy PDFs (32 files)
│   │   ├── company_info/        # IRDAI & company data
│   │   └── knowledge_base/      # Educational content
│   ├── processed/               # Processed data
│   │   ├── chunks/              # Chunked documents
│   │   └── embeddings/          # Pre-computed embeddings
│   ├── vector_store/            # ChromaDB storage
│   ├── cache/                   # Multi-level cache
│   ├── fine_tuning/             # Fine-tuning artifacts
│   ├── evaluation/              # Evaluation results
│   └── monitoring/              # Traces & metrics
├── src/                         # Source code
│   ├── core/                    # Config & constants
│   ├── models/                  # Pydantic schemas
│   ├── services/                # Business logic
│   │   ├── llm/                 # LLM service
│   │   ├── rag/                 # RAG pipeline
│   │   ├── cache/               # Caching layer
│   │   ├── guardrails/          # Safety guardrails
│   │   └── evaluation/          # Evaluation framework
│   ├── data_processing/         # Data preparation
│   ├── fine_tuning/             # Fine-tuning pipeline
│   ├── api/                     # FastAPI application
│   ├── monitoring/              # Observability
│   └── utils/                   # Utilities
├── scripts/                     # Utility scripts
│   ├── data_collection/         # Data collection
│   ├── data_processing/         # Data processing
│   ├── fine_tuning/             # Training scripts
│   ├── evaluation/              # Evaluation scripts
│   └── deployment/              # Deployment scripts
├── frontend/                    # Streamlit UI
│   ├── pages/                   # Multi-page app
│   ├── components/              # Reusable components
│   └── utils/                   # Frontend utilities
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   └── performance/             # Performance tests
├── notebooks/                   # Jupyter notebooks
├── docs/                        # Documentation
└── logs/                        # Application logs
"""
    print_section("FOLDER STRUCTURE OVERVIEW")
    print(tree)


def main():
    """Main execution"""
    print_section("HEALTH INSURANCE COPILOT - FOLDER SETUP")
    
    # Get current directory
    current_dir = Path.cwd()
    project_name = current_dir.name
    
    print_info(f"Creating folder structure in: {current_dir}")
    print_info(f"Project name: {project_name}")
    
    # Confirm with user
    response = input(f"\n✋ This will create folders in the CURRENT directory. Continue? (y/N): ")
    if response.lower() != 'y':
        print("\n❌ Setup cancelled.")
        return
    
    # Create folder structure
    print_section("CREATING FOLDERS")
    folder_count, init_count = create_folder_structure(current_dir, FOLDER_STRUCTURE)
    
    print(f"\n{Colors.GREEN}✅ Created {folder_count} folders{Colors.END}")
    print(f"{Colors.GREEN}✅ Created {init_count} __init__.py files{Colors.END}")
    
    # Create .gitkeep files
    print()
    create_placeholder_files(current_dir)
    
    # Print summary
    print_tree_summary()
    
    # Next steps
    print_section("SETUP COMPLETE! ✅")
    print(f"""
{Colors.GREEN}Folder structure created successfully!{Colors.END}

{Colors.BOLD}Next steps:{Colors.END}

1. Create configuration files:
   • pyproject.toml
   • .env and .env.example
   • .gitignore
   • README.md

2. Create virtual environment:
   uv venv

3. Activate virtual environment:
   source .venv/bin/activate  # Mac/Linux
   .venv\\Scripts\\activate     # Windows

4. Install dependencies:
   uv pip install -e ".[dev]"

5. Start building! 🚀

{Colors.BOLD}Folder Statistics:{Colors.END}
• Total folders: {folder_count}
• Python packages: {init_count}
• Ready for development!
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
