from pathlib import Path

def load_business_info(directory: str) -> str:
    base_path = Path(directory)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")

    content = ""
    for file_path in sorted(base_path.glob("*.txt")):
        section = file_path.read_text(encoding="utf-8").strip()
        content += f"\n\n# {file_path.stem.replace('_', ' ').title()}\n\n{section}"
    
    return content

info = load_business_info("corpus")
