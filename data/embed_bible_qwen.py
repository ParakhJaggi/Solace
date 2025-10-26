#!/usr/bin/env python3
"""
Script to create a ChromaDB vector database from Bible verses using Qwen3-Embedding-8B.
This creates a persistent database that can be loaded in production.
"""

import xml.etree.ElementTree as ET
import collections
import argparse
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Configuration
DEFAULT_INPUT_FILE = "./engwebp_vpl.xml"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = "Represent the emotional or spiritual concern described by the user to retrieve comforting Bible passages:"
DEFAULT_OUTPUT_DIR = "./output_bible_db_qwen"

# Bible book name mappings for better readability
BOOK_NAMES = {
    "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus", "NUM": "Numbers", "DEU": "Deuteronomy",
    "JOS": "Joshua", "JDG": "Judges", "RUT": "Ruth", "1SA": "1 Samuel", "2SA": "2 Samuel",
    "1KI": "1 Kings", "2KI": "2 Kings", "1CH": "1 Chronicles", "2CH": "2 Chronicles",
    "EZR": "Ezra", "NEH": "Nehemiah", "EST": "Esther", "JOB": "Job", "PSA": "Psalms",
    "PRO": "Proverbs", "ECC": "Ecclesiastes", "SNG": "Song of Solomon", "ISA": "Isaiah",
    "JER": "Jeremiah", "LAM": "Lamentations", "EZK": "Ezekiel", "DAN": "Daniel",
    "HOS": "Hosea", "JOL": "Joel", "AMO": "Amos", "OBA": "Obadiah", "JON": "Jonah",
    "MIC": "Micah", "NAM": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah", "HAG": "Haggai",
    "ZEC": "Zechariah", "MAL": "Malachi",
    "MAT": "Matthew", "MRK": "Mark", "LUK": "Luke", "JHN": "John", "ACT": "Acts",
    "ROM": "Romans", "1CO": "1 Corinthians", "2CO": "2 Corinthians", "GAL": "Galatians",
    "EPH": "Ephesians", "PHP": "Philippians", "COL": "Colossians", "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians", "1TI": "1 Timothy", "2TI": "2 Timothy", "TIT": "Titus",
    "PHM": "Philemon", "HEB": "Hebrews", "JAS": "James", "1PE": "1 Peter", "2PE": "2 Peter",
    "1JN": "1 John", "2JN": "2 John", "3JN": "3 John", "JUD": "Jude", "REV": "Revelation"
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Create ChromaDB vector database from Bible verses using Qwen3-Embedding-8B"
    )
    parser.add_argument(
        "-i", "--input_file",
        default=DEFAULT_INPUT_FILE,
        help=f"Path to input VPL XML file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "-m", "--model_name",
        default=DEFAULT_MODEL_NAME,
        help=f"HuggingFace model to use (default: {DEFAULT_MODEL_NAME})"
    )
    parser.add_argument(
        "-q", "--query_instruction",
        default=DEFAULT_QUERY_INSTRUCTION,
        help=f"Query instruction for embeddings (default: {DEFAULT_QUERY_INSTRUCTION})"
    )
    parser.add_argument(
        "-o", "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for ChromaDB (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Number of verses to group together (default: 3)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for embeddings (default: cpu, use 'cuda' for GPU)"
    )
    return parser.parse_args()


def parse_bible_xml(xml_file):
    """Parse the Bible XML file and group verses by chapter."""
    print(f"\n📖 Parsing XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    verses_by_chapter = collections.defaultdict(list)
    for verse in root.findall("v"):
        book = verse.attrib["b"]
        chapter = int(verse.attrib["c"])
        verse_num = int(verse.attrib["v"])
        text = verse.text if verse.text else ""
        
        verses_by_chapter[(book, chapter)].append((verse_num, text))
    
    total_verses = sum(len(verses) for verses in verses_by_chapter.values())
    print(f"   ✓ Found {total_verses:,} verses across {len(verses_by_chapter):,} chapters")
    
    return verses_by_chapter


def create_verse_chunks(verses_by_chapter, chunk_size=3):
    """
    Create chunks of verses for embedding.
    Each chunk contains a few verses for better context.
    """
    print(f"\n📝 Creating verse chunks (chunk_size={chunk_size})...")
    
    documents = []
    metadatas = []
    ids = []
    testament = "OT"
    
    for (book, chapter), verses in sorted(verses_by_chapter.items()):
        # Update testament marker
        if book.upper().startswith("MAT"):
            testament = "NT"
        
        book_name = BOOK_NAMES.get(book, book)
        
        # Group verses into chunks
        for i in range(0, len(verses), chunk_size):
            chunk_verses = verses[i:i + chunk_size]
            
            # Create the text content
            verse_texts = []
            verse_numbers = []
            for verse_num, text in chunk_verses:
                verse_texts.append(text.strip())
                verse_numbers.append(verse_num)
            
            chunk_text = " ".join(verse_texts)
            
            # Create verse reference
            if len(verse_numbers) == 1:
                verse_ref = f"{book_name} {chapter}:{verse_numbers[0]}"
            else:
                verse_ref = f"{book_name} {chapter}:{verse_numbers[0]}-{verse_numbers[-1]}"
            
            # Create document
            documents.append(chunk_text)
            metadatas.append({
                "book": book,
                "book_name": book_name,
                "chapter": chapter,
                "verse_start": verse_numbers[0],
                "verse_end": verse_numbers[-1],
                "reference": verse_ref,
                "testament": testament
            })
            ids.append(f"{book}_{chapter}_{verse_numbers[0]}_{verse_numbers[-1]}")
    
    print(f"   ✓ Created {len(documents):,} verse chunks")
    return documents, metadatas, ids


def create_embeddings_and_store(documents, metadatas, ids, model_name, query_instruction, 
                                output_dir, device="cpu"):
    """Create embeddings using Qwen3-Embedding-8B and store in ChromaDB."""
    
    print(f"\n🤖 Loading embedding model: {model_name}")
    print(f"   Using device: {device}")
    
    # Load the Qwen3 embedding model
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True
    )
    
    print(f"\n🔢 Creating embeddings for {len(documents):,} chunks...")
    print("   (This may take several minutes...)")
    
    start_time = datetime.now()
    
    # Create embeddings
    # For production queries, you'll use the query_instruction
    # For document embeddings, we use the default (no instruction prefix needed)
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )
    
    embed_time = datetime.now() - start_time
    print(f"   ✓ Embeddings created in {embed_time.total_seconds():.1f} seconds")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Create ChromaDB
    print(f"\n💾 Creating ChromaDB at: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=output_dir)
    
    # Create or get collection
    collection_name = "bible_verses"
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "model": model_name,
            "query_instruction": query_instruction,
            "description": "Bible verses embedded with Qwen3-Embedding-8B"
        }
    )
    
    # Add documents to collection in batches
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
        print(f"   Progress: {end_idx:,}/{len(documents):,} chunks added")
    
    print(f"   ✓ Database saved to {output_dir}")
    
    # Save configuration for production use
    config_file = Path(output_dir) / "config.txt"
    with open(config_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Query Instruction: {query_instruction}\n")
        f.write(f"Collection Name: {collection_name}\n")
        f.write(f"Total Chunks: {len(documents)}\n")
        f.write(f"Embedding Dimension: {embeddings.shape[1]}\n")
    
    return collection


def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("📚 Bible Vector Database Creator - Qwen3-Embedding-8B")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input File: {args.input_file}")
    print(f"  Model: {args.model_name}")
    print(f"  Query Instruction: {args.query_instruction}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Chunk Size: {args.chunk_size} verses")
    print(f"  Device: {args.device}")
    
    start_time = datetime.now()
    
    # Parse Bible XML
    verses_by_chapter = parse_bible_xml(args.input_file)
    
    # Create verse chunks
    documents, metadatas, ids = create_verse_chunks(verses_by_chapter, args.chunk_size)
    
    # Create embeddings and store
    collection = create_embeddings_and_store(
        documents, metadatas, ids,
        args.model_name,
        args.query_instruction,
        args.output_dir,
        args.device
    )
    
    total_time = datetime.now() - start_time
    
    print("\n" + "=" * 70)
    print("✅ COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Total time: {total_time.total_seconds():.1f} seconds")
    print(f"\n📁 Your database is ready at: {args.output_dir}")
    print(f"\n🚀 To use in production, load the database with:")
    print(f"   import chromadb")
    print(f"   client = chromadb.PersistentClient(path='{args.output_dir}')")
    print(f"   collection = client.get_collection('bible_verses')")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

