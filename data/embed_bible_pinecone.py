#!/usr/bin/env python3
"""
Upload Bible verses to Pinecone with serverless embeddings.
Pinecone will handle the embedding using llama-text-embed-v2.
"""

import xml.etree.ElementTree as ET
import collections
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Configuration
INPUT_FILE = "./engwebp_vpl.xml"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_HOST = "solace-t42ww4d.svc.aped-4627-b74a.pinecone.io"
CHUNK_SIZE = 3  # Verses per chunk

# Bible book name mappings
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

# New Testament books (for proper testament tagging)
NEW_TESTAMENT_BOOKS = {
    "MAT", "MRK", "LUK", "JHN", "ACT",
    "ROM", "1CO", "2CO", "GAL", "EPH", "PHP", "COL", "1TH", "2TH", "1TI", "2TI", "TIT", "PHM",
    "HEB", "JAS", "1PE", "2PE", "1JN", "2JN", "3JN", "JUD", "REV"
}


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


def create_verse_chunks(verses_by_chapter, chunk_size=CHUNK_SIZE):
    """Create chunks of verses for embedding."""
    print(f"\n📝 Creating verse chunks (chunk_size={chunk_size})...")
    
    records = []
    
    for (book, chapter), verses in sorted(verses_by_chapter.items()):
        # Determine testament based on book code
        testament = "NT" if book.upper() in NEW_TESTAMENT_BOOKS else "OT"
        
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
            
            # Skip empty verses (Pinecone can't embed empty strings)
            if not chunk_text.strip():
                continue
            
            # Create verse reference
            if len(verse_numbers) == 1:
                verse_ref = f"{book_name} {chapter}:{verse_numbers[0]}"
            else:
                verse_ref = f"{book_name} {chapter}:{verse_numbers[0]}-{verse_numbers[-1]}"
            
            # Create record ID
            record_id = f"{book}_{chapter}_{verse_numbers[0]}_{verse_numbers[-1]}"
            
            # Create record for Pinecone (integrated embedding)
            record = {
                "_id": record_id,  # Note: _id not id
                "text": chunk_text,  # Pinecone will auto-embed this
                "book": book,
                "book_name": book_name,
                "chapter": chapter,
                "verse_start": verse_numbers[0],
                "verse_end": verse_numbers[-1],
                "reference": verse_ref,
                "testament": testament,
                "translation": "WEB"
            }
            
            records.append(record)
    
    print(f"   ✓ Created {len(records):,} verse chunks")
    return records


def upload_to_pinecone(records):
    """Upload records to Pinecone with integrated embedding."""
    print(f"\n🚀 Connecting to Pinecone...")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in environment")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get the index using the index name
    print(f"📊 Connecting to index via host: {INDEX_HOST}")
    index = pc.Index(host=INDEX_HOST)
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"   Index current count: {stats.get('total_vector_count', 0):,}")
    
    # Upload in batches (max 96 for integrated embedding)
    batch_size = 96
    total_batches = (len(records) + batch_size - 1) // batch_size
    
    print(f"\n📤 Uploading {len(records):,} records in {total_batches} batches...")
    print(f"   (Pinecone will auto-embed the 'text' field)")
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        try:
            # Upsert with raw text - Pinecone handles embedding automatically!
            # Use keyword arguments (not positional)
            index.upsert_records(records=batch, namespace="__default__")
            print(f"   ✓ Uploaded batch {batch_num}/{total_batches} ({len(batch)} records)")
            
            # Rate limiting: sleep to avoid hitting 250k tokens/min limit
            if i + batch_size < len(records):  # Don't sleep after last batch
                time.sleep(6)  # ~10 batches/min = ~960 records/min
            
        except Exception as e:
            print(f"   ❌ Error uploading batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()
            
            # If rate limited, wait longer and retry
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                print(f"   ⏸️  Rate limited, waiting 60 seconds...")
                time.sleep(60)
                try:
                    index.upsert_records(records=batch, namespace="__default__")
                    print(f"   ✓ Retry succeeded for batch {batch_num}")
                except Exception as retry_error:
                    print(f"   ❌ Retry failed: {retry_error}")
            
            continue
    
    # Final stats
    stats = index.describe_index_stats()
    print(f"\n✅ Upload complete!")
    print(f"   Total vectors in index: {stats.get('total_vector_count', 0):,}")


def main():
    print("=" * 70)
    print("📚 Bible to Pinecone Uploader")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # Parse Bible XML
    verses_by_chapter = parse_bible_xml(INPUT_FILE)
    
    # Create verse chunks
    records = create_verse_chunks(verses_by_chapter, CHUNK_SIZE)
    
    # Upload to Pinecone
    upload_to_pinecone(records)
    
    elapsed = datetime.now() - start_time
    print(f"\n⏱️  Total time: {elapsed.total_seconds():.1f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()

