"""
Embed Harry Potter Books into Pinecone
Chunks consecutive lines for better context preservation
"""

import csv
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = "solace-t42ww4d.svc.aped-4627-b74a.pinecone.io"
CSV_PATH = "./books/harry_potter_books.csv"  # Download from: https://raw.githubusercontent.com/gastonstat/harry-potter-data/refs/heads/main/csv-data-file/harry_potter_books.csv

# Chunking settings
LINES_PER_CHUNK = 10  # Group 10 consecutive lines into one chunk (1-2 paragraphs)
BATCH_SIZE = 96  # Pinecone's limit for integrated embedding

# Pinecone uses llama-text-embed-v2 (1024 dimensions) with integrated embedding
# It requires a "text" field in the metadata for auto-embedding


def load_csv_data(csv_path):
    """Load Harry Potter CSV data"""
    print(f"📖 Loading Harry Potter data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found!")
        print(f"   Download it from: https://raw.githubusercontent.com/gastonstat/harry-potter-data/refs/heads/main/csv-data-file/harry_potter_books.csv")
        return None
    
    # Read CSV and group by (book, chapter)
    book_chapters = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text'].strip()
            book = row['book'].strip()
            chapter = row['chapter'].strip()
            
            if text:  # Skip empty lines
                key = (book, chapter)
                book_chapters[key].append(text)
    
    print(f"✓ Loaded {len(book_chapters)} unique (book, chapter) combinations")
    
    return book_chapters


def create_hp_chunks(book_chapters, lines_per_chunk=10):
    """Create chunks from consecutive lines"""
    chunks = []
    chunk_id = 0
    
    print(f"📝 Creating chunks ({lines_per_chunk} lines per chunk)...")
    
    for (book, chapter), lines in sorted(book_chapters.items()):
        # Split lines into chunks of N consecutive lines
        for i in range(0, len(lines), lines_per_chunk):
            chunk_lines = lines[i:i + lines_per_chunk]
            
            # Skip if chunk is too short (less than 3 lines)
            if len(chunk_lines) < 3:
                continue
            
            # Join lines into a single text block
            chunk_text = " ".join(chunk_lines)
            
            # Skip if chunk is empty or too short
            if not chunk_text.strip() or len(chunk_text) < 50:
                continue
            
            # Create a readable reference
            # e.g., "Philosopher's Stone, Chapter 1" or "Deathly Hallows, Chapter 37"
            book_short = book.split(": ")[-1] if ": " in book else book
            chapter_num = chapter.split("-")[-1] if "-" in chapter else chapter
            reference = f"{book_short}, Chapter {chapter_num}"
            
            chunk = {
                "id": f"HP_{chapter_num}_{chunk_id}",
                "text": chunk_text,
                "book": book,
                "book_name": book_short,
                "chapter": chapter,
                "chapter_num": int(chapter_num) if chapter_num.isdigit() else 0,
                "reference": reference,
                "testament": "HP",  # Harry Potter "testament"
                "translation": "Original"  # Consistency with Bible data
            }
            
            chunks.append(chunk)
            chunk_id += 1
    
    print(f"✓ Created {len(chunks)} chunks")
    return chunks


def upload_to_pinecone(chunks, batch_size=96):
    """Upload chunks to Pinecone with integrated embedding"""
    print(f"🔌 Connecting to Pinecone...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    print(f"✓ Connected to index: {PINECONE_INDEX_HOST}")
    print(f"📤 Uploading {len(chunks)} chunks in batches of {batch_size}...")
    
    total_uploaded = 0
    failed_batches = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        # Convert chunks to Pinecone records format
        records = []
        for chunk in batch:
            record = {
                "_id": chunk["id"],
                "text": chunk["text"],  # Required field for integrated embedding
                "book": chunk["book"],
                "book_name": chunk["book_name"],
                "chapter": chunk["chapter"],
                "chapter_num": chunk["chapter_num"],
                "reference": chunk["reference"],
                "testament": chunk["testament"],
                "translation": chunk["translation"]
            }
            records.append(record)
        
        # Upload batch with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                index.upsert_records(
                    namespace="__default__",
                    records=records
                )
                total_uploaded += len(batch)
                print(f"  ✓ Batch {batch_num}/{total_batches}: Uploaded {len(batch)} chunks ({total_uploaded}/{len(chunks)} total)")
                break
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "RESOURCE_EXHAUSTED" in error_str or "rate limit" in error_str.lower():
                    wait_time = 60 * (attempt + 1)
                    print(f"  ⚠️  Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Error uploading batch {batch_num}: {e}")
                    failed_batches.append(batch_num)
                    break
        else:
            # All retries failed
            print(f"  ❌ Failed to upload batch {batch_num} after {max_retries} retries")
            failed_batches.append(batch_num)
        
        # Rate limiting: sleep between batches to avoid hitting limits
        if i + batch_size < len(chunks):
            time.sleep(6)  # 6 seconds between batches
    
    print(f"\n✅ Upload complete!")
    print(f"   Total uploaded: {total_uploaded}/{len(chunks)} chunks")
    
    if failed_batches:
        print(f"   ⚠️  Failed batches: {failed_batches}")
    
    # Show index stats
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        print(f"   📊 Total vectors in index: {total_vectors}")
    except Exception as e:
        print(f"   ⚠️  Could not get index stats: {e}")


def main():
    print("=" * 60)
    print("Harry Potter → Pinecone Embedding Pipeline 🪄")
    print("=" * 60)
    print()
    
    # Step 1: Load CSV data
    book_chapters = load_csv_data(CSV_PATH)
    if not book_chapters:
        return
    
    print()
    
    # Step 2: Create chunks
    chunks = create_hp_chunks(book_chapters, LINES_PER_CHUNK)
    
    if not chunks:
        print("❌ No chunks created!")
        return
    
    print()
    print(f"Sample chunk:")
    print(f"  ID: {chunks[0]['id']}")
    print(f"  Reference: {chunks[0]['reference']}")
    print(f"  Text: {chunks[0]['text'][:200]}...")
    print()
    
    # Confirm before uploading
    response = input("Ready to upload to Pinecone? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print()
    
    # Step 3: Upload to Pinecone
    upload_to_pinecone(chunks, BATCH_SIZE)
    
    print()
    print("=" * 60)
    print("🎉 All done! Harry Potter wisdom is now searchable!")
    print("=" * 60)


if __name__ == "__main__":
    main()

