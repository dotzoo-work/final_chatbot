import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def check_data_chunks():
    print("🔍 Checking data chunks...")

    # 1. Check original text file
    try:
        with open("plain_text_crawled_data (1) (1).txt", 'r', encoding='utf-8') as f:
            text = f.read()

        words = text.split()
        chunk_size = 300
        total_chunks = len(words) // chunk_size + (1 if len(words) % chunk_size > 0 else 0)

        print(f"📄 Original file:")
        print(f"   - Total words: {len(words)}")
        print(f"   - Expected chunks (300 words each): {total_chunks}")

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None, None

    # 2. Check Pinecone database
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

        # Check available indexes
        indexes = pc.list_indexes().names()
        print(f"\n📊 Available Pinecone indexes: {indexes}")

        if "rag-index" in indexes:
            index = pc.Index("rag-index")
            stats = index.describe_index_stats()

            print(f"\n🗄️ Pinecone database:")
            print(f"   - Total vectors stored: {stats['total_vector_count']}")
            print(f"   - Index dimension: {stats.get('dimension', 'Unknown')}")

            # Compare
            stored_chunks = stats['total_vector_count']
            if stored_chunks == total_chunks:
                print(f"\n✅ Perfect match! {stored_chunks} chunks stored")
            elif stored_chunks < total_chunks:
                print(f"\n⚠️ Missing chunks: {total_chunks - stored_chunks} not stored")
            else:
                print(f"\n🤔 Extra chunks: {stored_chunks - total_chunks} more than expected")

            return len(words), stored_chunks

        else:
            print(f"\n❌ 'rag-index' not found in Pinecone")
            return len(words), 0

    except Exception as e:
        print(f"❌ Error checking Pinecone: {e}")
        return len(words), 0

if __name__ == "__main__":
    check_data_chunks()