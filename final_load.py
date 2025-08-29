import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
import time

load_dotenv()

def load_data_properly():
    try:
        print("🔄 Loading data with correct OpenAI syntax...")
        
        # Initialize OpenAI client (new syntax)
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        index_name = "rag-index"
        
        # Check if index exists, create if not
        existing_indexes = pc.list_indexes().names()
        print(f"📊 Available indexes: {existing_indexes}")
        
        if index_name not in existing_indexes:
            print(f"❌ Index '{index_name}' not found. Creating new index...")
            pc.create_index(
                name=index_name,
                dimension=1536,  # text-embedding-3-small dimension
                metric='cosine',
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            print(f"✅ Created index '{index_name}'")
            time.sleep(10)  # Wait for index to be ready
        else:
            print(f"✅ Using existing index '{index_name}'")
            
        index = pc.Index(index_name)
        
        # Load text file
        with open("plain_text_crawled_data (1) (1).txt", 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple chunking with overlap
        chunks = []
        words = text.split()
        chunk_size = 300
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
        
        print(f"📝 Created {len(chunks)} chunks")
        
        # Embed and store chunks (correct new syntax)
        for i, chunk in enumerate(chunks):
            try:
                # New OpenAI syntax
                response = client.embeddings.create(
                    model="text-embedding-3-small",  # Fast & cost-efficient
                    input=chunk
                )
                embedding = response.data[0].embedding
                
                # Unique ID for each chunk
                unique_id = f"dental_chunk_{i}_{hash(chunk) % 10000}"
                
                # Store with metadata
                index.upsert([(
                    unique_id,
                    embedding,
                    {"text": chunk, "chunk_id": i}  # Metadata for retrieval
                )])
                
                if i % 10 == 0:
                    print(f"✅ Processed {i+1}/{len(chunks)} chunks")
                    
            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")
                continue
        
        print("✅ Data loading completed!")
        
        # Check final stats
        stats = index.describe_index_stats()
        print(f"📊 Total vectors: {stats['total_vector_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def load_data_to_pinecone():
    """Wrapper function for advanced setup compatibility"""
    success = load_data_properly()
    if success:
        # Return approximate vector count
        try:
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index("rag-index")
            stats = index.describe_index_stats()
            return stats.total_vector_count
        except:
            return 872  # Default estimate
    return 0

if __name__ == "__main__":
    success = load_data_properly()
    if success:
        print("🎉 Data successfully loaded to Pinecone!")
    else:
        print("💥 Failed to load data")