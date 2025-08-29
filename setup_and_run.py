import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Dr. Chatbot server...")
    
    # First check if data is loaded
    print("🔍 Checking data...")
    from check_chunks import check_data_chunks
    check_data_chunks()
    
    # Load data only if needed
    print("\n🔄 Checking if data needs to be loaded...")
    import os
    from pinecone import Pinecone
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        if "rag-index" in pc.list_indexes().names():
            index = pc.Index("rag-index")
            stats = index.describe_index_stats()
            vector_count = stats['total_vector_count']
            
            if vector_count > 0:
                print(f"✅ Data already loaded: {vector_count} vectors found - SKIPPING LOAD")
            else:
                print("🔄 Index exists but empty, loading data...")
                from final_load import load_data_properly
                load_data_properly()
        else:
            print("🔄 Index not found, creating and loading data...")
            from final_load import load_data_properly
            load_data_properly()
    except Exception as e:
        print(f"⚠️ Error checking data: {e}")
        print("🔄 Loading data anyway...")
        from final_load import load_data_properly
        load_data_properly()
    
    # Start server
    print("\n🌐 Starting web server...")
    from simple_app import app
    uvicorn.run(app, host="127.0.0.1", port=8000)