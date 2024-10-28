import os
import getpass

os.environ['sk-PH7IzQpiAMGRPyKqz728dmXEXHa1fUMAQXinQTBr52T3BlbkFJ2dDGhagm4lpzGwlKnG3Wn2E0j12eEZS3LGBYfC5B0A'] = getpass.getpass('OpenAI API Key:sk-PH7IzQpiAMGRPyKqz728dmXEXHa1fUMAQXinQTBr52T3BlbkFJ2dDGhagm4lpzGwlKnG3Wn2E0j12eEZS3LGBYfC5B0A')

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

from langchain_chroma import Chroma

db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = user_input
docs = db.similarity_search(query)
print(docs[0].page_content)

embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
print(docs[0].page_content)

async def some_function():
    # Most már használhatod az await kulcsszót
    docs = await db.asimilarity_search(query)
    return docs
