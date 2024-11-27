# vector_db.py
import os
from dotenv import load_dotenv  # 추가
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma



load_dotenv()

# OpenAI API 키 로드
openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수로부터 API 키를 로드

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 텍스트 파일 경로
TXT_FILE_PATH = "/data/preprocessed_texts.txt"

# 벡터 DB 저장 경로
VECTOR_DB_PATH = "/vector_db"

print("TXT_FILE_PATH:", TXT_FILE_PATH)
print("VECTOR_DB_PATH:", VECTOR_DB_PATH)
print("Current Working Directory:", os.getcwd())

# 벡터 DB 생성 및 로드 함수
def get_vector_db():
    # 벡터 DB가 존재하는지 확인
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    if not os.path.exists(VECTOR_DB_PATH):        
        # 텍스트 파일 로드
        with open(TXT_FILE_PATH, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 텍스트 청크 생성
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )
        docs = text_splitter.create_documents([raw_text])

        
        # 벡터 저장소 생성 및 저장
        vectorstores = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,  # API 키 전달
            persist_directory=VECTOR_DB_PATH
        )
        vectorstores.persist()
        print(f"Vector DB 생성 완료: {VECTOR_DB_PATH}")
    else:
        pass

    # 벡터 DB 로드
    vectorstores = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)
    return vectorstores.as_retriever()

# 프롬프트 템플릿 생성
def get_template():
    template = '''
    아래는 인사 및 법률 관련 문서 내용입니다:
    {context}

    위 문서를 기반으로 다음 질문에 답해주세요.
    질문: {question}
    '''
    return template

if __name__ == "__main__":
    get_vector_db()  # 벡터 DB 생성 및 로드

