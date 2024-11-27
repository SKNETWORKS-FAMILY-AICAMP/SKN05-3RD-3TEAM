# app.py
import os
from dotenv import load_dotenv  # 추가
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_db import get_vector_db, get_template
from langchain.callbacks import get_openai_callback

# from chain_config import load_chain

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 로드
openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수로부터 API 키를 로드

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 벡터 DB 로드
def load_chain():
    retriever = get_vector_db()

    # 프롬프트 템플릿 설정
    template = get_template()
    prompt = ChatPromptTemplate.from_template(template)

    # ChatOpenAI 모델 불러오기
    fine_tuned_model_id = "ft:gpt-4o-mini-2024-07-18:personal::AXmDF5MY"
    model = ChatOpenAI(
        model_name=fine_tuned_model_id,
        temperature=0.7,
        max_tokens=1000,
        openai_api_key=openai_api_key  # API 키 전달
    )

    # 문서 포맷팅
    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    # 체인 구성
    chain = {'context': retriever | format_docs, 'question': RunnablePassthrough()} | prompt | model | StrOutputParser()

    return chain


# Streamlit UI 시작
st.title("한국공항공사 인사규정 챗봇")

# 익스펜더 넣기
with st.expander("**한국공항공사 인사 규정에 대한 챗봇입니다. 자세한 안내사항은 아래를 참고해주세요.**",expanded=True,icon="🚨"):
    st.markdown('''<small>[안내사항]<br>해당 챗봇은 한국공항공사의 인사규정을 바탕으로 제작된 챗봇입니다.<br>
        인사규정에 없는 내용은 법률 규정 텍스트 분석 데이터(ai hub) 를 토대로 답변하며,<br>
        인사규정은 11월 25일(월) 기준 공시되어있는 규정을 바탕으로 답변합니다.<br>
        자세한 인사규정 및 내용은 한국공항공사의 홈페이지를 이용해 주세요.</small>
    ''', unsafe_allow_html=True)

# 체인 로드
with st.spinner("챗봇을 로드 중입니다..."):
    chain = load_chain()

with st.container():
    prompt = st.chat_input("인사 규정에 대해 질문해주세요.")


# chat 히스토리 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt:
    # 사용자 메시지 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("답변을 생성 중입니다..."):
        with get_openai_callback() as cost:
            response = chain.stream(prompt)

    with st.chat_message("assistant"):
        st.write(response)

    response_history = ""
    for token in chain.stream(prompt):
        response_history += token

    st.session_state.messages.append({"role": "user", "content": response_history})

# # 사이드바 기업 정보
with st.sidebar:
    st.header(":blue-background[기업명]")
    st.subheader("한국공항공사")
    st.markdown('''
                **:blue-background[기업주소]** <br> 서울특별시 강서구 하늘길 78 <br> KAC한국공항공사 <br>
                **:blue-background[고객센터]** <br> 1661-2626
                ''', unsafe_allow_html=True)
    # 홈페이지 링크
    st.link_button("홈페이지 바로가기", "https://www.airport.co.kr/www/index.do")
    st.divider()
    # 규정 링크
    st.markdown('''
                **:blue-background[관련 인사 규정]**
                ''')
    st.markdown('''
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=3644&PWD=&SEARCH_FLD=&SEARCH=%25EC%259D%25B8%25EC%2582%25AC%25EA%25B7%259C%25EC%25A0%2595" style="color: skyblue;">인사규정</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=3645&PWD=&SEARCH_FLD=&SEARCH=%25EC%259D%25B8%25EC%2582%25AC%25EA%25B7%259C%25EC%25A0%2595" style="color: skyblue;">인사규정 시행세칙</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=3643&PWD=&SEARCH_FLD=&SEARCH=%25EC%25B7%25A8%25EC%2597%2585" style="color: skyblue;">취업규칙</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=3683&PWD=&SEARCH_FLD=&SEARCH=%25EC%259C%25A0%25EC%2597%25B0%25EA%25B7%25BC%25EB%25AC%25B4%25EC%25A0%259C" style="color: skyblue;">유연근무제 운영예규</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=41153&PWD=&SEARCH_FLD=&SEARCH=%25EA%25B8%25B0%25EA%25B0%2584%25EC%25A0%259C%25EA%25B7%25BC%25EB%25A1%259C%25EC%259E%2590" style="color: skyblue;">기간제근로자 관리예규</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=41154&PWD=&SEARCH_FLD=&SEARCH=%25EA%25B3%25B5%25EB%25AC%25B4%25EC%25A7%2581" style="color: skyblue;">공무직근로자 관리예규</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=3552062&PWD=&SEARCH_FLD=&SEARCH=%25EC%259E%2584%25EC%259B%2590" style="color: skyblue;">임원 복무 예규</a><br>
                <a href="https://www.airport.co.kr/www/cms/frBoardCon/boardView.do?pageNo=1&pagePerCnt=15&MENU_ID=2390&CONTENTS_NO=&SITE_NO=2&BOARD_SEQ=99&BBS_SEQ=3552743&PWD=&SEARCH_FLD=&SEARCH=%25EB%25AC%25B8%25EC%25B1%2585" style="color: skyblue;">안전사고에 관한 임원 문책규정</a>
                ''', unsafe_allow_html=True)
