import os
import sys
from datetime import datetime

# =====================================================================
# 0. 불필요한 알림 상자 및 로그 차단 & API 키 검증
# =====================================================================
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
    print("\n❌ [오류] GOOGLE_API_KEY가 설정되지 않았습니다.")
    print("👉 '.env' 파일을 열고 올바른 Gemini API 키를 입력해 주세요.\n")
    sys.exit(1)

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# =====================================================================
# 1. Gemini 3.1 Flash Lite 모델 설정
# =====================================================================
llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    temperature=0.3
)

# =====================================================================
# 2. 실무 부서용 탐색 도구 (Tools)
# =====================================================================
DOCS_DIR = "./docs"

@tool("문서 폴더 목록 조회 도구")
def list_doc_files() -> str:
    """'docs' 폴더 내 마크다운(.md) 참고 파일 목록을 확인합니다."""
    if os.path.exists(DOCS_DIR):
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.md')]
        if files:
            return f"docs 폴더 목록: {', '.join(files)}"
        return "docs 폴더에 .md 파일이 없습니다."
    return "docs 폴더가 존재하지 않습니다."

@tool("마크다운 문서 읽기 도구")
def read_doc_file(file_name: str) -> str:
    """docs 폴더 내 특정 마크다운 파일의 상세 내용을 읽습니다."""
    file_path = os.path.join(DOCS_DIR, file_name)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return f"⚠️ 오류: '{file_name}' 파일이 존재하지 않습니다."

@tool("웹사이트 템플릿(index.html) 읽기 도구")
def read_html_template() -> str:
    """기존 'index.html' 웹사이트 템플릿 소스코드를 읽어옵니다."""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ index.html 파일이 존재하지 않습니다."


# =====================================================================
# 3. 부서별 페르소나 및 PM 매니저 정의
# =====================================================================
tools_list = [list_doc_files, read_doc_file]

chief_pm = Agent(
    role='총괄 PM (Chief Project Manager)',
    goal='CEO의 업무 지시를 분석하여, 해당 지시를 수행하는 데 꼭 필요한 하위 부서(에이전트)에게만 업무를 할당하고 그 결과를 종합하여 보고한다.',
    backstory='스타트업 및 기술 기업의 오퍼레이션 총괄 책임자로, 불필요한 공수를 줄이고 필요한 부서를 적재적소에 배치하는 최고의 리더.',
    llm=llm,
    verbose=False
)

hardware_team = Agent(
    role='하드웨어 & 메카트로닉스 팀',
    goal='기계 구조, 센서 배치, 케이싱 설계 등 물리적 하드웨어 관련 업무를 수행한다.',
    backstory='기계 기구 설계, 센서 융합 및 물리 환경 대응 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=False
)

embedded_control_team = Agent(
    role='임베디드 & 로봇 제어 팀',
    goal='MCU 펌웨어, ROS2 로봇 제어, 통신 아키텍처 및 모터 제어 관련 업무를 수행한다.',
    backstory='MCU, ROS2, 산업용 통신 및 모터 제어 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=False
)

ai_vision_team = Agent(
    role='AI & 데이터 파이프라인 팀',
    goal='컴퓨터 비전, AI 모델 학습, 데이터 전처리 및 ONNX 경량화 추론 최적화 업무를 수행한다.',
    backstory='컴퓨터 비전, AI 모델 학습, ONNX/TensorRT 경량화 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=False
)

biz_pm_team = Agent(
    role='비즈니스 & 프로덕트 기획 팀',
    goal='제품 PRD 작성, 사업성 평가, ROI 산출, 사용자 페인포인트 분석 업무를 수행한다.',
    backstory='IT/AI 제품 기획 및 사업성/ROI 분석 전문가.',
    tools=tools_list,
    llm=llm,
    verbose=False
)

mlops_web_team = Agent(
    role='웹 개발 & MLOps 팀',
    goal='웹 프론트엔드/백엔드 소스코드 작성, 로그인/인증 기능, 시스템 통합 및 모니터링 화면 구축 업무를 수행한다.',
    backstory='풀스택 웹 개발, FastAPI 백엔드, DevOps/MLOps 전문가.',
    tools=[list_doc_files, read_doc_file, read_html_template],
    llm=llm,
    verbose=False
)


# =====================================================================
# 4. CEO(학생)의 업무 지시 입력 (유니코드 서러게이트 문자 제거 보완)
# =====================================================================
print("\n" + "="*60)
raw_instruction = input("💬 CEO님, 오늘 회사 부서들에게 내릴 업무 지시를 입력하세요:\n> ")
print("="*60 + "\n")

# [보완] 인코딩 예외 방지를 위해 안전한 UTF-8 문자열로 재정제
ceo_instruction = raw_instruction.encode('utf-8', 'ignore').decode('utf-8')


# =====================================================================
# 5. 계층형 프로젝트 태스크(Task) 정의 (동적 변수 {ceo_instruction} 매핑)
# =====================================================================
project_task = Task(
    description="""
    CEO의 지시사항: [{ceo_instruction}]
    
    1. 총괄 PM은 CEO의 지시사항을 분석하여, 이 지시를 완수하는 데 '어떤 하위 부서가 필요한지' 판단하라.
    2. 지시 내용과 직접 관계없는 부서는 호출하지 말고, '꼭 필요한 부서 에이전트'에게만 해당 업무를 할당(Assign)하여 결과를 제출받아라.
    3. 필요한 경우 각 부서가 'docs' 폴더 내 참고 문서나 'index.html' 웹 템플릿을 확인하도록 지시하라.
    4. 각 부서가 제출한 결과를 바탕으로 CEO에게 제출할 종합 [최종 업무 완료 보고서] 및 결과물을 작성하라.
    """,
    expected_output="PM의 부서 할당 내역 및 각 필요한 부서가 수행한 결과가 통합된 CEO 보고용 [최종 업무 완료 보고서]",
    agent=chief_pm
)


# =====================================================================
# 6. 계층형(Hierarchical) 크루 실행
# =====================================================================
company_crew = Crew(
    agents=[hardware_team, embedded_control_team, ai_vision_team, biz_pm_team, mlops_web_team],
    tasks=[project_task],
    process=Process.hierarchical,
    manager_agent=chief_pm,
    verbose=False
)

print("⏳ PM이 지시사항을 분석하고 필요한 부서를 호출하여 업무를 수행 중입니다...\n")

# [보완] kickoff 호출 시 inputs 딕셔너리로 안전하게 넘겨줍니다.
result = company_crew.kickoff(
    inputs={'ceo_instruction': ceo_instruction}
)

# =====================================================================
# 7. 최종 보고서 출력 및 저장
# =====================================================================
print("\n" + "="*60)
print("📋 [CEO 제출용 최종 업무 완료 보고서]")
print("="*60 + "\n")
print(result)

REPORTS_DIR = "./reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filepath = os.path.join(REPORTS_DIR, f"report_{timestamp}.md")

with open(report_filepath, "w", encoding="utf-8") as f:
    f.write(str(result))

print(f"\n📁 보고서 파일 저장 완료: '{report_filepath}'\n")