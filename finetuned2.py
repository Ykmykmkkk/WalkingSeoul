import pandas as pd
import requests
import re

def get_coordinates_with_logging_and_corrections(place_name, api_key, corrections_dict):
    """
    KakaoMap API를 사용하여 장소의 좌표를 검색하는 함수 (단계별 재시도, 로깅, 보정 기능 포함).

    Args:
        place_name (str): 검색할 장소 이름.
        api_key (str): Kakao REST API 키.
        corrections_dict (dict): 검색 실패 시 보정이 필요한 장소 이름과 보정값이 담긴 딕셔너리.

    Returns:
        tuple: (경도, 위도) 형식의 좌표. 실패 시 (None, None)을 반환.
    """
    def preprocess_place_name(name, attempt):
        if attempt == 1:
            # 괄호가 있는 경우 괄호 안의 내용을 우선적으로 사용
            if "(" in name and ")" in name:
                name = re.search(r"\((.*?)\)", name).group(1).strip()
                return name
            return name.strip()  # 그렇지 않으면 원래 이름을 그대로 반환
        elif attempt == 2:
            # 괄호를 제거하고 '역'으로 끝나는 키워드에 초점을 맞춤
            if "(" in name and ")" in name:
                parenthesized = re.search(r"\((.*?)\)", name).group(1)
                if "역" in parenthesized:
                    # '역'까지의 내용을 추출
                    name = re.search(r".*?역", parenthesized).group(0).strip()
                    return name
            # 괄호와 그 내용을 제거
            name = re.sub(r"\(.*?\)", "", name)
            # 불필요한 키워드 제거
            keywords_to_remove = ['출입구', '입구', '끝', '?', '버스정류장']
            for keyword in keywords_to_remove:
                name = name.replace(keyword, "")
            name = re.sub(r"\s+", " ", name).strip()  # 공백 정리
            return name
        elif attempt == 3:
            # 과감한 단순화: 이름의 첫 번째 부분만 사용
            name = re.split(r"[ ,]", name)[0].strip()
            return name

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}

    # 바로 수정 가능한 경우 체크
    if place_name in corrections_dict:
        corrected_name = corrections_dict[place_name]
        params = {"query": corrected_name}
        response = requests.get(url, headers=headers, params=params)

        attempts_log = [f"직접 보정: '{corrected_name}'로 검색 중..."]
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            if documents:
                attempts_log.append(f"성공: 좌표를 찾음 ({documents[0]['x']}, {documents[0]['y']}).")
                print("\n".join(attempts_log))
                return float(documents[0]['x']), float(documents[0]['y'])
            else:
                attempts_log.append("보정된 이름으로 결과가 없음.")
        elif response.status_code == 400:
            attempts_log.append("400 잘못된 요청.")
        else:
            attempts_log.append(f"HTTP 오류 {response.status_code}.")
        print("\n".join(attempts_log))
        return None, None

    # 보정 없이 일반적인 처리
    attempts_log = []
    for attempt in range(1, 4):  # 최대 3번의 시도, 각각 다른 전처리 방법 사용
        preprocessed_name = preprocess_place_name(place_name, attempt)
        params = {"query": preprocessed_name}
        response = requests.get(url, headers=headers, params=params)
        
        log_entry = f"{attempt}번째 시도: '{preprocessed_name}'로 검색 중..."
        
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            if documents:
                log_entry += f" 성공: 좌표를 찾음 ({documents[0]['x']}, {documents[0]['y']})."
                attempts_log.append(log_entry)
                print("\n".join(attempts_log))
                return float(documents[0]["x"]), float(documents[0]["y"])
            else:
                log_entry += " 결과가 없음."
        elif response.status_code == 400:
            log_entry += " 400 잘못된 요청."
        else:
            log_entry += f" HTTP 오류 {response.status_code}."

        attempts_log.append(log_entry)

    # 모든 시도가 실패한 경우
    attempts_log.append(f"'{place_name}'에 대한 좌표 검색 실패.")
    print("\n".join(attempts_log))  # 모든 로그 출력
    return None, None

# CSV 파일 로드
file_path = "course_nodes_output.csv"  # 파일 경로를 변경하세요
data = pd.read_csv(file_path)

# API 키
api_key = "f03538defb9fffd1f4da8d9e5b0353ea"

# "버스정류장" 예시를 포함한 보정 딕셔너리
corrections_dict = {
    '버스정류장 15-309': '금옥중학교',
    '버스정류장 정릉초교(08-350)': '정릉초등학교',
    '자하문고개?윤동주시인의언덕': '윤동주문학관',
    '전사자영비': '전사자명비',
    '오봉전망대': '우이령나들길',
    '서경대뒷길': '서경대',
    '근현대디자인박물관': '와우공원',
    '백석중학교후문': '백석중학교',
    '구로지양산숲길': '구로지양산나들길',
    '흔들바위전망대': '시흥계곡',
    '세곡2보금자리': '서울세곡2 공공주택지구',
    '문희공유창모역입구': '문희공 유창묘역',
    '팔도소나무 단지': '야생화공원',
    '조경인의 숲': '홍릉수목원',
    '수림대장미원': '수림대장미정원',
    '신망애정자': '오동공원',
    '범바위기점': '단군성전',
    '북촌한옥마을길': '북촌한옥마을',
    '안산초화원 포장 산책로': '서대문독립공원',
    '홍대걷고싶은길': '홍대걷고싶은거리',
    '등촌중학교후문길': '등촌중학교',
    '달거리약수터': '용왕정',
    '보호초식물화원': '달마을공원',
    '청룡상 산림욕장': '청룡산',
    '효시정': '효사정',
    '서달산산책로': '서달산',
    '둔촌이집선생 둔굴': '길동자연생태공원',
}

# 좌표 검색 및 실패 항목 로깅
failed_searches = []

def process_coordinates(name):
    coords = get_coordinates_with_logging_and_corrections(name, api_key, corrections_dict)
    if coords == (None, None):
        failed_searches.append(name)
    return coords

data['start_coordinates'] = data['start_point_name'].apply(process_coordinates)
data['end_coordinates'] = data['last_point'].apply(process_coordinates)

# 처리된 데이터 저장
output_path = "processed_course_data_with_coordinates_corrections.csv"
data.to_csv(output_path, index=False)
print(f"처리된 데이터가 {output_path}에 저장되었습니다.")

# 실패한 검색 로그 저장
failed_output_path = "failed_searches_corrections.csv"
pd.DataFrame({"failed_search": failed_searches}).to_csv(failed_output_path, index=False)
print(f"검색 실패 항목이 {failed_output_path}에 저장되었습니다.")
