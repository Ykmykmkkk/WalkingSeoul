import pandas as pd
import re

# 파일 경로: 확장자에 맞는 파일 경로 지정
file_path = r'C:\Users\user\Documents\algorithm project\walkdata.csv'  # CSV 파일인 경우

# CSV 파일 읽기
data = pd.read_csv(file_path, encoding='cp949')  # 한글 데이터 포함 시 CP949 인코딩 사용

class CourseNode:
    def __init__(self, course_name, distance, course_points, start_point_name, last_point, course_level):
        self.course_name = course_name
        self.distance = distance
        self.course_points = course_points
        self.start_point_name = start_point_name
        self.last_point = last_point
        self.course_level = course_level

    def __repr__(self):
        return (f"CourseNode(course_name='{self.course_name}', "
                f"distance='{self.distance}', "
                f"course_points='{self.course_points}', "
                f"start_point_name='{self.start_point_name}', "
                f"last_point='{self.last_point}', "
                f"course_level={self.course_level})")

def clean_location_name(location_name):
    """
    장소 이름에서 '인근'이라는 단어를 제거합니다.

    Args:
        location_name (str): 정리할 장소 이름.

    Returns:
        str: 정리된 장소 이름.
    """
    if "인근" in location_name:
        location_name = location_name.replace("인근", "").strip()
    return location_name

def extract_course_nodes(data):
    """
    제공된 DataFrame에서 코스 정보를 추출하여 CourseNode 객체 목록으로 반환합니다.

    Args:
        data (DataFrame): 코스 정보가 담긴 입력 데이터프레임.

    Returns:
        list: 추출된 정보를 포함하는 CourseNode 객체 목록.
    """
    nodes = []
    for _, row in data.iterrows():
        course_name = row['코스명']
        distance = row['거리']
        course_points = row['세부코스']  # 전체 코스 포인트 문자열 포함
        
        # 결측값 처리 및 문자열 변환
        start_point_name = row['포인트명칭']
        if pd.isna(course_points):
            course_points = start_point_name  # 세부코스가 없는 경우 기본값 설정
        else:
            course_points = str(course_points).strip()  # 문자열로 변환 및 공백 제거

        # '세부코스'에서 마지막 포인트 추출
        if course_points:
            # 다양한 구분자를 통일: '-', '~', '→' 등을 모두 '~'로 치환
            cleaned_points = re.sub(r'[\-~→]', '~', course_points)
            # '~'로 분리 후 마지막 요소 추출
            last_point = cleaned_points.split('~')[-1].strip()
        else:
            last_point = start_point_name  # 세부코스 정보가 없는 경우 시작 포인트 사용

        # '인근' 제거
        start_point_name = clean_location_name(start_point_name)
        last_point = clean_location_name(last_point)

        course_level = row['코스레벨']
        nodes.append(CourseNode(course_name, distance, course_points, start_point_name, last_point, course_level))
    return nodes

def save_nodes_to_csv(nodes, output_file_path):
    """
    CourseNode 객체 목록을 CSV 파일로 저장합니다.

    Args:
        nodes (list): CourseNode 객체 목록.
        output_file_path (str): 출력 CSV 파일 경로.
    """
    # CourseNode 객체를 딕셔너리 목록으로 변환
    data = [{
        "course_name": node.course_name,
        "distance": node.distance,
        "start_point_name": node.start_point_name,
        "last_point": node.last_point,
        "course_level": node.course_level,
        "course_points": node.course_points  # 이 열을 마지막으로 이동
    } for node in nodes]
    
    # 딕셔너리 목록을 데이터프레임으로 생성
    df = pd.DataFrame(data)
    
    # 데이터프레임을 CSV 파일로 저장
    df.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # 한글이 포함된 경우 UTF-8-SIG로 인코딩
    print(f"코스 노드가 {output_file_path}에 저장되었습니다.")

# 코스 노드 추출
nodes = extract_course_nodes(data)

# CSV 파일로 저장
output_file_path = r'C:\Users\user\Documents\algorithm project\course_nodes_output.csv'
save_nodes_to_csv(nodes, output_file_path)
