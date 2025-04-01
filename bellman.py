import csv
import time
import requests

from kakao_distance import get_coordinates
from kakao_distance import get_walking_distance
import math
import matplotlib.pyplot as plt
import re

# 구조체
class CourseNode:
    def __init__(self, course_name, distance, start_point_name, last_point, course_level, course_points=None, start_coordinates=None, end_coordinates=None):
        self.course_name = course_name
        self.distance = distance
        self.start_point_name = start_point_name
        self.last_point = last_point
        self.course_level = course_level
        self.course_points = course_points # 코스 세부 코스
        self.start_coordinates = self.convert_to_tuple(start_coordinates)
        self.end_coordinates = self.convert_to_tuple(end_coordinates)

    def __repr__(self):
        return (f"CourseNode(course_name='{self.course_name}', "
                f"distance='{self.distance}', "
                f"start_point_name='{self.start_point_name}', "
                f"last_point='{self.last_point}', "
                f"course_level={self.course_level}', "
                f"course_points='{self.course_points}', "
                f"start_coordinates={self.start_coordinates}', "
                f"end_coordinates='{self.end_coordinates}')")

    def convert_to_tuple(self, coord_str):
        # 문자열 좌표를 튜플로 변환
        coord_str = coord_str.strip("()")  # 괄호 제거
        coord_list = coord_str.split(",")  # 쉼표로 분리
        return float(coord_list[0]), float(coord_list[1])  # 튜플로 변환

#카카오 api 키
api_key = "f03538defb9fffd1f4da8d9e5b0353ea"

# csv 파일 읽기
f = open("processed_course_data_with_coordinates_corrections.csv",'r',encoding='utf-8')
rdr = csv.reader(f)
next(rdr)

#csv 데이터 읽고 저장할 리스트
course_nodes = []

for line in rdr:
    course_name = line[0]
    distance = line[1]
    start_point_name = line[2]
    last_point = line[3]
    course_level = line[4]
    course_points = line[5]
    start_coordinates = line[6]
    end_coordinates = line[7]

    # CourseNode 객체 생성
    course_node = CourseNode(course_name, distance, start_point_name, last_point, course_level, course_points, start_coordinates, end_coordinates)

    course_nodes.append(course_node)

f.close()


# 지구의 곡률을 고려한 거리 계산 (단위: km)
def haversine_distance(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return None  # 좌표가 None이면 거리 계산을 하지 않음

    R = 6371  # 지구 반지름 (단위: km)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])  # 각도를 라디안으로 변환

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# 두 객체 간의 거리 비교 (간선)
def create_edges(course_nodes):
    edges = {}  # 딕셔너리 형태로 수정
    for i in range(len(course_nodes)):
        for j in range(i + 1, len(course_nodes)):
            node1 = course_nodes[i]
            node2 = course_nodes[j]

            # 좌표가 None인 경우 건너뛰기
            if None in [node1.start_coordinates, node2.end_coordinates]:
                continue
            
            # 두 코스의 시작점과 끝점 좌표
            start_lat1, start_lon1 = node1.end_coordinates[0], node1.end_coordinates[1]
            end_lat2, end_lon2 = node2.start_coordinates[0], node2.start_coordinates[1]

            # 두 좌표 간 haversine 거리 계산
            distance = haversine_distance(start_lat1, start_lon1, end_lat2, end_lon2)

            # 코스 레벨에 따라 음수 가중치 설정
            level_weight = -0.1e-17 * int(node1.course_level)  # 레벨에 따라 음수 가중치 적용
            distance += level_weight

            # 거리 기준에 맞는 간선 추가 (distance가 None이 아니고 3km 이하일 경우)
            if distance is not None and distance <= 3:  # 3km 이하일 때 간선을 그린다.
                if node1.course_name not in edges:
                    edges[node1.course_name] = []
                if node2.course_name not in edges:
                    edges[node2.course_name] = []
                edges[node1.course_name].append((node2.course_name, distance))
                edges[node2.course_name].append((node1.course_name, distance))
    return edges

###############
# 벨만포드 알고리즘 구현
def bellman_ford(edges, start_node):
    # 음수 가중치 허용 및 음수 사이클 검출.
    # 노드 초기화
    distances = {node: float('inf') for node in edges}
    distances[start_node] = 0
    previous_nodes = {node: None for node in edges}

    # 벨만포드 거리 갱신 (노드 개수 - 1만큼 반복)
    for _ in range(len(edges) - 1):
        for node, neighbors in edges.items(): # 각 노드에 대해
            for neighbor, weight in neighbors: # 각 이웃에 대해
                if distances[node] + weight < distances[neighbor]: # 거리 갱신이 발생하면
                    distances[neighbor] = distances[node] + weight # 거리 갱신
                    previous_nodes[neighbor] = node

    # 음수 사이클 확인
    for node, neighbors in edges.items(): # edges.items란 노드와 이웃들을 반환
        for neighbor, weight in neighbors:
            if distances[node] + weight < distances[neighbor]: # 음수 사이클이 있으면(거리 갱신이 발생하면)
                print(f"Negative-weight cycle detected: {node} -> {neighbor}")
                # 음수 사이클이 있는 간선을 제거
                edges[node].remove((neighbor, weight))
                if not edges[node]:
                    del edges[node]
                return bellman_ford(edges, start_node)  # 다시 계산

    return distances, previous_nodes

# 경로 역추적 함수
def get_path(previous_nodes, start_node, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.reverse()
    return path

# 간선 생성
edges = create_edges(course_nodes)






# 출력: 3km 이하의 거리로 연결된 코스들
print("Edges created (3km 이하):")
for edge in edges:
    print(edge)

# 코스 이름과 번호 출력
print("Course List with Numbers:")
for i, node in enumerate(course_nodes, start=1):
    print(f"{i}: {node.course_name}")

# 두 코스 번호 입력받기
while True:
    try:
        course1_num = int(input("Enter the first course number: "))
        course2_num = int(input("Enter the second course number: "))

        if 1 <= course1_num <= len(course_nodes) and 1 <= course2_num <= len(course_nodes):
            break  # 유효한 번호가 입력되면 반복 종료
        else:
            print(f"Please enter a number between 1 and {len(course_nodes)}.")
    except ValueError:
        print("Invalid input! Please enter a valid number.")

# 선택된 코스 출력
course1 = course_nodes[course1_num - 1]
course2 = course_nodes[course2_num - 1]

# 벨만-포드 실행
start_time = time.perf_counter()
distances, previous_nodes = bellman_ford(edges, course1.course_name)
end_time = time.perf_counter()

print(f"You selected {course1.course_name} and {course2.course_name}.")

# 경로 역추적
path = get_path(previous_nodes, course1.course_name, course2.course_name)
path_courses = [node for node in path] # 경로에 포함된 코스 이름 리스트

# 경로에 포함된 코스들의 세부 정보 출력
total_dis = 0
print("Courses along the path:")
for i, course_name in enumerate(path_courses):
    course = next(node for node in course_nodes if node.course_name == course_name)
    print(f"{course.course_name}: 코스의 거리 {course.distance}")
    print(f"{course.course_name}: {course.course_points}")
    print()

    # 코스의 거리값에서 km 삭제 후 누적
    distance = re.findall(r"[\d.]+", course.distance)

    if distance: # 거리값이 존재하면 누적
        total_dis += float(distance[0]) # 거리값을 더함

    # 두 지점 간 거리 계산 및 추가
    if i > 0:  # 첫 번째 코스는 이전 코스가 없으므로 제외
        prev_course_name = path_courses[i - 1]
        prev_course = next(node for node in course_nodes if node.course_name == prev_course_name)

        # 이전 코스의 끝점과 현재 코스의 시작점 좌표
        prev_end_coords = prev_course.end_coordinates
        current_start_coords = course.start_coordinates

        # 두 지점 간 거리 계산 
        if prev_end_coords and current_start_coords:
            inter_course_distance = get_walking_distance(prev_end_coords, current_start_coords, api_key)
        if inter_course_distance["distance"] is not None:
            total_dis = total_dis + inter_course_distance["distance"]/1000
            print(f"{prev_course.course_name}와 {course_name} 사이 거리: {inter_course_distance["distance"]/1000} km")
            print()

print(f"당신이 산책할 총 거리 : {total_dis:.3f} km")
execution_time = end_time - start_time
print(f"벨만-포드 알고리즘 실행 시간: {execution_time:.4f} seconds")