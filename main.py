import csv
import time
import requests

from kakao_distance import get_coordinates
from kakao_distance import get_walking_distance
import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
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
            if None in [node1.end_coordinates, node2.start_coordinates]:
                continue

            # 출발 코스의 마지막 지점과 도착 코스의 시작 지점 사이 거리 계산
            end_lat1, end_lon1 = node1.end_coordinates
            start_lat2, start_lon2 = node2.start_coordinates

            distance = haversine_distance(end_lat1, end_lon1, start_lat2, start_lon2)

            # 거리 기준에 맞는 간선 추가 (distance가 None이 아니고 3km 이하일 경우)
            if distance is not None and distance <= 3:  # 3km 이하일 때 간선을 그린다.
                if node1.course_name not in edges:
                    edges[node1.course_name] = []
                if node2.course_name not in edges:
                    edges[node2.course_name] = []
                edges[node1.course_name].append((node2.course_name, distance))
                edges[node2.course_name].append((node1.course_name, distance))
    return edges


# 다익스트라 알고리즘 구현 (경로 추적 추가)
def dijkstra(edges, start_node):
    # 최단 거리 테이블 초기화 (모든 거리 초기값은 무한대, 출발 노드는 0)
    distances = {node: float('inf') for node in edges}
    distances[start_node] = 0
    # 우선순위 큐 (최소 힙) 초기화
    priority_queue = [(0, start_node)]  # (거리, 노드)

    # 이전 노드 추적용 딕셔너리
    previous_nodes = {node: None for node in edges}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # 현재 노드에 대해 이미 처리된 경우 건너뛰기
        if current_distance > distances[current_node]:
            continue

        # 인접한 노드들 탐색
        for neighbor, weight in edges.get(current_node, []):
            distance = current_distance + weight

            # 더 짧은 경로가 발견되면 갱신
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node  # 이전 노드 기록
                heapq.heappush(priority_queue, (distance, neighbor))

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

print(f"You selected {course1.course_name} and {course2.course_name}.")

# 다익스트라 알고리즘 실행
start_time = time.perf_counter()
distances_from_course1, previous_nodes_from_course1 = dijkstra(edges, course1.course_name)
distances_from_course2, previous_nodes_from_course2 = dijkstra(edges, course2.course_name)
end_time = time.perf_counter()

# 경로 역추적
path_from_course1_to_course2 = get_path(previous_nodes_from_course1, course1.course_name, course2.course_name)

# 경로에 포함된 코스들의 세부 정보 출력
total_dis = 0

print("Courses along the path:")
for i, course_name in enumerate(path_from_course1_to_course2):

    # 코스 이름에 해당하는 CourseNode 객체를 찾아서 그 세부 코스를 출력
    course = next(node for node in course_nodes if node.course_name == course_name)
    print(f"{course.course_name}: 코스의 거리 {course.distance}")
    print(f"{course.course_name}: {course.course_points}")
    print()

    #현 코스의 시작점
    start_place1 = course.start_coordinates

    # 코스의 거리값에서 km 삭제
    distance = re.findall(r"[\d.]+", course.distance)

    # 두번째 코스를 순회하게 될 때 부터 두 코스간 거리 측정
    if i>0:
        prev_course = next(node for node in course_nodes if node.course_name == path_from_course1_to_course2[i - 1])
        end_place1 = prev_course.end_coordinates

        # 두 코스 간 도보 거리 계산 (이전 코스 끝 지점과 현재 코스 시작 지점 간의 거리)
        distance_info = get_walking_distance(end_place1, start_place1, api_key)
        if distance_info["distance"] is not None:
            walking_distance_km = distance_info["distance"] / 1000  # 도보 거리(km)
            total_dis += walking_distance_km
            print(f"{prev_course.course_name}와 {course_name} 사이 거리: {walking_distance_km:.2f} km")
            print()

    if distance:
        total_dis += float(distance[0])  # 리스트에서 첫 번째 값만 사용하고 float으로 변환

print(f"당신이 산책할 총 거리 : {total_dis} km")
execution_time = end_time - start_time
print(f"다익스트라 알고리즘 실행 시간: {execution_time:.4f} seconds")