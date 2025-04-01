import requests

# 카카오맵 API: 장소명 -> 좌표 변환
def get_coordinates(place_name, api_key):
    """
    장소명을 입력받아 해당 장소의 좌표를 반환합니다.
    
    Args:
        place_name (str): 장소명
        api_key (str): 카카오 REST API 키
    
    Returns:
        tuple: (경도, 위도)
    """
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {"query": place_name}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        documents = response.json().get("documents", [])
        if documents:
            x = documents[0]["x"]  # 경도
            y = documents[0]["y"]  # 위도
            print(f"Coordinates for '{place_name}': x={x}, y={y}")
            return float(x), float(y)
        else:
            print(f"No coordinates found for '{place_name}'.")
    else:
        print(f"Failed to get coordinates for '{place_name}'. Status Code: {response.status_code}")
    return None, None

# 첫 번째 도보 경로 가져오기
def get_walking_distance(start_coords, end_coords, api_key):
    """
    두 좌표 간 첫 번째 도보 거리와 소요 시간을 반환합니다.
    """
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    headers = {"Authorization": f"KakaoAK {api_key}"}
    params = {
        "origin": f"{start_coords[0]},{start_coords[1]}",
        "destination": f"{end_coords[0]},{end_coords[1]}",
        "vehicleType": 1  # 도보 경로
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        routes = response.json().get("routes", [])
        if routes:
            # 첫 번째 경로 선택
            first_route = routes[0]
            distance = first_route["summary"]["distance"]  # 거리 (미터)
            duration = first_route["summary"]["duration"]  # 소요 시간 (초)
            print(f"Walking Distance: {distance} meters, Duration: {duration // 60} minutes")
            return {"distance": distance, "duration": duration}
        else:
            print("No walking route found.")
    else:
        print(f"Failed to calculate walking distance. Status Code: {response.status_code}")
    return {"distance": None, "duration": None}


def main():
    """
    사용자 입력을 받아 두 주소 간 첫 번째 도보 거리와 소요 시간을 계산합니다.
    """
    api_key = "f03538defb9fffd1f4da8d9e5b0353ea"  # 발급받은 API 키 입력
    start_place = input("출발지를 입력하세요: ")
    end_place = input("도착지를 입력하세요: ")
    
    # 좌표 검색
    start_coords = get_coordinates(start_place, api_key)
    end_coords = get_coordinates(end_place, api_key)
    
    if not start_coords or not end_coords:
        print("출발지 또는 도착지의 좌표를 찾을 수 없습니다.")
        return
    
    print(f"Start Coordinates: {start_coords}")
    print(f"End Coordinates: {end_coords}")

    # 첫 번째 도보 거리 계산
    distance_info = get_walking_distance(start_coords, end_coords, api_key)
    if distance_info["distance"] is not None:
        print(f"두 지점 간 도보 거리: {distance_info['distance']} 미터")
        print(f"예상 소요 시간: {distance_info['duration'] // 60} 분 {distance_info['duration'] % 60} 초")
    else:
        print("도보 경로를 계산할 수 없습니다.")

if __name__ == "__main__":
    main()
