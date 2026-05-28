```xml
kubectl get pods -A # pods를 전부 반환

kubectl get pods -n <네임 스페이스> # 네임 스페이스의 pods를 반환

kubectl logs -f <파드 이름> -n <네임 스페이스> # 실시간 로그 반환

kubectl logs -f <파드 이름> -n <네임 스페이스> 
 | grep -i -A 3 -B 1 #{keyword} #특정 키워드를 포함한 로그 반환  # -A n 뒤의 n줄 # -B n 앞의 n줄 
 | grep -i -v ${keyword} #특정 키워드를 제거한 로그 반환

kubectl exec -it <redis-pod-name> -n focus-jj -- redis-cli ## 쿠버네티스 레디스 커맨드라인 조회
KEYS *, GET *  ## KEY 전체 조회, 값 전체 조
```

```bash

# -yd 네임스페이스의 pods 조회
kubectl get pods -n <네임스페이스>
# 특정 pod의 상세 정보를 yaml 형식으로 조회
kubectl get pod <파드명> -o yaml

#  네임스페이스의 deployment 정보 조회
kubectl get deployment -n  <네임스페이스> <배포명> -o yaml
# deployment yaml 파일 적용
kubectl apply -f deploy.yaml

# 모든 네임스페이스의 pods 조회
kubectl get pods -A
```

| **명령어 (Command)** | **구문 (Syntax)** | **설명 (Description)** |
| --- | --- | --- |
| **적용/생성/업데이트** | `kubectl apply -f [FILENAME]` | 파일에 정의된 리소스(Pod, Service 등)의 원하는 상태를 **적용**합니다. (생성 및 업데이트 포함, 권장) |
| **생성** | `kubectl create -f [FILENAME]` | 파일로부터 리소스를 **생성**합니다. |
| **배포** | `kubectl run [NAME] --image=[IMAGE]` | 단일 Pod 또는 Deployment를 **배포**합니다. |
| **조회 (목록)** | `kubectl get [TYPE]` | 특정 리소스 타입의 **목록**을 확인합니다. (예: `kubectl get pods`, `kubectl get svc`) |
| **조회 (상세)** | `kubectl describe [TYPE] [NAME]` | 특정 리소스의 **상세 정보**와 상태를 확인합니다. (이벤트, IP, 컨테이너 정보 등) |
| **삭제** | `kubectl delete [TYPE] [NAME]` | 특정 리소스를 **삭제**합니다. (예: `kubectl delete pod my-pod`) |
| **로그** | `kubectl logs [POD_NAME]` | Pod 내 컨테이너의 **로그**를 출력합니다. (`-f` 옵션으로 실시간 확인 가능) |
| **접속/실행** | `kubectl exec -it [POD_NAME] -- bash` | Pod 내 컨테이너에 **접속**하여 명령을 실행합니다. (`-it`는 대화형 터미널을 의미) |
| **설정 편집** | `kubectl edit [TYPE] [NAME]` | 실행 중인 리소스의 설정을 **편집**합니다. (YAML 에디터 열림) |
| **CPU/Memory 사용량** | `kubectl top [TYPE]` | 노드 또는 Pod의 **CPU/메모리 사용량**을 확인합니다. (Metrics Server 필요) |
| **클러스터 정보** | `kubectl cluster-info` | 클러스터 마스터 및 서비스 **정보**를 확인합니다. |

```sql
# 서버 측에서 유효성 검사 및 병합 미리보기 (권장)
kubectl apply -f <수정한 설정파일> --dry-run=server

# 서버 측에 수정한 파일 등록
kubectl apply -f <수정한 설정파일> 
```