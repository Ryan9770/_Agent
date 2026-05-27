| 옵션 | 설명 |  |
| --- | --- | --- |
| `-f, --file` | 사용할 Compose 파일 지정 (여러 개 가능) |  |
| `-p, --project-name` | Compose 프로젝트 이름 지정 |  |
| `--env-file` | `.env` 말고 다른 환경변수 파일 지정 |  |
| `--profile` | 특정 profile을 활성화 |  |
| `--project-directory` | 프로젝트 디렉토리 강제 지정 |  |

| 명령어 | 설명 |
| --- | --- |
| `up` | 컨테이너 생성 및 실행 (`-d`로 백그라운드 실행 가능) |
| `down` | 컨테이너, 네트워크 제거 |
| `build` | 이미지 빌드 |
| `logs` | 로그 확인 |
| `exec` | 실행 중인 컨테이너에 명령어 실행 |
| `ps` | 현재 상태 확인 |
| `restart` | 컨테이너 재시작 |
| `watch` | 소스 변경 감지 후 자동 리빌드/리프레시 (최신 기능!) |

### 사용 예시

```bash
# 백그라운드로 서비스 실행
docker compose up -d

# 지정된 env 파일과 compose 파일로 실행
docker compose --env-file .env.dev -f docker-compose.dev.yml up -d

# 실행 중인 서비스에 명령 실행 (예: bash 진입)
docker compose exec web bash

# 전체 로그 확인
docker compose logs -f

# 컨테이너 종료 및 자원 제거
docker compose down

# 이미지 확인
docker images
```