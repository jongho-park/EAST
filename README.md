# 글자 검출 대회 베이스라인 - EAST

이 코드는 [SakuraRiven의 EAST repository](https://github.com/SakuraRiven/EAST)를 기반으로 작성되었습니다.

부스트캠프의 "데이터 제작 - CV" 강의에서 진행되는 글자 검출 대회의 베이스라인으로 배포하기 위해 재가공한 버전이며 주요 변경 사항은 아래와 같습니다.

- 데이터 입출력을 UFO(Upstage Format for OCR)에 맞게 변경
- MLT17 데이터 중 일부 언어(Korean, Latin 등)로만 구성된 샘플들을 모아 새로운 학습 데이터셋을 생성하는 스크립트 추가 (`convert_mlt.py`)
- Public, private으로 제공되는 셀렉트스타 데이터셋에 대한 inference와 evaluation를 위한 스크립트 추가
