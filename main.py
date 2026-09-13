import os
import torch 
import json

torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
import requests
import tempfile

app = FastAPI()

print("Loading model...")
whisper_model = WhisperModel("base", device="cuda", compute_type="int8")
print("Model loaded.")

OLLAMA_API_URL = "http://localhost:11434/api/chat"

@app.post("/api/v1/evaluate")
async def evaluate_audio(file: UploadFile = File(...)):
    
    # 1. 앱에서 보낸 오디오 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        try:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name
        finally:
            temp_audio.close()

    # 2. 스트리밍 데이터를 생성하는 제네레이터 함수
    def generate_feedback():
        try:
            # STT 변환
            print("STT 변환 시작")
            segments, info = whisper_model.transcribe(temp_audio_path, language="en")
            stt_text =  " ".join([segment.text for segment in segments]).strip()
            print(f"인식된 텍스트: {stt_text}")

            # LLM 스트리밍 채점
            print("Qwen AI 실시간 스트리밍 채점 시작...")

            messages = [
                {
                    "role": "system",
                    "content": "You are a professional TOEIC Speaking examiner for Korean students. You MUST write explanations in Korean (한국어). NEVER use Chinese characters. Do not use emojis or emoticons."
                },
                {
                    "role": "user",
                    "content": f"""Evaluate the following [User Text] based on TOEIC Speaking grading criteria. Correct the grammar and provide a natural native-level answer. Follow the EXACT format of the Example.

                    [Example]
                    문법 분석 및 교정:
                    - 오류 발견: 'I is' 부분이 잘못되었습니다.
                    - 상세 설명: 1인칭 단수 주어 'I' 뒤에는 'am'을 사용해야 합니다. 토익스피킹 채점 기준상 이러한 기본적인 주어-동사 수 일치 오류는 문법(Grammar) 항목에서 큰 감점 요인이 되므로 주의해야 합니다.

                    모범 답안:
                    I am a boy.

                    ---
                    [User Text]
                    {stt_text}

                    문법 분석 및 교정:
                    """
                }
            ]
            
            payload = {
                "model": "qwen2.5:3b",
                "messages": messages,
                "stream": True  # Ollama에게 단어 단위로 쪼개서 보내라고 지시
            }

            # Ollama API에 요청을 보내고 스트리밍 응답을 처리
            with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        try:
                            json_data = json.loads(decoded_line)
                            chunk_text = json_data.get("message", "").get("content", "")

                            print(chunk_text, end="", flush=True)

                            chunk_payload = json.dumps({"type": "chunk", "text": chunk_text})
                            yield f"data: {chunk_payload}\n\n"
                        
                        except json.JSONDecodeError:
                            continue

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            print("Qwen AI 실시간 스트리밍 채점 완료.")

        finally:
            try:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
            except Exception as e:
                print(f"임시 파일 삭제 실패: {e}")
                        
    return StreamingResponse(generate_feedback(), media_type="text/event-stream")