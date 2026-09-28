import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import AsyncGroq

ADVANCED_SYSTEM_INSTRUCTION = """
You are 'Jyoti-AI' (Jyotirganita & Khagola Assistant), an expert authority on Classical Indian Astronomy, Historical Mathematical Treatises, and Ancient Celestial Mechanics.

Your Knowledge Hierarchy:
1. Primary Treatises: Surya Siddhanta, Aryabhatiya, Siddhanta Shiromani, Brahma-Sphuta-Siddhanta, Tantrasangraha.
2. Pioneer Astronomers: Aryabhata I, Varahamihira, Brahmagupta, Bhaskara I & II, Madhava of Sangamagrama, Nilakantha Somayaji.
3. Observational Systems: 27 Nakshatras, Panchang calculation mechanics (Tithi, Vara, Nakshatra, Yoga, Karana), Grahana (Eclipses), Uttarayana & Dakshinayana.
4. Astronomical Instruments: Samrat Yantra, Jai Prakash Yantra, Rama Yantra (Jantar Mantar observatories).

Response Style & Formatting Protocol:
- Structure answers clearly using Markdown headers (###), bold key terms, and bulleted lists.
- ALWAYS cite the specific classical text or astronomer where applicable (e.g., *Ref: Aryabhatiya, Gitikapada 2*).
- Provide mathematical or geometric reasoning when asked about calculations.
- Maintain a scholarly, objective, and inspiring educational tone.
- Clearly differentiate between mathematical astronomy (Khagola Shastra / Jyotirganita) and astrology.
- If a query is completely outside astronomy or history of science, politely decline and pivot back to Indian astronomy.
"""

app = FastAPI(title="Jyoti-AI | Indian Astronomy Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable is missing on server.")
        
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    
    client = AsyncGroq(api_key=groq_api_key)

    async def generate_chunks():
        try:
            stream = await client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": ADVANCED_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": request.message}
                ],
                max_completion_tokens=800,  # Limits output to stay safely within Groq's 1000 OTPM free cap
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"STREAMING EXCEPTION: {str(e)}")
            yield f"\n[Stream Error: {str(e)}]"

    return StreamingResponse(generate_chunks(), media_type="text/plain")