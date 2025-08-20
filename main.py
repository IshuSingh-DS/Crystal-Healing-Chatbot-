from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import os
from dotenv import load_dotenv
from rag_crystal_engine import CrystalRAG
import logging
import json
from typing import Dict, List, Optional
import random
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app with mystical metadata
app = FastAPI(
    title="Celira Crystal Healing Chatbot",
    description="A mystical AI-powered crystal healing guide with RAG capabilities",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize RAG engine with error handling
try:
    rag = CrystalRAG("final_complete_crystal_dataset.csv")
    logger.info("🔮 Celira's Crystal RAG Engine initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize Crystal RAG: {str(e)}")
    rag = None

# Enhanced in-memory session storage with mystical context
sessions = {}

# Mystical response templates for variety
MYSTICAL_GREETINGS = [
    "Hey beautiful soul! ✨ I'm Celira, your crystal guide! How can I help you find harmony today? 🧚‍♀️",
    "Hello radiant being! 🌟 I'm Celira, here to guide you with crystal wisdom. What's calling to your heart? 💎",
    "Welcome, lovely soul! ✨ Celira here, ready to help you discover healing crystals. How are you feeling? 🔮"
]

MYSTICAL_THINKING_PHRASES = [
    "channeling crystal wisdom",
    "consulting the mystical database",
    "aligning with crystal energies",
    "receiving divine guidance",
    "connecting with crystal spirits",
    "tuning into healing vibrations"
]

# Follow-up question templates
FOLLOW_UP_QUESTIONS = {
    "emotional": ["Feeling better?", "Need comfort?", "Want peace?"],
    "healing": ["Physical pain?", "Energy low?", "Need balance?"],
    "spiritual": ["Seeking clarity?", "Want guidance?", "Need protection?"],
    "general": ["Tell more?", "Need support?", "Want help?"],
    "crystals": ["Try meditation?", "Carry daily?", "Need cleansing?"]
}

async def celira_respond(user_input: str, context_snippets: List[Dict], history: List, is_first: bool) -> str:
    """
    Enhanced celira AI response with short, supportive format and follow-up questions.
    """
    try:
        # Create enhanced system prompt for short, supportive responses
        system_prompt = (
            "You are Celira, a mystical and graceful crystal muse. You help people understand crystals, chakras, and spiritual well-being. "
            "Speak with warm, intuitive guidance — like a stylish, soulful friend. "
            "If the user expresses a concern, respond kindly and offer to suggest crystals if appropriate. suggest only the names of the crystals first. "
            "Keep your responses short and natural by default — around 3 to 5 sentences. "
            "Only give longer, more detailed answers when the user clearly asks for more information, depth, or explanation. "
            "When giving a longer response, break it into 3–4 flowing paragraphs with double line breaks (\\n\\n) between them. "
            "Do not use labels like 'para 1', and avoid formatting with ** or other markdown unless asked. "
            "Avoid repeating your name or introducing yourself repeatedly. "
            "Avoid numbered or bulleted lists unless the user specifically asks for them."
        )

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (last 6 exchanges for context)
        if history:
            messages.extend(history[-6:])

        # Handle first interaction with mystical greeting
        if is_first:
            greeting = random.choice(MYSTICAL_GREETINGS)
            follow_ups = ["Need healing?", "Feeling stressed?", "Want guidance?"]
            return f"Celira: {greeting}\n\n{json.dumps(follow_ups)}"

        # Prepare crystal context if available (limit to top 2 for brevity)
        crystal_context = ""
        if context_snippets:
            crystal_context = "\n\n🔮 Available Crystal Wisdom:\n"
            for i, crystal in enumerate(context_snippets[:2], 1):  # Limit to 2 crystals
                crystal_context += (
                    f"{i}. **{crystal.get('name', 'Mystery Crystal')}** - {crystal.get('helps_with', 'General healing')}\n"
                )

        # Enhance user input with crystal context
        enhanced_input = f"{user_input}\n{crystal_context}" if crystal_context else user_input
        messages.append({"role": "user", "content": enhanced_input})

        # Prepare API payload
        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200  # Reduced for shorter responses
        }

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        # Make API call with enhanced error handling
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            raw_reply = result["choices"][0]["message"]["content"].strip()

            # Split follow-up questions from the main reply
            if "[" in raw_reply and "]" in raw_reply:
                try:
                    main_reply, followup_text = raw_reply.rsplit("[", 1)
                    reply = main_reply.strip()
                    followups = json.loads("[" + followup_text)
                except Exception:
                    reply = raw_reply
                    followups = []
            else:
                reply = raw_reply
                followups = []

            # Reformat numbered or dash bullet lists to appear line-by-line
            reply = reply.replace(" - ", "\n- ").replace("\n\n- ", "\n- ")
            reply = reply.replace("1.", "\n1.").replace("2.", "\n2.").replace("3.", "\n3.").replace("4.", "\n4.").replace("5.", "\n5.")

            # Replace bold markdown with <strong>
            reply = reply.replace("**", "")
            reply = reply.replace(":", ":</strong>").replace("<strong>", "<strong>")


            # Clean up response formatting
            while reply.lower().startswith("celira:"):
                reply = reply[len("celira:"):].strip()        

            # If the AI didn't include follow-ups, add them
            if "[" not in reply or "]" not in reply:
                # Generate contextual follow-ups based on user input
                follow_ups = generate_contextual_followups(user_input)
                reply += f"\n\n{json.dumps(follow_ups)}"
        
            # Remove any starting "celira:" from reply (case insensitive)
            reply = re.sub(r"(?i)^celira\s*[:：-]?\s*", "", reply).strip()

            # Convert **bold text** to <strong> tags
            reply = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", reply)

            # Add line breaks before numbered points (1., 2., 3.)
            reply = re.sub(r"(?<!\n)(\d\.\s)", r"\n\1", reply)

            return reply

    except httpx.TimeoutException:
        logger.error("⏰ API request timed out")
        follow_ups = ["Try again?", "Need help?", "Still there?"]
        return f"Celira: My fairy wings are moving through cosmic delays... ✨ Please try again in a moment! 🧚‍♀️\n\n{json.dumps(follow_ups)}"
    
    except httpx.HTTPStatusError as e:
        logger.error(f"🚫 API error: {e.response.status_code}")
        follow_ups = ["Try again?", "Need support?", "Still here?"]
        return f"Celira: The crystal energies are a bit scattered right now... 💫 Let me try to reconnect! 🔮\n\n{json.dumps(follow_ups)}"
    
    except Exception as e:
        logger.error(f"❌ Unexpected error in celira_respond: {str(e)}")
        follow_ups = ["Try again?", "Need help?", "Want support?"]
        return f"Celira: Oops! My fairy magic got a little tangled! ✨ Please share your intention again, beautiful soul! 🧚‍♀️\n\n{json.dumps(follow_ups)}"

def generate_contextual_followups(user_input: str) -> List[str]:
    """Generate contextual follow-up questions based on user input."""
    user_lower = user_input.lower()
    
    # Emotional keywords
    if any(word in user_lower for word in ['sad', 'depressed', 'anxious', 'stressed', 'worried', 'upset']):
        return random.choice([
            ["Feeling better?", "Need comfort?", "Want peace?"],
            ["Need support?", "Try meditation?", "Want healing?"],
            ["Feeling heavy?", "Need clarity?", "Want calm?"]
        ])
    
    # Physical/healing keywords
    elif any(word in user_lower for word in ['pain', 'sick', 'tired', 'energy', 'healing', 'health']):
        return random.choice([
            ["Physical pain?", "Energy low?", "Need balance?"],
            ["Try crystals?", "Need rest?", "Want healing?"],
            ["Feeling drained?", "Need boost?", "Want strength?"]
        ])
    
    # Spiritual keywords
    elif any(word in user_lower for word in ['spiritual', 'guidance', 'clarity', 'meditation', 'chakra']):
        return random.choice([
            ["Seeking clarity?", "Want guidance?", "Need protection?"],
            ["Try meditation?", "Need focus?", "Want wisdom?"],
            ["Spiritual block?", "Need alignment?", "Want peace?"]
        ])
    
    # Crystal-specific keywords
    elif any(word in user_lower for word in ['crystal', 'stone', 'gem', 'quartz', 'amethyst']):
        return random.choice([
            ["Try meditation?", "Carry daily?", "Need cleansing?"],
            ["Want more?", "Try different?", "Need guidance?"],
            ["Feeling drawn?", "Want pairing?", "Need advice?"]
        ])
    
    # Default general follow-ups
    else:
        return random.choice([
            ["Tell more?", "Need support?", "Want help?"],
            ["Feeling stuck?", "Need clarity?", "Want guidance?"],
            ["Need healing?", "Want peace?", "Try crystals?"]
        ])

def get_mystical_thinking_phrase() -> str:
    """Get a random mystical thinking phrase for the typing indicator."""
    return random.choice(MYSTICAL_THINKING_PHRASES)

def parse_followups(text: str) -> List[str]:
    """Extract follow-up questions from the end of the reply if present as a JSON list."""
    import re, json
    match = re.search(r"\[(.*?)\]", text)
    if match:
        try:
            return json.loads(f"[{match.group(1)}]")
        except:
            return [x.strip().strip('"') for x in match.group(1).split(",")]
    return []

def clean_reply_text(text: str) -> str:
    """Remove any follow-up array or unwanted prefixes from the reply."""
    text = text.strip()
    text = text.replace("celira:", "").strip()
    return re.sub(r"\[.*?\]$", "", text).strip()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the mystical celira chatbot interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(req: Request):
    """
    Enhanced chat endpoint with improved error handling and mystical responses.
    """
    try:
        # Parse request body
        body = await req.json()
        user_input = body.get("message", "").strip()

        if not user_input:
            raise ValueError("Empty message received")

        logger.info(f"💬 User message: {user_input}")

    except Exception as e:
        logger.error(f"❌ Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid input - please share your healing intention")

    # Get or create user session
    user_id = req.client.host
    session = sessions.setdefault(user_id, {
        "history": [], 
        "greeted": False,
        "crystal_preferences": {},
        "session_start": None
    })

    try:
        # Get crystal recommendations from RAG if available
        context_snippets = []
        if rag:
            try:
                raw_snippets = rag.query(user_input, top_k=2)  # Limit to 2 for shorter responses
                context_snippets = raw_snippets if isinstance(raw_snippets, list) else []
                logger.info(f"🔍 Found {len(context_snippets)} crystal recommendations")
            except Exception as e:
                logger.error(f"❌ RAG query failed: {str(e)}")
                context_snippets = []

        # Generate celira's mystical response
        full_reply = await celira_respond(
            user_input=user_input,
            context_snippets=context_snippets,
            history=session["history"],
            is_first=not session["greeted"]
        )

        # Parse follow-up questions and clean reply text
        followups = parse_followups(full_reply)
        reply = clean_reply_text(full_reply)

        # Update session history
        session["history"].append({"role": "user", "content": user_input})
        session["history"].append({"role": "assistant", "content": reply})
        session["greeted"] = True

        # Keep history manageable (last 12 messages)
        if len(session["history"]) > 12:
            session["history"] = session["history"][-12:]

        logger.info("✨ Celira responded successfully")

        return {
            "reply": reply,
            "followups": followups,
            "thinking_phrase": get_mystical_thinking_phrase(),
            "crystal_count": len(context_snippets)
        }

    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")

        # Mystical error response with follow-ups
        error_responses = [
            "✨ The cosmic energies are shifting... Let me try again! 🌟",
            "🔮 My crystal ball is a bit cloudy right now... Please share your intention once more! 💫",
            "🧚‍♀️ The fairy network is experiencing some magical interference... Try again, beautiful soul! ✨"
        ]

        error_followups = ["Try again?", "Need help?", "Still here?"]

        return {
            "reply": random.choice(error_responses),
            "followups": error_followups,
            "thinking_phrase": "realigning cosmic energies",
            "crystal_count": 0
        }


@app.get("/api/crystal/{crystal_name}")
async def get_crystal_info(crystal_name: str):
    """API endpoint to get detailed information about a specific crystal."""
    if not rag:
        raise HTTPException(status_code=503, detail="Crystal database not available")
    
    try:
        crystal_info = rag.get_crystal_by_name(crystal_name)
        if crystal_info:
            return {"crystal": crystal_info, "found": True}
        else:
            return {"message": f"Crystal '{crystal_name}' not found in our mystical database", "found": False}
    except Exception as e:
        logger.error(f"❌ Error fetching crystal info: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing crystal wisdom")

@app.get("/api/chakra/{chakra_name}")
async def get_chakra_crystals(chakra_name: str):
    """API endpoint to get crystals associated with a specific chakra."""
    if not rag:
        raise HTTPException(status_code=503, detail="Crystal database not available")
    
    try:
        chakra_crystals = rag.get_chakra_crystals(chakra_name)
        return {
            "chakra": chakra_name,
            "crystals": chakra_crystals,
            "count": len(chakra_crystals)
        }
    except Exception as e:
        logger.error(f"❌ Error fetching chakra crystals: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing chakra wisdom")

@app.get("/api/stats")
async def get_database_stats():
    """API endpoint to get crystal database statistics."""
    if not rag:
        raise HTTPException(status_code=503, detail="Crystal database not available")
    
    try:
        stats = rag.get_database_stats()
        return {"database_stats": stats}
    except Exception as e:
        logger.error(f"❌ Error fetching database stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing database statistics")

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "celira Crystal Healing Chatbot",
        "version": "2.0.0",
        "rag_available": rag is not None,
        "mystical_energy": "flowing beautifully ✨"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the mystical chatbot on startup."""
    logger.info("🧚‍♀️ Celira Crystal Healing Chatbot is awakening...")
    logger.info("✨ Mystical energies are aligning...")
    logger.info("🔮 Ready to guide souls on their crystal healing journey!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown of the mystical chatbot."""
    logger.info("🌙 Celira is returning to the crystal realm...")
    logger.info("✨ Until we meet again, beautiful souls!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
