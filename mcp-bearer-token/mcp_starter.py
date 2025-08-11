import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

import random
import uuid
import asyncio
import difflib
from typing import Annotated
from pydantic import Field

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))


# Image inputs and sending images

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))


# In-memory quiz store (quiz_id -> dict with 'answer' and 'meta')
QUIZ_STORE: dict[str, dict] = {}
QUIZ_STORE_LOCK = asyncio.Lock()

# --- Rich descriptions ---
LYRICS_QUIZ_DESCRIPTION = RichToolDescription(
    description="Sends a random lyric snippet for the user to guess the song (multilingual).",
    use_when="Use when the user wants to play a music lyric guessing game.",
    side_effects="Returns a lyric snippet, a quiz_id to validate answers later, and optional hints.",
)

CHECK_ANSWER_DESCRIPTION = RichToolDescription(
    description="Check a user's guess for a previously-issued lyrics quiz using quiz_id.",
    use_when="Use after lyrics_quiz has provided a quiz_id and the user submits their guess.",
    side_effects="Returns whether the guess is correct, similarity score, and the correct answer if requested.",
)

# --- Expanded lyrics bank (multilingual) ---
# Each entry: (lyric_snippet, canonical_answer, language, category_tags)
LYRIC_BANK = [
    # EASY - English Pop / Classics
    ("Cause baby you're a firework", "Firework - Katy Perry", "en", ["easy","pop"]),
    ("Hello from the other side", "Hello - Adele", "en", ["easy","pop"]),
    ("Don't stop believin'", "Don't Stop Believin' - Journey", "en", ["easy","classic","rock"]),
    ("We will, we will rock you", "We Will Rock You - Queen", "en", ["easy","rock"]),
    ("Shake it off, shake it off", "Shake It Off - Taylor Swift", "en", ["easy","pop"]),
    ("I got a feeling that tonight's gonna be a good night", "I Gotta Feeling - Black Eyed Peas", "en", ["easy","party"]),

    # MEDIUM - English Rock / Indie / Rap / Ballads
    ("Is this the real life? Is this just fantasy?", "Bohemian Rhapsody - Queen", "en", ["medium","rock"]),
    ("We're just two lost souls swimming in a fish bowl", "Wish You Were Here - Pink Floyd", "en", ["medium","classic"]),
    ("Lose yourself in the music, the moment", "Lose Yourself - Eminem", "en", ["medium","rap"]),
    ("Every breath you take, every move you make", "Every Breath You Take - The Police", "en", ["medium","classic"]),
    ("If I lay here, if I just lay here", "Chasing Cars - Snow Patrol", "en", ["medium","indie"]),

    # HARD - English deeper cuts / older lyrics
    ("The dreams in which I'm dying are the best I've ever had", "Mad World - Tears for Fears (or Gary Jules cover)", "en", ["hard","alternative"]),
    ("Please allow me to introduce myself, I'm a man of wealth and taste", "Sympathy for the Devil - The Rolling Stones", "en", ["hard","rock"]),
    ("You can check out any time you like, but you can never leave", "Hotel California - Eagles", "en", ["hard","classic","rock"]),

    # HINDI - Classic Bollywood
    ("Lag jaa gale, ke phir ye haseen raat ho na ho", "Lag Jaa Gale - Lata Mangeshkar", "hi", ["easy","bollywood","classic"]),
    ("Kabhi kabhi mere dil mein khayal aata hai", "Kabhi Kabhie - Mukesh", "hi", ["easy","bollywood","classic"]),
    ("Mere sapno ki rani kab aayegi tu", "Mere Sapno Ki Rani - Kishore Kumar", "hi", ["easy","bollywood","classic"]),
    ("Chura liya hai tumne jo dil ko", "Chura Liya Hai - Asha Bhosle & Mohammed Rafi", "hi", ["medium","bollywood"]),
    ("Yeh dosti hum nahi todenge", "Yeh Dosti - Sholay (Kishore Kumar, Manna Dey)", "hi", ["easy","bollywood","friendship"]),

    # HINDI - Modern Bollywood / Indie
    ("Tum hi ho, ab tum hi ho", "Tum Hi Ho - Arijit Singh", "hi", ["medium","bollywood","romantic"]),
    ("Apna time aayega", "Apna Time Aayega - Gully Boy (Ranveer Singh)", "hi", ["medium","bollywood","rap"]),
    ("Kar har maidan fateh", "Kar Har Maidan Fateh - Sanju (Sukhwinder Singh, Shreya Ghoshal)", "hi", ["medium","bollywood","motivational"]),

    # MISC multilingual / global hits
    ("Despacito, quiero respirar tu cuello despacito", "Despacito - Luis Fonsi", "es", ["medium","latin","pop"]),
    ("Shape of you, I'm in love with the shape of you", "Shape of You - Ed Sheeran", "en", ["easy","pop"]),
    ("Bole chudiyan, bole kangana", "Bole Chudiyan - Kabhi Khushi Kabhie Gham", "hi", ["easy","bollywood","party"]),

    # Add more as you like...
]

# Utility: filter bank by requested filters
def _select_from_bank(language: str | None, category: str | None, difficulty_hint: str | None):
    candidates = LYRIC_BANK
    if language:
        language = language.lower()
        candidates = [e for e in candidates if e[2].lower() == language]
    if category:
        c = category.lower()
        candidates = [e for e in candidates if c in (tag.lower() for tag in e[3])]
    if difficulty_hint:
        hint = difficulty_hint.lower()
        candidates = [e for e in candidates if hint in (tag.lower() for tag in e[3])]
    return candidates or LYRIC_BANK  # fallback to full bank if filters too strict

# --- Tool: issue a quiz ---
@mcp.tool(description=LYRICS_QUIZ_DESCRIPTION.model_dump_json())
async def lyrics_quiz(
    difficulty: Annotated[str, Field(description="Difficulty: 'easy','medium','hard', or 'any'")] = "any",
    language: Annotated[str | None, Field(description="Language filter, e.g. 'en', 'hi', or 'any'")] = None,
    category: Annotated[str | None, Field(description="Optional category tag, e.g. 'romantic','party'")] = None,
    hint: Annotated[bool, Field(description="If True, return a short hint (artist or year)")] = False,
) -> str:
    """
    Create a lyrics quiz and return a quiz_id with the lyric snippet.
    Store the canonical answer server-side for later validation with check_lyrics_answer.
    """
    difficulty = (difficulty or "any").lower()
    if difficulty not in ("easy", "medium", "hard", "any"):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid difficulty. Use 'easy','medium','hard', or 'any'."))

    # map difficulty to tag usage
    diff_tag = None if difficulty == "any" else difficulty

    candidates = _select_from_bank(language if language and language.lower() != "any" else None,
                                   category if category and category.lower() != "any" else None,
                                   diff_tag)

    lyric_snippet, canonical_answer, lang, tags = random.choice(candidates)

    # Optionally craft a small hint (artist or short hint)
    hint_text = ""
    if hint:
        # try to extract an artist or year from the canonical_answer if present
        # canonical_answer usually formatted like "Song Title - Artist"
        parts = canonical_answer.split(" - ")
        if len(parts) >= 2:
            hint_text = f"Hint: Artist: **{parts[-1]}**"
        else:
            hint_text = "Hint: It's a well-known track."

    # Store quiz with UUID
    quiz_id = str(uuid.uuid4())
    async with QUIZ_STORE_LOCK:
        QUIZ_STORE[quiz_id] = {
            "answer": canonical_answer.lower(),
            "lyric": lyric_snippet,
            "lang": lang,
            "tags": tags,
        }

    response = (
        f"🎵 **Lyrics Quiz**\n\n"
        f"Quiz ID: `{quiz_id}`\n\n"
        f"Guess the song from this lyric:\n> *{lyric_snippet}*\n\n"
        f"_Language: {lang}_  _Tags: {', '.join(tags)}_\n"
    )
    if hint_text:
        response += f"\n{hint_text}\n"
    response += "\nReply with your guess and include the `quiz_id` so I can check it."

    return response

# --- Tool: check an answer ---
@mcp.tool(description=CHECK_ANSWER_DESCRIPTION.model_dump_json())
async def check_lyrics_answer(
    quiz_id: Annotated[str, Field(description="The quiz_id returned by lyrics_quiz")],
    user_guess: Annotated[str, Field(description="User's guess (song title or 'title - artist')")],
    reveal_on_fail: Annotated[bool, Field(description="Return the correct answer when guess is wrong")] = False,
) -> str:
    """
    Check the user's guess against stored canonical answer.
    Uses simple fuzzy matching to allow small variations.
    """
    async with QUIZ_STORE_LOCK:
        entry = QUIZ_STORE.get(quiz_id)

    if not entry:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid or expired quiz_id."))

    canonical = entry["answer"]
    guess = (user_guess or "").strip().lower()

    # direct inclusion or fuzzy match
    if guess == canonical or guess in canonical or canonical in guess:
        correct = True
        score = 1.0
    else:
        # Use difflib for a similarity ratio
        score = difflib.SequenceMatcher(None, guess, canonical).ratio()
        # threshold can be adjusted
        correct = score >= 0.65

    lyric = entry["lyric"]

    if correct:
        msg = (
            f"✅ Correct!\n\n"
            f"Lyric: *{lyric}*\n"
            f"Your guess: *{user_guess}*\n"
            f"Answer: *{entry['answer']}*\n"
            f"Match score: {score:.2f}"
        )
        # optionally remove the quiz to prevent replay
        async with QUIZ_STORE_LOCK:
            if quiz_id in QUIZ_STORE:
                del QUIZ_STORE[quiz_id]
        return msg

    # incorrect
    resp = (
        f"❌ Not quite.\n\n"
        f"Lyric: *{lyric}*\n"
        f"Your guess: *{user_guess}*\n"
        f"Match score: {score:.2f}\n"
    )
    if reveal_on_fail:
        resp += f"\nCorrect answer: *{entry['answer']}*\n"
        # remove quiz now that it's revealed
        async with QUIZ_STORE_LOCK:
            if quiz_id in QUIZ_STORE:
                del QUIZ_STORE[quiz_id]
    else:
        resp += "\nTry again or send `reveal_on_fail=true` to see the answer."

    return resp

# --- Run MCP Server ---
async def main():
    print("🚀 Starting MCP server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())

# import asyncio
# from typing import Annotated, Optional, Dict, Any
# import os
# import base64
# import json
# from pathlib import Path
# import io
# from dotenv import load_dotenv
# from fastmcp import FastMCP
# from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
# from mcp import ErrorData, McpError
# from mcp.server.auth.provider import AccessToken
# from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
# from pydantic import BaseModel, Field, AnyUrl

# import markdownify
# import httpx
# import readabilipy
# import openai
# from PIL import Image

# # --- Load environment variables ---
# load_dotenv()

# TOKEN = os.environ.get("AUTH_TOKEN")
# MY_NUMBER = os.environ.get("MY_NUMBER")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
# assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
# assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY in your .env file"

# # --- Auth Provider ---
# class SimpleBearerAuthProvider(BearerAuthProvider):
#     def __init__(self, token: str):
#         k = RSAKeyPair.generate()
#         super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
#         self.token = token

#     async def load_access_token(self, token: str) -> AccessToken | None:
#         if token == self.token:
#             return AccessToken(
#                 token=token,
#                 client_id="puch-client",
#                 scopes=["*"],
#                 expires_at=None,
#             )
#         return None

# # --- Rich Tool Description model ---
# class RichToolDescription(BaseModel):
#     description: str
#     use_when: str
#     side_effects: str | None = None

# # --- Room Optimizer Utility Class ---
# class RoomOptimizer:
#     def __init__(self, api_key: str):
#         self.client = openai.OpenAI(api_key=api_key)
    
#     async def process_image(self, base64_data: str) -> str:
#         """Process and resize image if needed for OpenAI API."""
#         try:
#             image_bytes = base64.b64decode(base64_data)
#             img = Image.open(io.BytesIO(image_bytes))
            
#             # Resize if too large
#             max_size = 2048
#             if max(img.size) > max_size:
#                 ratio = max_size / max(img.size)
#                 new_size = tuple(int(dim * ratio) for dim in img.size)
#                 img = img.resize(new_size, Image.Resampling.LANCZOS)
            
#             # Convert back to base64
#             buf = io.BytesIO()
#             img.save(buf, format="PNG", quality=85, optimize=True)
#             processed_bytes = buf.getvalue()
#             return base64.b64encode(processed_bytes).decode('utf-8')
            
#         except Exception as e:
#             raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Image processing error: {str(e)}"))

#     async def analyze_room(self, image_base64: str, desired_vibe: str, room_type: str) -> Dict[str, Any]:
#         """Analyze room and provide optimization suggestions."""
#         try:
#             processed_image = await self.process_image(image_base64)
            
#             prompt = f"""
#             Analyze this {room_type} photo and provide detailed suggestions to optimize it for a {desired_vibe} atmosphere. 

#             Please analyze the following aspects:

#             1. CURRENT STATE ANALYSIS:
#             - Room layout and furniture arrangement
#             - Lighting conditions (natural and artificial)
#             - Color scheme and visual harmony
#             - Clutter level and organization
#             - Comfort elements present

#             2. OPTIMIZATION SUGGESTIONS:
#             Focus on changes that DON'T require major renovations:
#             - Furniture rearrangement for better flow and productivity
#             - Lighting improvements (lamp placement, utilizing natural light)
#             - Organization and decluttering strategies
#             - Adding/moving decorative elements for aesthetics
#             - Creating designated zones for different activities
#             - Comfort improvements (cushions, plants, etc.)

#             3. SPECIFIC ACTIONABLE STEPS:
#             Provide a prioritized list of 5-8 specific actions they can take today, considering:
#             - What to move where and why
#             - What to add or remove
#             - How to better utilize natural light sources
#             - Organization systems to implement
#             - Small decorative changes for the desired vibe

#             4. PRODUCTIVITY ENHANCEMENTS:
#             - Ergonomic improvements
#             - Distraction reduction
#             - Creating focus zones
#             - Storage solutions

#             5. AESTHETIC IMPROVEMENTS:
#             - Visual balance and symmetry
#             - Color coordination with existing elements
#             - Creating focal points
#             - Adding warmth and personality

#             Constraints:
#             - No major construction or painting
#             - Work with existing furniture and major elements
#             - Consider budget-friendly solutions
#             - Respect the current room structure and windows

#             Please provide practical, actionable advice that can be implemented easily.
#             Format your response with clear sections and bullet points for easy reading.
#             """

#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": prompt},
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{processed_image}",
#                                     "detail": "high"
#                                 }
#                             }
#                         ]
#                     }
#                 ],
#                 max_tokens=2500,
#                 temperature=0.7
#             )
            
#             return {
#                 "success": True,
#                 "analysis": response.choices[0].message.content,
#                 "desired_vibe": desired_vibe,
#                 "room_type": room_type,
#                 "model_used": "gpt-4o"
#             }
            
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "analysis": None
#             }

#     async def get_quick_tips(self, image_base64: str, focus_area: str) -> Dict[str, Any]:
#         """Get quick focused tips for room improvement."""
#         try:
#             processed_image = await self.process_image(image_base64)
            
#             prompt = f"""
#             Look at this room photo and give me 5-7 quick, actionable tips specifically for improving {focus_area}.
            
#             Focus on:
#             - Simple changes that can be done in under 30 minutes
#             - Rearranging existing items
#             - Quick organizational fixes
#             - Immediate improvements for {focus_area}
#             - Budget-friendly solutions
            
#             Format as a numbered list with brief explanations.
#             Be specific about what to move, add, or change.
#             """
            
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": [
#                             {"type": "text", "text": prompt},
#                             {
#                                 "type": "image_url",
#                                 "image_url": {
#                                     "url": f"data:image/png;base64,{processed_image}",
#                                     "detail": "high"
#                                 }
#                             }
#                         ]
#                     }
#                 ],
#                 max_tokens=1000,
#                 temperature=0.7
#             )
            
#             return {
#                 "success": True,
#                 "tips": response.choices[0].message.content,
#                 "focus_area": focus_area
#             }
            
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "tips": None
#             }

# # --- Fetch Utility Class ---
# class Fetch:
#     USER_AGENT = "Puch/1.0 (Autonomous)"

#     @classmethod
#     async def fetch_url(
#         cls,
#         url: str,
#         user_agent: str,
#         force_raw: bool = False,
#     ) -> tuple[str, str]:
#         async with httpx.AsyncClient() as client:
#             try:
#                 response = await client.get(
#                     url,
#                     follow_redirects=True,
#                     headers={"User-Agent": user_agent},
#                     timeout=30,
#                 )
#             except httpx.HTTPError as e:
#                 raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

#             if response.status_code >= 400:
#                 raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

#             page_raw = response.text

#         content_type = response.headers.get("content-type", "")
#         is_page_html = "text/html" in content_type

#         if is_page_html and not force_raw:
#             return cls.extract_content_from_html(page_raw), ""

#         return (
#             page_raw,
#             f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
#         )

#     @staticmethod
#     def extract_content_from_html(html: str) -> str:
#         """Extract and convert HTML content to Markdown format."""
#         ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
#         if not ret or not ret.get("content"):
#             return "<error>Page failed to be simplified from HTML</error>"
#         content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
#         return content

#     @staticmethod
#     async def google_search_links(query: str, num_results: int = 5) -> list[str]:
#         """
#         Perform a scoped DuckDuckGo search and return a list of job posting URLs.
#         (Using DuckDuckGo because Google blocks most programmatic scraping.)
#         """
#         ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
#         links = []

#         async with httpx.AsyncClient() as client:
#             resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
#             if resp.status_code != 200:
#                 return ["<error>Failed to perform search.</error>"]

#         from bs4 import BeautifulSoup
#         soup = BeautifulSoup(resp.text, "html.parser")
#         for a in soup.find_all("a", class_="result__a", href=True):
#             href = a["href"]
#             if "http" in href:
#                 links.append(href)
#             if len(links) >= num_results:
#                 break

#         return links or ["<error>No results found.</error>"]

# # --- Initialize Room Optimizer ---
# room_optimizer = RoomOptimizer(OPENAI_API_KEY)

# # --- MCP Server Setup ---
# mcp = FastMCP(
#     "Enhanced Room & Job Finder MCP Server",
#     auth=SimpleBearerAuthProvider(TOKEN),
# )

# # --- Tool: validate (required by Puch) ---
# @mcp.tool
# async def validate() -> str:
#     return MY_NUMBER

# # --- Tool: job_finder ---
# JobFinderDescription = RichToolDescription(
#     description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
#     use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
#     side_effects="Returns insights, fetched job descriptions, or relevant job links.",
# )

# @mcp.tool(description=JobFinderDescription.model_dump_json())
# async def job_finder(
#     user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
#     job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
#     job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
#     raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
# ) -> str:
#     """
#     Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
#     """
#     if job_description:
#         return (
#             f"📝 **Job Description Analysis**\n\n"
#             f"---\n{job_description.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**\n\n"
#             f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
#         )

#     if job_url:
#         content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
#         return (
#             f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
#             f"---\n{content.strip()}\n---\n\n"
#             f"User Goal: **{user_goal}**"
#         )

#     if "look for" in user_goal.lower() or "find" in user_goal.lower():
#         links = await Fetch.google_search_links(user_goal)
#         return (
#             f"🔍 **Search Results for**: _{user_goal}_\n\n" +
#             "\n".join(f"- {link}" for link in links)
#         )

#     raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))

# # Room analysis tools

# ROOM_ANALYZER_DESCRIPTION = RichToolDescription(
#     description="Analyze room photos and provide detailed optimization suggestions for productivity and aesthetics.",
#     use_when="Use when user provides a room photo and wants comprehensive advice on improving the space layout, productivity, comfort, or visual appeal.",
#     side_effects="Generates detailed room analysis with actionable steps using OpenAI Vision API.",
# )

# @mcp.tool(description=ROOM_ANALYZER_DESCRIPTION.model_dump_json())
# async def room_analyzer(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded room image data to analyze")] = None,
#     desired_vibe: Annotated[str, Field(description="The atmosphere/vibe you want to achieve (e.g., 'cozy and productive', 'minimalist', 'warm and inviting')")] = "productive and cozy",
#     room_type: Annotated[str, Field(description="Type of room (bedroom, office, living room, kitchen, etc.)")] = "general",
# ) -> str:
#     """
#     Analyze room photo and provide comprehensive optimization suggestions for better productivity and aesthetics.
#     """
#     if not puch_image_data:
#         raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a room image to analyze."))
    
#     try:
#         result = await room_optimizer.analyze_room(puch_image_data, desired_vibe, room_type)
        
#         if result["success"]:
#             return (
#                 f"🏠 **Room Analysis Complete** 🏠\n\n"
#                 f"**Room Type**: {room_type}\n"
#                 f"**Desired Vibe**: {desired_vibe}\n"
#                 f"**Model Used**: {result['model_used']}\n\n"
#                 f"---\n\n"
#                 f"{result['analysis']}\n\n"
#                 f"---\n\n"
#                 f"💡 **Next Steps**: Review the actionable steps above and start with the highest priority items. "
#                 f"Small changes can make a big difference in your space's functionality and appeal!"
#             )
#         else:
#             raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Room analysis failed: {result['error']}"))
    
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error during room analysis: {str(e)}"))

# QUICK_ROOM_TIPS_DESCRIPTION = RichToolDescription(
#     description="Get quick, focused tips for specific room improvement areas that can be done in 30 minutes or less.",
#     use_when="Use when user wants fast, actionable advice for a specific aspect like productivity, organization, aesthetics, or comfort.",
#     side_effects="Provides 5-7 quick actionable tips focused on the specified area using OpenAI Vision API.",
# )

# @mcp.tool(description=QUICK_ROOM_TIPS_DESCRIPTION.model_dump_json())
# async def quick_room_tips(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded room image data to analyze")] = None,
#     focus_area: Annotated[str, Field(description="Specific focus area: 'productivity', 'aesthetics', 'comfort', 'organization', 'lighting'")] = "productivity",
# ) -> str:
#     """
#     Get quick, focused tips for improving a specific aspect of your room.
#     """
#     if not puch_image_data:
#         raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a room image to analyze."))
    
#     valid_focus_areas = ["productivity", "aesthetics", "comfort", "organization", "lighting"]
#     if focus_area.lower() not in valid_focus_areas:
#         focus_area = "productivity"
    
#     try:
#         result = await room_optimizer.get_quick_tips(puch_image_data, focus_area)
        
#         if result["success"]:
#             return (
#                 f"⚡ **Quick {focus_area.title()} Tips** ⚡\n\n"
#                 f"🎯 **Focus Area**: {focus_area.title()}\n"
#                 f"⏱️ **Time Required**: 30 minutes or less per tip\n\n"
#                 f"---\n\n"
#                 f"{result['tips']}\n\n"
#                 f"---\n\n"
#                 f"✅ **Pro Tip**: Start with tip #1 and work your way down. "
#                 f"These quick changes can have an immediate impact on your space!"
#             )
#         else:
#             raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Quick tips generation failed: {result['error']}"))
    
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error generating quick tips: {str(e)}"))

# # Image processing tools

# MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
#     description="Convert an image to black and white and save it.",
#     use_when="Use this tool when the user provides an image and requests it to be converted to black and white.",
#     side_effects="The image will be processed and saved in a black and white format.",
# )

# @mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
# async def make_img_black_and_white(
#     puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
# ) -> list[TextContent | ImageContent]:
#     try:
#         image_bytes = base64.b64decode(puch_image_data)
#         image = Image.open(io.BytesIO(image_bytes))

#         bw_image = image.convert("L")

#         buf = io.BytesIO()
#         bw_image.save(buf, format="PNG")
#         bw_bytes = buf.getvalue()
#         bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

#         return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
#     except Exception as e:
#         raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# # --- Run MCP Server ---
# async def main():
#     print("🚀 Starting Enhanced MCP server on http://0.0.0.0:8086")
#     print("🏠 Room optimization tools available!")
#     print("💼 Job finder tools available!")
#     print("🎨 Image processing tools available!")
#     await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

# if __name__ == "__main__":
#     asyncio.run(main())