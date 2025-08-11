import asyncio
from typing import Annotated, Optional, Dict, Any
import os
import base64
import json
from pathlib import Path
import io
from dotenv import load_dotenv
from fastmcp import FastMCP
from mcp import ErrorData, McpError
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy
import openai
from PIL import Image

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")



assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"
assert OPENAI_API_KEY is not None, "Please set OPENAI_API_KEY in your .env file"

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


class Hello:
    def __init__(self):
        pass

# --- Room Optimizer Utility Class ---
class RoomOptimizer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    async def process_image(self, base64_data: str) -> str:
        """Process and resize image if needed for OpenAI API."""
        try:
            image_bytes = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Resize if too large
            max_size = 2048
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert back to base64
            buf = io.BytesIO()
            img.save(buf, format="PNG", quality=85, optimize=True)
            processed_bytes = buf.getvalue()
            return base64.b64encode(processed_bytes).decode('utf-8')
            
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Image processing error: {str(e)}"))

    async def analyze_room(self, image_base64: str, desired_vibe: str, room_type: str) -> Dict[str, Any]:
        """Analyze room and provide optimization suggestions."""
        try:
            processed_image = await self.process_image(image_base64)
            
            prompt = f"""
            Analyze this {room_type} photo and provide detailed suggestions to optimize it for a {desired_vibe} atmosphere. 

            Please analyze the following aspects:

            1. CURRENT STATE ANALYSIS:
            - Room layout and furniture arrangement
            - Lighting conditions (natural and artificial)
            - Color scheme and visual harmony
            - Clutter level and organization
            - Comfort elements present

            2. OPTIMIZATION SUGGESTIONS:
            Focus on changes that DON'T require major renovations:
            - Furniture rearrangement for better flow and productivity
            - Lighting improvements (lamp placement, utilizing natural light)
            - Organization and decluttering strategies
            - Adding/moving decorative elements for aesthetics
            - Creating designated zones for different activities
            - Comfort improvements (cushions, plants, etc.)

            3. SPECIFIC ACTIONABLE STEPS:
            Provide a prioritized list of 5-8 specific actions they can take today, considering:
            - What to move where and why
            - What to add or remove
            - How to better utilize natural light sources
            - Organization systems to implement
            - Small decorative changes for the desired vibe

            4. PRODUCTIVITY ENHANCEMENTS:
            - Ergonomic improvements
            - Distraction reduction
            - Creating focus zones
            - Storage solutions

            5. AESTHETIC IMPROVEMENTS:
            - Visual balance and symmetry
            - Color coordination with existing elements
            - Creating focal points
            - Adding warmth and personality

            Constraints:
            - No major construction or painting
            - Work with existing furniture and major elements
            - Consider budget-friendly solutions
            - Respect the current room structure and windows

            Please provide practical, actionable advice that can be implemented easily.
            Format your response with clear sections and bullet points for easy reading.
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{processed_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2500,
                temperature=0.7
            )
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "desired_vibe": desired_vibe,
                "room_type": room_type,
                "model_used": "gpt-4o"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }

    async def get_quick_tips(self, image_base64: str, focus_area: str) -> Dict[str, Any]:
        """Get quick focused tips for room improvement."""
        try:
            processed_image = await self.process_image(image_base64)
            
            prompt = f"""
            Look at this room photo and give me 5-7 quick, actionable tips specifically for improving {focus_area}.
            
            Focus on:
            - Simple changes that can be done in under 30 minutes
            - Rearranging existing items
            - Quick organizational fixes
            - Immediate improvements for {focus_area}
            - Budget-friendly solutions
            
            Format as a numbered list with brief explanations.
            Be specific about what to move, add, or change.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{processed_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return {
                "success": True,
                "tips": response.choices[0].message.content,
                "focus_area": focus_area
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tips": None
            }

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

# --- Initialize Room Optimizer ---
room_optimizer = RoomOptimizer(OPENAI_API_KEY)

# --- MCP Server Setup ---
mcp = FastMCP(
    "Enhanced Room & Job Finder MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder ---
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
            f"📝 *Job Description Analysis*\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: *{user_goal}*\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 *Fetched Job Posting from URL*: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: *{user_goal}*"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 *Search Results for*: {user_goal}\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))

# Room analysis tools

ROOM_ANALYZER_DESCRIPTION = RichToolDescription(
    description="Analyze room photos and provide detailed optimization suggestions for productivity and aesthetics.",
    use_when="Use when user provides a room photo and wants comprehensive advice on improving the space layout, productivity, comfort, or visual appeal.",
    side_effects="Generates detailed room analysis with actionable steps using OpenAI Vision API.",
)

@mcp.tool(description=ROOM_ANALYZER_DESCRIPTION.model_dump_json())
async def room_analyzer(
    puch_image_data: Annotated[str, Field(description="Base64-encoded room image data to analyze")] = None,
    desired_vibe: Annotated[str, Field(description="The atmosphere/vibe you want to achieve (e.g., 'cozy and productive', 'minimalist', 'warm and inviting')")] = "productive and cozy",
    room_type: Annotated[str, Field(description="Type of room (bedroom, office, living room, kitchen, etc.)")] = "general",
) -> str:
    """
    Analyze room photo and provide comprehensive optimization suggestions for better productivity and aesthetics.
    """
    if not puch_image_data:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a room image to analyze."))
    
    try:
        result = await room_optimizer.analyze_room(puch_image_data, desired_vibe, room_type)
        
        if result["success"]:
            return (
                f"🏠 *Room Analysis Complete* 🏠\n\n"
                f"*Room Type*: {room_type}\n"
                f"*Desired Vibe*: {desired_vibe}\n"
                f"*Model Used*: {result['model_used']}\n\n"
                f"---\n\n"
                f"{result['analysis']}\n\n"
                f"---\n\n"
                f"💡 *Next Steps*: Review the actionable steps above and start with the highest priority items. "
                f"Small changes can make a big difference in your space's functionality and appeal!"
            )
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Room analysis failed: {result['error']}"))
    
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error during room analysis: {str(e)}"))

QUICK_ROOM_TIPS_DESCRIPTION = RichToolDescription(
    description="Get quick, focused tips for specific room improvement areas that can be done in 30 minutes or less.",
    use_when="Use when user wants fast, actionable advice for a specific aspect like productivity, organization, aesthetics, or comfort.",
    side_effects="Provides 5-7 quick actionable tips focused on the specified area using OpenAI Vision API.",
)

@mcp.tool(description=QUICK_ROOM_TIPS_DESCRIPTION.model_dump_json())
async def quick_room_tips(
    puch_image_data: Annotated[str, Field(description="Base64-encoded room image data to analyze")] = None,
    focus_area: Annotated[str, Field(description="Specific focus area: 'productivity', 'aesthetics', 'comfort', 'organization', 'lighting'")] = "productivity",
) -> str:
    """
    Get quick, focused tips for improving a specific aspect of your room.
    """
    if not puch_image_data:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide a room image to analyze."))
    
    valid_focus_areas = ["productivity", "aesthetics", "comfort", "organization", "lighting"]
    if focus_area.lower() not in valid_focus_areas:
        focus_area = "productivity"
    
    try:
        result = await room_optimizer.get_quick_tips(puch_image_data, focus_area)
        
        if result["success"]:
            return (
                f"⚡ *Quick {focus_area.title()} Tips* ⚡\n\n"
                f"🎯 *Focus Area*: {focus_area.title()}\n"
                f"⏱️ *Time Required*: 30 minutes or less per tip\n\n"
                f"---\n\n"
                f"{result['tips']}\n\n"
                f"---\n\n"
                f"✅ *Pro Tip*: Start with tip #1 and work your way down. "
                f"These quick changes can have an immediate impact on your space!"
            )
        else:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Quick tips generation failed: {result['error']}"))
    
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unexpected error generating quick tips: {str(e)}"))

# Image processing tools

MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")] = None,
) -> list[TextContent | ImageContent]:
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

# --- Run MCP Server ---
async def main():
    print("🚀 Starting Enhanced MCP server on http://0.0.0.0:8086")
    print("🏠 Room optimization tools available!")
    print("💼 Job finder tools available!")
    print("🎨 Image processing tools available!")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "_main_":
    asyncio.run(main())