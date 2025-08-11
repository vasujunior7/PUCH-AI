import openai
import base64
import json
from typing import Optional, Dict, Any
from pathlib import Path
import requests
from PIL import Image
import io

class RoomOptimizer:
    def __init__(self, api_key: str):
        """
        Initialize the Room Optimizer with OpenAI API key.
        
        Args:
            api_key (str): Your OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
        
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for OpenAI API.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def resize_image_if_needed(self, image_path: str, max_size: int = 2048) -> str:
        """
        Resize image if it's too large for API limits.
        
        Args:
            image_path (str): Path to the image
            max_size (int): Maximum dimension size
            
        Returns:
            str: Path to the processed image
        """
        with Image.open(image_path) as img:
            # Check if image needs resizing
            if max(img.size) > max_size:
                # Calculate new size maintaining aspect ratio
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                
                # Resize and save temporarily
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
                temp_path = f"temp_resized_{Path(image_path).name}"
                img_resized.save(temp_path, quality=85, optimize=True)
                return temp_path
        
        return image_path
    
    def analyze_room(self, image_path: str, desired_vibe: str = "productive and cozy", 
                    room_type: str = "general") -> Dict[str, Any]:
        """
        Analyze room photo and provide optimization suggestions.
        
        Args:
            image_path (str): Path to the room photo
            desired_vibe (str): The vibe/atmosphere user wants to achieve
            room_type (str): Type of room (bedroom, office, living room, etc.)
            
        Returns:
            Dict: Analysis results and suggestions
        """
        try:
            # Process image
            processed_image_path = self.resize_image_if_needed(image_path)
            base64_image = self.encode_image(processed_image_path)
            
            # Clean up temp file if created
            if processed_image_path != image_path:
                Path(processed_image_path).unlink(missing_ok=True)
            
            # Create detailed prompt for room analysis
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
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",  # GPT-4 Vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "success": True,
                "analysis": analysis,
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
    
    def get_quick_tips(self, image_path: str, focus_area: str = "productivity") -> Dict[str, Any]:
        """
        Get quick, focused tips for a specific area of improvement.
        
        Args:
            image_path (str): Path to the room photo
            focus_area (str): Specific focus (productivity, aesthetics, comfort, organization)
            
        Returns:
            Dict: Quick tips and suggestions
        """
        try:
            processed_image_path = self.resize_image_if_needed(image_path)
            base64_image = self.encode_image(processed_image_path)
            
            if processed_image_path != image_path:
                Path(processed_image_path).unlink(missing_ok=True)
            
            prompt = f"""
            Look at this room photo and give me 5 quick, actionable tips specifically for improving {focus_area}.
            
            Focus on:
            - Simple changes that can be done in under 30 minutes
            - Rearranging existing items
            - Quick organizational fixes
            - Immediate improvements for {focus_area}
            
            Format as a numbered list with brief explanations.
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
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
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

# Example usage
if __name__ == "__main__":
    # Initialize the room optimizer
    api_key = "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"  # Replace with your actual API key
    optimizer = RoomOptimizer(api_key)
    
    # Example 1: Full room analysis
    result = optimizer.analyze_room(
        image_path="room_photo.jpg",
        desired_vibe="minimalist and productive",
        room_type="home office"
    )
    
    if result["success"]:
        print("Room Analysis:")
        print("=" * 50)
        print(result["analysis"])
    else:
        print(f"Error: {result['error']}")
    
    # Example 2: Quick tips
    quick_tips = optimizer.get_quick_tips(
        image_path="room_photo.jpg",
        focus_area="productivity"
    )
    
    if quick_tips["success"]:
        print("\nQuick Tips:")
        print("=" * 30)
        print(quick_tips["tips"])