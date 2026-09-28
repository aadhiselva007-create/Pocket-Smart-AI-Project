import os
import json
from pathlib import Path
from fastapi.templating import Jinja2Templates

# Base directory setup
BASE_DIR = Path(__file__).resolve().parent

# Templates directory path
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
import re
import base64
import shutil
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import uuid
import asyncio
from io import BytesIO

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status, Cookie
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini AI Setup
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("No Google API key found in environment variables. Please set GEMINI_API_KEY in .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-3.8-flash")

# FastAPI app initialization
app = FastAPI(title="PocketSmart: AI Budget Planner")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_123456789")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates configuration
templates = Jinja2Templates(directory="templates")
os.makedirs("static/uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------- In-Memory Data Stores -----------------
users_db: Dict[str, dict] = {}
active_sessions: Dict[str, Any] = {}
blacklisted_tokens = set()
user_recommendations: Dict[str, List[Any]] = {}

# ----------------- Pydantic Models -----------------
class RegisterUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserSession:
    def __init__(self, username, login_time, last_activity, token, user_data=None):
        self.username = username
        self.login_time = login_time
        self.last_activity = last_activity
        self.token = token
        self.user_data = user_data or {}

class HomeBudgetInput(BaseModel):
    total_budget: float
    num_lights: int = 0
    num_fans: int = 0
    num_furniture: int = 0
    num_dining_tables: int = 0
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = "None"

class PartyBudgetInput(BaseModel):
    total_budget: float
    party_type: str
    num_guests: int
    venue_type: Optional[str] = "Not specified"
    needs_catering: bool = False
    needs_decoration: bool = False
    needs_entertainment: bool = False
    additional_requirements: Optional[str] = "None"

class JewelryBudgetInput(BaseModel):
    total_budget: float
    occasion: str
    preferences: Optional[str] = "Not specified"

class RecommendationHistoryItem:
    def __init__(self, rec_id, rec_type, input_data, full_result):
        self.id = rec_id
        self.timestamp = datetime.utcnow().strftime("%B %d, %Y, %I:%M %p")
        self.recommendation_type = rec_type
        self.input_data = input_data
        self.input_summary = f"Budget: ₹{input_data.get('total_budget', 0)}"
        self.result_summary = full_result
        self.full_result = full_result

# ----------------- Helper Functions -----------------
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return request.cookies.get("access_token")

async def get_current_user(request: Request, token: Optional[str] = Depends(get_token)):
    if not token or token in blacklisted_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        # Update session activity
        if username in active_sessions:
            active_sessions[username].last_activity = datetime.utcnow()
        return users_db[username]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

async def get_current_active_user(user: dict = Depends(get_current_user)):
    return user

def extract_json_from_response(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"error": "Failed to parse structured AI output", "raw_text": text}

def save_to_history(username: str, recommendation_type: str, input_data: dict, result: dict):
    if username not in user_recommendations:
        user_recommendations[username] = []
    rec_id = str(uuid.uuid4())[:8]
    item = RecommendationHistoryItem(rec_id, recommendation_type, input_data, result)
    user_recommendations[username].append(item)

# ----------------- AI Recommendation Functions -----------------
def get_home_recommendations(budget_input: HomeBudgetInput) -> dict:
    try:
        prompt = f"""
        I need interior design product recommendations for a home in India with a total budget of ₹{budget_input.total_budget:.2f}.
        Requirements:
        - {budget_input.num_lights} lights/lighting fixtures
        - {budget_input.num_fans} ceiling fans
        - {budget_input.num_furniture} furniture pieces
        - {budget_input.num_dining_tables} dining tables

        Additional rooms to consider:
        {"- Living room" if budget_input.has_living_room else ""}
        {"- Kitchen" if budget_input.has_kitchen else ""}
        {"- Bedroom" if budget_input.has_bedroom else ""}

        Additional requirements: {budget_input.additional_requirements or "None"}

        Please provide a detailed budget breakdown with product recommendations **available in India**.
        Use **Indian brands and pricing**. Include **search terms** suitable for Indian shopping platforms.

        Format your response strictly as JSON with this structure:
        {{
            "total_budget": {budget_input.total_budget:.2f},
            "budget_breakdown": [
                {{
                    "category": "lighting",
                    "allocation": 0.0,
                    "items": [
                        {{
                            "name": "",
                            "description": "",
                            "estimated_price": 0.0,
                            "quantity": 0,
                            "search_terms": ""
                        }}
                    ]
                }}
            ],
            "calculation_table": [
                {{
                    "category": "",
                    "items_count": 0,
                    "total_cost": 0.0,
                    "percentage_of_budget": 0.0
                }}
            ],
            "remaining_budget": 0.0,
            "additional_suggestions": []
        }}
        Ensure total costs stay within budget.
        """
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        # Add links
        for category in result.get("budget_breakdown", []):
            for item in category.get("items", []):
                st = item.get("search_terms", "")
                if st:
                    q = urllib.parse.quote_plus(st)
                    item["shopping_links"] = {
                        "amazon": f"https://www.amazon.in/s?k={q}",
                        "flipkart": f"https://www.flipkart.com/search?q={q}",
                        "ikea": f"https://www.ikea.com/in/en/search/?q={q}",
                        "myntra": f"https://www.myntra.com/search?q={q}",
                        "ajio": f"https://www.ajio.com/search/?text={q}"
                    }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def get_party_recommendations(budget_input: PartyBudgetInput) -> dict:
    try:
        prompt = f"""
        I need party planning recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.
        Party details:
        - Type: {budget_input.party_type}
        - Number of guests: {budget_input.num_guests}
        - Venue type: {budget_input.venue_type or "Not specified"}
        - Catering needed: {"Yes" if budget_input.needs_catering else "No"}
        - Decoration needed: {"Yes" if budget_input.needs_decoration else "No"}
        - Entertainment needed: {"Yes" if budget_input.needs_entertainment else "No"}

        Additional requirements: {budget_input.additional_requirements or "None"}

        Format your response strictly as JSON with this structure:
        {{
            "total_budget": {budget_input.total_budget:.2f},
            "budget_breakdown": [
                {{
                    "category": "venue",
                    "allocation": 0.0,
                    "items": [
                        {{
                            "name": "",
                            "description": "",
                            "estimated_price": 0.0,
                            "quantity": 0,
                            "search_terms": ""
                        }}
                    ]
                }}
            ],
            "venue_suggestions": [
                {{
                    "name": "",
                    "type": "",
                    "capacity": 0,
                    "estimated_cost": 0.0,
                    "search_terms": ""
                }}
            ],
            "remaining_budget": 0.0,
            "additional_suggestions": []
        }}
        Ensure all costs are in INR and total does not exceed the given budget.
        """
        response = model.generate_content(prompt)
        result = extract_json_from_response(response.text)

        result["calculation_table_inr"] = []
        categories = {}
        for category in result.get("budget_breakdown", []):
            cat_name = category.get("category", "Misc")
            if cat_name not in categories:
                categories[cat_name] = {"category": cat_name, "items_count": 0, "total_cost": 0, "percentage_of_budget": 0}
            for itm in category.get("items", []):
                categories[cat_name]["items_count"] += 1
                categories[cat_name]["total_cost"] += itm.get("estimated_price", 0)

        for c in categories.values():
            if result.get("total_budget", 0) > 0:
                c["percentage_of_budget"] = (c["total_cost"] / result["total_budget"]) * 100
            result["calculation_table_inr"].append(c)

        category_platforms = {
            "venue": ["google", "booking", "makemytrip", "oyorooms", "nobroker"],
            "catering": ["swiggy", "zomato"],
            "food": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
            "drinks": ["swiggy", "zomato", "bigbasket", "amazon", "flipkart"],
            "decoration": ["amazon", "flipkart", "meesho", "myntra"],
            "entertainment": ["bookmyshow", "amazon", "flipkart"]
        }

        for category in result.get("budget_breakdown", []):
            cat_name = category.get("category", "").lower()
            rel = category_platforms.get(cat_name, ["amazon", "flipkart", "google"])
            for item in category.get("items", []):
                st = item.get("search_terms", "")
                if st:
                    q = urllib.parse.quote_plus(st)
                    item["shopping_links"] = {}
                    if "amazon" in rel: item["shopping_links"]["amazon"] = f"https://www.amazon.in/s?k={q}"
                    if "flipkart" in rel: item["shopping_links"]["flipkart"] = f"https://www.flipkart.com/search?q={q}"
                    if "swiggy" in rel: item["shopping_links"]["swiggy"] = f"https://www.swiggy.com/search?query={q}"
                    if "zomato" in rel: item["shopping_links"]["zomato"] = f"https://www.zomato.com/search?q={q}"
                    if "bookmyshow" in rel: item["shopping_links"]["bookmyshow"] = f"https://in.bookmyshow.com/search?q={q}"

        for v in result.get("venue_suggestions", []):
            st = v.get("search_terms", "")
            if st:
                q = urllib.parse.quote_plus(st)
                v["search_links"] = {
                    "google": f"https://www.google.com/search?q={q}",
                    "booking": f"https://www.booking.com/search.html?ss={q}",
                    "makemytrip": f"https://www.makemytrip.com/hotels/hotel-listing/?searchText={q}",
                    "oyorooms": f"https://www.oyorooms.com/search/?location={q}"
                }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

def get_jewelry_recommendations(budget_input: JewelryBudgetInput, image_path: Optional[str] = None) -> dict:
    try:
        base_prompt = f"""
        I need jewelry recommendations for India with a total budget of ₹{budget_input.total_budget:.2f}.
        Occasion: {budget_input.occasion}
        Preferences: {budget_input.preferences or "Not specified"}
        Provide only India-relevant styles, availability, and price ranges in INR.
        """
        if image_path:
            img = Image.open(image_path)
            prompt = base_prompt + """
            An image of the outfit is uploaded. Suggest jewelry that complements it, considering color, design, and occasion appropriateness.
            Format the output strictly as JSON:
            {
                "outfit_analysis": {"colors": [], "style": "", "formality": ""},
                "total_budget": 0.0,
                "jewelry_recommendations": [
                    {"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_terms": ""}
                ],
                "remaining_budget": 0.0,
                "styling_tips": []
            }
            """
            response = model.generate_content([prompt, img])
        else:
            prompt = base_prompt + """
            Format the output strictly as JSON:
            {
                "total_budget": 0.0,
                "jewelry_recommendations": [
                    {"item_type": "", "description": "", "style": "", "estimated_price": 0.0, "search_terms": ""}
                ],
                "remaining_budget": 0.0,
                "styling_tips": []
            }
            """
            response = model.generate_content(prompt)

        result = extract_json_from_response(response.text)
        for item in result.get("jewelry_recommendations", []):
            st = item.get("search_terms", "")
            if st:
                q = urllib.parse.quote_plus(st)
                item["shopping_links"] = {
                    "amazon": f"https://www.amazon.in/s?k={q}",
                    "flipkart": f"https://www.flipkart.com/search?q={q}",
                    "bluestone": f"https://www.bluestone.com/search.html?query={q}",
                    "tanishq": f"https://www.tanishq.co.in/search?q={q}",
                    "caratlane": f"https://www.caratlane.com/search?q={q}",
                    "meesho": f"https://www.meesho.com/search?q={q}"
                }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# ----------------- Page UI Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    try:
        token = await get_token(request)
        if token:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass
    return templates.TemplateResponse(request=request, name="login.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    try:
        token = await get_token(request)
        if token:
            user = await get_current_user(request, token)
            if user:
                return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    except Exception:
        pass
    return templates.TemplateResponse(request=request, name="register.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, current_user: dict = Depends(get_current_active_user)):
    user_hist = user_recommendations.get(current_user["username"], [])
    return templates.TemplateResponse(
        request=request, 
        name="dashboard.html", 
        context={"user": current_user, "history": user_hist}
    )

@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request, current_user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="home_planner.html", context={"user": current_user})

@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request, current_user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="party_planner.html", context={"user": current_user})

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request, current_user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="jewelry_planner.html", context={"user": current_user})

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, current_user: dict = Depends(get_current_active_user)):
    return templates.TemplateResponse(request=request, name="history.html", context={"user": current_user})
# ----------------- Auth & Session Endpoints -----------------
@app.post("/register")
async def register_user_endpoint(user: RegisterUser):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")
    users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "password": get_password_hash(user.password)
    }
    return {"message": "User registered successfully"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user["username"]}, expires_delta=access_token_expires)

    if form_data.username in active_sessions:
        old_token = active_sessions[form_data.username].token
        blacklisted_tokens.add(old_token)

    active_sessions[form_data.username] = UserSession(
        username=user["username"],
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        token=access_token
    )

    response = JSONResponse(content={"access_token": access_token, "token_type": "bearer"})
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, samesite="lax")
    return response

@app.post("/logout")
async def logout(request: Request):
    token = await get_token(request)
    if token:
        blacklisted_tokens.add(token)
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username and username in active_sessions:
                del active_sessions[username]
        except JWTError:
            pass
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response

# ----------------- Planner Action Endpoints -----------------
@app.post("/home-budget")
async def plan_home_budget(budget_input: HomeBudgetInput, request: Request, current_user: dict = Depends(get_current_active_user)):
    result = get_home_recommendations(budget_input)
    save_to_history(current_user["username"], "home", budget_input.dict(), result)
    return result

@app.post("/party-budget")
async def plan_party_budget(budget_input: PartyBudgetInput, request: Request, current_user: dict = Depends(get_current_active_user)):
    result = get_party_recommendations(budget_input)
    save_to_history(current_user["username"], "party", budget_input.dict(), result)
    return result

@app.post("/jewelry-budget")
async def plan_jewelry_budget(
    total_budget: float = Form(...),
    occasion: str = Form(...),
    preferences: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    request: Request = None,
    current_user: dict = Depends(get_current_active_user)
):
    budget_input = JewelryBudgetInput(total_budget=total_budget, occasion=occasion, preferences=preferences)
    image_path = None
    if image and image.filename:
        image_path = f"static/uploads/{uuid.uuid4()}_{image.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    result = get_jewelry_recommendations(budget_input, image_path)
    input_data = budget_input.dict()
    if image_path:
        input_data["image"] = image_path
    save_to_history(current_user["username"], "jewelry", input_data, result)
    return result

@app.get("/recommendation-history")
async def get_recommendation_history(request: Request, current_user: dict = Depends(get_current_active_user)):
    history = sorted(user_recommendations.get(current_user["username"], []), key=lambda x: x.timestamp, reverse=True)
    return {"history": [{"id": itm.id, "timestamp": itm.timestamp, "type": itm.recommendation_type, "input": itm.input_summary, "summary": itm.result_summary} for itm in history]}

@app.get("/recommendation-details/{recommendation_id}")
async def get_recommendation_details(recommendation_id: str, current_user: dict = Depends(get_current_active_user)):
    for item in user_recommendations.get(current_user["username"], []):
        if item.id == recommendation_id:
            return {"id": item.id, "timestamp": item.timestamp, "type": item.recommendation_type, "input": item.input_summary, "full_result": item.full_result}
    raise HTTPException(status_code=404, detail="Recommendation not found")

# ----------------- Startup Background Cleanup -----------------
@app.on_event("startup")
async def setup_session_cleanup():
    async def cleanup_expired_sessions():
        while True:
            current_time = datetime.utcnow()
            expired = [u for u, s in list(active_sessions.items()) if (current_time - s.last_activity).total_seconds() > 1800]
            for u in expired:
                del active_sessions[u]
            await asyncio.sleep(300)
    asyncio.create_task(cleanup_expired_sessions())

# ----------------- Main Entry Point -----------------
if __name__ == "__main__":
    import uvicorn
    print("Starting PocketSmart: AI Budget Planner...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)