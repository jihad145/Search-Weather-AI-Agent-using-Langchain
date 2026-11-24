
import os
import requests
import certifi

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent


# Environment setup
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv(override=True)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# Weather function using Open-Meteo
def fetch_weather(city):

    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    geo_response = requests.get(
        geo_url,
        params=geo_params,
        timeout=15
    )
    geo_response.raise_for_status()

    geo_data = geo_response.json()

    if not geo_data.get("results"):
        return f"Could not find location: {city}"

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    country = location.get("country", "Unknown")
    city_name = location["name"]

    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "weather_code,"
            "wind_speed_10m,"
            "precipitation"
        ),
        "timezone": "auto"
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=15
    )
    weather_response.raise_for_status()

    weather_data = weather_response.json()
    current = weather_data["current"]

    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Heavy thunderstorm with hail"
    }

    weather = weather_codes.get(
        current["weather_code"],
        "Unknown weather"
    )

    return {
        "city": city_name,
        "country": country,
        "temperature": current["temperature_2m"],
        "feels_like": current["apparent_temperature"],
        "humidity": current["relative_humidity_2m"],
        "wind_speed": current["wind_speed_10m"],
        "precipitation": current["precipitation"],
        "weather": weather,
        "time": current["time"]
    }


# LangChain Weather Tool
@tool
def get_weather_data(city: str):
    """Get current weather information for a city using Open-Meteo."""

    try:
        data = fetch_weather(city)

        if isinstance(data, str):
            return data

        return (
            f"City: {data['city']}\n"
            f"Country: {data['country']}\n"
            f"Temperature: {data['temperature']}°C\n"
            f"Feels Like: {data['feels_like']}°C\n"
            f"Humidity: {data['humidity']}%\n"
            f"Wind Speed: {data['wind_speed']} km/h\n"
            f"Precipitation: {data['precipitation']} mm\n"
            f"Weather: {data['weather']}\n"
            f"Local Time: {data['time']}"
        )

    except requests.exceptions.RequestException as error:
        return f"Weather API connection error: {error}"

    except (KeyError, ValueError, TypeError) as error:
        return f"Weather data processing error: {error}"


# Tavily Search Tool
tools = [get_weather_data]

if TAVILY_API_KEY:
    search_tool = TavilySearchResults(max_results=2)
    tools.append(search_tool)


# Ollama Model
ollama_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url="http://localhost:11434"
)


# Create AI Agent
ollama_agent = create_agent(
    model=ollama_llm,
    tools=tools,
    system_prompt="""
    You are a helpful AI assistant.

    Instructions:
    1. Use get_weather_data for current weather questions.
    2. Use Tavily search for general web search questions when available.
    3. Analyze tool results carefully.
    4. Do not invent weather information.
    5. Always provide a clear and helpful answer.
    """
)


# Function for app.py
def ask_agent(question: str):

    response = ollama_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    return response["messages"][-1].content
