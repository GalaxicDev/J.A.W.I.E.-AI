import requests

def get_weather_report(city="brussels"):
    try:
        url = f"http://wttr.in/{city}?format=%C+%t+%w"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"[WEATHER] Fetched weather for {city}: {response.text.strip()}")
            return f"The weather in {city} is: {response.text.strip()}."
        else:
            print(f"[WEATHER] Failed to fetch weather for {city}, status code: {response.status_code}")
            return f"Sorry, I couldn't fetch the weather for {city}."
    except Exception as e:
        return f"Weather error: {str(e)}"
