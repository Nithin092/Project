


import requests

def get_temperature(api_key, city):
    """
    Fetches the current temperature for the given city using OpenWeatherMap API.
    Returns the temperature value.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  # Use 'imperial' for Fahrenheit
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx, 5xx)
        if response.status_code == 200:
            data = response.json()
            temperature = data['main']['temp']  # Get the temperature value
            return temperature  # Return the temperature value
        else:
            print(f"Failed to fetch temperature for {city}. Error: {response.status_code} - {response.text}")
            return None  # Return None if failed
    except requests.exceptions.RequestException as e:
        # Catch any error related to the network request
        print(f"Error fetching data: {e}")
        return None  # Return None in case of an error

if __name__ == "__main__":
    # Replace 'your_api_key' with your actual OpenWeatherMap API key
    API_KEY = "810366b35b463049ebc14f6b796cbbd7"  # Ensure it's the correct API key
    city = input("Enter the city name: ")

    # Store the fetched temperature in a variable
    temperature = get_temperature(API_KEY, city)
    print (temperature)
    # Check if the temperature was fetched successfully
    if temperature is not None:
        print(f"The current temperature in {city} is {temperature}°C.")
    else:
        print("Could not retrieve the temperature.")
