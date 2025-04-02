# ShelfAware - Real-time Food Detection and Pricing System

A Flask-based web application that uses computer vision to detect food items and display their prices in real-time.

## Features

- Real-time food detection using YOLOv8
- Live camera feed integration
- Automatic price calculation
- Support for multiple food items
- User-friendly web interface

## Supported Food Items and Prices

- Banana: $5
- Black Beans: $4
- Grilled Chicken Breast: $7
- Milk: $2
- Orange Juice: $3
- Pizza: $8
- Potato: $3
- Salad: $5
- Spaghetti: $10
- White Rice: $5

## Prerequisites

- Python 3.9 or higher
- Webcam
- Modern web browser

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ShelfAware
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

4. Allow camera access when prompted by your browser.

## Usage

1. Point your camera at food items
2. The system will detect items and display their prices in real-time
3. Use the "Menu" button to see the list of supported food items and their prices
4. Use the "Confirm" button to generate a bill with the detected items and total price

## Project Structure

- `app.py`: Main Flask application
- `model/`: Contains the trained YOLOv8 model
- `templates/`: HTML templates for the web interface
- `requirements.txt`: Python package dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.