
# Egyptian ID Data Extractor

This Django application helps extract data from Egyptian national IDs written in Arabic using pre-trained models. The application leverages **EasyOCR** and **OpenCV** to process images of IDs and extract important details such as name, ID number, birth date, and more.

## Features

- **OCR Processing:** Uses **EasyOCR** to recognize Arabic text on Egyptian IDs.
- **Image Processing:** Utilizes **OpenCV** for image enhancement and preprocessing to improve OCR accuracy.
- **Data Extraction:** Extracts relevant data fields from the ID, including:
  - First Name (in Arabic)
  - Last Name (in Arabic)
  - Address 1
  - Address 2
  - Job Title
  - Job Place
  - Marital Status
  - National ID number
  - Expiry Date
- **Django Integration:** Provides a web interface using Django, making it easy to upload ID images and view extracted information.
  
## Prerequisites

- Python 3.8+
- Django 3.0+
- OpenCV (`opencv-python`)
- EasyOCR

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/egyptian-id-extractor.git
   cd egyptian-id-extractor
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the Django application:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

5. Open the application in your browser at `http://127.0.0.1:8000`.

## Usage

1. Upload an image of an Egyptian ID through the web interface.
2. The application processes the image using OpenCV for pre-processing and EasyOCR for text recognition.
3. Extracted information is displayed on the web page for review.

## Example Output

After uploading an ID image, the application will display extracted information such as:

![screenshot][media/Screenshot from 2024-09-16 12-24-09.png]

## Technologies Used

- **Django:** Backend framework to handle web requests and render pages.
- **EasyOCR:** Pre-trained model for recognizing text in images, specifically tailored to support Arabic text.
- **OpenCV:** Image processing library for enhancing ID image quality before OCR.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request to contribute to the project.


## Contact

For any questions or feedback, please reach out to mohamedhassam2908@gmail.com.
