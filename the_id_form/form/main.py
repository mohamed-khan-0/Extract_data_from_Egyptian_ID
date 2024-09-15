import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
# import time

global reader 
reader = None

def crop_image(image, top_left, bottom_right):
    """
    Crops the image based on the provided top-left and bottom-right coordinates.
    
    Parameters:
        image (numpy.ndarray): The input image.
        top_left (tuple): Coordinates of the top-left corner of the rectangle (x1, y1).
        bottom_right (tuple): Coordinates of the bottom-right corner of the rectangle (x2, y2).
    
    Returns:
        cropped_image (numpy.ndarray): The cropped portion of the image.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Crop the image using the provided coordinates
    cropped_image = image[y1:y2, x1:x2]
    
    return cropped_image


def order_points(pts):
    # Initialize a list of coordinates that will be ordered: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, the bottom-right will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference, the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # or the top-right and top-left
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # or the top-left and bottom-left
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct the set of destination points to obtain a
    # "birds eye view", (i.e. top-down view of the image)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Apply the perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def extract_id_card(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # start_time = time.time()

    # i=0
    # j=255
    # while i<129 and j>127:
    #     edged = cv2.Canny(blurred, i, j)
    #     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 1:
    #         break
    #     i+=2
    #     edged = cv2.Canny(blurred, i, j)
    #     contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) == 1:
    #         break
    #     j-=2
   
    # end_time = time.time()
    # print("While time:", end_time - start_time)


    # start_time = time.time()
    for i in range(0, 129, 2):
        for j in range(255, 127, -2):
            # Use edge detection
            edged = cv2.Canny(blurred, i, j)
            # Find contours
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 1:
                break
        if len(contours) == 1:
            break
    # end_time = time.time()
    # print("For time:", end_time - start_time)


    if len(contours) > 1:
        print("improve quality")

    # cv2.imshow('edged', edged)
    # cv2.imwrite('edged.jpg', edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    
    # Draw contours
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # contourDraw = cv2.resize(image, (1920, 1080)) 
    # cv2.imshow('Contours', contourDraw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Find the largest rectangle-like contour
    largest_contour = None
    max_area = 0
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:
                largest_contour = approx
                max_area = area
    
    # If a rectangle was found
    if largest_contour is not None:
        # Extract the ID card by applying a four-point transform
        warped = four_point_transform(original, largest_contour.reshape(4, 2))
        # Resize the warped image to 1920x1080
        out = cv2.resize(warped, (1920, 1080))
        # Save the warped image (aligned and cropped ID card)
        cv2.imwrite(output_path, out)
        print(f"ID card extracted and saved to {output_path}")
        return out
    else:
        print("No ID card detected.")
        return original


def text_recognition(image, type=''):
    global reader
    if reader is None:
        # Initialize EasyOCR reader with Arabic language support
        reader = easyocr.Reader(['ar'], gpu=False)
    # Convert the image to RGB (OpenCV loads images in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if type == 'expiry_date' or type == 'id_number':
        result = reader.readtext(image_rgb, detail = 0, paragraph=True, width_ths=30, allowlist = '٠١٢٣٤٥٦٧٨٩', blocklist='0123456789')
    else:
        # Perform OCR on the image
        result = reader.readtext(image_rgb, detail = 0, paragraph=True, width_ths=25, blocklist='0123456789')
    if len(result) == 0:
        print("No text detected.")
        return None
    #print(" ".join(result))
    return result

def front_text_recognition(image_path):
    image = extract_id_card(image_path, 'media/front_id_card.jpg')
    # Define the dimensions of the first name
    first_name_dim = [(730, 265), (1840, 390)]
    # Define the dimensions of the last name
    last_name_dim = [(730, 375), (1840, 510)]
    # Define the dimensions of the address1
    address1_dim = [(730, 485), (1840, 630)]
    # Define the dimensions of the address2
    address2_dim = [(730, 610), (1840, 745)]
    # Define the dimensions of the ID number
    id_number_dim = [(730, 835), (1875, 970)]

    cv2.rectangle(image, first_name_dim[0], first_name_dim[1], (255, 0, 0), 1)
    cv2.rectangle(image, last_name_dim[0], last_name_dim[1], (0, 255, 0), 1)
    cv2.rectangle(image, address1_dim[0], address1_dim[1], (0, 0, 255), 1)
    cv2.rectangle(image, address2_dim[0], address2_dim[1], (0, 0, 0), 1)
    cv2.rectangle(image, id_number_dim[0], id_number_dim[1], (255, 255, 255), 1)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    first_name_image = crop_image(image, first_name_dim[0], first_name_dim[1])
    last_name_image = crop_image(image, last_name_dim[0], last_name_dim[1])
    address1_image = crop_image(image, address1_dim[0], address1_dim[1])
    address2_image = crop_image(image, address2_dim[0], address2_dim[1])
    id_number_image = crop_image(image, id_number_dim[0], id_number_dim[1])

    first_name = text_recognition(first_name_image)
    last_name = text_recognition(last_name_image)
    address1 = text_recognition(address1_image)
    address2 = text_recognition(address2_image)
    id_number = text_recognition(id_number_image, type='id_number')

    data = {
        "first_name": " ".join(first_name) if first_name else first_name,
        "last_name": " ".join(last_name) if last_name else last_name,
        "address1": " ".join(address1) if address1 else address1,
        "address2": " ".join(address2) if address2 else address2,
        "id_number": " ".join(id_number) if id_number else id_number
    }

    print(data)
    return data

def back_text_recognition(image_path):
    image = extract_id_card(image_path, 'media/back_id_card.jpg')
    # Define the dimensions of the job title
    job_title_dim = [(920, 120), (1600, 220)]
    # Define the dimensions of the job place
    job_place_dim = [(650, 210), (1600, 320)]
    # Define the dimensions of the marital status
    marital_status_dim = [(620, 300), (875, 420)]
    # Define the dimensions of the expire date
    expire_date_dim = [(600, 480), (1050, 600)]

    cv2.rectangle(image, job_title_dim[0], job_title_dim[1], (255, 0, 0), 1)
    cv2.rectangle(image, job_place_dim[0], job_place_dim[1], (0, 0, 0), 1)
    cv2.rectangle(image, marital_status_dim[0], marital_status_dim[1], (0, 255, 0), 1)
    cv2.rectangle(image, expire_date_dim[0], expire_date_dim[1], (0, 0, 255), 1)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    job_title_image = crop_image(image, job_title_dim[0], job_title_dim[1])
    job_place_image = crop_image(image, job_place_dim[0], job_place_dim[1])
    marital_status_image = crop_image(image, marital_status_dim[0], marital_status_dim[1])
    expire_date_image = crop_image(image, expire_date_dim[0], expire_date_dim[1])

    job_title = text_recognition(job_title_image)
    job_place = text_recognition(job_place_image)
    marital_status = text_recognition(marital_status_image)
    expire_date = text_recognition(expire_date_image, type='expiry_date')
    try:
        if len(expire_date[0]) == 10:
            expire_date = expire_date[0][0:4] + "/" + expire_date[0][5:7] + "/" + expire_date[0][8:10]

        if len(expire_date[0]) == 7:
            expire_date = "٢٠" + expire_date[0][0:2] + "/" + expire_date[0][3:5] + "/" + expire_date[0][6:7] + "٠"
        
        if len(expire_date[0]) == 8:
            expire_date = "٢٠" + expire_date[0][0:2] + "/" + expire_date[0][3:5] + "/" + expire_date[0][6:8]
    except:
        pass
    data = {
        "job_title": " ".join(job_title) if job_title is not None else job_title,
        "job_place": " ".join(job_place) if job_place is not None else job_place,
        "marital_status": " ".join(marital_status) if marital_status is not None else marital_status,
        "expire_date": "".join(expire_date) if expire_date is not None else expire_date
    }

    print(data)
    return data

# data = front_text_recognition('2.png')
# data.update(back_text_recognition('2.1.png'))
# print(data)
